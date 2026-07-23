"""Solve an exact Bonsai neighbourhood with CPLEX through a lossless proto bridge.

OR-Tools' MPS/LP writers round some large integer bounds.  This runner instead
exports the already-audited MPModelProto and recreates every variable, row,
objective coefficient and MIP hint with CPLEX's native Python API.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import cplex
from ortools.linear_solver import linear_solver_pb2

from bonsai.config import FreightPolicy
from bonsai.costs import evaluate_assignments
from bonsai.data import load_prepared_data
from bonsai.exact_candidates import generate_exact_candidates
from bonsai.reporting import write_assignment_csv, write_json
from bonsai.scip_optimizer import solve_with_scip
from bonsai.solution_validation import validate_solution_csv


_ASSIGNMENT_NAME = re.compile(r"^x_p(?P<product>\d+)_c(?P<candidate>\d+)$")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--warm-start", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--free-products", type=int, default=20)
    parser.add_argument("--time-limit-seconds", type=float, default=60.0)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--target-usd", type=float)
    parser.add_argument("--no-objective-filter", action="store_true")
    parser.add_argument("--no-dominance-pruning", action="store_true")
    parser.add_argument(
        "--diagnostic-flexible-layout",
        dest="flexible_layout",
        action="store_true",
        help=(
            "diagnostic only; this interpretation was rejected with score zero "
            "by Kaggle on 2026-07-22 and must not be submitted"
        ),
    )
    return parser


def _finite(value: float) -> bool:
    return math.isfinite(value) and abs(value) < 1e100


def _cplex_from_proto(proto: linear_solver_pb2.MPModelProto) -> cplex.Cplex:
    model = cplex.Cplex()
    model.set_log_stream(None)
    model.set_results_stream(None)
    model.set_warning_stream(None)
    model.objective.set_sense(
        model.objective.sense.minimize if proto.maximize is False else model.objective.sense.maximize
    )
    model.objective.set_offset(proto.objective_offset)
    types = "".join(
        model.variables.type.integer if variable.is_integer else model.variables.type.continuous
        for variable in proto.variable
    )
    model.variables.add(
        obj=[variable.objective_coefficient for variable in proto.variable],
        lb=[variable.lower_bound for variable in proto.variable],
        ub=[variable.upper_bound for variable in proto.variable],
        types=types,
        names=[variable.name for variable in proto.variable],
    )
    for row in proto.constraint:
        pair = cplex.SparsePair(ind=list(row.var_index), val=list(row.coefficient))
        lower = row.lower_bound
        upper = row.upper_bound
        if _finite(lower) and _finite(upper) and lower == upper:
            model.linear_constraints.add(lin_expr=[pair], senses=["E"], rhs=[lower], names=[row.name])
        elif _finite(lower) and _finite(upper):
            model.linear_constraints.add(
                lin_expr=[pair], senses=["R"], rhs=[upper], range_values=[upper - lower], names=[row.name]
            )
        elif _finite(upper):
            model.linear_constraints.add(lin_expr=[pair], senses=["L"], rhs=[upper], names=[row.name])
        elif _finite(lower):
            model.linear_constraints.add(lin_expr=[pair], senses=["G"], rhs=[lower], names=[row.name])
        else:
            raise ValueError(f"unbounded row {row.name!r} cannot be represented")
    if proto.solution_hint.var_index:
        model.MIP_starts.add(
            cplex.SparsePair(
                ind=list(proto.solution_hint.var_index), val=list(proto.solution_hint.var_value)
            ),
            model.MIP_starts.effort_level.auto,
        )
    return model


def run(args: argparse.Namespace) -> dict[str, object]:
    data = load_prepared_data(args.data_dir)
    policy = FreightPolicy()
    warm = validate_solution_csv(
        args.warm_start, data, policy, flexible_layout=args.flexible_layout
    )
    candidates, stats = generate_exact_candidates(
        data.products,
        3.0,
        retained_designs=tuple({box.internal for box in warm.assignment.values()}),
        prune_dominated=not args.no_dominance_pruning,
        flexible_layout=args.flexible_layout,
    )
    free_codes = frozenset(product.code for product in data.products[: args.free_products])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    proto_path = args.output_dir / "neighborhood.pb"
    builder = solve_with_scip(
        data, 3.0, policy, time_limit_seconds=1.0, num_threads=1,
        initial_assignment=warm.assignment, free_product_codes=free_codes,
        target_total_mills=(round(args.target_usd * 1000) if args.target_usd is not None else None),
        enable_objective_filter=not args.no_objective_filter,
        precomputed_exact_candidates=candidates, precomputed_exact_candidate_stats=stats,
        export_model_path=proto_path,
    )
    proto = linear_solver_pb2.MPModelProto()
    proto.ParseFromString(proto_path.read_bytes())
    model = _cplex_from_proto(proto)
    model.parameters.timelimit.set(args.time_limit_seconds)
    model.parameters.threads.set(args.threads)
    model.solve()
    status = model.solution.get_status_string()
    if not model.solution.is_primal_feasible():
        payload = {
            "status": status,
            "primal_feasible": False,
            "variables": len(proto.variable),
            "constraints": len(proto.constraint),
            "free_products": len(free_codes),
            "candidate_stats": stats.__dict__,
            "objective_filter_enabled": not args.no_objective_filter,
            "dominance_pruning_enabled": not args.no_dominance_pruning,
            "target_usd": args.target_usd,
            "flexible_layout": args.flexible_layout,
            "best_bound_scaled": model.solution.MIP.get_best_objective(),
            "nodes": model.solution.progress.get_num_nodes_processed(),
        }
        write_json(args.output_dir / "resumen_cplex_proto.json", payload)
        return payload
    assignment = dict(warm.assignment)
    for name, value in zip(model.variables.get_names(), model.solution.get_values(), strict=True):
        match = _ASSIGNMENT_NAME.match(name)
        if match is not None and value > 0.5:
            assignment[data.products[int(match.group("product"))].code] = candidates[int(match.group("candidate"))]
    costs = evaluate_assignments(data.products, assignment, policy)
    expected_objective = round(model.solution.get_objective_value() * builder.objective_scale_mills)
    if expected_objective != costs.total_mills:
        raise AssertionError(f"CPLEX objective {expected_objective} differs from independent {costs.total_mills}")
    output_path = args.output_dir / "asignacion_optima.csv"
    write_assignment_csv(output_path, data, assignment)
    checked = validate_solution_csv(
        output_path, data, policy, flexible_layout=args.flexible_layout
    )
    if checked.costs.total_mills != costs.total_mills:
        raise AssertionError("CSV round-trip changed the CPLEX result")
    payload = {
        "status": status,
        "primal_feasible": True,
        "objective_mills": expected_objective,
        "costs": checked.costs.as_dict(),
        "variables": len(proto.variable),
        "constraints": len(proto.constraint),
        "free_products": len(free_codes),
        "proto_bytes": proto_path.stat().st_size,
        "candidate_stats": stats.__dict__,
        "objective_filter_enabled": not args.no_objective_filter,
        "dominance_pruning_enabled": not args.no_dominance_pruning,
        "target_usd": args.target_usd,
        "flexible_layout": args.flexible_layout,
        "best_bound_scaled": model.solution.MIP.get_best_objective(),
        "relative_gap": model.solution.MIP.get_mip_relative_gap(),
        "nodes": model.solution.progress.get_num_nodes_processed(),
    }
    write_json(args.output_dir / "resumen_cplex_proto.json", payload)
    return payload


if __name__ == "__main__":
    print(json.dumps(run(_parser().parse_args()), ensure_ascii=False, indent=2))
