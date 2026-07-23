"""Solve a strict Bonsai neighbourhood with Gurobi through an MPModelProto bridge.

The model is built by the existing, independently-audited formulation and is
transferred directly to Gurobi.  This keeps all integer bounds, objective
coefficients and MIP starts intact, without relying on an MPS/LP export.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import gurobipy as gp
from gurobipy import GRB
from ortools.linear_solver import linear_solver_pb2

from bonsai.baseline import standardized_baseline
from bonsai.config import FreightPolicy
from bonsai.costs import evaluate_assignments
from bonsai.data import load_prepared_data
from bonsai.exact_candidates import generate_exact_candidates
from bonsai.reporting import write_assignment_csv, write_json
from bonsai.scip_optimizer import solve_with_scip
from bonsai.solution_validation import ValidationResult, validate_solution_csv


_ASSIGNMENT_NAME = re.compile(r"^x_p(?P<product>\d+)_c(?P<candidate>\d+)$")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--infer-internal-from-external",
        action="store_true",
        help=(
            "diagnostic alternate convention: derive historical internal dimensions "
            "as external minus twice historical thickness"
        ),
    )
    parser.add_argument("--warm-start", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--free-products", type=int, default=20)
    parser.add_argument("--time-limit-seconds", type=float, default=60.0)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--thickness-mm", type=float, choices=(3.0, 4.5, 5.0), default=3.0)
    parser.add_argument("--target-usd", type=float)
    parser.add_argument(
        "--best-obj-stop-usd",
        type=float,
        help="stop as soon as a feasible solution reaches this total cost; keeps the warm start usable",
    )
    parser.add_argument("--no-objective-filter", action="store_true")
    parser.add_argument("--no-dominance-pruning", action="store_true")
    parser.add_argument(
        "--mip-focus",
        type=int,
        choices=(0, 1, 2, 3),
        default=1,
        help="Gurobi MIPFocus: 1 favors feasible solutions; 2/3 favor bounds.",
    )
    parser.add_argument("--mip-gap", type=float, default=1e-4)
    parser.add_argument("--mip-gap-abs", type=float, default=1e-10)
    parser.add_argument("--heuristics", type=float)
    parser.add_argument("--no-rel-heur-time", type=float, default=0.0)
    parser.add_argument("--cuts", type=int, choices=(-1, 0, 1, 2), default=-1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--solution-limit", type=int)
    parser.add_argument("--log-file", type=Path)
    return parser


def _finite(value: float) -> bool:
    return math.isfinite(value) and abs(value) < 1e100


def _optional_model_attr(model: gp.Model, name: str) -> float | None:
    """Return a model attribute when it exists for the solved model type."""

    try:
        return float(model.getAttr(name))
    except (AttributeError, gp.GurobiError):
        return None


def _bound_usd(model: gp.Model, objective_scale_mills: int) -> float | None:
    bound = _optional_model_attr(model, "ObjBound")
    if bound is None:
        return None
    return bound * objective_scale_mills / 1000


def _status_name(status: int) -> str:
    names = {
        GRB.OPTIMAL: "OPTIMAL_TO_TOLERANCE",
        GRB.INFEASIBLE: "INFEASIBLE",
        GRB.INF_OR_UNBD: "INF_OR_UNBD",
        GRB.UNBOUNDED: "UNBOUNDED",
        GRB.CUTOFF: "CUTOFF",
        GRB.TIME_LIMIT: "TIME_LIMIT",
        GRB.SOLUTION_LIMIT: "SOLUTION_LIMIT",
        GRB.INTERRUPTED: "INTERRUPTED",
        GRB.NUMERIC: "NUMERIC",
        GRB.SUBOPTIMAL: "SUBOPTIMAL",
        GRB.USER_OBJ_LIMIT: "USER_OBJ_LIMIT",
        GRB.WORK_LIMIT: "WORK_LIMIT",
        GRB.MEM_LIMIT: "MEM_LIMIT",
    }
    return names.get(status, f"STATUS_{status}")


def _gurobi_from_proto(proto: linear_solver_pb2.MPModelProto) -> tuple[gp.Model, list[gp.Var]]:
    model = gp.Model("bonsai_proto")
    model.Params.OutputFlag = 0
    variables: list[gp.Var] = []
    for variable in proto.variable:
        vtype = GRB.INTEGER if variable.is_integer else GRB.CONTINUOUS
        lower = variable.lower_bound if _finite(variable.lower_bound) else -GRB.INFINITY
        upper = variable.upper_bound if _finite(variable.upper_bound) else GRB.INFINITY
        variables.append(model.addVar(lb=lower, ub=upper, vtype=vtype, name=variable.name))
    model.update()

    for row in proto.constraint:
        expr = gp.LinExpr(list(row.coefficient), [variables[i] for i in row.var_index])
        lower = row.lower_bound
        upper = row.upper_bound
        if _finite(lower) and _finite(upper) and lower == upper:
            model.addConstr(expr == lower, name=row.name)
        elif _finite(lower) and _finite(upper):
            model.addRange(expr, lower, upper, name=row.name)
        elif _finite(upper):
            model.addConstr(expr <= upper, name=row.name)
        elif _finite(lower):
            model.addConstr(expr >= lower, name=row.name)
        else:
            raise ValueError(f"unbounded row {row.name!r} cannot be represented")

    if proto.solution_hint.var_index:
        for index, value in zip(
            proto.solution_hint.var_index, proto.solution_hint.var_value, strict=True
        ):
            variables[index].Start = value

    objective = gp.LinExpr(
        [variable.objective_coefficient for variable in proto.variable], variables
    )
    objective += proto.objective_offset
    model.setObjective(objective, GRB.MAXIMIZE if proto.maximize else GRB.MINIMIZE)
    model.update()
    return model, variables


def run(args: argparse.Namespace) -> dict[str, object]:
    data = load_prepared_data(
        args.data_dir, infer_internal_from_external=args.infer_internal_from_external
    )
    policy = FreightPolicy()
    submitted_warm = validate_solution_csv(args.warm_start, data, policy)
    submitted_thicknesses = {
        candidate.thickness_mm for candidate in submitted_warm.assignment.values()
    }
    if submitted_thicknesses == {args.thickness_mm}:
        warm = submitted_warm
        warm_start_source = "submitted"
    else:
        assignment, costs = standardized_baseline(data, args.thickness_mm, policy)
        # The generated baseline is a direct construction from feasible
        # candidates.  It becomes the valid all-SKU start for a different
        # global carton thickness.
        warm = ValidationResult(assignment=assignment, costs=costs)
        warm_start_source = "standardized_baseline_for_requested_thickness"
    candidates, stats = generate_exact_candidates(
        data.products,
        args.thickness_mm,
        retained_designs=tuple({box.internal for box in warm.assignment.values()}),
        prune_dominated=not args.no_dominance_pruning,
    )
    free_codes = frozenset(product.code for product in data.products[: args.free_products])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    proto_path = args.output_dir / "neighborhood.pb"
    builder = solve_with_scip(
        data,
        args.thickness_mm,
        policy,
        time_limit_seconds=1.0,
        num_threads=1,
        initial_assignment=warm.assignment,
        free_product_codes=free_codes,
        target_total_mills=(round(args.target_usd * 1000) if args.target_usd is not None else None),
        enable_objective_filter=not args.no_objective_filter,
        precomputed_exact_candidates=candidates,
        precomputed_exact_candidate_stats=stats,
        export_model_path=proto_path,
    )
    proto = linear_solver_pb2.MPModelProto()
    proto.ParseFromString(proto_path.read_bytes())
    model, variables = _gurobi_from_proto(proto)
    model.Params.TimeLimit = args.time_limit_seconds
    model.Params.Threads = args.threads
    model.Params.MIPFocus = args.mip_focus
    model.Params.MIPGap = args.mip_gap
    model.Params.MIPGapAbs = args.mip_gap_abs
    model.Params.Cuts = args.cuts
    model.Params.Seed = args.seed
    if args.heuristics is not None:
        model.Params.Heuristics = args.heuristics
    if args.no_rel_heur_time:
        model.Params.NoRelHeurTime = args.no_rel_heur_time
    if args.solution_limit is not None:
        model.Params.SolutionLimit = args.solution_limit
    if args.best_obj_stop_usd is not None:
        # The MIP objective is in scaled mills.  The tiny positive allowance
        # prevents an otherwise qualifying integral solution from missing the
        # stop due solely to floating point conversion.
        model.Params.BestObjStop = (
            args.best_obj_stop_usd * 1000 / builder.objective_scale_mills + 1e-6
        )
    if args.log_file is not None:
        args.log_file.parent.mkdir(parents=True, exist_ok=True)
        model.Params.OutputFlag = 1
        model.Params.LogToConsole = 0
        model.Params.LogFile = str(args.log_file)
    model.optimize()

    has_solution = model.SolCount > 0
    if not has_solution:
        payload = {
            "status": int(model.Status),
            "status_name": _status_name(model.Status),
            "primal_feasible": False,
            "variables": len(proto.variable),
            "constraints": len(proto.constraint),
            "free_products": len(free_codes),
            "candidate_stats": stats.__dict__,
            "objective_filter_enabled": not args.no_objective_filter,
            "dominance_pruning_enabled": not args.no_dominance_pruning,
            "target_usd": args.target_usd,
            "thickness_mm": args.thickness_mm,
            "mip_focus": args.mip_focus,
            "mip_gap": args.mip_gap,
            "mip_gap_abs": args.mip_gap_abs,
            "best_obj_stop_usd": args.best_obj_stop_usd,
            "seed": args.seed,
            "warm_start_source": warm_start_source,
            "infer_internal_from_external": args.infer_internal_from_external,
            "objective_scale_mills": builder.objective_scale_mills,
            "best_bound_scaled": _optional_model_attr(model, "ObjBound"),
            "best_bound_usd": _bound_usd(model, builder.objective_scale_mills),
            "nodes": _optional_model_attr(model, "NodeCount"),
        }
        write_json(args.output_dir / "resumen_gurobi_proto.json", payload)
        return payload

    assignment = dict(warm.assignment)
    for variable in variables:
        match = _ASSIGNMENT_NAME.match(variable.VarName)
        if match is not None and variable.X > 0.5:
            assignment[data.products[int(match.group("product"))].code] = candidates[
                int(match.group("candidate"))
            ]
    costs = evaluate_assignments(data.products, assignment, policy)
    expected_objective = round(model.ObjVal * builder.objective_scale_mills)
    if expected_objective != costs.total_mills:
        raise AssertionError(
            f"Gurobi objective {expected_objective} differs from independent {costs.total_mills}"
        )
    output_path = args.output_dir / "asignacion_optima.csv"
    write_assignment_csv(output_path, data, assignment)
    checked = validate_solution_csv(output_path, data, policy)
    if checked.costs.total_mills != costs.total_mills:
        raise AssertionError("CSV round-trip changed the Gurobi result")
    payload = {
        "status": int(model.Status),
        "status_name": _status_name(model.Status),
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
        "thickness_mm": args.thickness_mm,
        "mip_focus": args.mip_focus,
        "mip_gap": args.mip_gap,
        "mip_gap_abs": args.mip_gap_abs,
        "best_obj_stop_usd": args.best_obj_stop_usd,
        "seed": args.seed,
        "warm_start_source": warm_start_source,
        "infer_internal_from_external": args.infer_internal_from_external,
        "objective_scale_mills": builder.objective_scale_mills,
        "best_bound_scaled": _optional_model_attr(model, "ObjBound"),
        "best_bound_usd": _bound_usd(model, builder.objective_scale_mills),
        "relative_gap": _optional_model_attr(model, "MIPGap"),
        "nodes": _optional_model_attr(model, "NodeCount"),
        "runtime_seconds": _optional_model_attr(model, "Runtime"),
    }
    write_json(args.output_dir / "resumen_gurobi_proto.json", payload)
    return payload


if __name__ == "__main__":
    print(json.dumps(run(_parser().parse_args()), ensure_ascii=False, indent=2))
