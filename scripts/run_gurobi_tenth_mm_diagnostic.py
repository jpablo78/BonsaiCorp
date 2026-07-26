"""Resuelve el modelo estricto global de 3 mm con dimensiones propuestas de 0,1 mm."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import gurobipy as gp
from ortools.linear_solver import linear_solver_pb2

from bonsai.config import FreightPolicy
from bonsai.costs import evaluate_assignments
from bonsai.data import load_prepared_data
from bonsai.decimal_candidates import generate_decimal_candidates
from bonsai.decimal_io import validate_decimal_solution_csv, write_decimal_assignment_csv
from bonsai.models import CandidateBox, Dimensions
from bonsai.reporting import write_json
from bonsai.scip_optimizer import solve_with_scip
from bonsai.solution_validation import validate_solution_csv
from run_gurobi_proto_neighborhood import _bound_usd, _gurobi_from_proto, _optional_model_attr, _status_name


ASSIGNMENT_NAME = re.compile(r"^x_p(?P<product>\d+)_c(?P<candidate>\d+)$")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument(
        "--warm-start",
        type=Path,
        default=Path("output_lp_pool_after_ba_15m/asignacion_optima.csv"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("output_gurobi_tenth_mm_diagnostic")
    )
    parser.add_argument("--time-limit-seconds", type=float, default=300.0)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument(
        "--decimal-places", type=int, choices=(1, 2, 3, 4, 5, 6), default=1
    )
    parser.add_argument("--mip-focus", type=int, choices=(0, 1, 2, 3), default=1)
    parser.add_argument("--cuts", type=int, choices=(-1, 0, 1, 2), default=-1)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument(
        "--require-novel-vs-decimal-places",
        type=int,
        choices=(1, 2, 3, 4, 5),
        default=None,
        help=(
            "require at least one assignment to a capacity/compatibility "
            "signature absent from the specified coarser decimal grid"
        ),
    )
    return parser


def run(args: argparse.Namespace) -> dict[str, object]:
    data = load_prepared_data(args.data_dir)
    policy = FreightPolicy()
    try:
        warm = validate_solution_csv(args.warm_start, data, policy)
        warm_start_kind = "official_integer"
    except ValueError:
        warm = validate_decimal_solution_csv(
            args.warm_start, data, policy, required_thickness_mm=3.0
        )
        warm_start_kind = "decimal"
    retained = tuple(
        sorted({box.internal for box in warm.assignment.values()}, key=Dimensions.as_tuple)
    )
    candidates, stats = generate_decimal_candidates(
        data.products,
        3.0,
        decimal_places=args.decimal_places,
        retained_designs=retained,
        prune_dominated=True,
    )
    novel_candidate_indices: tuple[int, ...] = ()
    if args.require_novel_vs_decimal_places is not None:
        reference_places = args.require_novel_vs_decimal_places
        if reference_places >= args.decimal_places:
            raise ValueError(
                "--require-novel-vs-decimal-places must be coarser than "
                "--decimal-places"
            )
        reference_candidates, _ = generate_decimal_candidates(
            data.products,
            3.0,
            decimal_places=reference_places,
            retained_designs=retained,
            prune_dominated=True,
        )
        reference_signatures = {
            (candidate.capacity_per_pallet, candidate.compatible_product_codes)
            for candidate in reference_candidates
        }
        novel_candidate_indices = tuple(
            index
            for index, candidate in enumerate(candidates)
            if (
                candidate.capacity_per_pallet,
                candidate.compatible_product_codes,
            )
            not in reference_signatures
        )
        if not novel_candidate_indices:
            raise ValueError("fine grid contains no novel candidate signatures")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    proto_path = args.output_dir / "decimal_master.pb"
    builder = solve_with_scip(
        data,
        3.0,
        policy,
        time_limit_seconds=1.0,
        num_threads=1,
        initial_assignment=warm.assignment,
        precomputed_exact_candidates=candidates,
        export_model_path=proto_path,
    )
    proto = linear_solver_pb2.MPModelProto()
    proto.ParseFromString(proto_path.read_bytes())
    model, variables = _gurobi_from_proto(proto)
    if novel_candidate_indices:
        novel_index_set = set(novel_candidate_indices)
        novel_assignment_variables = []
        for variable in variables:
            match = ASSIGNMENT_NAME.match(variable.VarName)
            if (
                match is not None
                and int(match.group("candidate")) in novel_index_set
            ):
                novel_assignment_variables.append(variable)
        if not novel_assignment_variables:
            raise RuntimeError(
                "all assignment arcs to novel signatures were removed before Gurobi"
            )
        model.addConstr(
            gp.quicksum(novel_assignment_variables) >= 1,
            name="require_novel_decimal_signature",
        )
    model.Params.TimeLimit = args.time_limit_seconds
    model.Params.Threads = args.threads
    model.Params.MIPFocus = args.mip_focus
    model.Params.MIPGap = 0.0
    model.Params.MIPGapAbs = 0.0
    model.Params.Heuristics = 0.20
    model.Params.Cuts = args.cuts
    model.Params.Seed = args.seed
    model.Params.OutputFlag = 1
    model.Params.LogToConsole = 0
    model.Params.LogFile = str(args.output_dir / "gurobi.log")
    model.optimize()
    if model.SolCount < 1:
        raise RuntimeError(f"Gurobi found no tenth-mm solution; status={model.Status}")

    assignment = dict(warm.assignment)
    for variable in variables:
        match = ASSIGNMENT_NAME.match(variable.VarName)
        if match is not None and variable.X > 0.5:
            assignment[data.products[int(match.group("product"))].code] = candidates[
                int(match.group("candidate"))
            ]
    costs = evaluate_assignments(data.products, assignment, policy)
    expected_mills = round(model.ObjVal * builder.objective_scale_mills)
    if expected_mills != costs.total_mills:
        raise AssertionError(
            f"Gurobi objective {expected_mills} differs from independent cost {costs.total_mills}"
        )
    output_path = args.output_dir / "asignacion_decimal.csv"
    write_decimal_assignment_csv(
        output_path, data, assignment, decimal_places=args.decimal_places
    )
    checked = validate_decimal_solution_csv(
        output_path, data, policy, required_thickness_mm=3.0
    )
    checked_assignment, checked_costs = checked.assignment, checked.costs
    if checked_costs.total_mills != costs.total_mills:
        raise AssertionError("decimal CSV round-trip changed total cost")
    if set(checked_assignment) != set(assignment):
        raise AssertionError("decimal CSV round-trip changed assigned products")

    baseline_usd = 209_235_093.6821712
    payload = {
        "diagnostic_only": True,
        "whole_millimetre_rule_relaxed": True,
        "precision_mm": 10 ** (-args.decimal_places),
        "decimal_places": args.decimal_places,
        "reference_decimal_places": args.require_novel_vs_decimal_places,
        "novel_candidate_signatures": len(novel_candidate_indices),
        "warm_start_kind": warm_start_kind,
        "status": int(model.Status),
        "status_name": _status_name(model.Status),
        "optimal": model.Status == gp.GRB.OPTIMAL,
        "runtime_seconds": _optional_model_attr(model, "Runtime"),
        "nodes": _optional_model_attr(model, "NodeCount"),
        "relative_gap": _optional_model_attr(model, "MIPGap"),
        "best_bound_usd": _bound_usd(model, builder.objective_scale_mills),
        "candidate_stats": stats.__dict__,
        "candidate_count": len(candidates),
        "variables": len(proto.variable),
        "constraints": len(proto.constraint),
        "costs": costs.as_dict(),
        "warm_start_costs": warm.costs.as_dict(),
        "savings_vs_warm_start_usd": (warm.costs.total_mills - costs.total_mills)
        / 1000,
        "estimated_score": 100
        * (baseline_usd - costs.total_mills / 1000)
        / baseline_usd,
        "changed_products": sum(
            assignment[p.code].external != warm.assignment[p.code].external
            for p in data.products
        ),
        "fractional_dimension_rows": sum(
            any(not float(value).is_integer() for value in assignment[p.code].external.as_tuple())
            for p in data.products
        ),
        "output_csv": str(output_path),
    }
    write_json(args.output_dir / "resumen_decimal.json", payload)
    return payload


if __name__ == "__main__":
    print(json.dumps(run(_parser().parse_args()), ensure_ascii=False, indent=2))
