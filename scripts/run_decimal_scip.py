"""Optimiza el universo de candidatos en milímetros decimales con OR-Tools MPSolver/SCIP.

Es la contraparte de código abierto del ejecutor decimal histórico de Gurobi.
Conserva el mismo generador de candidatos, validador decimal estricto y
evaluador independiente de costos; sólo cambia el backend MIP.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bonsai.config import FreightPolicy
from bonsai.data import load_prepared_data
from bonsai.decimal_candidates import generate_decimal_candidates
from bonsai.decimal_io import validate_decimal_solution_csv, write_decimal_assignment_csv
from bonsai.models import Dimensions
from bonsai.reporting import write_json
from bonsai.scip_optimizer import solve_with_scip
from bonsai.solution_validation import ValidationResult, validate_solution_csv


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument("--warm-start", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--time-limit-seconds", type=float, default=300.0)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument(
        "--decimal-places", type=int, choices=(1, 2, 3, 4, 5, 6), default=1
    )
    return parser


def _load_warm_start(path: Path, data, policy: FreightPolicy) -> tuple[ValidationResult, str]:
    try:
        return validate_solution_csv(path, data, policy), "integer"
    except ValueError:
        return (
            validate_decimal_solution_csv(path, data, policy, required_thickness_mm=3.0),
            "decimal",
        )


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.time_limit_seconds <= 0:
        raise ValueError("--time-limit-seconds must be positive")
    if args.threads < 1:
        raise ValueError("--threads must be at least one")
    data = load_prepared_data(args.data_dir)
    policy = FreightPolicy()
    warm, warm_start_kind = _load_warm_start(args.warm_start, data, policy)
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
    result = solve_with_scip(
        data,
        3.0,
        policy,
        time_limit_seconds=args.time_limit_seconds,
        num_threads=args.threads,
        initial_assignment=warm.assignment,
        precomputed_exact_candidates=candidates,
        precomputed_exact_candidate_stats=stats,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "asignacion_decimal.csv"
    write_decimal_assignment_csv(
        output_path,
        data,
        result.assignment,
        decimal_places=args.decimal_places,
    )
    checked = validate_decimal_solution_csv(
        output_path, data, policy, required_thickness_mm=3.0
    )
    if checked.costs.total_mills != result.costs.total_mills:
        raise AssertionError("decimal CSV round-trip changed total cost")
    payload = {
        "solver": "OR-Tools MPSolver/SCIP",
        "decimal_places": args.decimal_places,
        "precision_mm": 10 ** (-args.decimal_places),
        "warm_start_kind": warm_start_kind,
        "status": result.status,
        "optimal": result.status == "OPTIMAL",
        "runtime_seconds": result.wall_time_seconds,
        "best_bound_usd": (
            result.best_bound_mills / 1000 if result.best_bound_mills is not None else None
        ),
        "relative_gap": result.candidate_universe_relative_gap,
        "candidate_stats": stats.__dict__,
        "candidate_count": len(candidates),
        "costs": checked.costs.as_dict(),
        "output_csv": str(output_path),
    }
    write_json(args.output_dir / "resumen_decimal_scip.json", payload)
    return payload


if __name__ == "__main__":
    print(json.dumps(run(_parser().parse_args()), ensure_ascii=False, indent=2))
