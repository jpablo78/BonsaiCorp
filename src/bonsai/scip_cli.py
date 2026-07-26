"""Ejecutor de línea de comandos para el modelo maestro exacto alternativo con SCIP."""

from __future__ import annotations

import argparse
from decimal import Decimal, ROUND_HALF_UP
import hashlib
import json
import os
from pathlib import Path
import sys
from uuid import uuid4

from .config import FreightPolicy
from .data import load_prepared_data
from .reporting import write_assignment_csv, write_json
from .scip_optimizer import solve_with_scip
from .solution_validation import validate_solution_csv


def _usd_to_mills(value: Decimal | None) -> int | None:
    if value is None:
        return None
    return int((value * Decimal(1000)).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Alternative exact 3 mm Bonsai master solver with SCIP"
    )
    parser.add_argument("--data-dir", type=Path, default=Path("output/cleaned_data"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--warm-start", type=Path)
    parser.add_argument("--time-limit-seconds", type=float, default=300.0)
    parser.add_argument(
        "--num-threads",
        type=int,
        default=2,
        help="SCIP concurrent threads (default 2 to avoid model-copy memory spikes)",
    )
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--max-extra-pallets", type=int)
    parser.add_argument("--target-total-usd", type=Decimal)
    parser.add_argument("--export-model", type=Path)
    parser.add_argument(
        "--solver-output",
        action="store_true",
        help="stream SCIP's node/incumbent log while solving",
    )
    parser.add_argument(
        "--memory-limit-mb",
        type=int,
        help="SCIP memory limit; reduce threads first when memory is tight",
    )
    parser.add_argument(
        "--scip-parameter",
        action="append",
        default=[],
        help="advanced raw SCIP parameter line; may be repeated",
    )
    parser.add_argument(
        "--quiet-progress",
        action="store_true",
        help="suppress preparation/build/solve stage messages",
    )
    return parser


def run(args: argparse.Namespace) -> dict[str, object]:
    data = load_prepared_data(args.data_dir)
    policy = FreightPolicy()
    initial_assignment = None
    warm_costs = None
    if args.warm_start is not None:
        warm = validate_solution_csv(args.warm_start, data, policy)
        initial_assignment = warm.assignment
        warm_costs = warm.costs

    result = solve_with_scip(
        data,
        3.0,
        policy,
        time_limit_seconds=args.time_limit_seconds,
        num_threads=args.num_threads,
        random_seed=args.random_seed,
        initial_assignment=initial_assignment,
        max_extra_pallets=args.max_extra_pallets,
        target_total_mills=_usd_to_mills(args.target_total_usd),
        export_model_path=args.export_model,
        enable_solver_output=args.solver_output,
        memory_limit_mb=args.memory_limit_mb,
        scip_parameters="\n".join(args.scip_parameter) or None,
        progress_callback=(
            None
            if args.quiet_progress
            else lambda message: print(message, file=sys.stderr, flush=True)
        ),
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "asignacion_optima.csv"
    pending = args.output_dir / f".asignacion_optima.{os.getpid()}.{uuid4().hex}.pending.csv"
    write_assignment_csv(pending, data, result.assignment)
    checked = validate_solution_csv(pending, data, policy)
    if checked.costs.total_mills != result.costs.total_mills:
        pending.unlink(missing_ok=True)
        raise RuntimeError("written SCIP output failed independent cost validation")
    os.replace(pending, output_path)
    digest = hashlib.sha256(output_path.read_bytes()).hexdigest().upper()

    payload: dict[str, object] = {
        "solver": "SCIP via OR-Tools MPSolver",
        "status": result.status,
        "selected_source": result.selected_source,
        "improved_incumbent": result.improved_incumbent,
        "warm_start": str(args.warm_start) if args.warm_start is not None else None,
        "warm_start_costs": warm_costs.as_dict() if warm_costs is not None else None,
        "output_path": str(output_path),
        "output_sha256": digest,
        "independently_validated": True,
        "costs": checked.costs.as_dict(),
        "solver_objective_usd": (
            result.solver_objective_mills / 1000
            if result.solver_objective_mills is not None
            else None
        ),
        "best_bound_usd": (
            result.best_bound_mills / 1000
            if result.best_bound_mills is not None
            else None
        ),
        "candidate_universe_relative_gap": result.candidate_universe_relative_gap,
        "incumbent_usd": result.incumbent_mills / 1000,
        "candidate_count": result.candidate_count,
        "assignment_variable_count": result.assignment_variable_count,
        "threshold_variable_count": result.threshold_variable_count,
        "fixed_product_count": result.fixed_product_count,
        "pruned_assignment_count": result.pruned_assignment_count,
        "pallet_pruned_assignment_count": result.pallet_pruned_assignment_count,
        "objective_pruned_assignment_count": result.objective_pruned_assignment_count,
        "objective_scale_mills": result.objective_scale_mills,
        "candidate_stats": (
            vars(result.candidate_stats) if result.candidate_stats is not None else None
        ),
        "wall_time_seconds": result.wall_time_seconds,
        "preparation_time_seconds": result.preparation_time_seconds,
        "model_build_time_seconds": result.model_build_time_seconds,
        "solve_time_seconds": result.solve_time_seconds,
        "nodes": result.nodes,
        "num_threads": args.num_threads,
        "random_seed": args.random_seed,
        "minimum_possible_pallets": result.minimum_possible_pallets,
        "max_extra_pallets": result.max_extra_pallets,
        "target_total_usd": (
            result.target_total_mills / 1000
            if result.target_total_mills is not None
            else None
        ),
        "target_met": result.target_met,
        "assumptions": {
            "thickness_mm": 3,
            "demand_source": "operaciones_planta.csv",
            "freight_usd_per_pallet": 150,
            "procurement": "exact cumulative all-units thresholds",
            "candidate_universe": "exact integer-mm FAQ #10 grid",
        },
    }
    write_json(args.output_dir / "resumen_scip.json", payload)
    return payload


def main() -> int:
    payload = run(build_parser().parse_args())
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
