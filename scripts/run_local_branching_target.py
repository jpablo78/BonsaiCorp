"""Busca un vecindario exacto por distancia de Hamming bajo un costo objetivo.

This is a solver-search driver, not a different contest formulation.  It
keeps the documented candidate universe and cost evaluator unchanged, fixes a
validated warm start as the reference point, and asks SCIP for a solution that
changes only a bounded number of SKU assignments while meeting a stated total
cost target.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from bonsai.config import FreightPolicy, MILLS_PER_USD
from bonsai.data import load_prepared_data
from bonsai.exact_candidates import generate_exact_candidates
from bonsai.reporting import write_assignment_csv, write_json
from bonsai.scip_optimizer import solve_with_scip
from bonsai.solution_validation import validate_solution_csv


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--warm-start", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--time-limit-seconds", type=float, default=1_800.0)
    parser.add_argument("--num-threads", type=int, default=2)
    parser.add_argument("--random-seed", type=int, default=20260722)
    parser.add_argument(
        "--target-usd",
        type=float,
        default=188_078_500.0,
        help="Strict validation target, in USD (default: 188078500).",
    )
    parser.add_argument("--max-changed-products", type=int, required=True)
    parser.add_argument("--min-changed-products", type=int)
    parser.add_argument("--memory-limit-mb", type=int, default=10_000)
    parser.add_argument(
        "--scip-parameter",
        action="append",
        default=[],
        help="Raw SCIP parameter line; may be supplied more than once.",
    )
    return parser


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.time_limit_seconds <= 0:
        raise ValueError("time-limit-seconds must be positive")
    if args.target_usd < 0:
        raise ValueError("target-usd cannot be negative")

    data = load_prepared_data(args.data_dir)
    policy = FreightPolicy()
    warm = validate_solution_csv(args.warm_start, data, policy)
    candidates, candidate_stats = generate_exact_candidates(
        data.products,
        3.0,
        retained_designs=tuple({box.internal for box in warm.assignment.values()}),
    )
    target_mills = round(args.target_usd * MILLS_PER_USD)
    print(
        f"Local branching start: USD {warm.costs.total_mills / 1000:,.2f}; "
        f"target <= USD {target_mills / 1000:,.2f}; "
        f"changes {args.min_changed_products or 0}..{args.max_changed_products}",
        flush=True,
    )
    result = solve_with_scip(
        data,
        3.0,
        policy,
        time_limit_seconds=args.time_limit_seconds,
        num_threads=args.num_threads,
        random_seed=args.random_seed,
        initial_assignment=warm.assignment,
        target_total_mills=target_mills,
        min_changed_products=args.min_changed_products,
        max_changed_products=args.max_changed_products,
        precomputed_exact_candidates=candidates,
        precomputed_exact_candidate_stats=candidate_stats,
        memory_limit_mb=args.memory_limit_mb,
        scip_parameters="\n".join(args.scip_parameter) or None,
        progress_callback=lambda message: print(f"[local] {message}", flush=True),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "asignacion_optima.csv"
    write_assignment_csv(output_path, data, result.assignment)
    checked = validate_solution_csv(output_path, data, policy)
    if checked.costs.total_mills != result.costs.total_mills:
        raise AssertionError("CSV round trip changed the independently evaluated cost")
    payload: dict[str, object] = {
        "configuration": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "candidate_stats": asdict(candidate_stats),
        "warm_start": warm.costs.as_dict(),
        "result": {
            "status": result.status,
            "selected_source": result.selected_source,
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
            "costs": checked.costs.as_dict(),
            "target_met": checked.costs.total_mills <= target_mills,
            "changed_product_count": result.changed_product_count,
            "nodes": result.nodes,
            "solve_time_seconds": result.solve_time_seconds,
            "output_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest().upper(),
        },
    }
    write_json(args.output_dir / "resumen_local_branching.json", payload)
    print(
        f"Local branching end: USD {checked.costs.total_mills / 1000:,.2f}; "
        f"target met={payload['result']['target_met']}; "
        f"changes={result.changed_product_count}",
        flush=True,
    )
    return payload


if __name__ == "__main__":
    print(json.dumps(run(_parser().parse_args()), ensure_ascii=False, indent=2))
