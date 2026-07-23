"""Run a reproducible chain of exact SCIP neighbourhoods.

Each step frees the SKU with demand in a specified set of plants while every
other SKU remains fixed to the current incumbent.  A single global candidate
universe is reused across the chain.  Outputs are independently validated and
the best CSV is never overwritten by a worse incumbent.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import time
from uuid import uuid4

from bonsai.config import FreightPolicy
from bonsai.data import load_prepared_data
from bonsai.exact_candidates import generate_exact_candidates
from bonsai.models import PLANTS, CandidateBox, PreparedData
from bonsai.reporting import write_assignment_csv, write_json
from bonsai.scip_optimizer import solve_with_scip
from bonsai.solution_validation import validate_solution_csv


TARGET_TOTAL_MILLS = 188_079_000_000


def _slug(plants: tuple[str, ...]) -> str:
    return "_".join(plants)


def _default_schedule() -> list[tuple[tuple[str, ...], float]]:
    """Use about 7h45m if every limit is consumed.

    We put smaller exact neighbourhoods first so that any improvement becomes
    a warm start for the more difficult pairs, triples and full master.
    """

    singles = [(plant,) for plant in PLANTS]
    pairs = [
        ("curitiba", "monterrey"),
        ("curitiba", "bakersfield"),
        ("buenos_aires", "santiago"),
        ("buenos_aires", "monterrey"),
        ("buenos_aires", "bakersfield"),
        ("santiago", "monterrey"),
        ("santiago", "bakersfield"),
        ("monterrey", "bakersfield"),
        # These were explored with an older incumbent, so revisit them last.
        ("buenos_aires", "curitiba"),
        ("curitiba", "santiago"),
    ]
    triples = [
        ("curitiba", "monterrey", "bakersfield"),
        ("buenos_aires", "curitiba", "monterrey"),
        ("buenos_aires", "curitiba", "bakersfield"),
        ("curitiba", "santiago", "monterrey"),
        ("curitiba", "santiago", "bakersfield"),
        ("buenos_aires", "monterrey", "bakersfield"),
        ("buenos_aires", "santiago", "monterrey"),
        ("buenos_aires", "santiago", "bakersfield"),
        ("santiago", "monterrey", "bakersfield"),
        ("buenos_aires", "curitiba", "santiago"),
    ]
    # 15 min singles + 150 min pairs + 200 min triples + 100 min global.
    return (
        [(group, 180.0) for group in singles]
        + [(group, 900.0) for group in pairs]
        + [(group, 1200.0) for group in triples]
        + [(PLANTS, 7800.0)]
    )


def _free_codes(data: PreparedData, plants: tuple[str, ...]) -> frozenset[str]:
    return frozenset(
        product.code
        for product in data.products
        if any(product.annual_volume_by_plant[plant] > 0 for plant in plants)
    )


def _write_checked(
    path: Path,
    data: PreparedData,
    policy: FreightPolicy,
    assignment: dict[str, CandidateBox],
):
    pending = path.parent / f".{path.name}.{os.getpid()}.{uuid4().hex}.pending"
    write_assignment_csv(pending, data, assignment)
    checked = validate_solution_csv(pending, data, policy)
    os.replace(pending, path)
    return checked


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sequential exact SCIP neighbourhood chain for Bonsai"
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--warm-start", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--total-time-seconds", type=float, default=28_800.0)
    parser.add_argument("--num-threads", type=int, default=6)
    parser.add_argument("--memory-limit-mb", type=int, default=12_000)
    parser.add_argument("--max-extra-pallets", type=int, default=5_000)
    parser.add_argument("--random-seed", type=int, default=20260721)
    parser.add_argument(
        "--start-step",
        type=int,
        default=1,
        help="one-based schedule step to start from when resuming a chain",
    )
    parser.add_argument(
        "--stop-at-target", action="store_true", help="stop once USD 188,079,000 is met"
    )
    return parser.parse_args()


def _configuration_payload(args: argparse.Namespace) -> dict[str, object]:
    """Make argparse values JSON-safe without losing their exact spelling."""

    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.total_time_seconds <= 0:
        raise ValueError("total time must be positive")
    if args.start_step < 1:
        raise ValueError("start step must be at least one")
    started = time.perf_counter()
    data = load_prepared_data(args.data_dir)
    policy = FreightPolicy()
    current = validate_solution_csv(args.warm_start, data, policy)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_path = args.output_dir / "asignacion_optima.csv"
    _write_checked(best_path, data, policy, current.assignment)

    candidates, candidate_stats = generate_exact_candidates(
        data.products,
        3.0,
        retained_designs=tuple({box.internal for box in current.assignment.values()}),
    )
    schedule = _default_schedule()
    records: list[dict[str, object]] = []

    for ordinal, (plants, scheduled_seconds) in enumerate(schedule, start=1):
        if ordinal < args.start_step:
            continue
        elapsed = time.perf_counter() - started
        remaining = args.total_time_seconds - elapsed
        if remaining <= 1.0:
            break
        limit = min(scheduled_seconds, remaining)
        free_codes = _free_codes(data, plants)
        label = f"{ordinal:02d}_{_slug(plants)}"
        before = current.costs.total_mills
        result = solve_with_scip(
            data,
            3.0,
            policy,
            time_limit_seconds=limit,
            num_threads=args.num_threads,
            random_seed=args.random_seed + ordinal,
            initial_assignment=current.assignment,
            free_product_codes=free_codes,
            precomputed_exact_candidates=candidates,
            precomputed_exact_candidate_stats=candidate_stats,
            max_extra_pallets=args.max_extra_pallets,
            # ``target_total_mills`` is a hard MIP constraint.  The target is
            # only a chain-stopping condition: otherwise a valid partial
            # improvement would be falsely rejected as infeasible.
            target_total_mills=None,
            memory_limit_mb=args.memory_limit_mb,
            progress_callback=lambda message, step=label: print(f"[{step}] {message}", flush=True),
        )
        step_path = args.output_dir / "snapshots" / label / "asignacion_optima.csv"
        checked = _write_checked(step_path, data, policy, result.assignment)
        if checked.costs.total_mills > before:
            raise RuntimeError("protected incumbent unexpectedly became worse")
        improved = checked.costs.total_mills < before
        if improved:
            current = checked
            _write_checked(best_path, data, policy, current.assignment)
        record = {
            "step": ordinal,
            "plants": list(plants),
            "free_product_count": len(free_codes),
            "time_limit_seconds": limit,
            "status": result.status,
            "nodes": result.nodes,
            "solve_time_seconds": result.solve_time_seconds,
            "before_total_usd": before / 1000,
            "after_total_usd": checked.costs.total_mills / 1000,
            "saving_usd": (before - checked.costs.total_mills) / 1000,
            "best_bound_usd": (
                result.best_bound_mills / 1000
                if result.best_bound_mills is not None
                else None
            ),
            "improved": improved,
            "snapshot": str(step_path),
            "sha256": hashlib.sha256(step_path.read_bytes()).hexdigest().upper(),
        }
        records.append(record)
        write_json(
            args.output_dir / "resumen_cadena.json",
            {
                "configuration": _configuration_payload(args),
                "candidate_count": len(candidates),
                "candidate_stats": candidate_stats.__dict__,
                "elapsed_seconds": time.perf_counter() - started,
                "records": records,
                "best": current.costs.as_dict(),
                "best_path": str(best_path),
                "best_sha256": hashlib.sha256(best_path.read_bytes()).hexdigest().upper(),
                "target_total_usd": TARGET_TOTAL_MILLS / 1000,
                "target_met": current.costs.total_mills <= TARGET_TOTAL_MILLS,
            },
        )
        print(
            f"[{label}] total={checked.costs.total_mills / 1000:,.2f} "
            f"saving={(before - checked.costs.total_mills) / 1000:,.2f} "
            f"status={result.status}",
            flush=True,
        )
        if args.stop_at_target and current.costs.total_mills <= TARGET_TOTAL_MILLS:
            break

    # Re-read the final CSV rather than relying on in-memory solver state.
    final = validate_solution_csv(best_path, data, policy)
    return {
        "best": final.costs.as_dict(),
        "best_path": str(best_path),
        "target_met": final.costs.total_mills <= TARGET_TOTAL_MILLS,
        "elapsed_seconds": time.perf_counter() - started,
        "steps": len(records),
    }


if __name__ == "__main__":
    print(run(_parse_arguments()))
