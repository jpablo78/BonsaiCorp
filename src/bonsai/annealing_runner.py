"""Auditable command-line runner for guided incremental annealing."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
from uuid import uuid4

from .annealing import simulated_annealing
from .config import FreightPolicy
from .data import load_prepared_data
from .exact_candidates import generate_exact_candidates
from .models import CandidateBox, Dimensions
from .reporting import write_assignment_csv, write_json
from .solution_validation import validate_solution_csv


def _unique_internal_designs(
    assignment: dict[str, CandidateBox],
) -> tuple[Dimensions, ...]:
    return tuple(
        sorted(
            {candidate.internal for candidate in assignment.values()},
            key=lambda dimensions: dimensions.as_tuple(),
        )
    )


def _atomic_assignment(path: Path, data, assignment, policy):
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.parent / f".{path.name}.{os.getpid()}.{uuid4().hex}.pending"
    try:
        write_assignment_csv(pending, data, assignment)
        checked = validate_solution_csv(pending, data, policy)
        os.replace(pending, path)
        return checked
    finally:
        if pending.exists():
            pending.unlink()


def run_guided_annealing(args: argparse.Namespace) -> dict[str, object]:
    if args.runs < 1:
        raise ValueError("--runs must be positive")
    if args.duration_seconds is None and args.max_steps is None:
        raise ValueError("--duration-seconds or --max-steps is required")

    data = load_prepared_data(args.data_dir)
    policy = FreightPolicy(extra_region_share=args.extra_region_share)
    incumbent = validate_solution_csv(args.warm_start, data, policy)
    thicknesses = {
        candidate.thickness_mm for candidate in incumbent.assignment.values()
    }
    if len(thicknesses) != 1:
        raise ValueError("warm start must use one global thickness")
    thickness = next(iter(thicknesses))
    candidates, candidate_stats = generate_exact_candidates(
        data.products,
        thickness,
        retained_designs=_unique_internal_designs(incumbent.assignment),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_path = args.output_dir / "asignacion_optima.csv"
    existing = (
        validate_solution_csv(best_path, data, policy) if best_path.exists() else None
    )
    if existing is not None and existing.costs.total_mills < incumbent.costs.total_mills:
        incumbent = existing
    initial_cost = incumbent.costs.total_mills
    if existing is None or incumbent.costs.total_mills < existing.costs.total_mills:
        _atomic_assignment(best_path, data, incumbent.assignment, policy)

    attempts: list[dict[str, object]] = []
    for run_index in range(args.runs):
        before = incumbent.costs.total_mills
        seed = args.random_seed + run_index * 104_729
        result = simulated_annealing(
            data.products,
            incumbent.assignment,
            candidates,
            policy,
            duration_seconds=args.duration_seconds,
            max_steps=args.max_steps,
            random_seed=seed,
            initial_temperature_usd=args.initial_temperature_usd,
            final_temperature_usd=args.final_temperature_usd,
            max_extra_pallets=args.max_extra_pallets,
            validation_interval=args.validation_interval,
            proposal_strategy=args.proposal_strategy,
            proposal_sample_size=args.proposal_sample_size,
            used_target_probability=args.used_target_probability,
            proposal_greediness=args.proposal_greediness,
            group_proposal_probability=args.group_proposal_probability,
            max_group_size=args.max_group_size,
            restart_interval_steps=(
                args.restart_interval_steps or None
            ),
        )
        accepted = result.costs.total_mills < before
        if accepted:
            incumbent = _atomic_assignment(
                best_path, data, result.assignment, policy
            )
            if incumbent.costs.total_mills != result.costs.total_mills:
                raise RuntimeError("CSV validation changed annealing cost")
        attempt = {
            "run": run_index + 1,
            "seed": seed,
            "before_usd": before / 1000,
            "after_usd": result.costs.total_mills / 1000,
            "saving_usd": (before - result.costs.total_mills) / 1000,
            "accepted": accepted,
            "steps": result.steps,
            "proposed_moves": result.proposed_moves,
            "accepted_moves": result.accepted_moves,
            "accepted_worse_moves": result.accepted_worse_moves,
            "proposed_group_moves": result.proposed_group_moves,
            "accepted_group_moves": result.accepted_group_moves,
            "improvements": result.improvements,
            "restarts": result.restarts,
            "elapsed_seconds": result.elapsed_seconds,
            "current_usd": result.current_costs.total_mills / 1000,
        }
        attempts.append(attempt)
        print(
            f"Annealing {run_index + 1}/{args.runs}: "
            f"USD {result.costs.total_mills / 1000:,.2f}; "
            f"{result.steps:,} steps; {result.accepted_group_moves:,} group accepts",
            flush=True,
        )

    summary: dict[str, object] = {
        "configuration": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "candidate_stats": asdict(candidate_stats),
        "candidate_count": len(candidates),
        "initial_usd": initial_cost / 1000,
        "best": incumbent.costs.as_dict(),
        "saving_usd": (initial_cost - incumbent.costs.total_mills) / 1000,
        "attempts": attempts,
        "best_path": str(best_path),
    }
    write_json(args.output_dir / "resumen_annealing.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Guided annealing for Bonsai Corp")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--warm-start", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--duration-seconds", type=float, default=60.0)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--initial-temperature-usd", type=float, default=50_000.0)
    parser.add_argument("--final-temperature-usd", type=float, default=10.0)
    parser.add_argument("--max-extra-pallets", type=int, default=500)
    parser.add_argument("--extra-region-share", type=float, default=0.0)
    parser.add_argument("--proposal-strategy", choices=("guided", "uniform"), default="guided")
    parser.add_argument("--proposal-sample-size", type=int, default=12)
    parser.add_argument("--used-target-probability", type=float, default=0.70)
    parser.add_argument("--proposal-greediness", type=float, default=0.85)
    parser.add_argument("--group-proposal-probability", type=float, default=0.02)
    parser.add_argument("--max-group-size", type=int, default=4)
    parser.add_argument("--restart-interval-steps", type=int, default=250_000)
    parser.add_argument("--validation-interval", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> int:
    summary = run_guided_annealing(build_parser().parse_args(argv))
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
