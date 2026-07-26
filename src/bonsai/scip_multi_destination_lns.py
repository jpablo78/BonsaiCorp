"""LNS restringido multidestino resuelto con subproblemas SCIP de un hilo.

This runner shares the destination ranking and neighbourhood construction with
``multi_destination_lns`` but sends each incumbent-or-destinations model to the
SCIP backend.  Only released SKUs exist as assignment variables; every other
SKU is absorbed into fixed freight and procurement-tier constants.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from decimal import Decimal
import json
from pathlib import Path

from .config import FreightPolicy
from .data import load_prepared_data
from .destination_lns import rank_destination_work_items
from .exact_candidates import generate_exact_candidates
from .multi_destination_lns import (
    _multi_payload,
    allowed_internals_for_neighborhood,
    build_multi_destination_work_items,
)
from .scip_optimizer import ScipSolveResult, solve_with_scip
from .solution_validation import validate_solution_csv
from .tier_lns import (
    _atomic_write_assignment,
    _atomic_write_json,
    _cost_payload,
    _infer_thickness,
    _mills_from_usd,
    _next_snapshot_number,
    _unique_internal_designs,
)


def _scip_result_payload(result: ScipSolveResult) -> dict[str, object]:
    return {
        "status": result.status,
        "selected_source": result.selected_source,
        "improved_incumbent": result.improved_incumbent,
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
        "candidate_count": result.candidate_count,
        "assignment_variable_count": result.assignment_variable_count,
        "threshold_variable_count": result.threshold_variable_count,
        "fixed_product_count": result.fixed_product_count,
        "pruned_assignment_count": result.pruned_assignment_count,
        "wall_time_seconds": result.wall_time_seconds,
        "preparation_time_seconds": result.preparation_time_seconds,
        "model_build_time_seconds": result.model_build_time_seconds,
        "solve_time_seconds": result.solve_time_seconds,
        "nodes": result.nodes,
        "target_met": result.target_met,
    }


def run_scip_multi_destination_lns(args: argparse.Namespace) -> dict[str, object]:
    if args.rounds < 1:
        raise ValueError("--rounds must be at least 1")
    if args.time_per_neighborhood <= 0:
        raise ValueError("--time-per-neighborhood must be positive")
    if args.max_neighborhoods is not None and args.max_neighborhoods < 1:
        raise ValueError("--max-neighborhoods must be positive")

    data = load_prepared_data(args.data_dir)
    policy = FreightPolicy(extra_region_share=args.extra_region_share)
    incumbent = validate_solution_csv(args.warm_start, data, policy)
    thickness = _infer_thickness(incumbent)
    if thickness != 3.0:
        raise ValueError("SCIP LNS currently supports only a 3 mm incumbent")
    target_mills = _mills_from_usd(args.target_total_usd)
    min_opportunity_mills = _mills_from_usd(args.min_gross_opportunity_usd) or 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_path = args.output_dir / "asignacion_optima.csv"
    if best_path.exists():
        existing = validate_solution_csv(best_path, data, policy)
        if _infer_thickness(existing) != thickness:
            raise ValueError("existing output uses a different global thickness")
        if existing.costs.total_mills < incumbent.costs.total_mills:
            incumbent = existing

    start_costs = incumbent.costs
    snapshot_number = _next_snapshot_number(args.output_dir)
    _atomic_write_assignment(
        args.output_dir / f"incumbent_{snapshot_number:04d}.csv",
        data,
        incumbent.assignment,
        policy,
    )
    snapshot_number += 1
    _atomic_write_assignment(best_path, data, incumbent.assignment, policy)

    exact_candidates, candidate_stats = generate_exact_candidates(
        data.products,
        thickness,
        retained_designs=_unique_internal_designs(incumbent.assignment),
    )
    summary_path = args.output_dir / "resumen_scip_multi_destination_lns.json"
    attempts: list[dict[str, object]] = []
    improvements: list[dict[str, object]] = []
    summary: dict[str, object] = {
        "solver": "SCIP restricted LNS via OR-Tools MPSolver",
        "configuration": {
            "data_dir": str(args.data_dir),
            "warm_start": str(args.warm_start),
            "time_per_neighborhood_seconds": args.time_per_neighborhood,
            "num_threads": 1,
            "rounds": args.rounds,
            "destination_pool_size": args.destination_pool_size,
            "min_destinations_per_neighborhood": args.min_destinations_per_neighborhood,
            "max_destinations_per_neighborhood": args.max_destinations_per_neighborhood,
            "min_destination_choices_per_sku": args.min_destination_choices_per_sku,
            "max_neighborhoods": args.max_neighborhoods,
            "min_skus_per_neighborhood": args.min_skus_per_neighborhood,
            "max_skus_per_neighborhood": args.max_skus_per_neighborhood,
            "max_extra_pallets": args.max_extra_pallets,
            "target_total_usd": target_mills / 1000 if target_mills is not None else None,
            "random_seed": args.random_seed,
        },
        "exact_candidate_stats": asdict(candidate_stats),
        "initial": _cost_payload(start_costs),
        "attempts": attempts,
        "improvements": improvements,
    }
    attempted_at_cost: set[tuple[int, frozenset[tuple[int, int, int]]]] = set()
    termination = "round_limit"
    print(
        f"SCIP multi-destination LNS start: USD {incumbent.costs.total_mills / 1000:,.2f}; "
        f"{len(exact_candidates):,} exact candidates",
        flush=True,
    )

    if target_mills is not None and incumbent.costs.total_mills <= target_mills:
        termination = "target_already_met"
    else:
        for round_index in range(args.rounds):
            ranked = rank_destination_work_items(
                data,
                incumbent.assignment,
                exact_candidates,
                policy,
                min_skus=args.min_skus_per_destination,
                max_skus=args.max_skus_per_neighborhood,
                min_gross_opportunity_mills=min_opportunity_mills,
                max_destinations=args.destination_pool_size,
            )
            neighborhoods = build_multi_destination_work_items(
                ranked,
                incumbent.assignment,
                min_destinations=args.min_destinations_per_neighborhood,
                max_destinations=args.max_destinations_per_neighborhood,
                min_destination_choices_per_sku=args.min_destination_choices_per_sku,
                min_skus=args.min_skus_per_neighborhood,
                max_skus=args.max_skus_per_neighborhood,
                max_neighborhoods=args.max_neighborhoods,
            )
            work_items = tuple(
                item
                for item in neighborhoods
                if (
                    incumbent.costs.total_mills,
                    frozenset(
                        destination.candidate.internal.as_tuple()
                        for destination in item.destinations
                    ),
                )
                not in attempted_at_cost
            )
            if not work_items:
                termination = "no_neighborhoods"
                break
            print(
                f"Round {round_index + 1}/{args.rounds}: "
                f"{len(ranked)} destinations -> {len(work_items)} neighborhoods",
                flush=True,
            )
            improved_this_round = False
            for item_index, item in enumerate(work_items):
                before_mills = incumbent.costs.total_mills
                destination_key = frozenset(
                    destination.candidate.internal.as_tuple()
                    for destination in item.destinations
                )
                attempted_at_cost.add((before_mills, destination_key))
                seed = args.random_seed + round_index * 104_729 + item_index * 1_009
                result = solve_with_scip(
                    data,
                    thickness,
                    policy,
                    time_limit_seconds=args.time_per_neighborhood,
                    num_threads=1,
                    random_seed=seed,
                    initial_assignment=incumbent.assignment,
                    max_extra_pallets=args.max_extra_pallets,
                    free_product_codes=item.product_codes,
                    allowed_internals_by_product=allowed_internals_for_neighborhood(item),
                    precomputed_exact_candidates=exact_candidates,
                    precomputed_exact_candidate_stats=candidate_stats,
                    memory_limit_mb=args.memory_limit_mb,
                    scip_parameters=("\n".join(args.scip_parameter) or None),
                )
                attempt = {
                    "round": round_index + 1,
                    "sequence": len(attempts) + 1,
                    **_multi_payload(item),
                    "before_usd": before_mills / 1000,
                    "seed": seed,
                    **_scip_result_payload(result),
                    "accepted": False,
                }
                if result.costs.total_mills < before_mills:
                    candidate_path = args.output_dir / ".candidate_validation.csv"
                    try:
                        checked = _atomic_write_assignment(
                            candidate_path, data, result.assignment, policy
                        )
                        if checked.costs.total_mills != result.costs.total_mills:
                            raise RuntimeError("independent validation changed SCIP cost")
                        incumbent = checked
                        snapshot_path = (
                            args.output_dir / f"incumbent_{snapshot_number:04d}.csv"
                        )
                        _atomic_write_assignment(
                            snapshot_path, data, incumbent.assignment, policy
                        )
                        _atomic_write_assignment(
                            best_path, data, incumbent.assignment, policy
                        )
                        snapshot_number += 1
                        improvement = {
                            "round": round_index + 1,
                            "attempt": len(attempts) + 1,
                            "neighborhood_id": item.neighborhood_id,
                            "before_usd": before_mills / 1000,
                            "after_usd": incumbent.costs.total_mills / 1000,
                            "saving_usd": (before_mills - incumbent.costs.total_mills) / 1000,
                            "snapshot_path": str(snapshot_path),
                        }
                        improvements.append(improvement)
                        attempt["accepted"] = True
                        attempt["snapshot_path"] = str(snapshot_path)
                        improved_this_round = True
                        print(
                            f"  {item.neighborhood_id}: accepted "
                            f"USD {incumbent.costs.total_mills / 1000:,.2f}",
                            flush=True,
                        )
                    finally:
                        candidate_path.unlink(missing_ok=True)
                attempts.append(attempt)
                summary["best"] = _cost_payload(incumbent.costs)
                summary["saving_usd"] = (
                    start_costs.total_mills - incumbent.costs.total_mills
                ) / 1000
                summary["target_met"] = (
                    incumbent.costs.total_mills <= target_mills
                    if target_mills is not None
                    else None
                )
                _atomic_write_json(summary_path, summary)
                if not attempt["accepted"]:
                    print(
                        f"  {item.neighborhood_id}: {result.status}, no improvement "
                        f"({item.destination_count} destinations, {item.sku_count} SKUs, "
                        f"{result.wall_time_seconds:.1f}s)",
                        flush=True,
                    )
                if target_mills is not None and incumbent.costs.total_mills <= target_mills:
                    termination = "target_met"
                    break
    # Se reordena inmediatamente tras una mejora: cambió la economía de destinos
    # y orígenes, por lo que la lista restante quedó desactualizada.
                if improved_this_round:
                    break
            if termination == "target_met":
                break

    summary["best"] = _cost_payload(incumbent.costs)
    summary["saving_usd"] = (
        start_costs.total_mills - incumbent.costs.total_mills
    ) / 1000
    summary["target_met"] = (
        incumbent.costs.total_mills <= target_mills
        if target_mills is not None
        else None
    )
    summary["termination"] = termination
    summary["best_path"] = str(best_path)
    _atomic_write_json(summary_path, summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="One-thread SCIP multi-destination LNS for Bonsai Corp"
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--warm-start", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--time-per-neighborhood", type=float, default=10.0)
    parser.add_argument("--max-extra-pallets", type=int, default=1000)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--destination-pool-size", type=int, default=96)
    parser.add_argument("--min-destinations-per-neighborhood", type=int, default=2)
    parser.add_argument("--max-destinations-per-neighborhood", type=int, default=8)
    parser.add_argument("--min-destination-choices-per-sku", type=int, default=2)
    parser.add_argument("--max-neighborhoods", type=int, default=256)
    parser.add_argument("--min-skus-per-destination", type=int, default=1)
    parser.add_argument("--min-skus-per-neighborhood", type=int, default=2)
    parser.add_argument("--max-skus-per-neighborhood", type=int, default=220)
    parser.add_argument("--min-gross-opportunity-usd", type=Decimal, default=Decimal("1"))
    parser.add_argument("--target-total-usd", type=Decimal)
    parser.add_argument("--extra-region-share", type=float, default=0.0)
    parser.add_argument("--memory-limit-mb", type=int)
    parser.add_argument("--scip-parameter", action="append", default=[])
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run_scip_multi_destination_lns(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
