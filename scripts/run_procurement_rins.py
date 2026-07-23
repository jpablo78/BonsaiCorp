"""Exact Procurement-tier and LP-RINS large-neighbourhood search for Bonsai."""

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
from bonsai.lp_pool import build_lp_candidate_pools
from bonsai.models import CandidateBox, Dimensions
from bonsai.procurement_lns import (
    ProcurementExposure,
    rank_procurement_exposures,
    rins_disagreement_order,
    threshold_free_codes,
)
from bonsai.reporting import write_assignment_csv, write_json
from bonsai.scip_optimizer import solve_with_scip
from bonsai.solution_validation import validate_solution_csv


def _write_checked(path, data, policy, assignment):
    pending = path.parent / f".{path.name}.{os.getpid()}.{uuid4().hex}.pending"
    write_assignment_csv(pending, data, assignment)
    checked = validate_solution_csv(pending, data, policy)
    os.replace(pending, path)
    return checked


def _exposure_payload(exposure: ProcurementExposure) -> dict[str, object]:
    return {
        "internal": exposure.internal.as_tuple(),
        "plant": exposure.plant,
        "current_volume": exposure.current_volume,
        "current_tier_index": exposure.current_tier_index,
        "next_threshold": exposure.next_threshold,
        "gap_to_next": exposure.gap_to_next,
        "potential_saving_usd": exposure.potential_saving_mills / 1000,
        "priority": exposure.priority,
        "current_user_count": len(exposure.current_user_codes),
        "eligible_incoming_count": len(exposure.eligible_incoming_codes),
    }


def _merge_pool(
    pools: dict[str, frozenset[Dimensions]],
    code: str,
    targets: tuple[Dimensions, ...],
    candidates_by_internal: dict[Dimensions, CandidateBox],
) -> frozenset[Dimensions]:
    merged = set(pools[code])
    for target in targets:
        if code in candidates_by_internal[target].compatible_product_codes:
            merged.add(target)
    return frozenset(merged)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Threshold-centred and LP-RINS exact neighbourhood search"
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--warm-start", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--total-time-seconds", type=float, default=7200.0)
    parser.add_argument("--lp-time-seconds", type=float, default=120.0)
    parser.add_argument("--top-thresholds", type=int, default=24)
    parser.add_argument("--threshold-max-skus", type=int, default=160)
    parser.add_argument(
        "--threshold-bundle-sizes", type=int, nargs="+", default=[2, 3, 5]
    )
    parser.add_argument("--rins-sizes", type=int, nargs="+", default=[80, 160, 260])
    parser.add_argument("--rins-threshold-combos", type=int, default=10)
    parser.add_argument("--pool-size", type=int, default=16)
    parser.add_argument("--num-threads", type=int, default=6)
    parser.add_argument("--memory-limit-mb", type=int, default=12_000)
    parser.add_argument("--max-extra-pallets", type=int, default=5_000)
    parser.add_argument("--random-seed", type=int, default=20260727)
    return parser


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.total_time_seconds <= 0 or args.lp_time_seconds <= 0:
        raise ValueError("time limits must be positive")
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
    candidate_by_internal = {candidate.internal: candidate for candidate in candidates}
    lp_limit = min(args.lp_time_seconds, args.total_time_seconds)
    lp = solve_with_scip(
        data,
        3.0,
        policy,
        time_limit_seconds=lp_limit,
        num_threads=1,
        random_seed=args.random_seed,
        initial_assignment=current.assignment,
        max_extra_pallets=args.max_extra_pallets,
        precomputed_exact_candidates=candidates,
        precomputed_exact_candidate_stats=candidate_stats,
        relax_integrality=True,
        memory_limit_mb=args.memory_limit_mb,
        progress_callback=lambda message: print(f"[LP] {message}", flush=True),
    )
    if lp.status not in {"OPTIMAL", "FEASIBLE"} or not lp.assignment_arc_values:
        raise RuntimeError(f"LP relaxation did not return arcs: {lp.status}")
    pools, pool_stats = build_lp_candidate_pools(
        data.products,
        candidates,
        current.assignment,
        lp.assignment_arc_values,
        lp.assignment_arc_reduced_costs_mills,
        pool_size=args.pool_size,
    )
    exposures = rank_procurement_exposures(data.products, current.assignment, candidates)
    disagreement = rins_disagreement_order(
        data.products,
        current.assignment,
        lp.assignment_arc_values,
        lp.assignment_arc_reduced_costs_mills,
    )

    # Specs hold only the economics necessary to reproduce a neighbourhood.
    specs: list[
        tuple[str, frozenset[str], tuple[Dimensions, ...], bool, dict[str, object]]
    ] = []
    seen_specs: set[tuple[tuple[str, ...], tuple[Dimensions, ...], bool]] = set()

    def add_spec(
        label: str,
        free_codes: frozenset[str],
        targets: tuple[Dimensions, ...],
        restricted: bool,
        metadata,
    ):
        key = (tuple(sorted(free_codes)), targets, restricted)
        if not free_codes or key in seen_specs:
            return
        seen_specs.add(key)
        specs.append((label, free_codes, targets, restricted, metadata))

    for index, exposure in enumerate(exposures[: args.top_thresholds], start=1):
        free_codes = threshold_free_codes(
            exposure, data.products, max_codes=args.threshold_max_skus
        )
        add_spec(
            f"threshold_{index:02d}_{exposure.plant}",
            free_codes,
            (exposure.internal,),
            True,
            {"kind": "threshold", "exposure": _exposure_payload(exposure)},
        )

    selected_exposures = exposures[: args.top_thresholds]
    for bundle_size in args.threshold_bundle_sizes:
        if bundle_size < 2:
            continue
        # Consecutive priority windows combine related near-tier opportunities
        # while remaining much smaller than a plant-wide neighbourhood.
        for start in range(0, len(selected_exposures) - bundle_size + 1, bundle_size):
            bundle = selected_exposures[start : start + bundle_size]
            free_codes = frozenset(
                code
                for exposure in bundle
                for code in threshold_free_codes(
                    exposure, data.products, max_codes=args.threshold_max_skus
                )
            )
            add_spec(
                f"threshold_bundle_{bundle_size}_{start // bundle_size + 1:02d}",
                free_codes,
                tuple(exposure.internal for exposure in bundle),
                True,
                {
                    "kind": "threshold_bundle",
                    "exposures": [_exposure_payload(exposure) for exposure in bundle],
                },
            )

    for size in args.rins_sizes:
        add_spec(
            f"rins_{size}",
            frozenset(disagreement[: min(size, len(disagreement))]),
            (),
            True,
            {"kind": "rins", "requested_size": size},
        )
    for index, exposure in enumerate(exposures[: args.rins_threshold_combos], start=1):
        threshold_codes = threshold_free_codes(
            exposure, data.products, max_codes=args.threshold_max_skus
        )
        combined = frozenset(set(disagreement[:160]) | set(threshold_codes))
        add_spec(
            f"rins_threshold_{index:02d}_{exposure.plant}",
            combined,
            (exposure.internal,),
            True,
            {"kind": "rins_threshold", "exposure": _exposure_payload(exposure)},
        )

    # Complete RINS variants: LP fixes the complementary SKU, but the free
    # rows retain the entire documented candidate universe rather than a pool.
    for size in sorted({min(229, len(disagreement)), min(320, len(disagreement))}):
        add_spec(
            f"rins_full_{size}",
            frozenset(disagreement[:size]),
            (),
            False,
            {"kind": "rins_full", "requested_size": size},
        )
    for start in (0, 5):
        bundle = selected_exposures[start : start + 5]
        if not bundle:
            continue
        free_codes = frozenset(
            code
            for exposure in bundle
            for code in threshold_free_codes(
                exposure, data.products, max_codes=args.threshold_max_skus
            )
        )
        add_spec(
            f"threshold_full_bundle_{start // 5 + 1:02d}",
            free_codes,
            tuple(exposure.internal for exposure in bundle),
            False,
            {
                "kind": "threshold_full_bundle",
                "exposures": [_exposure_payload(exposure) for exposure in bundle],
            },
        )
    if selected_exposures:
        exposure = selected_exposures[0]
        combined = frozenset(
            set(disagreement[: min(260, len(disagreement))])
            | set(
                threshold_free_codes(
                    exposure, data.products, max_codes=args.threshold_max_skus
                )
            )
        )
        add_spec(
            "rins_full_threshold_01",
            combined,
            (exposure.internal,),
            False,
            {"kind": "rins_full_threshold", "exposure": _exposure_payload(exposure)},
        )

    records: list[dict[str, object]] = []
    for ordinal, (label, free_codes, targets, restricted, metadata) in enumerate(specs, start=1):
        elapsed = time.perf_counter() - started
        remaining = args.total_time_seconds - elapsed
        remaining_specs = len(specs) - ordinal + 1
        if remaining <= 2.0:
            break
        limit = max(1.0, remaining / remaining_specs)
        allowed = (
            {
                code: _merge_pool(pools, code, targets, candidate_by_internal)
                for code in free_codes
            }
            if restricted
            else None
        )
        before = current.costs.total_mills
        print(
            f"[{label}] free={len(free_codes)} arcs="
            f"{sum(map(len, allowed.values())) if allowed is not None else 'all'} "
            f"SCIP={limit:.1f}s",
            flush=True,
        )
        result = solve_with_scip(
            data,
            3.0,
            policy,
            time_limit_seconds=limit,
            num_threads=args.num_threads,
            random_seed=args.random_seed + ordinal * 1009,
            initial_assignment=current.assignment,
            free_product_codes=free_codes,
            allowed_internals_by_product=allowed,
            precomputed_exact_candidates=candidates,
            precomputed_exact_candidate_stats=candidate_stats,
            max_extra_pallets=args.max_extra_pallets,
            memory_limit_mb=args.memory_limit_mb,
            progress_callback=lambda message, name=label: print(f"[{name}] {message}", flush=True),
        )
        snapshot = args.output_dir / "snapshots" / f"{ordinal:02d}_{label}" / "asignacion_optima.csv"
        checked = _write_checked(snapshot, data, policy, result.assignment)
        if checked.costs.total_mills > before:
            raise RuntimeError("protected incumbent became worse")
        improved = checked.costs.total_mills < before
        if improved:
            current = checked
            _write_checked(best_path, data, policy, current.assignment)
        records.append(
            {
                "step": ordinal,
                "label": label,
                "free_product_count": len(free_codes),
                "restricted_candidate_pools": restricted,
                "allowed_arc_count": (
                    sum(map(len, allowed.values())) if allowed is not None else None
                ),
                "time_limit_seconds": limit,
                "before_total_usd": before / 1000,
                "after_total_usd": checked.costs.total_mills / 1000,
                "saving_usd": (before - checked.costs.total_mills) / 1000,
                "improved": improved,
                "status": result.status,
                "nodes": result.nodes,
                "solve_time_seconds": result.solve_time_seconds,
                "best_bound_usd": (
                    result.best_bound_mills / 1000
                    if result.best_bound_mills is not None
                    else None
                ),
                "snapshot": str(snapshot),
                "sha256": hashlib.sha256(snapshot.read_bytes()).hexdigest().upper(),
                **metadata,
            }
        )
        write_json(
            args.output_dir / "resumen_procurement_rins.json",
            {
                "configuration": {
                    key: str(value) if isinstance(value, Path) else value
                    for key, value in vars(args).items()
                },
                "candidate_count": len(candidates),
                "candidate_stats": candidate_stats.__dict__,
                "lp": {
                    "status": lp.status,
                    "objective_usd": (
                        lp.solver_objective_mills / 1000
                        if lp.solver_objective_mills is not None
                        else None
                    ),
                    "positive_arcs": sum(value > 1e-7 for value in lp.assignment_arc_values.values()),
                    "pool_stats": pool_stats.__dict__,
                },
                "top_exposures": [_exposure_payload(item) for item in exposures[:25]],
                "records": records,
                "best": current.costs.as_dict(),
                "best_path": str(best_path),
                "best_sha256": hashlib.sha256(best_path.read_bytes()).hexdigest().upper(),
            },
        )
        print(
            f"[{label}] total={checked.costs.total_mills / 1000:,.2f} "
            f"saving={(before - checked.costs.total_mills) / 1000:,.2f}",
            flush=True,
        )

    final = validate_solution_csv(best_path, data, policy)
    return {
        "best": final.costs.as_dict(),
        "best_path": str(best_path),
        "steps": len(records),
        "elapsed_seconds": time.perf_counter() - started,
    }


if __name__ == "__main__":
    print(run(_parser().parse_args()))
