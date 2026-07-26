"""Reparaciones SCIP exactas para movimientos coordinados de tiers de Procurement.

Each repair frees a deliberately different set of SKUs but gives every freed
SKU the complete documented 3-mm candidate universe.  The target tier signals
only select a neighbourhood; the SCIP master and independent evaluator decide
whether a combination is actually economical.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from bonsai.config import FreightPolicy
from bonsai.costs import freight_pallets
from bonsai.data import load_prepared_data
from bonsai.exact_candidates import generate_exact_candidates
from bonsai.models import CandidateBox, Dimensions, PLANTS, PreparedData, Product
from bonsai.procurement_lns import ProcurementExposure, rank_procurement_exposures
from bonsai.reporting import write_assignment_csv, write_json
from bonsai.scip_optimizer import solve_with_scip
from bonsai.solution_validation import validate_solution_csv


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one exact Procurement-synergy SCIP neighbourhood"
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--warm-start", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--neighborhood",
        choices=("neutral", "crossplant", "pairs"),
        required=True,
    )
    parser.add_argument("--time-limit-seconds", type=float, default=720.0)
    parser.add_argument("--num-threads", type=int, default=2)
    parser.add_argument("--max-free-skus", type=int, default=240)
    parser.add_argument("--max-extra-pallets", type=int, default=5_000)
    parser.add_argument(
        "--no-pallet-cap",
        action="store_true",
        help="Remove the solver-only pallet cap; this is not a contest rule.",
    )
    parser.add_argument("--random-seed", type=int, default=20260722)
    parser.add_argument("--memory-limit-mb", type=int, default=10_000)
    parser.add_argument(
        "--scip-parameter",
        action="append",
        default=[],
        help="Raw SCIP parameter line; may be supplied more than once.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _pallets(product: Product, candidate: CandidateBox) -> int:
    return sum(freight_pallets(product, candidate, plant) for plant in PLANTS)


def _exposure_payload(exposure: ProcurementExposure) -> dict[str, object]:
    return {
        "internal": exposure.internal.as_tuple(),
        "plant": exposure.plant,
        "current_volume": exposure.current_volume,
        "current_tier_index": exposure.current_tier_index,
        "next_threshold": exposure.next_threshold,
        "gap_to_next": exposure.gap_to_next,
        "potential_saving_usd": exposure.potential_saving_mills / 1000,
        "current_user_count": len(exposure.current_user_codes),
    }


def _pallet_audit(
    data: PreparedData,
    incumbent: dict[str, CandidateBox],
    candidates: tuple[CandidateBox, ...],
    max_extra_pallets: int,
) -> dict[str, int]:
    """Informa el límite de pallets sólo de búsqueda sin tratarlo como regla de negocio."""

    minimum_total = 0
    current_total = 0
    compatible_arc_count = 0
    per_sku_filtered_count = 0
    deltas_from_incumbent: list[int] = []
    for product in data.products:
        counts = [
            _pallets(product, candidate)
            for candidate in candidates
            if product.code in candidate.compatible_product_codes
        ]
        current = _pallets(product, incumbent[product.code])
        minimum = min(counts)
        minimum_total += minimum
        current_total += current
        compatible_arc_count += len(counts)
        per_sku_filtered_count += sum(
            count > minimum + max_extra_pallets for count in counts
        )
        deltas_from_incumbent.extend(count - current for count in counts)
    cap = minimum_total + max_extra_pallets
    headroom = cap - current_total
    return {
        "minimum_pallets": minimum_total,
        "incumbent_pallets": current_total,
        "search_cap_pallets": cap,
        "incumbent_excess_over_minimum": current_total - minimum_total,
        "headroom_from_incumbent": headroom,
        "compatible_arcs": compatible_arc_count,
        "per_sku_filtered_arcs": per_sku_filtered_count,
        "single_move_arcs_within_headroom": sum(
            delta <= headroom for delta in deltas_from_incumbent
        ),
    }


def _move_options(
    exposure: ProcurementExposure,
    data: PreparedData,
    incumbent: dict[str, CandidateBox],
    candidate_by_internal: dict[Dimensions, CandidateBox],
) -> list[tuple[float, int, int, str]]:
    """Ordena SKU potencialmente entrantes para un tipo/planta objetivo.

    The score is intentionally only a selector.  It credits the fractional
    threshold saving that a SKU could unlock and charges only its direct freight
    increase.  The later MIP uses the exact all-units cost of every plant/type.
    """

    target = candidate_by_internal[exposure.internal]
    options: list[tuple[float, int, int, str]] = []
    for product in data.products:
        if product.code in exposure.current_user_codes:
            continue
        incoming = product.annual_volume_by_plant[exposure.plant]
        if not incoming or product.code not in target.compatible_product_codes:
            continue
        pallet_delta = _pallets(product, target) - _pallets(
            product, incumbent[product.code]
        )
        coverage = min(incoming, exposure.gap_to_next) / exposure.gap_to_next
        estimated_mills = (
            exposure.potential_saving_mills * coverage
            - max(0, pallet_delta) * 150_000
        )
        options.append((estimated_mills, incoming, pallet_delta, product.code))
    return sorted(
        options,
        key=lambda item: (
            -item[0],
            abs(item[1] - exposure.gap_to_next),
            item[2],
            item[3],
        ),
    )


def _select_codes(
    scores: dict[str, float],
    mandatory: set[str],
    max_free_skus: int,
) -> frozenset[str]:
    if max_free_skus < 1:
        raise ValueError("max_free_skus must be positive")
    # Cuando un objetivo ya tiene muchos usuarios, se retienen primero los más
    # sólidos económicamente. Así se evita una expansión oculta del MIP declarado.
    ordered_mandatory = sorted(
        mandatory, key=lambda code: (-scores.get(code, 0.0), code)
    )
    selected = set(ordered_mandatory[:max_free_skus])
    for code, _ in sorted(scores.items(), key=lambda item: (-item[1], item[0])):
        if len(selected) >= max_free_skus:
            break
        selected.add(code)
    if not selected:
        raise RuntimeError("synergy selector produced an empty neighbourhood")
    return frozenset(selected)


def _build_neighborhood(
    kind: str,
    data: PreparedData,
    incumbent: dict[str, CandidateBox],
    candidate_by_internal: dict[Dimensions, CandidateBox],
    exposures: tuple[ProcurementExposure, ...],
    max_free_skus: int,
) -> tuple[frozenset[str], dict[str, object]]:
    if kind == "neutral":
        chosen = exposures[:15]
        scores: dict[str, float] = defaultdict(float)
        mandatory: set[str] = set()
        for exposure in chosen:
            mandatory.update(exposure.current_user_codes)
            for score, _, pallet_delta, code in _move_options(
                exposure, data, incumbent, candidate_by_internal
            ):
                if pallet_delta <= 0:
                    scores[code] += score
            for code in exposure.current_user_codes:
                scores[code] += exposure.potential_saving_mills * 0.05
        return _select_codes(scores, mandatory, max_free_skus), {
            "kind": kind,
            "signals": [_exposure_payload(exposure) for exposure in chosen],
            "selection": "moves to active types with non-increasing total pallets",
        }

    if kind == "crossplant":
    # Agrega la contribución de un SKU a cada tipo/planta objetivo seleccionada y
    # favorece cajas que pueden mejorar conjuntamente más de una geografía.
        chosen = exposures[:24]
        scores: dict[str, float] = defaultdict(float)
        mandatory: set[str] = set()
        for exposure in chosen:
            mandatory.update(exposure.current_user_codes)
            for score, _, pallet_delta, code in _move_options(
                exposure, data, incumbent, candidate_by_internal
            ):
                if pallet_delta <= 1_500:
                    scores[code] += max(0.0, score)
            for code in exposure.current_user_codes:
                scores[code] += exposure.potential_saving_mills * 0.03
        return _select_codes(scores, mandatory, max_free_skus), {
            "kind": kind,
            "signals": [_exposure_payload(exposure) for exposure in chosen],
            "selection": "aggregate multi-plant Procurement exposure",
        }

    if kind == "pairs":
        chosen = exposures[:12]
        scores: dict[str, float] = defaultdict(float)
        mandatory: set[str] = set()
        pair_summaries: list[dict[str, object]] = []
        for exposure in chosen:
            mandatory.update(exposure.current_user_codes)
            options = _move_options(exposure, data, incumbent, candidate_by_internal)[:36]
            pairs: list[tuple[float, tuple[float, int, int, str], tuple[float, int, int, str]]] = []
            for left_index, left in enumerate(options):
                for right in options[left_index + 1 :]:
                    combined_volume = left[1] + right[1]
                    coverage = min(1.0, combined_volume / exposure.gap_to_next)
                    overshoot = abs(combined_volume - exposure.gap_to_next) / exposure.gap_to_next
                    pair_score = (
                        exposure.potential_saving_mills * coverage
                        - max(0, left[2] + right[2]) * 150_000
                        - overshoot * exposure.potential_saving_mills * 0.08
                    )
                    pairs.append((pair_score, left, right))
            best_pairs = sorted(pairs, key=lambda item: (-item[0], item[1][3], item[2][3]))[:10]
            for pair_score, left, right in best_pairs:
                scores[left[3]] += max(0.0, pair_score)
                scores[right[3]] += max(0.0, pair_score)
            for code in exposure.current_user_codes:
                scores[code] += exposure.potential_saving_mills * 0.05
            if best_pairs:
                pair_summaries.append(
                    {
                        "exposure": _exposure_payload(exposure),
                        "best_pair_codes": [best_pairs[0][1][3], best_pairs[0][2][3]],
                        "best_pair_score_usd": best_pairs[0][0] / 1000,
                    }
                )
        return _select_codes(scores, mandatory, max_free_skus), {
            "kind": kind,
            "signals": pair_summaries,
            "selection": "two-SKU combinations near a Procurement threshold",
        }

    raise ValueError(f"unknown neighbourhood kind: {kind}")


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.time_limit_seconds <= 0:
        raise ValueError("time limit must be positive")
    data = load_prepared_data(args.data_dir)
    policy = FreightPolicy()
    warm = validate_solution_csv(args.warm_start, data, policy)
    candidates, candidate_stats = generate_exact_candidates(
        data.products,
        3.0,
        retained_designs=tuple({box.internal for box in warm.assignment.values()}),
    )
    candidate_by_internal = {candidate.internal: candidate for candidate in candidates}
    exposures = rank_procurement_exposures(data.products, warm.assignment, candidates)
    free_codes, metadata = _build_neighborhood(
        args.neighborhood,
        data,
        warm.assignment,
        candidate_by_internal,
        exposures,
        args.max_free_skus,
    )
    effective_pallet_cap = None if args.no_pallet_cap else args.max_extra_pallets
    audit: dict[str, object] = (
        {"enabled": False, "reason": "no documented pallet cap was imposed"}
        if effective_pallet_cap is None
        else {
            "enabled": True,
            **_pallet_audit(
                data, warm.assignment, candidates, effective_pallet_cap
            ),
        }
    )
    payload: dict[str, object] = {
        "neighborhood": args.neighborhood,
        "free_product_count": len(free_codes),
        "free_product_codes": sorted(free_codes),
        "metadata": metadata,
        "candidate_count": len(candidates),
        "candidate_stats": candidate_stats.__dict__,
        "pallet_cap_audit": audit,
        "warm_start_costs": warm.costs.as_dict(),
    }
    if args.dry_run:
        return payload

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = solve_with_scip(
        data,
        3.0,
        policy,
        time_limit_seconds=args.time_limit_seconds,
        num_threads=args.num_threads,
        random_seed=args.random_seed,
        initial_assignment=warm.assignment,
        free_product_codes=free_codes,
        precomputed_exact_candidates=candidates,
        precomputed_exact_candidate_stats=candidate_stats,
        max_extra_pallets=effective_pallet_cap,
        memory_limit_mb=args.memory_limit_mb,
        scip_parameters="\n".join(args.scip_parameter) or None,
        progress_callback=lambda message: print(
            f"[{args.neighborhood}] {message}", flush=True
        ),
    )
    output_path = args.output_dir / "asignacion_optima.csv"
    write_assignment_csv(output_path, data, result.assignment)
    checked = validate_solution_csv(output_path, data, policy)
    if checked.costs.total_mills > warm.costs.total_mills:
        raise RuntimeError("protected incumbent became worse")
    payload.update(
        {
            "result": {
                "status": result.status,
                "nodes": result.nodes,
                "solve_time_seconds": result.solve_time_seconds,
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
                "saving_usd": (
                    warm.costs.total_mills - checked.costs.total_mills
                )
                / 1000,
                "improved": checked.costs.total_mills < warm.costs.total_mills,
                "output_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest().upper(),
            }
        }
    )
    write_json(args.output_dir / "resumen_procurement_synergy.json", payload)
    return payload


if __name__ == "__main__":
    print(json.dumps(run(_parser().parse_args()), ensure_ascii=False, indent=2))
