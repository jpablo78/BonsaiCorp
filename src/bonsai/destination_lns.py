"""Búsqueda de gran vecindario binaria con un único destino.

Cada subproblema elige un diseño exacto de caja y permite a cada SKU compatible
elegir sólo entre su diseño incumbente y ese destino. Es un complemento pequeño
y dirigido del LNS de tiers: expone cruces coordinados de tiers sin dar a
CP-SAT el universo completo de diseños para cada SKU liberado.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from decimal import Decimal
import json
from pathlib import Path
from typing import Iterable, Mapping

from .config import FreightPolicy
from .costs import box_type_key, freight_pallets, tier_index, unit_price_mills
from .data import load_prepared_data
from .exact_candidates import generate_exact_candidates
from .models import CandidateBox, Dimensions, PLANTS, PreparedData
from .optimizer import solve_for_thickness
from .solution_validation import validate_solution_csv
from .tier_lns import (
    _atomic_write_assignment,
    _atomic_write_json,
    _cost_payload,
    _infer_thickness,
    _mills_from_usd,
    _next_snapshot_number,
    _result_payload,
    _unique_internal_designs,
)


@dataclass(frozen=True)
class DestinationWorkItem:
    """Un subproblema binario entre la incumbente y un destino."""

    destination_id: str
    candidate: CandidateBox
    product_codes: tuple[str, ...]
    procurement_opportunity_mills: int
    freight_saving_opportunity_mills: int
    freight_penalty_if_all_move_mills: int
    tier_crossings: int
    potential_volume: int

    @property
    def sku_count(self) -> int:
        return len(self.product_codes)

    @property
    def gross_opportunity_mills(self) -> int:
        return (
            self.procurement_opportunity_mills
            + self.freight_saving_opportunity_mills
        )

    @property
    def net_all_move_opportunity_mills(self) -> int:
        """Oportunidad bruta optimista menos la pérdida conocida de flete al mover todo."""

        return self.gross_opportunity_mills - self.freight_penalty_if_all_move_mills


def _physical_candidate_rank(candidate: CandidateBox) -> tuple[object, ...]:
    return (
        -len(candidate.compatible_product_codes),
        -candidate.capacity_per_pallet,
        candidate.candidate_id,
    )


def _representative_candidates(
    exact_candidates: Iterable[CandidateBox],
) -> tuple[CandidateBox, ...]:
    """Consolida de forma determinista diseños físicos duplicados accidentalmente."""

    grouped: dict[tuple[float, float, float, float], list[CandidateBox]] = {}
    for candidate in exact_candidates:
        grouped.setdefault(box_type_key(candidate), []).append(candidate)

    representatives: list[CandidateBox] = []
    for type_key in sorted(grouped):
        same_type = grouped[type_key]
        representative = min(same_type, key=_physical_candidate_rank)
        compatible_codes = frozenset().union(
            *(candidate.compatible_product_codes for candidate in same_type)
        )
        representatives.append(
            CandidateBox(
                candidate_id=representative.candidate_id,
                thickness_mm=representative.thickness_mm,
                internal=representative.internal,
                external=representative.external,
                capacity_per_pallet=representative.capacity_per_pallet,
                compatible_product_codes=compatible_codes,
            )
        )
    return tuple(representatives)


def rank_destination_work_items(
    data: PreparedData,
    incumbent_assignment: Mapping[str, CandidateBox],
    exact_candidates: Iterable[CandidateBox],
    freight_policy: FreightPolicy,
    *,
    min_skus: int = 1,
    max_skus: int | None = None,
    min_gross_opportunity_mills: int = 1,
    max_destinations: int | None = None,
) -> tuple[DestinationWorkItem, ...]:
    """Ordena destinos exactos por oportunidad optimista de tier y flete.

    La métrica de Procurement valora el volumen potencial consolidado en el
    tier alcanzable e ignora deliberadamente el deterioro de tiers origen. El
    flete usa sólo ahorros positivos de pallets por SKU. Es una priorización
    determinista, no una promesa de mejora neta; el optimizador y el validador
    independiente deciden si se acepta el movimiento.
    """

    if min_skus < 1:
        raise ValueError("min_skus must be positive")
    if max_skus is not None and max_skus < min_skus:
        raise ValueError("max_skus cannot be smaller than min_skus")
    if min_gross_opportunity_mills < 0:
        raise ValueError("min_gross_opportunity_mills cannot be negative")
    if max_destinations is not None and max_destinations < 1:
        raise ValueError("max_destinations must be positive")

    products = tuple(sorted(data.products, key=lambda product: product.code))
    product_by_code = {product.code: product for product in products}
    expected_codes = set(product_by_code)
    if set(incumbent_assignment) != expected_codes:
        raise ValueError("incumbent assignment must contain exactly one box per SKU")
    thicknesses = {box.thickness_mm for box in incumbent_assignment.values()}
    if len(thicknesses) != 1:
        raise ValueError("incumbent assignment must use one global thickness")
    thickness = next(iter(thicknesses))

    incumbent_groups: dict[tuple[float, float, float, float], list[str]] = {}
    for code in sorted(expected_codes):
        incumbent_groups.setdefault(box_type_key(incumbent_assignment[code]), []).append(code)
    volume_by_type_and_plant: dict[
        tuple[tuple[float, float, float, float], str], int
    ] = {}
    for type_key, codes in incumbent_groups.items():
        for plant in PLANTS:
            volume_by_type_and_plant[(type_key, plant)] = sum(
                product_by_code[code].annual_volume_by_plant[plant] for code in codes
            )

    items: list[DestinationWorkItem] = []
    for candidate in _representative_candidates(exact_candidates):
        if candidate.thickness_mm != thickness:
            continue
        target_type = box_type_key(candidate)
        movable_codes = tuple(
            sorted(
                code
                for code in candidate.compatible_product_codes & expected_codes
                if incumbent_assignment[code].internal != candidate.internal
            )
        )
        if len(movable_codes) < min_skus:
            continue
        if max_skus is not None and len(movable_codes) > max_skus:
            continue

        procurement_opportunity = 0
        tier_crossings = 0
        potential_volume = 0
        for plant in PLANTS:
            current_target_volume = volume_by_type_and_plant.get((target_type, plant), 0)
            movable_volume = sum(
                product_by_code[code].annual_volume_by_plant[plant]
                for code in movable_codes
            )
            consolidated_volume = current_target_volume + movable_volume
            potential_volume += consolidated_volume
            if consolidated_volume <= 0:
                continue
            target_price = unit_price_mills(thickness, consolidated_volume)
            if current_target_volume:
                old_target_price = unit_price_mills(thickness, current_target_volume)
                procurement_opportunity += current_target_volume * max(
                    0, old_target_price - target_price
                )
                tier_crossings += max(
                    0,
                    tier_index(consolidated_volume) - tier_index(current_target_volume),
                )
            else:
                tier_crossings += tier_index(consolidated_volume)

            for code in movable_codes:
                volume = product_by_code[code].annual_volume_by_plant[plant]
                if not volume:
                    continue
                source_type = box_type_key(incumbent_assignment[code])
                source_volume = volume_by_type_and_plant[(source_type, plant)]
                source_price = unit_price_mills(thickness, source_volume)
                procurement_opportunity += volume * max(0, source_price - target_price)

        freight_saving = 0
        freight_penalty = 0
        for code in movable_codes:
            product = product_by_code[code]
            incumbent = incumbent_assignment[code]
            pallet_delta = sum(
                freight_pallets(product, incumbent, plant)
                - freight_pallets(product, candidate, plant)
                for plant in PLANTS
            )
            if pallet_delta >= 0:
                freight_saving += pallet_delta * freight_policy.expected_mills_per_pallet
            else:
                freight_penalty += -pallet_delta * freight_policy.expected_mills_per_pallet

        item = DestinationWorkItem(
            destination_id="",
            candidate=candidate,
            product_codes=movable_codes,
            procurement_opportunity_mills=procurement_opportunity,
            freight_saving_opportunity_mills=freight_saving,
            freight_penalty_if_all_move_mills=freight_penalty,
            tier_crossings=tier_crossings,
            potential_volume=potential_volume,
        )
        if item.gross_opportunity_mills >= min_gross_opportunity_mills:
            items.append(item)

    items.sort(
        key=lambda item: (
            -item.net_all_move_opportunity_mills,
            -item.gross_opportunity_mills,
            -item.tier_crossings,
            -item.procurement_opportunity_mills,
            -item.freight_saving_opportunity_mills,
            item.freight_penalty_if_all_move_mills,
            item.sku_count,
            item.candidate.internal.as_tuple(),
            item.candidate.candidate_id,
        )
    )
    if max_destinations is not None:
        items = items[:max_destinations]
    return tuple(
        DestinationWorkItem(
            destination_id=f"destination_{index:04d}",
            candidate=item.candidate,
            product_codes=item.product_codes,
            procurement_opportunity_mills=item.procurement_opportunity_mills,
            freight_saving_opportunity_mills=item.freight_saving_opportunity_mills,
            freight_penalty_if_all_move_mills=item.freight_penalty_if_all_move_mills,
            tier_crossings=item.tier_crossings,
            potential_volume=item.potential_volume,
        )
        for index, item in enumerate(items)
    )


def allowed_internals_for_destination(
    item: DestinationWorkItem,
) -> dict[str, tuple[Dimensions, ...]]:
    """Devuelve el lado destino de cada elección entre incumbente y objetivo."""

    return {code: (item.candidate.internal,) for code in item.product_codes}


def _work_item_payload(item: DestinationWorkItem) -> dict[str, object]:
    return {
        "destination_id": item.destination_id,
        "candidate_id": item.candidate.candidate_id,
        "internal_mm": item.candidate.internal.as_tuple(),
        "external_mm": item.candidate.external.as_tuple(),
        "capacity_per_pallet": item.candidate.capacity_per_pallet,
        "sku_count": item.sku_count,
        "product_codes": item.product_codes,
        "procurement_opportunity_usd": item.procurement_opportunity_mills / 1000,
        "freight_saving_opportunity_usd": item.freight_saving_opportunity_mills / 1000,
        "freight_penalty_if_all_move_usd": item.freight_penalty_if_all_move_mills / 1000,
        "gross_opportunity_usd": item.gross_opportunity_mills / 1000,
        "net_all_move_opportunity_usd": item.net_all_move_opportunity_mills / 1000,
        "tier_crossings": item.tier_crossings,
        "potential_volume": item.potential_volume,
    }


def run_destination_lns(args: argparse.Namespace) -> dict[str, object]:
    if args.rounds < 1:
        raise ValueError("--rounds must be at least 1")
    if args.time_per_destination <= 0:
        raise ValueError("--time-per-destination must be positive")
    if args.num_search_workers < 1:
        raise ValueError("--num-search-workers must be positive")
    if args.target_mode == "hard" and args.target_total_usd is None:
        raise ValueError("--target-mode hard requires --target-total-usd")

    data = load_prepared_data(args.data_dir)
    policy = FreightPolicy(extra_region_share=args.extra_region_share)
    incumbent = validate_solution_csv(args.warm_start, data, policy)
    thickness = _infer_thickness(incumbent)
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
    summary_path = args.output_dir / "resumen_destination_lns.json"
    attempts: list[dict[str, object]] = []
    improvements: list[dict[str, object]] = []
    summary: dict[str, object] = {
        "configuration": {
            "data_dir": str(args.data_dir),
            "warm_start": str(args.warm_start),
            "thickness_mm": thickness,
            "time_per_destination_seconds": args.time_per_destination,
            "num_search_workers": args.num_search_workers,
            "rounds": args.rounds,
            "max_destinations": args.max_destinations,
            "min_skus": args.min_skus,
            "max_skus": args.max_skus,
            "min_gross_opportunity_usd": min_opportunity_mills / 1000,
            "target_total_usd": target_mills / 1000 if target_mills is not None else None,
            "target_mode": args.target_mode,
            "max_extra_pallets": args.max_extra_pallets,
            "random_seed": args.random_seed,
        },
        "exact_candidate_stats": asdict(candidate_stats),
        "initial": _cost_payload(start_costs),
        "attempts": attempts,
        "improvements": improvements,
    }
    termination = "round_limit"
    attempted_at_cost: set[tuple[int, tuple[int, int, int]]] = set()
    print(
        f"Destination LNS start: USD {incumbent.costs.total_mills / 1000:,.2f}; "
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
                min_skus=args.min_skus,
                max_skus=args.max_skus,
                min_gross_opportunity_mills=min_opportunity_mills,
                max_destinations=args.max_destinations,
            )
            work_items = tuple(
                item
                for item in ranked
                if (
                    incumbent.costs.total_mills,
                    item.candidate.internal.as_tuple(),
                )
                not in attempted_at_cost
            )
            if not work_items:
                termination = "no_destinations"
                break
            print(
                f"Round {round_index + 1}/{args.rounds}: "
                f"{len(work_items)} ranked destinations",
                flush=True,
            )
            improved_this_round = False
            for item_index, item in enumerate(work_items):
                before_mills = incumbent.costs.total_mills
                attempted_at_cost.add((before_mills, item.candidate.internal.as_tuple()))
                seed = args.random_seed + round_index * 104_729 + item_index * 1_009
                result = solve_for_thickness(
                    data,
                    thickness,
                    policy,
                    time_limit_seconds=args.time_per_destination,
                    num_search_workers=args.num_search_workers,
                    random_seed=seed,
                    initial_assignment=incumbent.assignment,
                    candidate_strategy="exact",
                    max_extra_pallets=args.max_extra_pallets,
                    target_total_mills=(
                        target_mills if args.target_mode == "hard" else None
                    ),
                    free_product_codes=item.product_codes,
                    allowed_internals_by_product=allowed_internals_for_destination(item),
                    precomputed_exact_candidates=exact_candidates,
                    precomputed_exact_candidate_stats=candidate_stats,
                )
                attempt = {
                    "round": round_index + 1,
                    "sequence": len(attempts) + 1,
                    **_work_item_payload(item),
                    "before_usd": before_mills / 1000,
                    "seed": seed,
                    **_result_payload(result),
                    "accepted": False,
                }
                if result.costs.total_mills < before_mills:
                    candidate_path = args.output_dir / ".candidate_validation.csv"
                    try:
                        checked = _atomic_write_assignment(
                            candidate_path, data, result.assignment, policy
                        )
                        if checked.costs.total_mills != result.costs.total_mills:
                            raise RuntimeError("independent validation changed solver cost")
                    except (KeyError, RuntimeError, TypeError, ValueError) as exc:
                        attempt["validation_error"] = str(exc)
                    else:
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
                            "destination_id": item.destination_id,
                            "internal_mm": item.candidate.internal.as_tuple(),
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
                            f"  {item.destination_id} {item.candidate.internal.as_tuple()}: "
                            f"accepted USD {incumbent.costs.total_mills / 1000:,.2f}",
                            flush=True,
                        )
                    finally:
                        if candidate_path.exists():
                            candidate_path.unlink()

                attempts.append(attempt)
                summary["best"] = _cost_payload(incumbent.costs)
                summary["target_met"] = (
                    incumbent.costs.total_mills <= target_mills
                    if target_mills is not None
                    else None
                )
                _atomic_write_json(summary_path, summary)
                if not attempt["accepted"]:
                    print(
                        f"  {item.destination_id}: {result.status}, no improvement "
                        f"({item.sku_count} SKUs, {result.wall_time_seconds:.1f}s)",
                        flush=True,
                    )
                if target_mills is not None and incumbent.costs.total_mills <= target_mills:
                    termination = "target_met"
                    break
                if improved_this_round:
                    break
            if termination == "target_met":
                break

    summary["best"] = _cost_payload(incumbent.costs)
    summary["saving_usd"] = (start_costs.total_mills - incumbent.costs.total_mills) / 1000
    summary["target_met"] = (
        incumbent.costs.total_mills <= target_mills if target_mills is not None else None
    )
    summary["termination"] = termination
    summary["best_path"] = str(best_path)
    _atomic_write_json(summary_path, summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Exact incumbent-or-single-destination LNS for Bonsai Corp"
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--warm-start", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--time-per-destination", type=float, default=30.0)
    parser.add_argument("--num-search-workers", type=int, default=6)
    parser.add_argument("--max-extra-pallets", type=int, default=500)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--max-destinations", type=int, default=64)
    parser.add_argument("--min-skus", type=int, default=2)
    parser.add_argument("--max-skus", type=int, default=220)
    parser.add_argument("--min-gross-opportunity-usd", type=Decimal, default=Decimal("1"))
    parser.add_argument("--target-total-usd", type=Decimal)
    parser.add_argument("--target-mode", choices=("stop", "hard"), default="stop")
    parser.add_argument("--extra-region-share", type=float, default=0.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run_destination_lns(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
