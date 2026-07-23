"""Budget-aware destination ranking for high-pallet SCIP neighborhoods.

The ordinary destination ranking scores the hypothetical move of every
compatible SKU.  That is deliberately conservative for low-capacity designs:
a useful subset can be hidden behind the freight cost of many irrelevant
SKUs.  This module builds an optimistic subset under the *same global pallet
budget* used by the exact solvers and is intended only to choose an LNS pool;
SCIP and the independent evaluator remain authoritative for acceptance.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Mapping

from .costs import box_type_key, evaluate_assignments, freight_pallets, unit_price_mills
from .models import CandidateBox, Dimensions, PLANTS, PreparedData
from .config import FreightPolicy


@dataclass(frozen=True)
class BudgetDestinationRank:
    candidate: CandidateBox
    selected_product_codes: tuple[str, ...]
    optimistic_packaging_saving_mills: int
    freight_delta_mills: int
    extra_pallets: int
    budget_net_score_mills: int


def _global_minimum_pallets(
    data: PreparedData, candidates: tuple[CandidateBox, ...]
) -> int:
    return sum(
        min(
            sum(freight_pallets(product, candidate, plant) for plant in PLANTS)
            for candidate in candidates
            if product.code in candidate.compatible_product_codes
        )
        for product in data.products
    )


def rank_budget_destinations(
    data: PreparedData,
    incumbent_assignment: Mapping[str, CandidateBox],
    exact_candidates: Iterable[CandidateBox],
    freight_policy: FreightPolicy,
    *,
    max_extra_pallets: int,
    min_optimistic_packaging_saving_mills: int = 0,
    max_destinations: int | None = None,
) -> tuple[BudgetDestinationRank, ...]:
    """Rank destinations by an optimistic, pallet-budgeted subset move.

    For each destination we price compatible SKUs at the tier attainable if
    every compatible SKU moved there.  We then greedily retain positive-net
    SKU moves within the remaining global pallet budget.  Source-tier
    deterioration and target-tier activation are intentionally relaxed, so
    the score is an upper-bound-style search signal, not a feasible saving.
    """

    if max_extra_pallets < 0:
        raise ValueError("max_extra_pallets cannot be negative")
    if min_optimistic_packaging_saving_mills < 0:
        raise ValueError("minimum packaging saving cannot be negative")
    if max_destinations is not None and max_destinations < 1:
        raise ValueError("max_destinations must be positive")

    candidates = tuple(exact_candidates)
    expected_codes = {product.code for product in data.products}
    if set(incumbent_assignment) != expected_codes:
        raise ValueError("incumbent assignment must contain exactly one box per SKU")

    incumbent_costs = evaluate_assignments(
        data.products, dict(incumbent_assignment), freight_policy
    )
    minimum_pallets = _global_minimum_pallets(data, candidates)
    remaining_budget = max_extra_pallets - (
        incumbent_costs.pallets - minimum_pallets
    )
    if remaining_budget < 0:
        raise ValueError("incumbent assignment already exceeds the global pallet budget")

    products_by_type: dict[tuple[float, float, float, float], list[object]] = defaultdict(list)
    for product in data.products:
        products_by_type[box_type_key(incumbent_assignment[product.code])].append(product)
    volume_by_type_plant = {
        (type_key, plant): sum(
            product.annual_volume_by_plant[plant] for product in products
        )
        for type_key, products in products_by_type.items()
        for plant in PLANTS
    }

    ranked: list[BudgetDestinationRank] = []
    for candidate in candidates:
        target_type = box_type_key(candidate)
        movable = tuple(
            product
            for product in data.products
            if product.code in candidate.compatible_product_codes
            and incumbent_assignment[product.code].internal != candidate.internal
        )
        if not movable:
            continue
        final_target_volume = {
            plant: volume_by_type_plant.get((target_type, plant), 0)
            + sum(product.annual_volume_by_plant[plant] for product in movable)
            for plant in PLANTS
        }

        choices: list[tuple[int, int, int, str]] = []
        for product in movable:
            source_type = box_type_key(incumbent_assignment[product.code])
            packaging_saving = 0
            for plant in PLANTS:
                volume = product.annual_volume_by_plant[plant]
                if not volume:
                    continue
                source_volume = volume_by_type_plant[(source_type, plant)]
                source_price = unit_price_mills(candidate.thickness_mm, source_volume)
                target_price = unit_price_mills(
                    candidate.thickness_mm, final_target_volume[plant]
                )
                packaging_saving += volume * (source_price - target_price)
            pallet_delta = sum(
                freight_pallets(product, candidate, plant)
                - freight_pallets(product, incumbent_assignment[product.code], plant)
                for plant in PLANTS
            )
            freight_delta = pallet_delta * freight_policy.expected_mills_per_pallet
            choices.append(
                (packaging_saving - freight_delta, pallet_delta, packaging_saving, product.code)
            )

        # Free/saving pallet moves first, then positive-pallet moves by their
        # estimated net return per pallet.  Deterministic ties make runs auditable.
        choices.sort(
            key=lambda item: (
                item[1] > 0,
                -(item[0] / max(item[1], 1)),
                -item[0],
                item[3],
            )
        )
        selected: list[tuple[int, int, int, str]] = []
        used_extra = 0
        for choice in choices:
            net, pallet_delta, _, _ = choice
            additional = max(pallet_delta, 0)
            if net > 0 and used_extra + additional <= remaining_budget:
                selected.append(choice)
                used_extra += additional
        if not selected:
            continue
        packaging_saving = sum(item[2] for item in selected)
        if packaging_saving < min_optimistic_packaging_saving_mills:
            continue
        freight_delta = sum(item[1] for item in selected) * freight_policy.expected_mills_per_pallet
        ranked.append(
            BudgetDestinationRank(
                candidate=candidate,
                selected_product_codes=tuple(sorted(item[3] for item in selected)),
                optimistic_packaging_saving_mills=packaging_saving,
                freight_delta_mills=freight_delta,
                extra_pallets=used_extra,
                budget_net_score_mills=packaging_saving - freight_delta,
            )
        )

    ranked.sort(
        key=lambda item: (
            -item.budget_net_score_mills,
            -item.optimistic_packaging_saving_mills,
            item.extra_pallets,
            item.candidate.internal.as_tuple(),
        )
    )
    if max_destinations is not None:
        ranked = ranked[:max_destinations]
    return tuple(ranked)


def restrictions_for_ranked_pool(
    ranked: Iterable[BudgetDestinationRank],
    incumbent_assignment: Mapping[str, CandidateBox],
    exact_candidates: Iterable[CandidateBox],
) -> tuple[frozenset[str], dict[str, tuple[Dimensions, ...]]]:
    """Build SCIP LNS restrictions from ranked destinations plus used designs."""

    selected = tuple(ranked)
    pool = {item.candidate.internal for item in selected}
    pool.update(candidate.internal for candidate in incumbent_assignment.values())
    free_codes = frozenset().union(
        *(item.candidate.compatible_product_codes for item in selected)
    )
    candidates = tuple(exact_candidates)
    allowed = {
        code: tuple(
            candidate.internal
            for candidate in candidates
            if candidate.internal in pool and code in candidate.compatible_product_codes
        )
        for code in sorted(free_codes)
    }
    return free_codes, allowed
