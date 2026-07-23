"""Economic large neighbourhoods centred on Procurement discount tiers.

The useful couplings in this problem are not necessarily geographic.  A box
design is procured separately per plant, and moving a set of SKUs onto or off
one design can cross an all-units discount threshold.  These helpers expose
those couplings for exact restricted SCIP repairs.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Mapping, Sequence

from .config import DISCOUNT_TIERS
from .costs import tier_index, unit_price_mills
from .models import CandidateBox, Dimensions, PLANTS, Product


@dataclass(frozen=True)
class ProcurementExposure:
    """A current design/plant volume with an attainable next discount tier."""

    internal: Dimensions
    plant: str
    current_volume: int
    current_tier_index: int
    next_threshold: int
    gap_to_next: int
    potential_saving_mills: int
    current_user_codes: tuple[str, ...]
    eligible_incoming_codes: tuple[str, ...]

    @property
    def priority(self) -> float:
        """Potential saving weighted toward thresholds that are easier to reach."""

        return self.potential_saving_mills / max(1, self.gap_to_next)


def rank_procurement_exposures(
    products: Sequence[Product],
    incumbent: Mapping[str, CandidateBox],
    candidates: Sequence[CandidateBox],
) -> tuple[ProcurementExposure, ...]:
    """Rank current type/plant volumes that could reach their next price tier.

    The potential saving is exact *conditional on reaching the threshold*: the
    lower price then applies to every unit of that physical type at that plant.
    It is only a priority signal, never an objective approximation.
    """

    product_by_code = {product.code: product for product in products}
    if set(incumbent) != set(product_by_code):
        raise ValueError("incumbent must cover exactly the supplied products")
    candidate_by_internal = {candidate.internal: candidate for candidate in candidates}

    users_by_internal: dict[Dimensions, list[str]] = defaultdict(list)
    volume_by_internal_plant: dict[tuple[Dimensions, str], int] = defaultdict(int)
    for product in products:
        internal = incumbent[product.code].internal
        users_by_internal[internal].append(product.code)
        for plant in PLANTS:
            volume_by_internal_plant[(internal, plant)] += product.annual_volume_by_plant[plant]

    exposures: list[ProcurementExposure] = []
    for internal, current_users in users_by_internal.items():
        candidate = candidate_by_internal.get(internal)
        if candidate is None:
            # Retained incumbent designs are always part of the exact universe.
            raise ValueError(f"incumbent internal design missing from candidates: {internal}")
        for plant in PLANTS:
            volume = volume_by_internal_plant[(internal, plant)]
            if volume < 1:
                continue
            current_tier = tier_index(volume)
            if current_tier + 1 >= len(DISCOUNT_TIERS):
                continue
            next_threshold = DISCOUNT_TIERS[current_tier + 1].lower_inclusive
            gap = next_threshold - volume
            if gap <= 0:
                raise AssertionError("tier ordering must be strictly increasing")
            current_price = unit_price_mills(candidate.thickness_mm, volume)
            next_price = unit_price_mills(candidate.thickness_mm, next_threshold)
            potential_saving = volume * (current_price - next_price)
            if potential_saving <= 0:
                continue
            eligible = sorted(
                (
                    product.code
                    for product in products
                    if product.code in candidate.compatible_product_codes
                    and product.annual_volume_by_plant[plant] > 0
                ),
                key=lambda code: (
                    code not in current_users,
                    -product_by_code[code].annual_volume_by_plant[plant],
                    code,
                ),
            )
            exposures.append(
                ProcurementExposure(
                    internal=internal,
                    plant=plant,
                    current_volume=volume,
                    current_tier_index=current_tier,
                    next_threshold=next_threshold,
                    gap_to_next=gap,
                    potential_saving_mills=potential_saving,
                    current_user_codes=tuple(sorted(current_users)),
                    eligible_incoming_codes=tuple(eligible),
                )
            )
    return tuple(
        sorted(
            exposures,
            key=lambda exposure: (
                -exposure.priority,
                -exposure.potential_saving_mills,
                exposure.gap_to_next,
                exposure.plant,
                exposure.internal.as_tuple(),
            ),
        )
    )


def threshold_free_codes(
    exposure: ProcurementExposure,
    products: Sequence[Product],
    *,
    max_codes: int,
) -> frozenset[str]:
    """Select a compact cross-plant repair set for one threshold exposure."""

    if max_codes < 1:
        raise ValueError("max codes must be positive")
    product_by_code = {product.code: product for product in products}
    chosen: list[str] = list(exposure.current_user_codes)
    seen = set(chosen)

    # Prefer incoming SKUs whose plant demand most efficiently closes the gap.
    incoming = sorted(
        (code for code in exposure.eligible_incoming_codes if code not in seen),
        key=lambda code: (
            abs(product_by_code[code].annual_volume_by_plant[exposure.plant] - exposure.gap_to_next),
            -product_by_code[code].annual_volume_by_plant[exposure.plant],
            code,
        ),
    )
    for code in incoming:
        if len(chosen) >= max_codes:
            break
        chosen.append(code)
        seen.add(code)
    return frozenset(chosen)


def rins_disagreement_order(
    products: Sequence[Product],
    incumbent: Mapping[str, CandidateBox],
    arc_values: Mapping[tuple[str, Dimensions], float],
    reduced_costs_mills: Mapping[tuple[str, Dimensions], float],
) -> tuple[str, ...]:
    """Rank SKU whose LP row most strongly disagrees with the incumbent.

    A zero LP value on the incumbent arc is a standard RINS signal.  Reduced
    costs break ties so rows with promising alternative designs are considered
    before merely fractional but economically inactive rows.
    """

    ranked: list[tuple[float, float, str]] = []
    for product in products:
        code = product.code
        incumbent_internal = incumbent[code].internal
        incumbent_value = arc_values.get((code, incumbent_internal), 0.0)
        alternatives = [
            reduced
            for (arc_code, internal), reduced in reduced_costs_mills.items()
            if arc_code == code and internal != incumbent_internal
        ]
        best_alternative_reduced = min(alternatives, default=0.0)
        ranked.append((1.0 - incumbent_value, -best_alternative_reduced, code))
    ranked.sort(key=lambda item: (-item[0], -item[1], item[2]))
    return tuple(code for _, _, code in ranked)
