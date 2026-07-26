"""Cotas inferiores rigurosas de costo por grosor."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .config import FreightPolicy
from .costs import freight_pallets, unit_price_mills
from .exact_candidates import ExactCandidateStats, generate_exact_candidates
from .models import PLANTS, PreparedData


@dataclass(frozen=True)
class ThicknessLowerBound:
    thickness_mm: float
    minimum_pallets: int
    freight_lower_bound_mills: int
    packaging_lower_bound_mills: int
    total_lower_bound_mills: int
    candidate_stats: ExactCandidateStats

    def as_dict(self) -> dict[str, object]:
        return {
            "thickness_mm": self.thickness_mm,
            "minimum_pallets": self.minimum_pallets,
            "freight_lower_bound_usd": self.freight_lower_bound_mills / 1000,
            "packaging_lower_bound_usd": self.packaging_lower_bound_mills / 1000,
            "total_lower_bound_usd": self.total_lower_bound_mills / 1000,
            "candidate_stats": asdict(self.candidate_stats),
        }


def thickness_cost_lower_bound(
    data: PreparedData, thickness_mm: float, freight_policy: FreightPolicy
) -> ThicknessLowerBound:
    """Combina el flete mínimo exacto con una cota optimista de Procurement.

    El flete se minimiza de forma independiente para cada SKU en la grilla
    completa de candidatos enteros. Procurement supone, de manera optimista,
    que toda la demanda de una planta puede usar un tipo común. Ese tipo no
    necesita ser geométricamente factible, por lo que la suma resultante es
    una cota inferior válida, no una solución propuesta.
    """

    candidates, candidate_stats = generate_exact_candidates(data.products, thickness_mm)
    minimum_pallets = 0
    for product in data.products:
        maximum_capacity = max(
            candidate.capacity_per_pallet
            for candidate in candidates
            if product.code in candidate.compatible_product_codes
        )
        # Cualquier candidato de capacidad máxima minimiza el término de techo de cada planta.
        representative = next(
            candidate
            for candidate in candidates
            if product.code in candidate.compatible_product_codes
            and candidate.capacity_per_pallet == maximum_capacity
        )
        minimum_pallets += sum(
            freight_pallets(product, representative, plant) for plant in PLANTS
        )

    packaging_lower_bound_mills = 0
    for plant in PLANTS:
        plant_volume = sum(
            product.annual_volume_by_plant[plant] for product in data.products
        )
        if plant_volume:
            packaging_lower_bound_mills += plant_volume * unit_price_mills(
                thickness_mm, plant_volume
            )
    freight_lower_bound_mills = (
        minimum_pallets * freight_policy.expected_mills_per_pallet
    )
    return ThicknessLowerBound(
        thickness_mm=thickness_mm,
        minimum_pallets=minimum_pallets,
        freight_lower_bound_mills=freight_lower_bound_mills,
        packaging_lower_bound_mills=packaging_lower_bound_mills,
        total_lower_bound_mills=freight_lower_bound_mills + packaging_lower_bound_mills,
        candidate_stats=candidate_stats,
    )
