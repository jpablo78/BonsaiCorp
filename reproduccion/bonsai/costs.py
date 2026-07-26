"""Evaluación de costos independiente del modelo de optimización."""

from __future__ import annotations

import math
from collections import defaultdict

from .config import BASE_PRICE_USD, DISCOUNT_TIERS, FreightPolicy, MILLS_PER_USD
from .geometry import pallet_utilization
from .models import CandidateBox, CostBreakdown, PLANTS, Product


BoxTypeKey = tuple[float, float, float, float]


def box_type_key(candidate: CandidateBox) -> BoxTypeKey:
    """Devuelve la identidad comercial de un diseño físico de caja.

    Los ID de candidatos son etiquetas locales al solver. Los descuentos de
    Procurement y la cantidad informada de tipos se determinan por la caja
    física; por eso, diseños iguales deben consolidarse aunque provengan de
    universos de candidatos o archivos de solución distintos.
    """

    return (candidate.thickness_mm, *candidate.external.as_tuple())


def tier_index(volume: int) -> int:
    if volume < 1:
        raise ValueError("a selected box type must have positive annual volume")
    for index, tier in enumerate(DISCOUNT_TIERS):
        if tier.contains(volume):
            return index
    raise AssertionError("all positive volumes must fall into a discount tier")


def unit_price_mills(thickness_mm: float, volume: int) -> int:
    base_mills = round(BASE_PRICE_USD[thickness_mm] * MILLS_PER_USD)
    percentage = DISCOUNT_TIERS[tier_index(volume)].factor_percent
    return base_mills * percentage // 100


def freight_pallets(product: Product, candidate: CandidateBox, plant: str) -> int:
    volume = product.annual_volume_by_plant[plant]
    return math.ceil(volume / candidate.capacity_per_pallet) if volume else 0


def evaluate_assignments(
    products: tuple[Product, ...],
    assignment_by_code: dict[str, CandidateBox],
    freight_policy: FreightPolicy,
) -> CostBreakdown:
    """Recalcula todos los costos; no confía en el objetivo del solucionador."""

    if set(assignment_by_code) != {product.code for product in products}:
        raise ValueError("assignment must contain exactly one row for every product")

    volumes: dict[BoxTypeKey, dict[str, int]] = defaultdict(
        lambda: {plant: 0 for plant in PLANTS}
    )
    products_by_candidate: dict[BoxTypeKey, list[Product]] = defaultdict(list)
    candidates: dict[BoxTypeKey, CandidateBox] = {}
    for product in products:
        candidate = assignment_by_code[product.code]
        if product.code not in candidate.compatible_product_codes:
            raise ValueError(f"infeasible candidate {candidate.candidate_id} for {product.code}")
        type_key = box_type_key(candidate)
        candidates[type_key] = candidate
        products_by_candidate[type_key].append(product)
        for plant, volume in product.annual_volume_by_plant.items():
            volumes[type_key][plant] += volume

    packaging_mills = 0
    for type_key, by_plant in volumes.items():
        candidate = candidates[type_key]
        for plant, volume in by_plant.items():
            if volume:
                packaging_mills += volume * unit_price_mills(candidate.thickness_mm, volume)

    freight_mills = 0
    total_pallets = 0
    occupied_volume_by_plant = {plant: 0.0 for plant in PLANTS}
    capacity_volume_by_plant = {plant: 0.0 for plant in PLANTS}
    pallet_volume = 1200 * 800 * 1800
    for type_key, assigned_products in products_by_candidate.items():
        candidate = candidates[type_key]
        for product in assigned_products:
            for plant in PLANTS:
                volume = product.annual_volume_by_plant[plant]
                pallets = freight_pallets(product, candidate, plant)
                total_pallets += pallets
                freight_mills += pallets * freight_policy.expected_mills_per_pallet
                occupied_volume_by_plant[plant] += volume * candidate.external.volume_mm3
                capacity_volume_by_plant[plant] += pallets * pallet_volume

    utilization = {
        plant: (
            occupied_volume_by_plant[plant] / capacity_volume_by_plant[plant]
            if capacity_volume_by_plant[plant]
            else 0.0
        )
        for plant in PLANTS
    }
    return CostBreakdown(
        packaging_mills=packaging_mills,
        freight_mills=freight_mills,
        pallets=total_pallets,
        types=len(candidates),
        pallet_utilization_by_plant=utilization,
    )
