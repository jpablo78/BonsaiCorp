"""Grandes vecindarios económicos centrados en tiers de descuento de Procurement.

Los acoplamientos útiles de este problema no son necesariamente geográficos.
Un diseño de caja se compra por separado en cada planta y mover un conjunto de
SKU hacia o desde un diseño puede cruzar un umbral de descuento sobre todas las
unidades. Estas utilidades exponen esos acoplamientos para reparaciones SCIP
exactas y restringidas.
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
    """Un volumen actual de diseño/planta con un próximo tier de descuento alcanzable."""

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
        """Ahorro potencial ponderado hacia umbrales más fáciles de alcanzar."""

        return self.potential_saving_mills / max(1, self.gap_to_next)


def rank_procurement_exposures(
    products: Sequence[Product],
    incumbent: Mapping[str, CandidateBox],
    candidates: Sequence[CandidateBox],
) -> tuple[ProcurementExposure, ...]:
    """Ordena volúmenes actuales tipo/planta que podrían alcanzar su próximo tier de precio.

    El ahorro potencial es exacto *condicionado a alcanzar el umbral*: el
    menor precio se aplica entonces a cada unidad de ese tipo físico en la planta.
    Es sólo una señal de prioridad, nunca una aproximación del objetivo.
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
    # Los diseños retenidos de la incumbente siempre pertenecen al universo exacto.
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
    """Selecciona un conjunto compacto de reparación multiplanta para una exposición a umbral."""

    if max_codes < 1:
        raise ValueError("max codes must be positive")
    product_by_code = {product.code: product for product in products}
    chosen: list[str] = list(exposure.current_user_codes)
    seen = set(chosen)

    # Se prefieren SKU entrantes cuya demanda en planta cierra la brecha con mayor eficiencia.
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
    """Ordena SKU cuya fila LP discrepa con mayor fuerza de la incumbente.

    Un valor LP nulo en el arco incumbente es una señal estándar de RINS. Los
    costos reducidos desempatan para considerar antes filas con diseños
    alternativos prometedores que filas sólo fraccionarias e inactivas económicamente.
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
