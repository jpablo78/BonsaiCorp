"""Reglas decimales mínimas requeridas para validar el CSV final."""

from __future__ import annotations

from .geometry import (
    boxes_per_pallet,
    compression_feasible,
    faq_reconciled_headspace_feasible,
    respects_dimension_adjustment,
)
from .models import Dimensions, Product


def decimal_external_from_internal(
    internal: Dimensions, thickness_mm: float
) -> Dimensions:
    """Convierte dimensiones internas en exteriores sin perder precisión decimal."""

    return Dimensions(
        round(internal.length + 2 * thickness_mm, 6),
        round(internal.width + 2 * thickness_mm, 6),
        round(internal.height + 2 * thickness_mm, 6),
    )


def decimal_product_fits_candidate(
    product: Product, internal: Dimensions, thickness_mm: float
) -> bool:
    """Verifica ajuste, headspace, ECT y palletización para una caja decimal."""

    if not respects_dimension_adjustment(product, internal):
        return False
    if not faq_reconciled_headspace_feasible(product, internal, thickness_mm):
        return False
    external = decimal_external_from_internal(internal, thickness_mm)
    return boxes_per_pallet(external) > 0 and compression_feasible(
        product, external, thickness_mm
    )
