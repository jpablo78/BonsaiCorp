"""Reglas de factibilidad de geometría, headspace, pallet y compresión."""

from __future__ import annotations

import math

from .config import (
    ECT_N_PER_M,
    GRAVITY_M_PER_S2,
    PALLET_LENGTH_MM,
    PALLET_MAX_HEIGHT_MM,
    PALLET_WIDTH_MM,
)
from .models import Dimensions, Product


EPSILON = 1e-8


def external_from_internal(internal: Dimensions, thickness_mm: float) -> Dimensions:
    """Calcula dimensiones exteriores; la salida estándar usa milímetros enteros."""

    external = Dimensions(
        internal.length + 2 * thickness_mm,
        internal.width + 2 * thickness_mm,
        internal.height + 2 * thickness_mm,
    )
    if any(not float(value).is_integer() for value in external.as_tuple()):
        raise ValueError("integer-mm output requires integer external dimensions")
    return external


def headspace_percentage(thickness_mm: float) -> float:
    if thickness_mm == 3.0:
        return 0.06
    if thickness_mm == 4.5:
        return 0.08
    if thickness_mm == 5.0:
        return 0.10
    raise ValueError(f"unsupported thickness: {thickness_mm}")


def headspace_maxima(internal: Dimensions, thickness_mm: float) -> Dimensions:
    percentage = headspace_percentage(thickness_mm)
    return Dimensions(
        min(percentage * internal.length, 40.0),
        min(percentage * internal.width, 40.0),
        min(percentage * internal.height, 40.0),
    )


def respects_dimension_adjustment(product: Product, internal: Dimensions) -> bool:
    """Aplica la banda de ajuste ±10% por eje sobre las dimensiones internas actuales."""

    return all(
        0.90 * original - EPSILON <= proposed <= 1.10 * original + EPSILON
        for original, proposed in zip(product.current_internal.as_tuple(), internal.as_tuple())
    )


def flexible_volume_headspace_feasible(
    product_volume_mm3: float, internal: Dimensions, thickness_mm: float
) -> bool:
    """Verifica si existe una asignación válida de headspace en tres ejes.

    El producto se considera flexible: sólo su volumen es invariante. Sea
    `p_a` su dimensión efectiva y `hs_a` su headspace en cada eje. Una
    configuración factible requiere p_a + hs_a = caja_a, product(p_a) = V y
    0 <= hs_a <= HSmax_a. Como cada eje es continuo y positivo, equivale a
    `product(caja_a - HSmax_a) <= V <= product(caja_a)`.

    Es crucial que `p_a` no sea la dimensión física actual en ese eje. Una
    altura propuesta mayor podría compensarse exactamente con un ancho menor,
    sin headspace, si los dos volúmenes fueran iguales.
    """

    hs_max = headspace_maxima(internal, thickness_mm)
    lower = Dimensions(
        internal.length - hs_max.length,
        internal.width - hs_max.width,
        internal.height - hs_max.height,
    )
    if min(lower.as_tuple()) <= 0:
        return False
    return lower.volume_mm3 - EPSILON <= product_volume_mm3 <= internal.volume_mm3 + EPSILON


def documented_axis_headspace_feasible(
    product: Product, internal: Dimensions, thickness_mm: float
) -> bool:
    """Aplica la regla documentada de headspace usada por el validador Kaggle.

    Las dimensiones internas actuales son las dimensiones del producto en
    cada eje. El headspace es entonces `eje_interno - eje_producto`: no puede
    ser negativo y no debe exceder el porcentaje por grosor ni el tope de
    40 mm. Esto es independiente de la banda ±10% de ajuste del FAQ.
    """

    maxima = headspace_maxima(internal, thickness_mm)
    return all(
        -EPSILON <= proposed - original <= maximum + EPSILON
        for original, proposed, maximum in zip(
            product.current_internal.as_tuple(), internal.as_tuple(), maxima.as_tuple()
        )
    )


def faq_reconciled_headspace_feasible(
    product: Product, internal: Dimensions, thickness_mm: float
) -> bool:
    """Aplica el FAQ #10 con tope positivo de headspace por eje.

    El FAQ #10 permite expresamente reducir una dimensión interna hasta 10%
    cuando el volumen interno resultante siga alcanzando para el producto. Por
    eso, el tope de headspace se aplica sólo a un aumento positivo respecto de
    la dimensión interna actual; una reducción permitida no se interpreta como
    headspace negativo.
    """

    if internal.volume_mm3 + EPSILON < product.product_volume_mm3:
        return False
    maxima = headspace_maxima(internal, thickness_mm)
    return all(
        proposed - original <= maximum + EPSILON
        for original, proposed, maximum in zip(
            product.current_internal.as_tuple(), internal.as_tuple(), maxima.as_tuple()
        )
    )


def pallet_layout(external: Dimensions) -> tuple[int, int, int]:
    """Devuelve capas y cajas sobre los lados de 800 y 1.200 mm del pallet."""

    layers = math.floor(PALLET_MAX_HEIGHT_MM / external.height)
    along_short_side = math.floor(PALLET_WIDTH_MM / external.length)
    along_long_side = math.floor(PALLET_LENGTH_MM / external.width)
    return layers, along_short_side, along_long_side


def boxes_per_pallet(external: Dimensions) -> int:
    layers, along_short_side, along_long_side = pallet_layout(external)
    return layers * along_short_side * along_long_side


def pallet_utilization(external: Dimensions) -> float:
    capacity = boxes_per_pallet(external)
    pallet_volume = PALLET_LENGTH_MM * PALLET_WIDTH_MM * PALLET_MAX_HEIGHT_MM
    return capacity * external.volume_mm3 / pallet_volume


def compression_capacity_kg(external: Dimensions, thickness_mm: float) -> float:
    """Capacidad ECT usando el perímetro exterior L/A aprobado."""

    perimeter_m = 2 * (external.length + external.width) / 1000.0
    return ECT_N_PER_M[thickness_mm] * perimeter_m / GRAVITY_M_PER_S2


def compression_feasible(product: Product, external: Dimensions, thickness_mm: float) -> bool:
    """Un pallet lleva un solo SKU: cada caja soporta sus capas superiores."""

    layers, _, _ = pallet_layout(external)
    if layers < 1:
        return False
    upper_layer_load = max(layers - 1, 0) * product.net_weight_kg
    return compression_capacity_kg(external, thickness_mm) + EPSILON >= upper_layer_load


def product_fits_candidate(product: Product, internal: Dimensions, thickness_mm: float) -> bool:
    """Aplica restricciones de FAQ #10, headspace, pallet y compresión."""

    if not respects_dimension_adjustment(product, internal):
        return False
    if not faq_reconciled_headspace_feasible(product, internal, thickness_mm):
        return False
    external = external_from_internal(internal, thickness_mm)
    return boxes_per_pallet(external) > 0 and compression_feasible(product, external, thickness_mm)


def flexible_product_fits_candidate(
    product: Product, internal: Dimensions, thickness_mm: float
) -> bool:
    """Oráculo diagnóstico para una interpretación de producto tipo líquido.

    Las dimensiones internas actuales determinan el volumen invariante y la
    banda documentada +/-10% de la caja, pero no la disposición futura por
    eje del producto. El headspace puede asignarse a cualquier combinación de
    los tres ejes.

    No usar este oráculo en submissions: una entrega de 2026-07-22 basada en
    él obtuvo score cero. El validador requiere empíricamente la regla más
    estricta de headspace positivo por eje de ``product_fits_candidate``.
    """

    if not respects_dimension_adjustment(product, internal):
        return False
    if not flexible_volume_headspace_feasible(
        product.product_volume_mm3, internal, thickness_mm
    ):
        return False
    external = external_from_internal(internal, thickness_mm)
    return boxes_per_pallet(external) > 0 and compression_feasible(
        product, external, thickness_mm
    )
