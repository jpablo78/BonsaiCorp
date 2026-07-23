"""Geometry, headspace, pallet and compression feasibility rules."""

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
    """Calculate external dimensions; approved output dimensions must be integer mm."""

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
    """Apply the ±10% per-axis fit band around the current internal dimensions."""

    return all(
        0.90 * original - EPSILON <= proposed <= 1.10 * original + EPSILON
        for original, proposed in zip(product.current_internal.as_tuple(), internal.as_tuple())
    )


def flexible_volume_headspace_feasible(
    product_volume_mm3: float, internal: Dimensions, thickness_mm: float
) -> bool:
    """Verify existence of a valid three-axis headspace allocation.

    The product is flexible: only its volume is invariant.  Let `p_a` be its
    effective dimension and `hs_a` its headspace in each axis.  A feasible
    configuration requires p_a + hs_a = box_a, product(p_a) = V, and
    0 <= hs_a <= HSmax_a.  As every axis is continuous and positive, this is
    equivalent to `product(box_a - HSmax_a) <= V <= product(box_a)`.

    Crucially, `p_a` is not the current physical dimension on that axis.  A
    larger proposed height may be exactly offset by a smaller proposed width,
    with no headspace at all when the two volumes are equal.
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
    """Apply the documented headspace rule used by the Kaggle checker.

    The current internal dimensions are the product dimensions on each axis.
    Headspace is therefore `internal_axis - product_axis`; it cannot be
    negative and it must not exceed the thickness-specific percentage or the
    40-mm cap.  This is independent from the FAQ's ±10% box-fit band.
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
    """Apply FAQ #10 with a positive per-axis headspace cap.

    FAQ #10 expressly allows reducing an internal box dimension by up to 10%
    when the resulting internal volume remains enough for the product.  The
    headspace cap is therefore applied only to a positive enlargement versus
    the current internal dimension; a permitted reduction is not interpreted
    as negative headspace.
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
    """Return layers, boxes along pallet's 800-mm side, and along 1200-mm side."""

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
    """ECT capacity using the approved *external* L/W perimeter."""

    perimeter_m = 2 * (external.length + external.width) / 1000.0
    return ECT_N_PER_M[thickness_mm] * perimeter_m / GRAVITY_M_PER_S2


def compression_feasible(product: Product, external: Dimensions, thickness_mm: float) -> bool:
    """A pallet carries only one SKU, so every box supports its own layers above."""

    layers, _, _ = pallet_layout(external)
    if layers < 1:
        return False
    upper_layer_load = max(layers - 1, 0) * product.net_weight_kg
    return compression_capacity_kg(external, thickness_mm) + EPSILON >= upper_layer_load


def product_fits_candidate(product: Product, internal: Dimensions, thickness_mm: float) -> bool:
    """Apply FAQ #10 fit, positive headspace, pallet and compression constraints."""

    if not respects_dimension_adjustment(product, internal):
        return False
    if not faq_reconciled_headspace_feasible(product, internal, thickness_mm):
        return False
    external = external_from_internal(internal, thickness_mm)
    return boxes_per_pallet(external) > 0 and compression_feasible(product, external, thickness_mm)


def flexible_product_fits_candidate(
    product: Product, internal: Dimensions, thickness_mm: float
) -> bool:
    """Diagnostic oracle for a liquid-like product interpretation.

    Current internal dimensions determine invariant product volume and the
    documented +/-10% box band, but not the product's future per-axis layout.
    Headspace can be allocated on any combination of the three axes.

    Do not use this oracle for submissions: a 2026-07-22 Kaggle submission
    built with it scored zero.  The checker empirically requires the stricter
    positive per-axis headspace rule implemented by ``product_fits_candidate``.
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
