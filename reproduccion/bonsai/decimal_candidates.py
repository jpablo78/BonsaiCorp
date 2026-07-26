"""Enumeración exacta sobre una grilla decimal configurable de dimensiones internas.

La relajación decimal conserva las reglas estrictas de fit/headspace por eje,
ECT, orientación fija del pallet y volumen del producto. Los valores de cada
eje se generan a partir de eventos donde cambia el estado, en lugar de recorrer
cada punto. Esto vuelve práctica una grilla de micromilímetros e incluye los
vecinos discretos de cada umbral de fit, headspace y capacidad de pallet.
"""

from __future__ import annotations

from bisect import bisect_right
from collections import defaultdict
from dataclasses import dataclass
from fractions import Fraction
import math

from .config import ECT_N_PER_M, GRAVITY_M_PER_S2
from .exact_candidates import HEADSPACE_FRACTIONS, _Signature, _remove_dominated
from .geometry import (
    EPSILON,
    boxes_per_pallet,
    compression_feasible,
    faq_reconciled_headspace_feasible,
    respects_dimension_adjustment,
)
from .models import CandidateBox, Dimensions, Product


@dataclass(frozen=True)
class DecimalCandidateStats:
    precision_mm: float
    decimal_places: int
    raw_grid_points: int
    compressed_length_values: int
    compressed_width_values: int
    compressed_height_values: int
    compressed_grid_points: int
    feasible_signatures: int
    nondominated_signatures: int
    retained_designs_added: int
    compatibility_links: int


def decimal_external_from_internal(
    internal: Dimensions, thickness_mm: float
) -> Dimensions:
    return Dimensions(
        round(internal.length + 2 * thickness_mm, 6),
        round(internal.width + 2 * thickness_mm, 6),
        round(internal.height + 2 * thickness_mm, 6),
    )


def decimal_product_fits_candidate(
    product: Product, internal: Dimensions, thickness_mm: float
) -> bool:
    if not respects_dimension_adjustment(product, internal):
        return False
    if not faq_reconciled_headspace_feasible(product, internal, thickness_mm):
        return False
    external = decimal_external_from_internal(internal, thickness_mm)
    return boxes_per_pallet(external) > 0 and compression_feasible(
        product, external, thickness_mm
    )


def _axis_bounds_units(
    original: float, thickness_mm: float, scale: int
) -> tuple[int, int]:
    numerator, denominator = HEADSPACE_FRACTIONS[thickness_mm]
    original_fraction = Fraction(str(original))
    lower_fraction = Fraction(9, 10) * original_fraction * scale
    lower = -(-lower_fraction.numerator // lower_fraction.denominator)
    upper_fraction = min(
        Fraction(11, 10) * original_fraction,
        Fraction(denominator, denominator - numerator) * original_fraction,
        original_fraction + 40,
    )
    scaled_upper = upper_fraction * scale
    upper = scaled_upper.numerator // scaled_upper.denominator
    return lower, upper


def _compressed_axis_values(
    products: tuple[Product, ...],
    thickness_mm: float,
    axis: int,
    pallet_side: float,
    scale: int,
) -> tuple[tuple[float, int, int], ...]:
    """Conserva el mayor valor de grilla en cada celda de estado del eje.

    Dentro de una celda, una dimensión mayor conserva compatibilidad por eje y
    capacidad de pallet, y mejora débilmente el volumen y, en ejes
    horizontales, ECT. Por eso domina todo valor menor de esa misma celda.

    Un estado sólo puede cambiar en un límite inferior de producto, justo
    después de un límite superior o justo después de un umbral de cantidad en
    pallet. Basta evaluar esos eventos y sus vecinos; no se recorre la enorme
    grilla completa del eje.
    """

    bounds = tuple(
        _axis_bounds_units(
            product.current_internal.as_tuple()[axis], thickness_mm, scale
        )
        for product in products
    )
    minimum = min(lower for lower, _ in bounds)
    maximum = max(upper for _, upper in bounds)
    pallet_units = round(pallet_side * scale)
    twice_thickness_units = round(2 * thickness_mm * scale)
    if not math.isclose(
        twice_thickness_units, 2 * thickness_mm * scale, abs_tol=EPSILON
    ):
        raise ValueError("thickness is not exactly representable on decimal grid")

    event_units = {minimum, maximum}
    for lower, upper in bounds:
        event_units.update((lower - 1, lower, upper, upper + 1))

    maximum_count = pallet_units // (minimum + twice_thickness_units)
    for count in range(1, maximum_count + 1):
        # count * unidades_exteriores <= unidades_pallet equivale exactamente
        # a que entren al menos ``count`` cajas sobre este eje del pallet.
        threshold = pallet_units // count - twice_thickness_units
        event_units.update((threshold, threshold + 1))

    representatives: dict[tuple[int, int], int] = {}
    for value_units in sorted(
        value for value in event_units if minimum <= value <= maximum
    ):
        mask = 0
        for product_index, (lower, upper) in enumerate(bounds):
            if lower <= value_units <= upper:
                mask |= 1 << product_index
        if not mask:
            continue
        count = pallet_units // (value_units + twice_thickness_units)
        if count > 0:
            representatives[(mask, count)] = value_units
    return tuple(
        sorted(
            (value_units / scale, mask, count)
            for (mask, count), value_units in representatives.items()
        )
    )


def _prefix_masks(values_by_product: tuple[float, ...]) -> tuple[tuple[float, ...], tuple[int, ...]]:
    grouped: dict[float, int] = defaultdict(int)
    for product_index, value in enumerate(values_by_product):
        grouped[value] |= 1 << product_index
    values: list[float] = []
    masks: list[int] = []
    running = 0
    for value in sorted(grouped):
        running |= grouped[value]
        values.append(value)
        masks.append(running)
    return tuple(values), tuple(masks)


def _mask_at_most(
    sorted_values: tuple[float, ...], prefix_masks: tuple[int, ...], threshold: float
) -> int:
    index = bisect_right(sorted_values, threshold) - 1
    return prefix_masks[index] if index >= 0 else 0


def _set_bits(mask: int):
    while mask:
        bit = mask & -mask
        yield bit.bit_length() - 1
        mask ^= bit


def generate_decimal_candidates(
    products: tuple[Product, ...],
    thickness_mm: float,
    *,
    decimal_places: int,
    retained_designs: tuple[Dimensions, ...] = (),
    prune_dominated: bool = True,
) -> tuple[tuple[CandidateBox, ...], DecimalCandidateStats]:
    if decimal_places < 1 or decimal_places > 6:
        raise ValueError("decimal_places must be between 1 and 6")
    scale = 10**decimal_places
    length_values = _compressed_axis_values(products, thickness_mm, 0, 800, scale)
    width_values = _compressed_axis_values(products, thickness_mm, 1, 1200, scale)
    height_values = _compressed_axis_values(products, thickness_mm, 2, 1800, scale)
    volume_values, volume_masks = _prefix_masks(
        tuple(product.product_volume_mm3 for product in products)
    )
    weight_values, weight_masks = _prefix_masks(
        tuple(product.net_weight_kg for product in products)
    )

    representatives: dict[tuple[int, int], tuple[float, float, float]] = {}
    weight_mask_cache: dict[tuple[float, float, int], int] = {}
    for length, length_mask, along_short in length_values:
        external_length = length + 2 * thickness_mm
        for width, width_mask, along_long in width_values:
            partial_mask = length_mask & width_mask
            if not partial_mask:
                continue
            external_width = width + 2 * thickness_mm
            compression_capacity = (
                ECT_N_PER_M[thickness_mm]
                * 2
                * (external_length + external_width)
                / 1000
                / GRAVITY_M_PER_S2
            )
            for height, height_mask, layers in height_values:
                compatible_mask = partial_mask & height_mask
                if not compatible_mask:
                    continue
                compatible_mask &= _mask_at_most(
                    volume_values,
                    volume_masks,
                    length * width * height + EPSILON,
                )
                if not compatible_mask:
                    continue
                if layers > 1:
                    cache_key = (length, width, layers)
                    weight_mask = weight_mask_cache.get(cache_key)
                    if weight_mask is None:
                        maximum_weight = (compression_capacity + EPSILON) / (layers - 1)
                        weight_mask = _mask_at_most(
                            weight_values, weight_masks, maximum_weight
                        )
                        weight_mask_cache[cache_key] = weight_mask
                    compatible_mask &= weight_mask
                    if not compatible_mask:
                        continue
                capacity = along_short * along_long * layers
                internal = (length, width, height)
                signature = (capacity, compatible_mask)
                current = representatives.get(signature)
                if current is None or (math.prod(internal), internal) < (
                    math.prod(current),
                    current,
                ):
                    representatives[signature] = internal

    signatures = [
        _Signature(capacity, compatible_mask, internal)
        for (capacity, compatible_mask), internal in representatives.items()
    ]
    feasible_count = len(signatures)
    if prune_dominated:
        signatures = _remove_dominated(signatures)

    by_internal = {signature.internal: signature for signature in signatures}
    retained_added = 0
    for retained in retained_designs:
        key = tuple(round(value, decimal_places) for value in retained.as_tuple())
        if key in by_internal:
            continue
        internal = Dimensions(*key)
        compatible_mask = 0
        for product_index, product in enumerate(products):
            if decimal_product_fits_candidate(product, internal, thickness_mm):
                compatible_mask |= 1 << product_index
        if not compatible_mask:
            raise ValueError(f"retained decimal design {key} covers no product")
        external = decimal_external_from_internal(internal, thickness_mm)
        signature = _Signature(boxes_per_pallet(external), compatible_mask, key)
        signatures.append(signature)
        by_internal[key] = signature
        retained_added += 1

    signatures.sort(key=lambda signature: signature.internal)
    candidates: list[CandidateBox] = []
    for ordinal, signature in enumerate(signatures):
        internal = Dimensions(*signature.internal)
        external = decimal_external_from_internal(internal, thickness_mm)
        compatible_codes = frozenset(
            products[index].code for index in _set_bits(signature.compatible_mask)
        )
        oracle_codes = frozenset(
            product.code
            for product in products
            if decimal_product_fits_candidate(product, internal, thickness_mm)
        )
        if oracle_codes != compatible_codes:
            raise AssertionError(
                f"decimal compatibility acceleration mismatch for {signature.internal}"
            )
        candidates.append(
            CandidateBox(
                candidate_id=f"decimal_t{thickness_mm:g}_{ordinal:05d}",
                thickness_mm=thickness_mm,
                internal=internal,
                external=external,
                capacity_per_pallet=signature.capacity,
                compatible_product_codes=compatible_codes,
            )
        )

    covered = set().union(*(candidate.compatible_product_codes for candidate in candidates))
    missing = {product.code for product in products} - covered
    if missing:
        raise ValueError(f"decimal candidate generation leaves SKUs uncovered: {sorted(missing)}")

    raw_axis_sizes = []
    for axis in range(3):
        bounds = tuple(
            _axis_bounds_units(
                product.current_internal.as_tuple()[axis], thickness_mm, scale
            )
            for product in products
        )
        raw_axis_sizes.append(
            max(upper for _, upper in bounds) - min(lower for lower, _ in bounds) + 1
        )
    stats = DecimalCandidateStats(
        precision_mm=1 / scale,
        decimal_places=decimal_places,
        raw_grid_points=math.prod(raw_axis_sizes),
        compressed_length_values=len(length_values),
        compressed_width_values=len(width_values),
        compressed_height_values=len(height_values),
        compressed_grid_points=len(length_values) * len(width_values) * len(height_values),
        feasible_signatures=feasible_count,
        nondominated_signatures=len(signatures) - retained_added,
        retained_designs_added=retained_added,
        compatibility_links=sum(len(candidate.compatible_product_codes) for candidate in candidates),
    )
    return tuple(candidates), stats


def generate_tenth_mm_candidates(
    products: tuple[Product, ...],
    thickness_mm: float,
    *,
    retained_designs: tuple[Dimensions, ...] = (),
    prune_dominated: bool = True,
) -> tuple[tuple[CandidateBox, ...], DecimalCandidateStats]:
    """Adaptador de compatibilidad para la grilla de 0,1 mm."""

    return generate_decimal_candidates(
        products,
        thickness_mm,
        decimal_places=1,
        retained_designs=retained_designs,
        prune_dominated=prune_dominated,
    )
