"""Enumeración exacta de candidatos enteros para el modelo del FAQ #10.

La grilla global de dimensiones factibles es lo bastante pequeña para
enumerarse. Las máscaras de bits permiten calcular el conjunto completo de SKU
compatibles de cada caja sin probar cada producto en el bucle interno. Los
diseños con igual capacidad y compatibilidad son equivalentes para el objetivo;
luego se eliminan con seguridad los diseños dominados.
"""

from __future__ import annotations

from bisect import bisect_right
from collections import defaultdict
from dataclasses import dataclass
import math

from .config import (
    ECT_N_PER_M,
    GRAVITY_M_PER_S2,
    PALLET_LENGTH_MM,
    PALLET_MAX_HEIGHT_MM,
    PALLET_WIDTH_MM,
)
from .geometry import (
    EPSILON,
    boxes_per_pallet,
    external_from_internal,
    flexible_product_fits_candidate,
    headspace_maxima,
    product_fits_candidate,
)
from .models import CandidateBox, Dimensions, Product


# Formas racionales exactas de los porcentajes documentados por grosor.
HEADSPACE_FRACTIONS: dict[float, tuple[int, int]] = {
    3.0: (3, 50),
    4.5: (2, 25),
    5.0: (1, 10),
}


@dataclass(frozen=True)
class ExactCandidateStats:
    grid_size: int
    feasible_signatures: int
    nondominated_signatures: int
    retained_designs_added: int
    compatibility_links: int


@dataclass(frozen=True)
class _Signature:
    capacity: int
    compatible_mask: int
    internal: tuple[int, int, int]


def _whole_mm(value: float) -> int:
    rounded = round(value)
    if abs(value - rounded) > EPSILON:
        raise ValueError(f"exact enumeration requires whole-mm source dimensions, got {value}")
    return int(rounded)


def exact_axis_bounds(original_mm: float, thickness_mm: float) -> tuple[int, int]:
    """Límites enteros inclusivos de un eje fuente posiblemente fraccionario."""

    original = float(original_mm)
    if original <= 0:
        raise ValueError(f"source dimensions must be positive, got {original_mm}")
    try:
        numerator, denominator = HEADSPACE_FRACTIONS[thickness_mm]
    except KeyError as exc:
        raise ValueError(f"unsupported thickness: {thickness_mm}") from exc
    lower = math.ceil(0.90 * original - EPSILON)
    upper = math.floor(
        min(
            1.10 * original,
            denominator * original / (denominator - numerator),
            original + 40,
        )
        + EPSILON
    )
    return int(lower), int(upper)


def _prefix_masks(values_by_product: tuple[float, ...]) -> tuple[tuple[float, ...], tuple[int, ...]]:
    """Construye máscaras acumuladas para consultas `valor <= umbral`."""

    grouped: dict[float, int] = defaultdict(int)
    for product_index, value in enumerate(values_by_product):
        grouped[value] |= 1 << product_index
    values: list[float] = []
    masks: list[int] = []
    running_mask = 0
    for value in sorted(grouped):
        running_mask |= grouped[value]
        values.append(value)
        masks.append(running_mask)
    return tuple(values), tuple(masks)


def _mask_at_most(
    sorted_values: tuple[float, ...], prefix_masks: tuple[int, ...], threshold: float
) -> int:
    index = bisect_right(sorted_values, threshold) - 1
    return prefix_masks[index] if index >= 0 else 0


def _axis_masks(
    products: tuple[Product, ...], thickness_mm: float, axis: int, *, flexible_layout: bool
) -> tuple[int, tuple[int, ...]]:
    bounds = tuple(
        (
            (
                math.ceil(
                    0.90 * product.current_internal.as_tuple()[axis] - EPSILON
                ),
                math.floor(
                    1.10 * product.current_internal.as_tuple()[axis] + EPSILON
                ),
            )
            if flexible_layout
            else exact_axis_bounds(product.current_internal.as_tuple()[axis], thickness_mm)
        )
        for product in products
    )
    minimum = min(lower for lower, _ in bounds)
    maximum = max(upper for _, upper in bounds)
    masks = []
    for dimension in range(minimum, maximum + 1):
        mask = 0
        for product_index, (lower, upper) in enumerate(bounds):
            if lower <= dimension <= upper:
                mask |= 1 << product_index
        masks.append(mask)
    return minimum, tuple(masks)


def _iter_set_bits(mask: int):
    while mask:
        least_significant = mask & -mask
        yield least_significant.bit_length() - 1
        mask ^= least_significant


def _representative_rank(internal: tuple[int, int, int]) -> tuple[int, tuple[int, int, int]]:
    return math.prod(internal), internal


def _enumerate_signatures(
    products: tuple[Product, ...], thickness_mm: float, *, flexible_layout: bool
) -> tuple[list[_Signature], int]:
    length_start, length_masks = _axis_masks(products, thickness_mm, 0, flexible_layout=flexible_layout)
    width_start, width_masks = _axis_masks(products, thickness_mm, 1, flexible_layout=flexible_layout)
    height_start, height_masks = _axis_masks(products, thickness_mm, 2, flexible_layout=flexible_layout)
    grid_size = len(length_masks) * len(width_masks) * len(height_masks)

    volume_values, volume_masks = _prefix_masks(
        tuple(product.product_volume_mm3 for product in products)
    )
    weight_values, weight_masks = _prefix_masks(
        tuple(product.net_weight_kg for product in products)
    )
    twice_thickness = _whole_mm(2 * thickness_mm)
    representatives: dict[tuple[int, int], tuple[int, int, int]] = {}

    for length_offset, length_mask in enumerate(length_masks):
        if not length_mask:
            continue
        length = length_start + length_offset
        external_length = length + twice_thickness
        along_short_side = PALLET_WIDTH_MM // external_length
        if along_short_side < 1:
            continue

        for width_offset, width_mask in enumerate(width_masks):
            partial_mask = length_mask & width_mask
            if not partial_mask:
                continue
            width = width_start + width_offset
            external_width = width + twice_thickness
            along_long_side = PALLET_LENGTH_MM // external_width
            if along_long_side < 1:
                continue

            compression_capacity = (
                ECT_N_PER_M[thickness_mm]
                * 2
                * (external_length + external_width)
                / 1000.0
                / GRAVITY_M_PER_S2
            )
            weight_mask_by_layers: dict[int, int] = {}
            for height_offset, height_mask in enumerate(height_masks):
                compatible_mask = partial_mask & height_mask
                if not compatible_mask:
                    continue
                height = height_start + height_offset
                external_height = height + twice_thickness
                layers = PALLET_MAX_HEIGHT_MM // external_height
                if layers < 1:
                    continue

                compatible_mask &= _mask_at_most(
                    volume_values, volume_masks, length * width * height + EPSILON
                )
                if not compatible_mask:
                    continue
                if flexible_layout:
                    internal_dimensions = Dimensions(length, width, height)
                    headspace = headspace_maxima(internal_dimensions, thickness_mm)
                    minimum_product_volume = math.prod(
                        proposed - maximum
                        for proposed, maximum in zip(
                            internal_dimensions.as_tuple(), headspace.as_tuple()
                        )
                    )
                    compatible_mask &= ~_mask_at_most(
                        volume_values,
                        volume_masks,
                        minimum_product_volume - EPSILON,
                    )
                    if not compatible_mask:
                        continue
                if layers > 1:
                    if layers not in weight_mask_by_layers:
                        maximum_weight = (compression_capacity + EPSILON) / (layers - 1)
                        weight_mask_by_layers[layers] = _mask_at_most(
                            weight_values, weight_masks, maximum_weight
                        )
                    compatible_mask &= weight_mask_by_layers[layers]
                    if not compatible_mask:
                        continue

                capacity = along_short_side * along_long_side * layers
                signature = (capacity, compatible_mask)
                internal = (length, width, height)
                current = representatives.get(signature)
                if current is None or _representative_rank(internal) < _representative_rank(current):
                    representatives[signature] = internal

    signatures = [
        _Signature(capacity, compatible_mask, internal)
        for (capacity, compatible_mask), internal in representatives.items()
    ]
    return signatures, grid_size


def _remove_dominated(signatures: list[_Signature]) -> list[_Signature]:
    """Elimina cajas dominadas por un superconjunto de mayor capacidad."""

    if not signatures:
        return []
    ordered_indices = sorted(
        range(len(signatures)),
        key=lambda index: (
            -signatures[index].capacity,
            -signatures[index].compatible_mask.bit_count(),
            signatures[index].internal,
        ),
    )
    candidates_by_product: dict[int, int] = defaultdict(int)
    eligible_candidates = 0
    dominated: set[int] = set()
    cursor = 0
    while cursor < len(ordered_indices):
        capacity = signatures[ordered_indices[cursor]].capacity
        group_end = cursor
        while (
            group_end < len(ordered_indices)
            and signatures[ordered_indices[group_end]].capacity == capacity
        ):
            candidate_index = ordered_indices[group_end]
            candidate_bit = 1 << candidate_index
            eligible_candidates |= candidate_bit
            for product_index in _iter_set_bits(signatures[candidate_index].compatible_mask):
                candidates_by_product[product_index] |= candidate_bit
            group_end += 1

        for position in range(cursor, group_end):
            candidate_index = ordered_indices[position]
            possible_dominators = eligible_candidates & ~(1 << candidate_index)
            product_indices = list(_iter_set_bits(signatures[candidate_index].compatible_mask))
            product_indices.sort(key=lambda index: candidates_by_product[index].bit_count())
            for product_index in product_indices:
                possible_dominators &= candidates_by_product[product_index]
                if not possible_dominators:
                    break
            if possible_dominators:
                dominated.add(candidate_index)
        cursor = group_end

    return [
        signature for index, signature in enumerate(signatures) if index not in dominated
    ]


def _mask_to_codes(mask: int, products: tuple[Product, ...]) -> frozenset[str]:
    return frozenset(products[index].code for index in _iter_set_bits(mask))


def generate_exact_candidates(
    products: tuple[Product, ...],
    thickness_mm: float,
    *,
    retained_designs: tuple[Dimensions, ...] = (),
    prune_dominated: bool = True,
    flexible_layout: bool = False,
) -> tuple[tuple[CandidateBox, ...], ExactCandidateStats]:
    """Enumera la grilla entera y devuelve los diseños relevantes al objetivo."""

    signatures, grid_size = _enumerate_signatures(
        products, thickness_mm, flexible_layout=flexible_layout
    )
    feasible_signature_count = len(signatures)
    if prune_dominated:
        signatures = _remove_dominated(signatures)

    by_internal: dict[tuple[int, int, int], _Signature] = {
        signature.internal: signature for signature in signatures
    }
    retained_added = 0
    for retained in retained_designs:
        internal_tuple = tuple(_whole_mm(value) for value in retained.as_tuple())
        if internal_tuple in by_internal:
            continue
        internal = Dimensions(*internal_tuple)
        compatible_mask = 0
        for product_index, product in enumerate(products):
            fit_oracle = (
                flexible_product_fits_candidate if flexible_layout else product_fits_candidate
            )
            if fit_oracle(product, internal, thickness_mm):
                compatible_mask |= 1 << product_index
        if not compatible_mask:
            raise ValueError(f"retained design {internal_tuple} is infeasible for every SKU")
        external = external_from_internal(internal, thickness_mm)
        signature = _Signature(boxes_per_pallet(external), compatible_mask, internal_tuple)
        signatures.append(signature)
        by_internal[internal_tuple] = signature
        retained_added += 1

    signatures.sort(key=lambda signature: signature.internal)
    candidates: list[CandidateBox] = []
    for ordinal, signature in enumerate(signatures):
        internal = Dimensions(*signature.internal)
        external = external_from_internal(internal, thickness_mm)
        compatible_codes = _mask_to_codes(signature.compatible_mask, products)
        # Esta redundancia es deliberada: hace fallar cualquier discrepancia
        # entre las máscaras aceleradas y las reglas geométricas independientes
    # antes de que un solucionador o un archivo de entrega use el candidato.
        fit_oracle = (
            flexible_product_fits_candidate if flexible_layout else product_fits_candidate
        )
        if any(
            not fit_oracle(products[index], internal, thickness_mm)
            for index in _iter_set_bits(signature.compatible_mask)
        ):
            raise AssertionError(f"accelerated compatibility mismatch for {signature.internal}")
        candidates.append(
            CandidateBox(
                candidate_id=f"exact_t{thickness_mm:g}_{ordinal:05d}",
                thickness_mm=thickness_mm,
                internal=internal,
                external=external,
                capacity_per_pallet=signature.capacity,
                compatible_product_codes=compatible_codes,
            )
        )

    covered_codes = set().union(
        *(candidate.compatible_product_codes for candidate in candidates)
    )
    missing = {product.code for product in products} - covered_codes
    if missing:
        raise ValueError(f"exact candidate generation left SKUs uncovered: {sorted(missing)}")
    stats = ExactCandidateStats(
        grid_size=grid_size,
        feasible_signatures=feasible_signature_count,
        nondominated_signatures=len(signatures) - retained_added,
        retained_designs_added=retained_added,
        compatibility_links=sum(len(candidate.compatible_product_codes) for candidate in candidates),
    )
    return tuple(candidates), stats
