"""Genera un universo compacto y auditable de cajas factibles enteras."""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations_with_replacement
import math

from .config import PALLET_LENGTH_MM, PALLET_MAX_HEIGHT_MM, PALLET_WIDTH_MM
from .geometry import boxes_per_pallet, external_from_internal, product_fits_candidate
from .models import CandidateBox, Dimensions, Product


def _integer_dimensions(dimensions: Dimensions) -> Dimensions:
    values = tuple(round(value) for value in dimensions.as_tuple())
    if any(abs(original - rounded) > 1e-8 for original, rounded in zip(dimensions.as_tuple(), values)):
        raise ValueError("current product dimensions must be whole millimetres")
    return Dimensions(*values)


def _componentwise_max(left: Dimensions, right: Dimensions) -> Dimensions:
    return Dimensions(
        max(left.length, right.length),
        max(left.width, right.width),
        max(left.height, right.height),
    )


def _integer_fit_bounds(value: float) -> tuple[int, int]:
    """Devuelve los límites enteros inclusivos para el ajuste ±10% del FAQ."""

    return math.ceil(0.90 * value), math.floor(1.10 * value)


def _pallet_edge_values(
    pallet_dimension_mm: int,
    lower_internal_mm: int,
    upper_internal_mm: int,
    thickness_mm: float,
) -> tuple[int, ...]:
    """Tamaños internos donde puede entrar una caja adicional en un eje del pallet."""

    values = {lower_internal_mm, upper_internal_mm}
    external_lower = lower_internal_mm + 2 * thickness_mm
    external_upper = upper_internal_mm + 2 * thickness_mm
    minimum_count = max(1, math.floor(pallet_dimension_mm / external_upper))
    maximum_count = max(1, math.floor(pallet_dimension_mm / external_lower))
    for count in range(minimum_count, maximum_count + 1):
        largest_internal = math.floor(pallet_dimension_mm / count - 2 * thickness_mm)
        if lower_internal_mm <= largest_internal <= upper_internal_mm:
            values.add(largest_internal)
    return tuple(sorted(values))


def _pallet_aligned_variants(
    profile: Dimensions,
    thickness_mm: float,
    *,
    limit: int,
) -> tuple[Dimensions, ...]:
    """Genera diseños enteros de igual volumen cerca de saltos de capacidad.

    Dos dimensiones internas se ubican en un borde factible de pallet. La
    tercera es el menor entero que preserva el volumen del producto. Se repite
    para cada elección de eje dependiente, de modo que el headspace pueda
    aparecer en cualquier eje.
    """

    if limit <= 0:
        return ()
    bounds = tuple(_integer_fit_bounds(value) for value in profile.as_tuple())
    edge_values = (
        _pallet_edge_values(PALLET_WIDTH_MM, *bounds[0], thickness_mm),
        _pallet_edge_values(PALLET_LENGTH_MM, *bounds[1], thickness_mm),
        _pallet_edge_values(PALLET_MAX_HEIGHT_MM, *bounds[2], thickness_mm),
    )
    profile_volume = profile.volume_mm3
    variants: set[Dimensions] = set()

    def add_with_dependent_axis(first_axis: int, second_axis: int, dependent_axis: int) -> None:
        for first_value in edge_values[first_axis]:
            for second_value in edge_values[second_axis]:
                dependent_value = math.ceil(
                    profile_volume / (first_value * second_value)
                )
                lower, upper = bounds[dependent_axis]
                if not lower <= dependent_value <= upper:
                    continue
                values = [0, 0, 0]
                values[first_axis] = first_value
                values[second_axis] = second_value
                values[dependent_axis] = dependent_value
                variants.add(Dimensions(*values))

    add_with_dependent_axis(0, 1, 2)
    add_with_dependent_axis(0, 2, 1)
    add_with_dependent_axis(1, 2, 0)

    def rank(internal: Dimensions) -> tuple[int, tuple[float, float, float]]:
        external = external_from_internal(internal, thickness_mm)
        return -boxes_per_pallet(external), internal.as_tuple()

    return tuple(sorted(variants, key=rank)[:limit])


def _group_compromise_variants(
    products: tuple[Product, ...],
    thickness_mm: float,
    *,
    limit: int,
) -> tuple[Dimensions, ...]:
    """Busca cajas alineadas al pallet factibles para cada SKU de un grupo.

    La incumbente actual define grupos de SKU que ya comparten diseño. Esta
    rutina busca la intersección de sus bandas ±10% y usa el mayor volumen al
    construir una caja. Los diseños retenidos se verifican contra cada SKU,
    incluyendo headspace y compresión.
    """

    if limit <= 0 or not products:
        return ()
    per_axis_bounds = tuple(
        (
            max(_integer_fit_bounds(product.current_internal.as_tuple()[axis])[0] for product in products),
            min(_integer_fit_bounds(product.current_internal.as_tuple()[axis])[1] for product in products),
        )
        for axis in range(3)
    )
    if any(lower > upper for lower, upper in per_axis_bounds):
        return ()
    edge_values = (
        _pallet_edge_values(PALLET_WIDTH_MM, *per_axis_bounds[0], thickness_mm),
        _pallet_edge_values(PALLET_LENGTH_MM, *per_axis_bounds[1], thickness_mm),
        _pallet_edge_values(PALLET_MAX_HEIGHT_MM, *per_axis_bounds[2], thickness_mm),
    )
    largest_product_volume = max(product.product_volume_mm3 for product in products)
    variants: set[Dimensions] = set()
    minimum_common = Dimensions(
        *(max(product.current_internal.as_tuple()[axis] for product in products) for axis in range(3))
    )
    if all(product_fits_candidate(product, minimum_common, thickness_mm) for product in products):
        variants.add(minimum_common)

    def add_with_dependent_axis(first_axis: int, second_axis: int, dependent_axis: int) -> None:
        for first_value in edge_values[first_axis]:
            for second_value in edge_values[second_axis]:
                dependent_value = max(
                    per_axis_bounds[dependent_axis][0],
                    math.ceil(largest_product_volume / (first_value * second_value)),
                )
                if dependent_value > per_axis_bounds[dependent_axis][1]:
                    continue
                values = [0, 0, 0]
                values[first_axis] = first_value
                values[second_axis] = second_value
                values[dependent_axis] = dependent_value
                internal = Dimensions(*values)
                if all(product_fits_candidate(product, internal, thickness_mm) for product in products):
                    variants.add(internal)

    add_with_dependent_axis(0, 1, 2)
    add_with_dependent_axis(0, 2, 1)
    add_with_dependent_axis(1, 2, 0)

    def rank(internal: Dimensions) -> tuple[int, tuple[float, float, float]]:
        external = external_from_internal(internal, thickness_mm)
        return -boxes_per_pallet(external), internal.as_tuple()

    return tuple(sorted(variants, key=rank)[:limit])


def generate_candidates(
    products: tuple[Product, ...],
    thickness_mm: float,
    *,
    pair_profile_limit: int = 90,
    max_extra_pair_designs: int = 1_000,
    pallet_variant_profile_limit: int = 90,
    max_pallet_variants_per_profile: int = 18,
    seed_pallet_variant_profiles: tuple[Dimensions, ...] = (),
    retained_designs: tuple[Dimensions, ...] = (),
    seed_compromise_groups: tuple[tuple[Product, ...], ...] = (),
    max_compromise_variants_per_group: int = 18,
) -> tuple[CandidateBox, ...]:
    """Construye un universo determinista de candidatos.

    Se retiene cada perfil interno actual, garantizando una solución factible
    sin consolidación. Las cajas adicionales son máximos por componente de
    pares de alto volumen y variantes alineadas al pallet que preservan
    volumen. Perfiles y grupos semilla pueden enfocar esas variantes alrededor
    de diseños de una incumbente. Capturan saltos discretos, como una caja más
    por eje de pallet. CP-SAT es exacto sobre este universo explícito y
    reproducible.
    """

    profile_volume: dict[Dimensions, int] = defaultdict(int)
    for product in products:
        profile_volume[_integer_dimensions(product.current_internal)] += product.annual_volume
    profiles = sorted(profile_volume, key=lambda dims: (-profile_volume[dims], dims.as_tuple()))

    anchor_designs = set(profiles)
    pair_profiles = profiles[:pair_profile_limit]
    extra_design_scores: dict[Dimensions, int] = {}
    for left, right in combinations_with_replacement(pair_profiles, 2):
        merged = _componentwise_max(left, right)
        if merged not in anchor_designs:
            source_volume = profile_volume[left] + profile_volume[right]
            extra_design_scores[merged] = max(
                extra_design_scores.get(merged, 0), source_volume
            )
    extra_designs = set(
        sorted(
            extra_design_scores,
            key=lambda dims: (-extra_design_scores[dims], dims.as_tuple()),
        )[:max_extra_pair_designs]
    )

    pallet_variant_designs: set[Dimensions] = set()
    variant_profiles = tuple(
        dict.fromkeys((*profiles[:pallet_variant_profile_limit], *seed_pallet_variant_profiles))
    )
    for profile in variant_profiles:
        pallet_variant_designs.update(
            _pallet_aligned_variants(
                profile,
                thickness_mm,
                limit=max_pallet_variants_per_profile,
            )
        )

    compromise_designs: set[Dimensions] = set()
    for group in seed_compromise_groups:
        compromise_designs.update(
            _group_compromise_variants(
                group,
                thickness_mm,
                limit=max_compromise_variants_per_group,
            )
        )

    candidates: list[CandidateBox] = []
    # Retiene cada diseño del warm start y su vecindario. Así una incumbente
    # válida queda disponible como indicación inicial para CP-SAT.
    all_designs = (
        anchor_designs
        | extra_designs
        | pallet_variant_designs
        | compromise_designs
        | set(seed_pallet_variant_profiles)
        | set(retained_designs)
    )
    for ordinal, internal in enumerate(sorted(all_designs, key=lambda dims: dims.as_tuple())):
        external = external_from_internal(internal, thickness_mm)
        compatible = frozenset(
            product.code
            for product in products
            if product_fits_candidate(product, internal, thickness_mm)
        )
        if compatible:
            candidates.append(
                CandidateBox(
                    candidate_id=f"t{thickness_mm:g}_{ordinal:05d}",
                    thickness_mm=thickness_mm,
                    internal=internal,
                    external=external,
                    capacity_per_pallet=boxes_per_pallet(external),
                    compatible_product_codes=compatible,
                )
            )

    uncovered = {product.code for product in products} - set().union(
        *(candidate.compatible_product_codes for candidate in candidates)
    )
    if uncovered:
        raise ValueError(f"candidate generation left SKUs uncovered: {sorted(uncovered)}")
    return tuple(candidates)
