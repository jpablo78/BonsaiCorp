"""Vecindarios deterministas de SKU centrados en tiers para búsqueda de gran vecindario.

La tabla de Procurement se aplica a un tipo físico de caja en cada planta,
mientras que asignar un SKU es una decisión global. Por eso, un vecindario LNS
útil no puede contener sólo el SKU que completa un tier objetivo: debe incluir
todos los SKU hoy asignados al tipo origen de ese SKU. De otro modo el modelo
local congela la mayor parte del tier origen y puede omitir o valorar mal el
movimiento coordinado.

Este módulo sólo identifica vecindarios prometedores. No depende
deliberadamente de OR-Tools ni modifica la asignación incumbente.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Literal, Mapping

from .config import DISCOUNT_TIERS
from .costs import BoxTypeKey, box_type_key, unit_price_mills
from .models import CandidateBox, PLANTS, PreparedData, Product


@dataclass(frozen=True)
class DonorGroup:
    """Un tipo origen completo de la incumbente que puede alimentar un objetivo de tier.

    ``source_codes`` siempre contiene *todos* los SKU asignados a
    ``source_type``. Sólo ``eligible_codes`` puede moverse directamente al
    diseño receptor y sólo su demanda se cuenta en
    ``eligible_volume_at_target_plant``.
    """

    source_type: BoxTypeKey
    source_codes: tuple[str, ...]
    eligible_codes: tuple[str, ...]
    eligible_volume_at_target_plant: int
    source_volume_at_target_plant: int


@dataclass(frozen=True)
class TierTarget:
    """Un tipo físico de caja/planta de la incumbente bajo su próximo tier de descuento."""

    box_type: BoxTypeKey
    plant: str
    candidate: CandidateBox
    receiver_codes: tuple[str, ...]
    current_volume: int
    current_tier_index: int
    current_tier_name: str
    current_factor_percent: int
    next_tier_index: int
    next_tier_name: str
    next_factor_percent: int
    next_threshold: int
    gap_units: int
    gap_ratio: float
    discount_per_unit_mills: int
    incumbent_discount_value_mills: int
    threshold_discount_value_mills: int
    donor_groups: tuple[DonorGroup, ...]
    eligible_donor_volume: int
    reachable_from_eligible_donors: bool

    @property
    def key(self) -> tuple[BoxTypeKey, str, int]:
        return self.box_type, self.plant, self.next_threshold

    @property
    def donor_coverage_ratio(self) -> float:
        return self.eligible_donor_volume / self.gap_units


@dataclass(frozen=True)
class Neighborhood:
    """Un conjunto de grupos origen completos de la incumbente para liberar juntos en LNS."""

    neighborhood_id: str
    kind: Literal["star", "component"]
    product_codes: tuple[str, ...]
    source_types: tuple[BoxTypeKey, ...]
    targets: tuple[TierTarget, ...]
    selected_donor_volume: int
    reaches_primary_target: bool

    @property
    def sku_count(self) -> int:
        return len(self.product_codes)

    @property
    def target_count(self) -> int:
        return len(self.targets)

    @property
    def minimum_gap_units(self) -> int:
        return min(target.gap_units for target in self.targets)

    @property
    def gross_incumbent_discount_value_mills(self) -> int:
        """Valor bruto de tier antes del flete y de efectos secundarios en los tiers origen."""

        return sum(target.incumbent_discount_value_mills for target in self.targets)


@dataclass(frozen=True)
class TierNeighborhoodPlan:
    """Salida determinista completa usada por un ejecutor LNS posterior."""

    targets: tuple[TierTarget, ...]
    stars: tuple[Neighborhood, ...]
    components: tuple[Neighborhood, ...]


def _candidate_rank(candidate: CandidateBox) -> tuple[int, str, tuple[int, int, int]]:
    # Se prefiere el representante exacto de compatibilidad más amplia. Los
    # campos restantes hacen que la elección no dependa del orden iterable.
    return (
        -len(candidate.compatible_product_codes),
        candidate.candidate_id,
        candidate.internal.as_tuple(),
    )


def _validate_inputs(
    data: PreparedData,
    incumbent_assignment: Mapping[str, CandidateBox],
) -> float:
    product_codes = [product.code for product in data.products]
    if len(product_codes) != len(set(product_codes)):
        raise ValueError("prepared data contains duplicate product codes")
    if set(incumbent_assignment) != set(product_codes):
        raise ValueError("incumbent assignment must contain exactly one box for every SKU")
    thicknesses = {
        incumbent_assignment[code].thickness_mm for code in product_codes
    }
    if len(thicknesses) != 1:
        raise ValueError("tier neighborhoods require one global carton thickness")
    return next(iter(thicknesses))


def _groups_by_type(
    products: tuple[Product, ...],
    incumbent_assignment: Mapping[str, CandidateBox],
) -> dict[BoxTypeKey, tuple[str, ...]]:
    groups: dict[BoxTypeKey, list[str]] = defaultdict(list)
    for product in products:
        groups[box_type_key(incumbent_assignment[product.code])].append(product.code)
    return {
        type_key: tuple(sorted(codes))
        for type_key, codes in sorted(groups.items())
    }


def _exact_representatives(
    candidates: Iterable[CandidateBox],
    thickness_mm: float,
) -> tuple[dict[BoxTypeKey, CandidateBox], dict[BoxTypeKey, frozenset[str]]]:
    grouped: dict[BoxTypeKey, list[CandidateBox]] = defaultdict(list)
    for candidate in candidates:
        if candidate.thickness_mm == thickness_mm:
            grouped[box_type_key(candidate)].append(candidate)

    representatives: dict[BoxTypeKey, CandidateBox] = {}
    compatible_codes: dict[BoxTypeKey, frozenset[str]] = {}
    for type_key, same_type in grouped.items():
        representatives[type_key] = min(same_type, key=_candidate_rank)
        compatible_codes[type_key] = frozenset().union(
            *(candidate.compatible_product_codes for candidate in same_type)
        )
    return representatives, compatible_codes


def _donor_rank(group: DonorGroup, gap_units: int) -> tuple[object, ...]:
    """Prefiere un grupo origen pequeño que pueda cubrir por sí solo la brecha.

    If no group is sufficient, larger contributions rank first.  This keeps a
    capped star useful without introducing randomness.
    """

    contribution = group.eligible_volume_at_target_plant
    if contribution >= gap_units:
        return (0, contribution - gap_units, len(group.source_codes), group.source_type)
    return (1, -contribution, len(group.source_codes), group.source_type)


def _target_rank(target: TierTarget) -> tuple[object, ...]:
    return (
        not target.reachable_from_eligible_donors,
        target.gap_ratio,
        target.gap_units,
        -target.incumbent_discount_value_mills,
        target.box_type,
        target.plant,
    )


def identify_tier_targets(
    data: PreparedData,
    incumbent_assignment: Mapping[str, CandidateBox],
    exact_candidates: Iterable[CandidateBox],
    *,
    plants: Iterable[str] = PLANTS,
    max_gap_units: int | None = None,
    max_gap_ratio: float | None = None,
    require_reachable: bool = False,
    max_targets: int | None = None,
) -> tuple[TierTarget, ...]:
    """Devuelve pares tipo/planta de la incumbente por proximidad al próximo tier.

    Zero-volume type/plant pairs and pairs already in the last tier are not
    targets.  ``max_gap_*`` filters are optional and conjunctive.  Reachability
    counts only demand from outside the receiver group whose SKU is compatible
    with the exact receiving design; it is an opportunity metric, not a claim
    that moving those SKUs is cost-improving.
    """

    if max_gap_units is not None and max_gap_units < 0:
        raise ValueError("max_gap_units cannot be negative")
    if max_gap_ratio is not None and max_gap_ratio < 0:
        raise ValueError("max_gap_ratio cannot be negative")
    if max_targets is not None and max_targets < 0:
        raise ValueError("max_targets cannot be negative")

    requested_plants = tuple(plants)
    unknown_plants = set(requested_plants) - set(PLANTS)
    if unknown_plants:
        raise ValueError(f"unknown plants: {sorted(unknown_plants)}")
    if len(requested_plants) != len(set(requested_plants)):
        raise ValueError("plants cannot contain duplicates")

    thickness_mm = _validate_inputs(data, incumbent_assignment)
    products = tuple(sorted(data.products, key=lambda product: product.code))
    product_by_code = {product.code: product for product in products}
    groups = _groups_by_type(products, incumbent_assignment)
    representatives, compatible_by_type = _exact_representatives(
        exact_candidates, thickness_mm
    )
    missing_types = set(groups) - set(representatives)
    if missing_types:
        raise ValueError(
            "exact candidate universe does not contain incumbent physical types: "
            f"{sorted(missing_types)}"
        )

    targets: list[TierTarget] = []
    for target_type, receiver_codes in groups.items():
        candidate = representatives[target_type]
        compatible_codes = compatible_by_type[target_type]
        for plant in requested_plants:
            current_volume = sum(
                product_by_code[code].annual_volume_by_plant[plant]
                for code in receiver_codes
            )
            if current_volume <= 0:
                continue

            current_tier_index = next(
                index
                for index, tier in enumerate(DISCOUNT_TIERS)
                if tier.contains(current_volume)
            )
            if current_tier_index + 1 == len(DISCOUNT_TIERS):
                continue
            current_tier = DISCOUNT_TIERS[current_tier_index]
            next_tier_index = current_tier_index + 1
            next_tier = DISCOUNT_TIERS[next_tier_index]
            next_threshold = next_tier.lower_inclusive
            gap_units = next_threshold - current_volume
            gap_ratio = gap_units / next_threshold
            if max_gap_units is not None and gap_units > max_gap_units:
                continue
            if max_gap_ratio is not None and gap_ratio > max_gap_ratio:
                continue

            donor_groups: list[DonorGroup] = []
            for source_type, source_codes in groups.items():
                if source_type == target_type:
                    continue
                eligible_codes = tuple(
                    code
                    for code in source_codes
                    if code in compatible_codes
                    and product_by_code[code].annual_volume_by_plant[plant] > 0
                )
                if not eligible_codes:
                    continue
                donor_groups.append(
                    DonorGroup(
                        source_type=source_type,
                        source_codes=source_codes,
                        eligible_codes=eligible_codes,
                        eligible_volume_at_target_plant=sum(
                            product_by_code[code].annual_volume_by_plant[plant]
                            for code in eligible_codes
                        ),
                        source_volume_at_target_plant=sum(
                            product_by_code[code].annual_volume_by_plant[plant]
                            for code in source_codes
                        ),
                    )
                )
            donor_groups.sort(key=lambda group: _donor_rank(group, gap_units))
            eligible_donor_volume = sum(
                group.eligible_volume_at_target_plant for group in donor_groups
            )
            reachable = eligible_donor_volume >= gap_units
            if require_reachable and not reachable:
                continue

            current_unit_mills = unit_price_mills(thickness_mm, current_volume)
            next_unit_mills = unit_price_mills(thickness_mm, next_threshold)
            discount_per_unit_mills = current_unit_mills - next_unit_mills
            targets.append(
                TierTarget(
                    box_type=target_type,
                    plant=plant,
                    candidate=candidate,
                    receiver_codes=receiver_codes,
                    current_volume=current_volume,
                    current_tier_index=current_tier_index,
                    current_tier_name=current_tier.name,
                    current_factor_percent=current_tier.factor_percent,
                    next_tier_index=next_tier_index,
                    next_tier_name=next_tier.name,
                    next_factor_percent=next_tier.factor_percent,
                    next_threshold=next_threshold,
                    gap_units=gap_units,
                    gap_ratio=gap_ratio,
                    discount_per_unit_mills=discount_per_unit_mills,
                    incumbent_discount_value_mills=(
                        current_volume * discount_per_unit_mills
                    ),
                    threshold_discount_value_mills=(
                        next_threshold * discount_per_unit_mills
                    ),
                    donor_groups=tuple(donor_groups),
                    eligible_donor_volume=eligible_donor_volume,
                    reachable_from_eligible_donors=reachable,
                )
            )

    targets.sort(key=_target_rank)
    if max_targets is not None:
        targets = targets[:max_targets]
    return tuple(targets)


def build_star_neighborhoods(
    targets: Iterable[TierTarget],
    *,
    max_source_groups: int | None = None,
    max_skus: int | None = None,
    require_selected_donor: bool = True,
) -> tuple[Neighborhood, ...]:
    """Crea una estrella centrada en el receptor para cada objetivo.

    Limits never split an incumbent source group.  A target whose receiver
    group alone exceeds ``max_skus`` is omitted; donor groups that would exceed
    the cap are skipped.  ``max_source_groups`` counts donor groups and does
    not count the mandatory receiver group.
    """

    if max_source_groups is not None and max_source_groups < 0:
        raise ValueError("max_source_groups cannot be negative")
    if max_skus is not None and max_skus < 1:
        raise ValueError("max_skus must be positive")

    ordered_targets = sorted(targets, key=_target_rank)
    stars: list[Neighborhood] = []
    for target in ordered_targets:
        selected_codes = set(target.receiver_codes)
        if max_skus is not None and len(selected_codes) > max_skus:
            continue
        selected_types = {target.box_type}
        selected_donor_volume = 0
        selected_group_count = 0
        for donor in target.donor_groups:
            if (
                max_source_groups is not None
                and selected_group_count >= max_source_groups
            ):
                break
            new_codes = set(donor.source_codes) - selected_codes
            if max_skus is not None and len(selected_codes) + len(new_codes) > max_skus:
                continue
            selected_codes.update(donor.source_codes)
            selected_types.add(donor.source_type)
            selected_donor_volume += donor.eligible_volume_at_target_plant
            selected_group_count += 1

        if require_selected_donor and selected_group_count == 0:
            continue
        stars.append(
            Neighborhood(
                neighborhood_id="",  # assigned after deterministic sorting
                kind="star",
                product_codes=tuple(sorted(selected_codes)),
                source_types=tuple(sorted(selected_types)),
                targets=(target,),
                selected_donor_volume=selected_donor_volume,
                reaches_primary_target=selected_donor_volume >= target.gap_units,
            )
        )

    return tuple(
        Neighborhood(
            neighborhood_id=f"star_{index:04d}",
            kind=star.kind,
            product_codes=star.product_codes,
            source_types=star.source_types,
            targets=star.targets,
            selected_donor_volume=star.selected_donor_volume,
            reaches_primary_target=star.reaches_primary_target,
        )
        for index, star in enumerate(stars)
    )


def build_component_neighborhoods(
    stars: Iterable[Neighborhood],
) -> tuple[Neighborhood, ...]:
    """Une estrellas conectadas mediante al menos un grupo origen de la incumbente."""

    ordered_stars = tuple(
        sorted(
            stars,
            key=lambda star: (
                star.targets[0].key if star.targets else ((), "", 0),
                star.product_codes,
            ),
        )
    )
    if any(star.kind != "star" for star in ordered_stars):
        raise ValueError("component construction accepts star neighborhoods only")

    parent = list(range(len(ordered_stars)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root == right_root:
            return
        if left_root < right_root:
            parent[right_root] = left_root
        else:
            parent[left_root] = right_root

    first_star_by_type: dict[BoxTypeKey, int] = {}
    for star_index, star in enumerate(ordered_stars):
        for source_type in star.source_types:
            previous = first_star_by_type.setdefault(source_type, star_index)
            union(previous, star_index)

    stars_by_root: dict[int, list[Neighborhood]] = defaultdict(list)
    for star_index, star in enumerate(ordered_stars):
        stars_by_root[find(star_index)].append(star)

    raw_components: list[Neighborhood] = []
    for component_stars in stars_by_root.values():
        product_codes = tuple(
            sorted(
                set().union(*(set(star.product_codes) for star in component_stars))
            )
        )
        source_types = tuple(
            sorted(set().union(*(set(star.source_types) for star in component_stars)))
        )
        targets_by_key = {
            target.key: target
            for star in component_stars
            for target in star.targets
        }
        component_targets = tuple(
            targets_by_key[key] for key in sorted(targets_by_key)
        )
        raw_components.append(
            Neighborhood(
                neighborhood_id="",
                kind="component",
                product_codes=product_codes,
                source_types=source_types,
                targets=component_targets,
                selected_donor_volume=sum(
                    star.selected_donor_volume for star in component_stars
                ),
                reaches_primary_target=all(
                    star.reaches_primary_target for star in component_stars
                ),
            )
        )

    raw_components.sort(
        key=lambda component: (
            component.targets[0].key,
            component.product_codes,
        )
    )
    return tuple(
        Neighborhood(
            neighborhood_id=f"component_{index:04d}",
            kind=component.kind,
            product_codes=component.product_codes,
            source_types=component.source_types,
            targets=component.targets,
            selected_donor_volume=component.selected_donor_volume,
            reaches_primary_target=component.reaches_primary_target,
        )
        for index, component in enumerate(raw_components)
    )


def build_tier_neighborhoods(
    data: PreparedData,
    incumbent_assignment: Mapping[str, CandidateBox],
    exact_candidates: Iterable[CandidateBox],
    *,
    plants: Iterable[str] = PLANTS,
    max_gap_units: int | None = None,
    max_gap_ratio: float | None = None,
    require_reachable: bool = True,
    max_targets: int | None = None,
    max_source_groups: int | None = None,
    max_skus: int | None = None,
) -> TierNeighborhoodPlan:
    """Identifica objetivos y construye vistas de estrellas y componentes conexos."""

    targets = identify_tier_targets(
        data,
        incumbent_assignment,
        exact_candidates,
        plants=plants,
        max_gap_units=max_gap_units,
        max_gap_ratio=max_gap_ratio,
        require_reachable=require_reachable,
        max_targets=max_targets,
    )
    stars = build_star_neighborhoods(
        targets,
        max_source_groups=max_source_groups,
        max_skus=max_skus,
    )
    return TierNeighborhoodPlan(
        targets=targets,
        stars=stars,
        components=build_component_neighborhoods(stars),
    )
