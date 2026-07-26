"""Recocido simulado incremental sobre el espacio exacto de asignación de candidatos.

El modelo CP-SAT sirve para demostrar y coordinar cambios de tiers, pero una
asignación inicial también puede explorarse económicamente con movimientos de
un SKU. Este módulo conserva en memoria el estado comercial exacto, por tipo
físico de caja y planta, de modo que un movimiento cuesta O(cantidad de plantas).

El recorrido de recocido puede aceptar deterioros temporales. Sólo devuelve la
mejor asignación hallada y la verifica de forma independiente con
:func:`bonsai.costs.evaluate_assignments` antes de devolverla.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math
import random
import time
from typing import Iterable

from .config import FreightPolicy
from .costs import (
    BoxTypeKey,
    box_type_key,
    evaluate_assignments,
    freight_pallets,
    unit_price_mills,
)
from .models import CandidateBox, CostBreakdown, PLANTS, Product


def _packaging_cost_mills(thickness_mm: float, volume: int) -> int:
    if volume < 0:
        raise ValueError("box-type volume cannot be negative")
    return volume * unit_price_mills(thickness_mm, volume) if volume else 0


def _candidate_rank(candidate: CandidateBox) -> tuple[BoxTypeKey, str]:
    return box_type_key(candidate), candidate.candidate_id


@dataclass(frozen=True)
class AnnealingMove:
    """Delta exacto de costo y pallets para reasignar un SKU."""

    code: str
    source_type: BoxTypeKey
    target_type: BoxTypeKey
    target_candidate: CandidateBox
    packaging_delta_mills: int
    freight_delta_mills: int
    pallet_delta: int
    type_delta: int

    @property
    def total_delta_mills(self) -> int:
        return self.packaging_delta_mills + self.freight_delta_mills


@dataclass(frozen=True)
class AnnealingGroupMove:
    """Reasignación simultánea exacta de varios SKU a un tipo físico."""

    codes: tuple[str, ...]
    source_types: tuple[BoxTypeKey, ...]
    target_type: BoxTypeKey
    target_candidate: CandidateBox
    packaging_delta_mills: int
    freight_delta_mills: int
    pallet_delta: int
    type_delta: int

    @property
    def total_delta_mills(self) -> int:
        return self.packaging_delta_mills + self.freight_delta_mills


@dataclass(frozen=True)
class AnnealingResult:
    """Mejor estado validado y diagnósticos sobre el recorrido exploratorio."""

    assignment: dict[str, CandidateBox]
    costs: CostBreakdown
    initial_costs: CostBreakdown
    current_costs: CostBreakdown
    steps: int
    proposed_moves: int
    accepted_moves: int
    accepted_worse_moves: int
    proposed_group_moves: int
    accepted_group_moves: int
    improvements: int
    restarts: int
    elapsed_seconds: float
    random_seed: int
    minimum_pallets: int
    max_pallets: int | None


class IncrementalAssignmentState:
    """Asignación mutable con contabilidad exacta O(plantas) por movimiento de un SKU.

    Su uso público sirve principalmente para probar estrategias de propuesta
    personalizadas. Un movimiento calculado queda obsoleto si otro movimiento
    cambia alguno de los tipos físicos involucrados; :meth:`apply` lo rechaza
    para no corromper los totales.
    """

    def __init__(
        self,
        products: tuple[Product, ...],
        assignment_by_code: dict[str, CandidateBox],
        freight_policy: FreightPolicy,
    ) -> None:
        reference = evaluate_assignments(products, assignment_by_code, freight_policy)
        thicknesses = {
            candidate.thickness_mm for candidate in assignment_by_code.values()
        }
        if len(thicknesses) != 1:
            raise ValueError("annealing requires one global carton thickness")

        self.products = products
        self.product_by_code = {product.code: product for product in products}
        self.assignment = dict(assignment_by_code)
        self.freight_policy = freight_policy
        self.thickness_mm = next(iter(thicknesses))
        self.volumes: dict[BoxTypeKey, dict[str, int]] = defaultdict(
            lambda: {plant: 0 for plant in PLANTS}
        )
        self.product_counts: dict[BoxTypeKey, int] = defaultdict(int)
        for product in products:
            type_key = box_type_key(self.assignment[product.code])
            self.product_counts[type_key] += 1
            for plant in PLANTS:
                self.volumes[type_key][plant] += product.annual_volume_by_plant[plant]

        self.packaging_mills = reference.packaging_mills
        self.freight_mills = reference.freight_mills
        self.pallets = reference.pallets
        self.types = reference.types

    @property
    def total_mills(self) -> int:
        return self.packaging_mills + self.freight_mills

    def calculate_move(self, code: str, target: CandidateBox) -> AnnealingMove | None:
        if code not in self.product_by_code:
            raise ValueError(f"unknown product code: {code}")
        product = self.product_by_code[code]
        source = self.assignment[code]
        source_type = box_type_key(source)
        target_type = box_type_key(target)
        if source_type == target_type:
            return None
        if target.thickness_mm != self.thickness_mm:
            return None
        if code not in target.compatible_product_codes:
            return None

        packaging_delta = 0
        source_volumes = self.volumes[source_type]
        target_volumes = self.volumes[target_type]
        for plant in PLANTS:
            moved_volume = product.annual_volume_by_plant[plant]
            if not moved_volume:
                continue
            source_before = source_volumes[plant]
            target_before = target_volumes[plant]
            source_after = source_before - moved_volume
            target_after = target_before + moved_volume
            if source_after < 0:
                raise AssertionError(f"negative source volume for {code} at {plant}")
            packaging_delta += (
                _packaging_cost_mills(self.thickness_mm, source_after)
                - _packaging_cost_mills(self.thickness_mm, source_before)
                + _packaging_cost_mills(self.thickness_mm, target_after)
                - _packaging_cost_mills(self.thickness_mm, target_before)
            )

        source_pallets = sum(
            freight_pallets(product, source, plant) for plant in PLANTS
        )
        target_pallets = sum(
            freight_pallets(product, target, plant) for plant in PLANTS
        )
        pallet_delta = target_pallets - source_pallets
        type_delta = -int(self.product_counts[source_type] == 1) + int(
            self.product_counts[target_type] == 0
        )
        return AnnealingMove(
            code=code,
            source_type=source_type,
            target_type=target_type,
            target_candidate=target,
            packaging_delta_mills=packaging_delta,
            freight_delta_mills=(
                pallet_delta * self.freight_policy.expected_mills_per_pallet
            ),
            pallet_delta=pallet_delta,
            type_delta=type_delta,
        )

    def apply(self, move: AnnealingMove) -> None:
        current_source = self.assignment.get(move.code)
        if current_source is None or box_type_key(current_source) != move.source_type:
            raise ValueError(f"stale annealing move for {move.code}")
        recalculated = self.calculate_move(move.code, move.target_candidate)
        if recalculated != move:
            raise ValueError(f"annealing move for {move.code} is stale")

        product = self.product_by_code[move.code]
        for plant in PLANTS:
            volume = product.annual_volume_by_plant[plant]
            self.volumes[move.source_type][plant] -= volume
            self.volumes[move.target_type][plant] += volume
        self.product_counts[move.source_type] -= 1
        self.product_counts[move.target_type] += 1
        self.assignment[move.code] = move.target_candidate
        self.packaging_mills += move.packaging_delta_mills
        self.freight_mills += move.freight_delta_mills
        self.pallets += move.pallet_delta
        self.types += move.type_delta

    def calculate_group_move(
        self, codes: Iterable[str], target: CandidateBox
    ) -> AnnealingGroupMove | None:
        """Calcula un movimiento simultáneo exacto sin alterar el estado.

        Evaluar los SKU en conjunto es esencial cerca de los umbrales de
        Procurement: movimientos individualmente poco atractivos pueden ser
        rentables cuando su volumen combinado alcanza un tier de descuento.
        """

        unique_codes = tuple(dict.fromkeys(codes))
        if len(unique_codes) < 2:
            return None
        if target.thickness_mm != self.thickness_mm:
            return None
        target_type = box_type_key(target)
        moved_codes: list[str] = []
        source_types: list[BoxTypeKey] = []
        moved_by_source: dict[BoxTypeKey, dict[str, int]] = defaultdict(
            lambda: {plant: 0 for plant in PLANTS}
        )
        pallet_delta = 0
        for code in unique_codes:
            if code not in self.product_by_code:
                raise ValueError(f"unknown product code: {code}")
            if code not in target.compatible_product_codes:
                return None
            source = self.assignment[code]
            source_type = box_type_key(source)
            if source_type == target_type:
                continue
            product = self.product_by_code[code]
            moved_codes.append(code)
            source_types.append(source_type)
            for plant in PLANTS:
                moved_by_source[source_type][plant] += product.annual_volume_by_plant[
                    plant
                ]
            pallet_delta += sum(
                freight_pallets(product, target, plant)
                - freight_pallets(product, source, plant)
                for plant in PLANTS
            )
        if len(moved_codes) < 2:
            return None

        packaging_delta = 0
        incoming = {plant: 0 for plant in PLANTS}
        for source_type, moved_volumes in moved_by_source.items():
            for plant in PLANTS:
                moved_volume = moved_volumes[plant]
                if not moved_volume:
                    continue
                before = self.volumes[source_type][plant]
                after = before - moved_volume
                if after < 0:
                    raise AssertionError(
                        f"negative source volume for group at {plant}"
                    )
                packaging_delta += _packaging_cost_mills(
                    self.thickness_mm, after
                ) - _packaging_cost_mills(self.thickness_mm, before)
                incoming[plant] += moved_volume
        for plant in PLANTS:
            if not incoming[plant]:
                continue
            before = self.volumes[target_type][plant]
            packaging_delta += _packaging_cost_mills(
                self.thickness_mm, before + incoming[plant]
            ) - _packaging_cost_mills(self.thickness_mm, before)

        counts_after = dict(self.product_counts)
        for source_type in source_types:
            counts_after[source_type] = counts_after.get(source_type, 0) - 1
        counts_after[target_type] = counts_after.get(target_type, 0) + len(moved_codes)
        affected_types = set(source_types) | {target_type}
        type_delta = sum(
            int(counts_after.get(key, 0) > 0)
            - int(self.product_counts.get(key, 0) > 0)
            for key in affected_types
        )
        return AnnealingGroupMove(
            codes=tuple(moved_codes),
            source_types=tuple(source_types),
            target_type=target_type,
            target_candidate=target,
            packaging_delta_mills=packaging_delta,
            freight_delta_mills=(
                pallet_delta * self.freight_policy.expected_mills_per_pallet
            ),
            pallet_delta=pallet_delta,
            type_delta=type_delta,
        )

    def apply_group(self, move: AnnealingGroupMove) -> None:
        recalculated = self.calculate_group_move(move.codes, move.target_candidate)
        if recalculated != move:
            raise ValueError("annealing group move is stale")
        for code, source_type in zip(move.codes, move.source_types, strict=True):
            product = self.product_by_code[code]
            for plant in PLANTS:
                volume = product.annual_volume_by_plant[plant]
                self.volumes[source_type][plant] -= volume
                self.volumes[move.target_type][plant] += volume
            self.product_counts[source_type] -= 1
            self.product_counts[move.target_type] += 1
            self.assignment[code] = move.target_candidate
        self.packaging_mills += move.packaging_delta_mills
        self.freight_mills += move.freight_delta_mills
        self.pallets += move.pallet_delta
        self.types += move.type_delta

    def validate(self) -> CostBreakdown:
        """Recalcula y verifica de forma independiente todas las cantidades del objetivo."""

        checked = evaluate_assignments(
            self.products, self.assignment, self.freight_policy
        )
        incremental = (
            self.packaging_mills,
            self.freight_mills,
            self.pallets,
            self.types,
        )
        independent = (
            checked.packaging_mills,
            checked.freight_mills,
            checked.pallets,
            checked.types,
        )
        if incremental != independent:
            raise AssertionError(
                "incremental annealing state differs from independent evaluation: "
                f"{incremental} != {independent}"
            )
        return checked


def build_targets_by_code(
    products: tuple[Product, ...],
    assignment_by_code: dict[str, CandidateBox],
    candidates: Iterable[CandidateBox],
    *,
    free_product_codes: Iterable[str] | None = None,
) -> dict[str, tuple[CandidateBox, ...]]:
    """Precalcula y deduplica físicamente objetivos factibles para cada SKU libre."""

    product_codes = {product.code for product in products}
    if free_product_codes is None:
        free_codes = product_codes
    else:
        free_codes = set(free_product_codes)
        unknown = free_codes - product_codes
        if unknown:
            raise ValueError(f"unknown free product codes: {sorted(unknown)}")
    thicknesses = {
        candidate.thickness_mm for candidate in assignment_by_code.values()
    }
    if len(thicknesses) != 1:
        raise ValueError("annealing requires one global carton thickness")
    thickness_mm = next(iter(thicknesses))
    by_code: dict[str, dict[BoxTypeKey, CandidateBox]] = {
        code: {} for code in free_codes
    }

    # Incluye el diseño incumbente para que todo movimiento aceptado sea reversible.
    for code in free_codes:
        incumbent = assignment_by_code[code]
        by_code[code][box_type_key(incumbent)] = incumbent
    for candidate in candidates:
        if candidate.thickness_mm != thickness_mm:
            continue
        for code in candidate.compatible_product_codes & free_codes:
            key = box_type_key(candidate)
            current = by_code[code].get(key)
            if current is None or _candidate_rank(candidate) < _candidate_rank(current):
                by_code[code][key] = candidate
    return {
        code: tuple(sorted(targets.values(), key=_candidate_rank))
        for code, targets in by_code.items()
    }


def minimum_pallets_for_candidates(
    products: tuple[Product, ...],
    targets_by_code: dict[str, tuple[CandidateBox, ...]],
    assignment_by_code: dict[str, CandidateBox],
) -> int:
    """Devuelve la cota inferior independiente de pallets por SKU de este espacio."""

    minimum = 0
    for product in products:
        targets = targets_by_code.get(product.code)
        if not targets:
    # Los productos fijos siguen aportando sus pallets de la incumbente.
            targets = (assignment_by_code[product.code],)
        minimum += min(
            sum(freight_pallets(product, candidate, plant) for plant in PLANTS)
            for candidate in targets
        )
    return minimum


def _progress(step: int, max_steps: int | None, elapsed: float, duration: float | None) -> float:
    fractions: list[float] = []
    if max_steps is not None:
        fractions.append(step / max_steps)
    if duration is not None:
        fractions.append(elapsed / duration)
    return min(1.0, max(fractions, default=0.0))


def _random_alternative(
    state: IncrementalAssignmentState,
    code: str,
    choices: tuple[CandidateBox, ...],
    rng: random.Random,
    *,
    prefer_used: bool,
) -> CandidateBox | None:
    """Elige una alternativa física y opcionalmente prefiere un tipo activo."""

    current_type = box_type_key(state.assignment[code])
    if prefer_used:
    # El muestreo por rechazo evita reconstruir una tupla filtrada potencialmente
    # grande en cada propuesta. Doce intentos dan un sesgo fuerte a tipos activos
    # y luego se vuelve ordenadamente al universo completo.
        for _ in range(12):
            candidate = rng.choice(choices)
            candidate_type = box_type_key(candidate)
            if (
                candidate_type != current_type
                and state.product_counts[candidate_type] > 0
            ):
                return candidate
    for _ in range(4):
        candidate = rng.choice(choices)
        if box_type_key(candidate) != current_type:
            return candidate
    alternatives = tuple(
        candidate
        for candidate in choices
        if box_type_key(candidate) != current_type
    )
    return rng.choice(alternatives) if alternatives else None


def guided_single_proposal(
    state: IncrementalAssignmentState,
    targets_by_code: dict[str, tuple[CandidateBox, ...]],
    movable_codes: tuple[str, ...],
    rng: random.Random,
    *,
    sample_size: int,
    used_target_probability: float,
    greediness: float,
) -> AnnealingMove | None:
    """Muestrea movimientos exactos y normalmente devuelve el de menor delta.

    El muestreo conserva diversidad sin recorrer cada par SKU/candidato. La
    preferencia por tipos activos vuelve mucho menos infrecuentes la
    consolidación y los cruces de tier que un muestreo uniforme.
    """

    sampled: list[AnnealingMove] = []
    for _ in range(sample_size):
        code = rng.choice(movable_codes)
        target = _random_alternative(
            state,
            code,
            targets_by_code[code],
            rng,
            prefer_used=rng.random() < used_target_probability,
        )
        if target is None:
            continue
        move = state.calculate_move(code, target)
        if move is not None:
            sampled.append(move)
    if not sampled:
        return None
    if rng.random() < greediness:
        return min(
            sampled,
            key=lambda move: (
                move.total_delta_mills,
                move.pallet_delta,
                move.code,
                _candidate_rank(move.target_candidate),
            ),
        )
    return rng.choice(sampled)


def guided_group_proposal(
    state: IncrementalAssignmentState,
    targets_by_code: dict[str, tuple[CandidateBox, ...]],
    movable_codes: tuple[str, ...],
    rng: random.Random,
    *,
    sample_size: int,
    used_target_probability: float,
    greediness: float,
    max_group_size: int,
) -> AnnealingGroupMove | None:
    """Propone una pequeña consolidación coordinada hacia un destino compartido."""

    anchor = guided_single_proposal(
        state,
        targets_by_code,
        movable_codes,
        rng,
        sample_size=sample_size,
        used_target_probability=used_target_probability,
        greediness=greediness,
    )
    if anchor is None:
        return None
    target = anchor.target_candidate
    eligible = [
        code
        for code in movable_codes
        if code != anchor.code
        and code in target.compatible_product_codes
        and box_type_key(state.assignment[code]) != anchor.target_type
    ]
    if not eligible:
        return None
    group_size = min(rng.randint(2, max_group_size), len(eligible) + 1)
    chosen = [anchor.code]
    best_group: AnnealingGroupMove | None = None
    # Evalúa vorazmente el delta *conjunto* después de cada adición posible. Así
    # detecta saltos de descuento sobre todas las unidades que un ranking de
    # acompañantes individuales no puede ver. El muestreo acota el trabajo sin
    # depender del tamaño del dataset.
    remaining = eligible
    while len(chosen) < group_size and remaining:
        companion_sample = rng.sample(
            remaining, min(len(remaining), max(sample_size, group_size - 1))
        )
        alternatives: list[AnnealingGroupMove] = []
        for code in companion_sample:
            trial = state.calculate_group_move((*chosen, code), target)
            if trial is not None:
                alternatives.append(trial)
        if not alternatives:
            break
        selected = min(
            alternatives,
            key=lambda move: (
                move.total_delta_mills,
                move.pallet_delta,
                move.codes,
            ),
        )
        chosen = list(selected.codes)
        if best_group is None or selected.total_delta_mills < best_group.total_delta_mills:
            best_group = selected
        chosen_set = set(chosen)
        remaining = [code for code in remaining if code not in chosen_set]
    return best_group


def simulated_annealing(
    products: tuple[Product, ...],
    assignment_by_code: dict[str, CandidateBox],
    candidates: Iterable[CandidateBox],
    freight_policy: FreightPolicy,
    *,
    duration_seconds: float | None = None,
    max_steps: int | None = 1_000_000,
    random_seed: int = 42,
    initial_temperature_usd: float = 50_000.0,
    final_temperature_usd: float = 10.0,
    max_extra_pallets: int | None = None,
    max_pallets: int | None = None,
    free_product_codes: Iterable[str] | None = None,
    validation_interval: int = 0,
    proposal_strategy: str = "guided",
    proposal_sample_size: int = 12,
    used_target_probability: float = 0.70,
    proposal_greediness: float = 0.85,
    group_proposal_probability: float = 0.02,
    max_group_size: int = 4,
    restart_interval_steps: int | None = 250_000,
) -> AnnealingResult:
    """Explora asignaciones exactas con recocido simulado reproducible por SKU.

    ``max_extra_pallets`` tiene el mismo significado que en el optimizador
    exacto: se suma al mínimo de pallets de cada SKU dentro del espacio de
    búsqueda provisto. ``max_pallets`` es un límite absoluto alternativo. La
    incumbente ya debe cumplir el límite elegido.

    Pueden indicarse a la vez límites de duración y de pasos; el primero que se
    alcance detiene el recorrido. ``validation_interval`` permite auditar
    periódicamente los estados aceptados. El mejor estado devuelto siempre se audita.
    """

    if duration_seconds is not None and duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive")
    if max_steps is not None and max_steps < 0:
        raise ValueError("max_steps cannot be negative")
    if duration_seconds is None and max_steps is None:
        raise ValueError("either duration_seconds or max_steps is required")
    if initial_temperature_usd <= 0 or final_temperature_usd <= 0:
        raise ValueError("annealing temperatures must be positive")
    if final_temperature_usd > initial_temperature_usd:
        raise ValueError("final temperature cannot exceed initial temperature")
    if max_extra_pallets is not None and max_extra_pallets < 0:
        raise ValueError("max_extra_pallets cannot be negative")
    if max_pallets is not None and max_pallets < 0:
        raise ValueError("max_pallets cannot be negative")
    if max_extra_pallets is not None and max_pallets is not None:
        raise ValueError("use max_extra_pallets or max_pallets, not both")
    if validation_interval < 0:
        raise ValueError("validation_interval cannot be negative")
    if proposal_strategy not in {"uniform", "guided"}:
        raise ValueError("proposal_strategy must be 'uniform' or 'guided'")
    if proposal_sample_size < 1:
        raise ValueError("proposal_sample_size must be positive")
    for name, value in (
        ("used_target_probability", used_target_probability),
        ("proposal_greediness", proposal_greediness),
        ("group_proposal_probability", group_proposal_probability),
    ):
        if not 0 <= value <= 1:
            raise ValueError(f"{name} must be between zero and one")
    if max_group_size < 2:
        raise ValueError("max_group_size must be at least 2")
    if restart_interval_steps is not None and restart_interval_steps < 1:
        raise ValueError("restart_interval_steps must be positive")

    state = IncrementalAssignmentState(products, assignment_by_code, freight_policy)
    # Construye el universo completo una sola vez. La semántica del presupuesto
    # de pallets usa deliberadamente el mínimo global por SKU, aun si sólo un
    # subconjunto LNS está libre, igual que la definición del optimizador exacto.
    all_targets = build_targets_by_code(
        products,
        assignment_by_code,
        candidates,
    )
    if free_product_codes is None:
        targets = all_targets
    else:
        free_codes = set(free_product_codes)
        unknown = free_codes - set(all_targets)
        if unknown:
            raise ValueError(f"unknown free product codes: {sorted(unknown)}")
        targets = {code: all_targets[code] for code in free_codes}
    movable_codes = tuple(
        sorted(code for code, choices in targets.items() if len(choices) > 1)
    )
    minimum_pallets = minimum_pallets_for_candidates(
        products, all_targets, assignment_by_code
    )
    pallet_limit = max_pallets
    if max_extra_pallets is not None:
        pallet_limit = minimum_pallets + max_extra_pallets
    if pallet_limit is not None and state.pallets > pallet_limit:
        raise ValueError(
            f"incumbent has {state.pallets} pallets, above limit {pallet_limit}"
        )

    initial_costs = state.validate()
    best_assignment = dict(state.assignment)
    best_total_mills = state.total_mills
    rng = random.Random(random_seed)
    start = time.perf_counter()
    steps = proposed = accepted = accepted_worse = improvements = 0
    proposed_groups = accepted_groups = restarts = 0
    initial_temperature_mills = initial_temperature_usd * 1000.0
    final_temperature_mills = final_temperature_usd * 1000.0
    temperature_ratio = final_temperature_mills / initial_temperature_mills

    while movable_codes:
        elapsed = time.perf_counter() - start
        if max_steps is not None and steps >= max_steps:
            break
        if duration_seconds is not None and elapsed >= duration_seconds:
            break
        if (
            restart_interval_steps is not None
            and steps > 0
            and steps % restart_interval_steps == 0
        ):
    # Reiniciar en el mejor estado conocido descarta una excursión improductiva
    # y recalienta el ciclo siguiente.
            state = IncrementalAssignmentState(
                products, best_assignment, freight_policy
            )
            restarts += 1
        steps += 1

        is_group = (
            proposal_strategy == "guided"
            and group_proposal_probability > 0
            and rng.random() < group_proposal_probability
        )
        if is_group:
            move = guided_group_proposal(
                state,
                targets,
                movable_codes,
                rng,
                sample_size=proposal_sample_size,
                used_target_probability=used_target_probability,
                greediness=proposal_greediness,
                max_group_size=max_group_size,
            )
            proposed_groups += int(move is not None)
        elif proposal_strategy == "guided":
            move = guided_single_proposal(
                state,
                targets,
                movable_codes,
                rng,
                sample_size=proposal_sample_size,
                used_target_probability=used_target_probability,
                greediness=proposal_greediness,
            )
        else:
            code = rng.choice(movable_codes)
            target = _random_alternative(
                state, code, targets[code], rng, prefer_used=False
            )
            move = state.calculate_move(code, target) if target else None
        if move is None:
            continue
        proposed += 1
        if pallet_limit is not None and state.pallets + move.pallet_delta > pallet_limit:
            continue

        if restart_interval_steps is None:
            progress = _progress(steps, max_steps, elapsed, duration_seconds)
        else:
            progress = (steps % restart_interval_steps) / restart_interval_steps
        temperature = initial_temperature_mills * (temperature_ratio**progress)
        delta = move.total_delta_mills
        accept = delta <= 0 or rng.random() < math.exp(-delta / temperature)
        if not accept:
            continue

        if isinstance(move, AnnealingGroupMove):
            state.apply_group(move)
            accepted_groups += 1
        else:
            state.apply(move)
        accepted += 1
        if delta > 0:
            accepted_worse += 1
        if validation_interval and accepted % validation_interval == 0:
            state.validate()
        if state.total_mills < best_total_mills:
            best_total_mills = state.total_mills
            best_assignment = dict(state.assignment)
            improvements += 1

    current_costs = state.validate()
    best_costs = evaluate_assignments(products, best_assignment, freight_policy)
    if best_costs.total_mills != best_total_mills:
        raise AssertionError(
            "best incremental cost differs from independent evaluation: "
            f"{best_total_mills} != {best_costs.total_mills}"
        )
    if best_costs.total_mills > initial_costs.total_mills:
        raise AssertionError("annealing returned a worse assignment than its incumbent")
    return AnnealingResult(
        assignment=best_assignment,
        costs=best_costs,
        initial_costs=initial_costs,
        current_costs=current_costs,
        steps=steps,
        proposed_moves=proposed,
        accepted_moves=accepted,
        accepted_worse_moves=accepted_worse,
        proposed_group_moves=proposed_groups,
        accepted_group_moves=accepted_groups,
        improvements=improvements,
        restarts=restarts,
        elapsed_seconds=time.perf_counter() - start,
        random_seed=random_seed,
        minimum_pallets=minimum_pallets,
        max_pallets=pallet_limit,
    )
