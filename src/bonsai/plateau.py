"""Diversification over the exact zero-cost assignment plateau.

The master problem often has many assignments with the same objective value.
Those alternatives are useful as genuinely different CP-SAT or LNS seeds, but
they must be identified with the same commercial accounting used by the final
validator.  In particular, procurement volume is consolidated by *physical
box geometry* and plant, regardless of solver-local candidate IDs.

This module deliberately does not depend on the optimizer.  It can enumerate
all currently available one-SKU zero-cost moves and perform a reproducible
random walk that returns the most diversified state visited.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import random
from typing import Iterable

from .config import FreightPolicy
from .costs import BoxTypeKey, box_type_key, evaluate_assignments, freight_pallets, unit_price_mills
from .models import CandidateBox, CostBreakdown, PLANTS, Product


@dataclass(frozen=True)
class PlateauMove:
    """One feasible reassignment of a SKU to a different physical box type."""

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
class PlateauWalkResult:
    """Most diversified assignment found during a zero-cost random walk."""

    assignment: dict[str, CandidateBox]
    moves: tuple[PlateauMove, ...]
    initial_costs: CostBreakdown
    final_costs: CostBreakdown
    distance_from_start: int
    walk_steps: int
    visited_states: int


def _packaging_cost_mills(thickness_mm: float, volume: int) -> int:
    if volume < 0:
        raise ValueError("box-type volume cannot be negative")
    return volume * unit_price_mills(thickness_mm, volume) if volume else 0


def _candidate_rank(candidate: CandidateBox) -> tuple[str, tuple[int, int, int]]:
    """Stable choice when one geometry has several solver-local candidates."""

    return candidate.candidate_id, candidate.internal.as_tuple()


class _IncrementalPlateauState:
    """Mutable internal state whose deltas mirror ``evaluate_assignments``."""

    def __init__(
        self,
        products: tuple[Product, ...],
        assignment_by_code: dict[str, CandidateBox],
        freight_policy: FreightPolicy,
    ) -> None:
        # The independent evaluator is both an input validation step and the
        # reference value against which incremental accounting is checked.
        self.reference_costs = evaluate_assignments(
            products, assignment_by_code, freight_policy
        )
        thicknesses = {
            candidate.thickness_mm for candidate in assignment_by_code.values()
        }
        if len(thicknesses) != 1:
            raise ValueError("plateau walks require one global carton thickness")

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

        self.packaging_mills = self.reference_costs.packaging_mills
        self.freight_mills = self.reference_costs.freight_mills
        self.pallets = self.reference_costs.pallets

    def calculate_move(self, code: str, target: CandidateBox) -> PlateauMove | None:
        product = self.product_by_code[code]
        source = self.assignment[code]
        source_type = box_type_key(source)
        target_type = box_type_key(target)
        if source_type == target_type:
            # Changing only a solver-local candidate ID is not diversification.
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
        freight_delta = (
            pallet_delta * self.freight_policy.expected_mills_per_pallet
        )
        type_delta = -int(self.product_counts[source_type] == 1) + int(
            self.product_counts[target_type] == 0
        )
        return PlateauMove(
            code=code,
            source_type=source_type,
            target_type=target_type,
            target_candidate=target,
            packaging_delta_mills=packaging_delta,
            freight_delta_mills=freight_delta,
            pallet_delta=pallet_delta,
            type_delta=type_delta,
        )

    def apply(self, move: PlateauMove) -> None:
        current_source = self.assignment[move.code]
        if box_type_key(current_source) != move.source_type:
            raise ValueError(f"stale plateau move for {move.code}")
        recalculated = self.calculate_move(move.code, move.target_candidate)
        if recalculated != move:
            raise ValueError(f"plateau move for {move.code} is stale after another move")

        product = self.product_by_code[move.code]
        for plant in PLANTS:
            moved_volume = product.annual_volume_by_plant[plant]
            self.volumes[move.source_type][plant] -= moved_volume
            self.volumes[move.target_type][plant] += moved_volume
        self.product_counts[move.source_type] -= 1
        self.product_counts[move.target_type] += 1
        self.assignment[move.code] = move.target_candidate
        self.packaging_mills += move.packaging_delta_mills
        self.freight_mills += move.freight_delta_mills
        self.pallets += move.pallet_delta


def _targets_by_code(
    products: tuple[Product, ...],
    assignment_by_code: dict[str, CandidateBox],
    candidates: Iterable[CandidateBox],
) -> dict[str, tuple[CandidateBox, ...]]:
    """Deduplicate eligible targets by physical geometry for every SKU."""

    product_codes = {product.code for product in products}
    thicknesses = {
        candidate.thickness_mm for candidate in assignment_by_code.values()
    }
    if len(thicknesses) != 1:
        raise ValueError("plateau walks require one global carton thickness")
    thickness_mm = next(iter(thicknesses))
    by_code: dict[str, dict[BoxTypeKey, CandidateBox]] = {
        code: {} for code in product_codes
    }

    # Always retain each SKU's starting design.  This makes reverse moves
    # available even when the external candidate universe omitted incumbents.
    for code, candidate in assignment_by_code.items():
        by_code[code][box_type_key(candidate)] = candidate

    for candidate in candidates:
        if candidate.thickness_mm != thickness_mm:
            continue
        for code in candidate.compatible_product_codes & product_codes:
            key = box_type_key(candidate)
            current = by_code[code].get(key)
            if current is None or _candidate_rank(candidate) < _candidate_rank(current):
                by_code[code][key] = candidate
    return {
        code: tuple(sorted(targets.values(), key=lambda candidate: box_type_key(candidate)))
        for code, targets in by_code.items()
    }


def enumerate_zero_cost_moves(
    products: tuple[Product, ...],
    assignment_by_code: dict[str, CandidateBox],
    candidates: Iterable[CandidateBox],
    freight_policy: FreightPolicy,
) -> tuple[PlateauMove, ...]:
    """Return all feasible one-SKU moves whose total delta is exactly zero.

    Packaging deltas include tier changes at both the source and destination
    physical designs independently for every plant.  Freight remains SKU-level
    and is calculated with the candidate capacities and configured pallet rate.
    """

    state = _IncrementalPlateauState(products, assignment_by_code, freight_policy)
    targets = _targets_by_code(products, assignment_by_code, candidates)
    zero_cost_moves: list[PlateauMove] = []
    for product in products:
        for target in targets[product.code]:
            move = state.calculate_move(product.code, target)
            if move is not None and move.total_delta_mills == 0:
                zero_cost_moves.append(move)
    zero_cost_moves.sort(
        key=lambda move: (move.code, move.target_type, move.target_candidate.candidate_id)
    )
    return tuple(zero_cost_moves)


def apply_plateau_move(
    assignment_by_code: dict[str, CandidateBox], move: PlateauMove
) -> dict[str, CandidateBox]:
    """Return an assignment copy with ``move`` applied after a stale check."""

    if move.code not in assignment_by_code:
        raise ValueError(f"unknown product code in plateau move: {move.code}")
    if box_type_key(assignment_by_code[move.code]) != move.source_type:
        raise ValueError(f"stale plateau move for {move.code}")
    result = dict(assignment_by_code)
    result[move.code] = move.target_candidate
    return result


def diversify_assignment(
    products: tuple[Product, ...],
    assignment_by_code: dict[str, CandidateBox],
    candidates: Iterable[CandidateBox],
    freight_policy: FreightPolicy,
    *,
    max_steps: int = 1_000,
    random_seed: int = 42,
    avoid_revisits: bool = True,
    max_pallets: int | None = None,
) -> PlateauWalkResult:
    """Walk across exact zero-cost moves and return the farthest state visited.

    Distance is the number of SKUs assigned to a different physical geometry
    than at the start.  Returning the farthest state, instead of blindly the
    final state, prevents a random backtrack from erasing useful diversity.
    With ``avoid_revisits=True`` the walk stops when every zero-cost neighbour
    has already been seen.
    """

    if max_steps < 0:
        raise ValueError("max_steps cannot be negative")
    if max_pallets is not None and max_pallets < 0:
        raise ValueError("max_pallets cannot be negative")
    state = _IncrementalPlateauState(products, assignment_by_code, freight_policy)
    targets = _targets_by_code(products, assignment_by_code, candidates)
    rng = random.Random(random_seed)
    ordered_codes = tuple(product.code for product in products)
    code_index = {code: index for index, code in enumerate(ordered_codes)}
    initial_fingerprint = tuple(
        box_type_key(assignment_by_code[code]) for code in ordered_codes
    )
    current_fingerprint = initial_fingerprint
    visited = {initial_fingerprint}
    path: list[PlateauMove] = []
    best_assignment = dict(assignment_by_code)
    best_path: tuple[PlateauMove, ...] = ()
    best_distance = 0
    walk_steps = 0

    for _ in range(max_steps):
        available: list[tuple[PlateauMove, tuple[BoxTypeKey, ...]]] = []
        for product in products:
            product_position = code_index[product.code]
            for target in targets[product.code]:
                move = state.calculate_move(product.code, target)
                if move is None or move.total_delta_mills != 0:
                    continue
                if max_pallets is not None and state.pallets + move.pallet_delta > max_pallets:
                    continue
                next_fingerprint_list = list(current_fingerprint)
                next_fingerprint_list[product_position] = move.target_type
                next_fingerprint = tuple(next_fingerprint_list)
                if avoid_revisits and next_fingerprint in visited:
                    continue
                available.append((move, next_fingerprint))
        if not available:
            break

        move, next_fingerprint = rng.choice(available)
        state.apply(move)
        path.append(move)
        walk_steps += 1
        current_fingerprint = next_fingerprint
        visited.add(current_fingerprint)
        distance = sum(
            current != initial
            for current, initial in zip(current_fingerprint, initial_fingerprint)
        )
        if distance > best_distance:
            best_distance = distance
            best_assignment = dict(state.assignment)
            best_path = tuple(path)

    final_costs = evaluate_assignments(products, best_assignment, freight_policy)
    if final_costs.total_mills != state.reference_costs.total_mills:
        raise AssertionError(
            "zero-cost plateau walk changed the independently evaluated objective: "
            f"{state.reference_costs.total_mills} -> {final_costs.total_mills}"
        )
    return PlateauWalkResult(
        assignment=best_assignment,
        moves=best_path,
        initial_costs=state.reference_costs,
        final_costs=final_costs,
        distance_from_start=best_distance,
        walk_steps=walk_steps,
        visited_states=len(visited),
    )
