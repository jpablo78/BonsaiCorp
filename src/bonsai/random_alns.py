"""Stochastic adaptive large-neighbourhood search with short SCIP repairs.

The deterministic destination and tier neighbourhoods are useful once, but
repeating the same ranked pools from a strong incumbent quickly becomes
unproductive.  This module deliberately samples a fresh neighbourhood on
every iteration:

* complete incumbent source box types are released (never a partial source);
* five to thirty exact destination geometries are sampled with a mixture of
  uniform and procurement-tier guidance;
* a small feasible ``ruin`` perturbs the released products; and
* :func:`bonsai.scip_optimizer.solve_with_scip` repairs that state while all
  products outside the neighbourhood remain objective constants.

The search may carry a controlled non-best state between iterations.  The
best assignment is nevertheless monotone, independently CSV-validated, and
atomically snapshotted after every strict improvement.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict, dataclass
from decimal import Decimal
import json
import math
from pathlib import Path
import random
import time
from typing import Iterable, Mapping, Sequence, TypeVar

from .annealing import IncrementalAssignmentState
from .config import DISCOUNT_TIERS, FreightPolicy
from .costs import BoxTypeKey, box_type_key, evaluate_assignments, freight_pallets
from .data import load_prepared_data
from .exact_candidates import generate_exact_candidates
from .models import CandidateBox, CostBreakdown, Dimensions, PLANTS, PreparedData, Product
from .scip_multi_destination_lns import _scip_result_payload
from .scip_optimizer import solve_with_scip
from .solution_validation import ValidationResult, validate_solution_csv
from .tier_lns import (
    _atomic_write_assignment,
    _atomic_write_json,
    _cost_payload,
    _infer_thickness,
    _mills_from_usd,
    _next_snapshot_number,
    _unique_internal_designs,
)


T = TypeVar("T")


@dataclass(frozen=True)
class RandomNeighborhood:
    """A stochastic restricted master problem.

    ``product_codes`` is exactly the union of ``source_types`` in the state
    from which the neighbourhood was built.  Consequently a source type is
    never only partly released, which is important when procurement discounts
    depend on the type's complete volume at a plant.
    """

    source_types: tuple[BoxTypeKey, ...]
    product_codes: tuple[str, ...]
    destinations: tuple[CandidateBox, ...]
    source_strategy: str
    destination_strategy: str
    uncovered_product_codes: tuple[str, ...]

    @property
    def source_type_count(self) -> int:
        return len(self.source_types)

    @property
    def destination_count(self) -> int:
        return len(self.destinations)

    @property
    def sku_count(self) -> int:
        return len(self.product_codes)

    def allowed_internals(
        self, assignment: Mapping[str, CandidateBox]
    ) -> dict[str, tuple[Dimensions, ...]]:
        """Return incumbent plus every sampled compatible destination per SKU."""

        allowed: dict[str, set[Dimensions]] = {
            code: {assignment[code].internal} for code in self.product_codes
        }
        for candidate in self.destinations:
            for code in candidate.compatible_product_codes & allowed.keys():
                allowed[code].add(candidate.internal)
        return {
            code: tuple(sorted(internals, key=Dimensions.as_tuple))
            for code, internals in sorted(allowed.items())
        }


@dataclass(frozen=True)
class RuinResult:
    assignment: dict[str, CandidateBox]
    costs: CostBreakdown
    attempted_moves: int
    applied_moves: int


def _weighted_sample_without_replacement(
    items: Sequence[T], weights: Sequence[float], count: int, rng: random.Random
) -> tuple[T, ...]:
    """Sample without replacement while tolerating zero/invalid weights."""

    if len(items) != len(weights):
        raise ValueError("items and weights must have equal length")
    if count < 0:
        raise ValueError("count cannot be negative")
    remaining_items = list(items)
    remaining_weights = [
        weight if math.isfinite(weight) and weight > 0 else 0.0
        for weight in weights
    ]
    selected: list[T] = []
    for _ in range(min(count, len(remaining_items))):
        total = sum(remaining_weights)
        if total <= 0:
            index = rng.randrange(len(remaining_items))
        else:
            point = rng.random() * total
            cumulative = 0.0
            index = len(remaining_items) - 1
            for candidate_index, weight in enumerate(remaining_weights):
                cumulative += weight
                if point <= cumulative:
                    index = candidate_index
                    break
        selected.append(remaining_items.pop(index))
        remaining_weights.pop(index)
    return tuple(selected)


def _groups_by_type(
    assignment: Mapping[str, CandidateBox],
) -> dict[BoxTypeKey, tuple[str, ...]]:
    groups: dict[BoxTypeKey, list[str]] = defaultdict(list)
    for code, candidate in assignment.items():
        groups[box_type_key(candidate)].append(code)
    return {
        key: tuple(sorted(codes))
        for key, codes in sorted(groups.items(), key=lambda item: item[0])
    }


def _volumes_by_type(
    products_by_code: Mapping[str, Product],
    groups: Mapping[BoxTypeKey, tuple[str, ...]],
) -> dict[BoxTypeKey, dict[str, int]]:
    return {
        key: {
            plant: sum(
                products_by_code[code].annual_volume_by_plant[plant]
                for code in codes
            )
            for plant in PLANTS
        }
        for key, codes in groups.items()
    }


def _tier_proximity(volumes: Mapping[str, int]) -> float:
    """Smooth preference for types near any documented discount threshold."""

    score = 1.0
    thresholds = tuple(tier.lower_inclusive for tier in DISCOUNT_TIERS[1:])
    for volume in volumes.values():
        if volume <= 0:
            continue
        nearest = min(abs(volume - threshold) for threshold in thresholds)
        # The cap prevents an exactly-on-tier type from monopolising sampling.
        score += min(20.0, 20_000.0 / (nearest + 1_000.0))
    return score


def _destination_weight(
    candidate: CandidateBox,
    released_codes: set[str],
    assignment: Mapping[str, CandidateBox],
    products_by_code: Mapping[str, Product],
    active_volumes: Mapping[BoxTypeKey, Mapping[str, int]],
    *,
    tier_guided: bool,
) -> float:
    compatible = candidate.compatible_product_codes & released_codes
    if not compatible:
        return 0.0
    if not tier_guided:
        return 1.0

    key = box_type_key(candidate)
    current = active_volumes.get(key, {plant: 0 for plant in PLANTS})
    incoming = {
        plant: sum(
            products_by_code[code].annual_volume_by_plant[plant]
            for code in compatible
            if box_type_key(assignment[code]) != key
        )
        for plant in PLANTS
    }
    crossings = 0
    closeness = 0.0
    for plant in PLANTS:
        before = current[plant]
        reachable = before + incoming[plant]
        for tier in DISCOUNT_TIERS[1:]:
            threshold = tier.lower_inclusive
            if before < threshold <= reachable:
                crossings += 1
                closeness += threshold / max(threshold - before, 1)

    # Positive pallet deltas are deliberately not forbidden: the global USD
    # objective decides whether a procurement tier compensates for them.  A
    # modest preference for freight-saving destinations improves repair speed.
    pallet_saving = 0
    for code in compatible:
        product = products_by_code[code]
        source = assignment[code]
        pallet_saving += sum(
            freight_pallets(product, source, plant)
            - freight_pallets(product, candidate, plant)
            for plant in PLANTS
        )
    overlap = len(compatible)
    active_bonus = 3.0 if key in active_volumes else 0.0
    return max(
        0.05,
        1.0
        + overlap ** 1.25
        + 12.0 * crossings
        + min(closeness, 50.0)
        + active_bonus
        + max(0.0, pallet_saving / 20.0),
    )


def build_random_neighborhood(
    products: tuple[Product, ...],
    assignment: Mapping[str, CandidateBox],
    candidates: tuple[CandidateBox, ...],
    rng: random.Random,
    *,
    min_source_types: int = 5,
    max_source_types: int = 30,
    min_destinations: int = 5,
    max_destinations: int = 30,
    max_skus: int = 140,
    tier_guided_probability: float = 0.70,
) -> RandomNeighborhood:
    """Create a fresh connected-ish random neighbourhood.

    The first source group seeds an exact destination anchor.  Additional full
    source groups are biased toward products compatible with that anchor, but
    retain a non-zero uniform probability.  Destination sampling first covers
    as many released SKUs as possible and then fills the requested pool.  No
    global ranked destination list is constructed or reused.
    """

    if min_source_types < 1:
        raise ValueError("min_source_types must be positive")
    if max_source_types < min_source_types:
        raise ValueError("max_source_types cannot be below min_source_types")
    if min_destinations < 2:
        raise ValueError("min_destinations must be at least two")
    if max_destinations < min_destinations:
        raise ValueError("max_destinations cannot be below min_destinations")
    if max_skus < 1:
        raise ValueError("max_skus must be positive")
    if not 0 <= tier_guided_probability <= 1:
        raise ValueError("tier_guided_probability must be between zero and one")

    product_by_code = {product.code: product for product in products}
    groups = _groups_by_type(assignment)
    if len(groups) < min_source_types:
        raise ValueError("incumbent has fewer source types than requested")
    volumes = _volumes_by_type(product_by_code, groups)
    source_strategy = (
        "tier_linked" if rng.random() < tier_guided_probability else "uniform_linked"
    )
    destination_strategy = (
        "tier_mixed" if rng.random() < tier_guided_probability else "uniform"
    )

    group_keys = tuple(groups)
    source_weights = [
        (
            _tier_proximity(volumes[key])
            if source_strategy == "tier_linked"
            else 1.0
        )
        / math.sqrt(max(len(groups[key]), 1))
        for key in group_keys
    ]
    seed_type = _weighted_sample_without_replacement(
        group_keys, source_weights, 1, rng
    )[0]
    seed_codes = set(groups[seed_type])
    anchor_pool = tuple(
        candidate
        for candidate in candidates
        if candidate.compatible_product_codes & seed_codes
        and box_type_key(candidate) != seed_type
    )
    if anchor_pool:
        anchor_weights = [
            _destination_weight(
                candidate,
                seed_codes,
                assignment,
                product_by_code,
                volumes,
                tier_guided=source_strategy == "tier_linked",
            )
            for candidate in anchor_pool
        ]
        anchor = _weighted_sample_without_replacement(
            anchor_pool, anchor_weights, 1, rng
        )[0]
    else:
        anchor = None

    desired_source_count = rng.randint(
        min_source_types, min(max_source_types, len(group_keys))
    )
    selected_types = [seed_type]
    selected_codes = set(seed_codes)
    remaining_types = [key for key in group_keys if key != seed_type]
    while remaining_types and len(selected_types) < desired_source_count:
        viable = [
            key
            for key in remaining_types
            if len(selected_codes) + len(groups[key]) <= max_skus
        ]
        if not viable:
            break
        weights: list[float] = []
        for key in viable:
            base = (
                _tier_proximity(volumes[key])
                if source_strategy == "tier_linked"
                else 1.0
            ) / math.sqrt(max(len(groups[key]), 1))
            linked = (
                len(set(groups[key]) & anchor.compatible_product_codes)
                if anchor is not None
                else 0
            )
            weights.append(base * (1.0 + 5.0 * linked))
        chosen = _weighted_sample_without_replacement(viable, weights, 1, rng)[0]
        selected_types.append(chosen)
        selected_codes.update(groups[chosen])
        remaining_types.remove(chosen)

    # If large early groups made the random target unreachable, fill with the
    # smallest complete groups.  We still never split a source type.
    if len(selected_types) < min_source_types:
        for key in sorted(remaining_types, key=lambda item: (len(groups[item]), item)):
            if len(selected_codes) + len(groups[key]) > max_skus:
                continue
            selected_types.append(key)
            selected_codes.update(groups[key])
            if len(selected_types) >= min_source_types:
                break
    if len(selected_types) < min_source_types:
        raise RuntimeError("max_skus cannot accommodate the minimum source types")

    destination_pool = tuple(
        candidate
        for candidate in candidates
        if candidate.compatible_product_codes & selected_codes
    )
    if len(destination_pool) < min_destinations:
        raise RuntimeError("too few compatible exact destinations")
    desired_destinations = rng.randint(
        min_destinations, min(max_destinations, len(destination_pool))
    )
    destination_weights = {
        candidate.internal: _destination_weight(
            candidate,
            selected_codes,
            assignment,
            product_by_code,
            volumes,
            tier_guided=destination_strategy == "tier_mixed",
        )
        for candidate in destination_pool
    }

    # Randomised set cover: prioritise choices that give a SKU an alternative
    # to its current type.  The stochastic weight prevents this from becoming
    # another fixed, repeatedly exhausted destination ranking.
    uncovered = set(selected_codes)
    selected_destinations: list[CandidateBox] = []
    remaining_destinations = list(destination_pool)
    while uncovered and len(selected_destinations) < desired_destinations:
        coverages = [
            {
                code
                for code in candidate.compatible_product_codes & uncovered
                if box_type_key(assignment[code]) != box_type_key(candidate)
            }
            for candidate in remaining_destinations
        ]
        if not any(coverages):
            break
        weights = [
            destination_weights[candidate.internal] * len(coverage) ** 2
            for candidate, coverage in zip(
                remaining_destinations, coverages, strict=True
            )
        ]
        chosen = _weighted_sample_without_replacement(
            remaining_destinations, weights, 1, rng
        )[0]
        index = remaining_destinations.index(chosen)
        uncovered -= coverages[index]
        selected_destinations.append(chosen)
        remaining_destinations.pop(index)

    fill_count = desired_destinations - len(selected_destinations)
    if fill_count:
        fill_weights = [
            destination_weights[candidate.internal]
            if destination_strategy == "tier_mixed" or rng.random() < 0.5
            else 1.0
            for candidate in remaining_destinations
        ]
        selected_destinations.extend(
            _weighted_sample_without_replacement(
                remaining_destinations, fill_weights, fill_count, rng
            )
        )

    # Recalculate actual uncovered products after all fill destinations.
    movable = set()
    for candidate in selected_destinations:
        target_type = box_type_key(candidate)
        movable.update(
            code
            for code in candidate.compatible_product_codes & selected_codes
            if box_type_key(assignment[code]) != target_type
        )
    uncovered = selected_codes - movable
    return RandomNeighborhood(
        source_types=tuple(sorted(selected_types)),
        product_codes=tuple(sorted(selected_codes)),
        destinations=tuple(
            sorted(selected_destinations, key=lambda candidate: candidate.internal.as_tuple())
        ),
        source_strategy=source_strategy,
        destination_strategy=destination_strategy,
        uncovered_product_codes=tuple(sorted(uncovered)),
    )


def ruin_assignment(
    products: tuple[Product, ...],
    assignment: Mapping[str, CandidateBox],
    neighborhood: RandomNeighborhood,
    freight_policy: FreightPolicy,
    rng: random.Random,
    *,
    move_fraction: float,
    max_total_mills: int,
    max_pallets: int,
    proposal_temperature_usd: float = 5_000.0,
) -> RuinResult:
    """Apply feasible stochastic moves before SCIP recreates the neighbourhood."""

    if not 0 <= move_fraction <= 1:
        raise ValueError("move_fraction must be between zero and one")
    if proposal_temperature_usd <= 0:
        raise ValueError("proposal_temperature_usd must be positive")
    state = IncrementalAssignmentState(products, dict(assignment), freight_policy)
    by_code: dict[str, list[CandidateBox]] = {
        code: [
            candidate
            for candidate in neighborhood.destinations
            if code in candidate.compatible_product_codes
            and box_type_key(candidate) != box_type_key(assignment[code])
        ]
        for code in neighborhood.product_codes
    }
    eligible = [code for code, choices in by_code.items() if choices]
    rng.shuffle(eligible)
    requested = max(1, round(len(eligible) * move_fraction)) if eligible else 0
    attempted = 0
    applied = 0
    for code in eligible:
        if applied >= requested:
            break
        choices = by_code[code]
        moves = [state.calculate_move(code, candidate) for candidate in choices]
        moves = [move for move in moves if move is not None]
        if not moves:
            continue
        attempted += 1
        # A softmax over exact deltas makes destructive moves varied but avoids
        # spending most short SCIP calls merely undoing catastrophic freight.
        minimum_delta = min(move.total_delta_mills for move in moves)
        scale_mills = max(proposal_temperature_usd * 1000.0, 1.0)
        weights = [
            math.exp(
                -min(
                    max((move.total_delta_mills - minimum_delta) / scale_mills, 0.0),
                    50.0,
                )
            )
            for move in moves
        ]
        move = _weighted_sample_without_replacement(moves, weights, 1, rng)[0]
        if state.total_mills + move.total_delta_mills > max_total_mills:
            continue
        if state.pallets + move.pallet_delta > max_pallets:
            continue
        state.apply(move)
        applied += 1

    costs = state.validate()
    if costs.total_mills > max_total_mills:
        raise AssertionError("ruin exceeded its objective excursion cap")
    if costs.pallets > max_pallets:
        raise AssertionError("ruin exceeded the global pallet budget")
    return RuinResult(dict(state.assignment), costs, attempted, applied)


def _minimum_exact_pallets(
    products: tuple[Product, ...], candidates: tuple[CandidateBox, ...]
) -> int:
    minimum = 0
    for product in products:
        compatible = (
            candidate
            for candidate in candidates
            if product.code in candidate.compatible_product_codes
        )
        minimum += min(
            sum(freight_pallets(product, candidate, plant) for plant in PLANTS)
            for candidate in compatible
        )
    return minimum


def _temperature(initial: float, final: float, progress: float) -> float:
    if initial <= 0 or final <= 0:
        raise ValueError("temperatures must be positive")
    progress = min(max(progress, 0.0), 1.0)
    return initial * (final / initial) ** progress


def _accept(delta_mills: int, temperature_usd: float, rng: random.Random) -> bool:
    if delta_mills <= 0:
        return True
    return rng.random() < math.exp(-delta_mills / (temperature_usd * 1000.0))


def _neighborhood_payload(item: RandomNeighborhood) -> dict[str, object]:
    return {
        "source_strategy": item.source_strategy,
        "destination_strategy": item.destination_strategy,
        "source_type_count": item.source_type_count,
        "destination_count": item.destination_count,
        "sku_count": item.sku_count,
        "uncovered_sku_count": len(item.uncovered_product_codes),
        "source_types": item.source_types,
        "destination_internals": tuple(
            candidate.internal.as_tuple() for candidate in item.destinations
        ),
    }


def run_random_alns(args: argparse.Namespace) -> dict[str, object]:
    """Run several stochastic SCIP trajectories within one wall-clock budget."""

    if args.duration_seconds <= 0:
        raise ValueError("--duration-seconds must be positive")
    if args.runs < 1:
        raise ValueError("--runs must be positive")
    if not 0.05 <= args.time_per_neighborhood_min <= args.time_per_neighborhood_max:
        raise ValueError("invalid per-neighborhood time interval")
    if args.num_threads < 1:
        raise ValueError("--num-threads must be positive")
    if args.max_extra_pallets < 0:
        raise ValueError("--max-extra-pallets cannot be negative")

    data: PreparedData = load_prepared_data(args.data_dir)
    policy = FreightPolicy(extra_region_share=args.extra_region_share)
    warm = validate_solution_csv(args.warm_start, data, policy)
    thickness = _infer_thickness(warm)
    if thickness != 3.0:
        raise ValueError("stochastic SCIP ALNS currently supports only 3 mm")
    target_mills = _mills_from_usd(args.target_total_usd)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_path = args.output_dir / "asignacion_optima.csv"
    best: ValidationResult = warm
    if best_path.exists():
        existing = validate_solution_csv(best_path, data, policy)
        if _infer_thickness(existing) != thickness:
            raise ValueError("existing ALNS output uses another thickness")
        if existing.costs.total_mills < best.costs.total_mills:
            best = existing
    initial_costs = best.costs
    snapshot_number = _next_snapshot_number(args.output_dir)
    _atomic_write_assignment(
        args.output_dir / f"incumbent_{snapshot_number:04d}.csv",
        data,
        best.assignment,
        policy,
    )
    snapshot_number += 1
    _atomic_write_assignment(best_path, data, best.assignment, policy)

    candidates, candidate_stats = generate_exact_candidates(
        data.products,
        thickness,
        retained_designs=_unique_internal_designs(best.assignment),
    )
    minimum_pallets = _minimum_exact_pallets(data.products, candidates)
    maximum_pallets = minimum_pallets + args.max_extra_pallets
    if best.costs.pallets > maximum_pallets:
        raise ValueError("warm start exceeds the requested global pallet budget")

    started_at = time.perf_counter()
    deadline = started_at + args.duration_seconds
    attempts: list[dict[str, object]] = []
    improvements: list[dict[str, object]] = []
    summary_path = args.output_dir / "resumen_random_alns.json"
    summary: dict[str, object] = {
        "solver": "stochastic ALNS with restricted SCIP repair",
        "configuration": {
            key: str(value) if isinstance(value, (Path, Decimal)) else value
            for key, value in vars(args).items()
        },
        "candidate_stats": asdict(candidate_stats),
        "candidate_count": len(candidates),
        "minimum_exact_pallets": minimum_pallets,
        "maximum_pallets": maximum_pallets,
        "initial": _cost_payload(initial_costs),
        "attempts": attempts,
        "improvements": improvements,
    }
    print(
        f"Random ALNS start: USD {best.costs.total_mills / 1000:,.2f}; "
        f"{len(candidates):,} exact candidates; {args.runs} seeds; "
        f"{args.duration_seconds:.0f}s total",
        flush=True,
    )

    termination = "time_limit"
    for run_index in range(args.runs):
        now = time.perf_counter()
        if now >= deadline:
            break
        # Reserve an equal share of the remaining time for every remaining
        # seed.  Each new trajectory starts from the best validated incumbent.
        run_deadline = now + (deadline - now) / (args.runs - run_index)
        seed = args.random_seed + run_index * 1_000_003
        rng = random.Random(seed)
        current_assignment = dict(best.assignment)
        current_costs = best.costs
        stagnation = 0
        iteration = 0
        while time.perf_counter() < run_deadline:
            iteration += 1
            elapsed = time.perf_counter() - started_at
            progress = min(elapsed / args.duration_seconds, 1.0)
            excursion_usd = _temperature(
                args.max_excursion_initial_usd,
                args.max_excursion_final_usd,
                progress,
            )
            acceptance_temperature = _temperature(
                args.acceptance_temperature_initial_usd,
                args.acceptance_temperature_final_usd,
                progress,
            )
            proposal_temperature = _temperature(
                args.proposal_temperature_initial_usd,
                args.proposal_temperature_final_usd,
                progress,
            )
            # The admissible excursion cools continuously.  A state accepted
            # under yesterday's wider cap can therefore sit outside the new
            # cap even before the next ruin applies a move.
            if (
                current_costs.total_mills
                > best.costs.total_mills + round(excursion_usd * 1000)
            ):
                current_assignment = dict(best.assignment)
                current_costs = best.costs
                stagnation = 0
            try:
                neighborhood = build_random_neighborhood(
                    data.products,
                    current_assignment,
                    candidates,
                    rng,
                    min_source_types=args.min_source_types,
                    max_source_types=args.max_source_types,
                    min_destinations=args.min_destinations,
                    max_destinations=args.max_destinations,
                    max_skus=args.max_skus,
                    tier_guided_probability=args.tier_guided_probability,
                )
            except RuntimeError as exc:
                attempts.append(
                    {
                        "run": run_index + 1,
                        "iteration": iteration,
                        "seed": seed,
                        "construction_error": str(exc),
                    }
                )
                continue

            move_fraction = rng.uniform(
                args.ruin_fraction_min, args.ruin_fraction_max
            )
            ruin = ruin_assignment(
                data.products,
                current_assignment,
                neighborhood,
                policy,
                rng,
                move_fraction=move_fraction,
                max_total_mills=best.costs.total_mills
                + round(excursion_usd * 1000),
                max_pallets=maximum_pallets,
                proposal_temperature_usd=proposal_temperature,
            )
            remaining = run_deadline - time.perf_counter()
            if remaining <= 0:
                break
            # Log-uniform solve times create many cheap probes while retaining
            # occasional two-second repairs for the harder random ruins.
            low = args.time_per_neighborhood_min
            high = min(args.time_per_neighborhood_max, max(remaining, low))
            solve_seconds = (
                low
                if high <= low
                else math.exp(rng.uniform(math.log(low), math.log(high)))
            )
            allowed = neighborhood.allowed_internals(current_assignment)
            result = solve_with_scip(
                data,
                thickness,
                policy,
                time_limit_seconds=solve_seconds,
                num_threads=args.num_threads,
                random_seed=rng.randrange(1, 2_000_000_000),
                initial_assignment=ruin.assignment,
                max_extra_pallets=args.max_extra_pallets,
                free_product_codes=neighborhood.product_codes,
                allowed_internals_by_product=allowed,
                precomputed_exact_candidates=candidates,
                precomputed_exact_candidate_stats=candidate_stats,
                memory_limit_mb=args.memory_limit_mb,
                scip_parameters=("\n".join(args.scip_parameter) or None),
            )
            candidate_costs = result.costs
            delta_current = candidate_costs.total_mills - current_costs.total_mills
            accepted = (
                candidate_costs.total_mills
                <= best.costs.total_mills + round(excursion_usd * 1000)
                and _accept(delta_current, acceptance_temperature, rng)
            )
            improved_best = candidate_costs.total_mills < best.costs.total_mills
            attempt: dict[str, object] = {
                "run": run_index + 1,
                "iteration": iteration,
                "sequence": len(attempts) + 1,
                "seed": seed,
                "before_current_usd": current_costs.total_mills / 1000,
                "best_before_usd": best.costs.total_mills / 1000,
                "excursion_cap_usd": excursion_usd,
                "acceptance_temperature_usd": acceptance_temperature,
                "move_fraction": move_fraction,
                "ruin_applied_moves": ruin.applied_moves,
                "ruin_attempted_moves": ruin.attempted_moves,
                "ruined_usd": ruin.costs.total_mills / 1000,
                "solve_limit_seconds": solve_seconds,
                **_neighborhood_payload(neighborhood),
                **_scip_result_payload(result),
                "accepted_current": accepted,
                "improved_best": improved_best,
            }

            if improved_best:
                candidate_path = args.output_dir / ".candidate_validation.csv"
                try:
                    checked = _atomic_write_assignment(
                        candidate_path, data, result.assignment, policy
                    )
                    if checked.costs.total_mills != candidate_costs.total_mills:
                        raise RuntimeError("CSV validation changed the SCIP cost")
                    before_best = best.costs.total_mills
                    best = checked
                    current_assignment = dict(best.assignment)
                    current_costs = best.costs
                    snapshot_path = (
                        args.output_dir / f"incumbent_{snapshot_number:04d}.csv"
                    )
                    _atomic_write_assignment(
                        snapshot_path, data, best.assignment, policy
                    )
                    _atomic_write_assignment(best_path, data, best.assignment, policy)
                    snapshot_number += 1
                    improvement = {
                        "run": run_index + 1,
                        "iteration": iteration,
                        "attempt": len(attempts) + 1,
                        "before_usd": before_best / 1000,
                        "after_usd": best.costs.total_mills / 1000,
                        "saving_usd": (before_best - best.costs.total_mills) / 1000,
                        "snapshot_path": str(snapshot_path),
                    }
                    improvements.append(improvement)
                    attempt["snapshot_path"] = str(snapshot_path)
                    stagnation = 0
                    print(
                        f"  seed {run_index + 1} iter {iteration}: accepted best "
                        f"USD {best.costs.total_mills / 1000:,.2f} "
                        f"(saved {improvement['saving_usd']:,.2f})",
                        flush=True,
                    )
                finally:
                    candidate_path.unlink(missing_ok=True)
            elif accepted:
                checked_costs = evaluate_assignments(
                    data.products, result.assignment, policy
                )
                if checked_costs.total_mills != candidate_costs.total_mills:
                    raise RuntimeError("accepted ALNS state failed independent validation")
                current_assignment = dict(result.assignment)
                current_costs = checked_costs
                stagnation += 1
            else:
                stagnation += 1

            if stagnation >= args.reset_after_stagnation:
                current_assignment = dict(best.assignment)
                current_costs = best.costs
                stagnation = 0
                attempt["reset_to_best"] = True

            attempts.append(attempt)
            summary["best"] = _cost_payload(best.costs)
            summary["saving_usd"] = (
                initial_costs.total_mills - best.costs.total_mills
            ) / 1000
            summary["elapsed_seconds"] = time.perf_counter() - started_at
            _atomic_write_json(summary_path, summary)
            if len(attempts) % 20 == 0:
                print(
                    f"  {len(attempts)} repairs; best USD "
                    f"{best.costs.total_mills / 1000:,.2f}; "
                    f"elapsed {summary['elapsed_seconds']:.0f}s",
                    flush=True,
                )
            if target_mills is not None and best.costs.total_mills <= target_mills:
                termination = "target_met"
                break
        if termination == "target_met":
            break

    final_checked = validate_solution_csv(best_path, data, policy)
    if final_checked.costs.total_mills != best.costs.total_mills:
        raise RuntimeError("final best CSV does not match the in-memory result")
    summary["best"] = _cost_payload(final_checked.costs)
    summary["saving_usd"] = (
        initial_costs.total_mills - final_checked.costs.total_mills
    ) / 1000
    summary["elapsed_seconds"] = time.perf_counter() - started_at
    summary["termination"] = termination
    summary["target_met"] = (
        final_checked.costs.total_mills <= target_mills
        if target_mills is not None
        else None
    )
    summary["best_path"] = str(best_path)
    _atomic_write_json(summary_path, summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stochastic ruin-and-recreate ALNS with short SCIP repairs"
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--warm-start", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--duration-seconds", type=float, default=600.0)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--random-seed", type=int, default=73_921)
    parser.add_argument("--time-per-neighborhood-min", type=float, default=0.2)
    parser.add_argument("--time-per-neighborhood-max", type=float, default=2.0)
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument("--max-extra-pallets", type=int, default=5_000)
    parser.add_argument("--min-source-types", type=int, default=5)
    parser.add_argument("--max-source-types", type=int, default=30)
    parser.add_argument("--min-destinations", type=int, default=5)
    parser.add_argument("--max-destinations", type=int, default=30)
    parser.add_argument("--max-skus", type=int, default=140)
    parser.add_argument("--tier-guided-probability", type=float, default=0.70)
    parser.add_argument("--ruin-fraction-min", type=float, default=0.10)
    parser.add_argument("--ruin-fraction-max", type=float, default=0.55)
    parser.add_argument("--max-excursion-initial-usd", type=float, default=30_000.0)
    parser.add_argument("--max-excursion-final-usd", type=float, default=500.0)
    parser.add_argument(
        "--acceptance-temperature-initial-usd", type=float, default=8_000.0
    )
    parser.add_argument(
        "--acceptance-temperature-final-usd", type=float, default=25.0
    )
    parser.add_argument(
        "--proposal-temperature-initial-usd", type=float, default=5_000.0
    )
    parser.add_argument(
        "--proposal-temperature-final-usd", type=float, default=100.0
    )
    parser.add_argument("--reset-after-stagnation", type=int, default=35)
    parser.add_argument("--target-total-usd", type=Decimal)
    parser.add_argument("--extra-region-share", type=float, default=0.0)
    parser.add_argument("--memory-limit-mb", type=int)
    parser.add_argument("--scip-parameter", action="append", default=[])
    return parser


def main(argv: list[str] | None = None) -> int:
    summary = run_random_alns(build_parser().parse_args(argv))
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
