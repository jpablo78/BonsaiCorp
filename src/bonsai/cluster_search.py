"""Tabu/beam search over coordinated multi-facility reassignment chains.

The exact and destination LNS models protect the incumbent, which is normally
desirable but prevents an *ejection chain*: a temporarily costly consolidation
into one box type can make a second consolidation profitable by restoring a
source procurement tier.  This module deliberately keeps a bounded beam of
such intermediate states.

The first operation in every chain must move at least four SKUs drawn from at
least three physical incumbent types.  Later operations may be smaller because
they act as the refill/swap legs of the same multi-type chain.  A tabu set over
complete assignments prevents cycling, and only an independently evaluated
strict improvement is ever written as the final answer.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time
from typing import Iterable, Mapping

from .annealing import IncrementalAssignmentState
from .config import FreightPolicy
from .costs import BoxTypeKey, box_type_key, evaluate_assignments
from .data import load_prepared_data
from .destination_lns import rank_destination_work_items
from .exact_candidates import generate_exact_candidates
from .models import CandidateBox, CostBreakdown, PreparedData, Product
from .solution_validation import validate_solution_csv
from .tier_lns import (
    _atomic_write_assignment,
    _atomic_write_json,
    _cost_payload,
    _infer_thickness,
    _unique_internal_designs,
)


@dataclass(frozen=True)
class FacilityMove:
    """One exact reassignment into a common physical destination."""

    codes: tuple[str, ...]
    target: CandidateBox
    source_types: frozenset[BoxTypeKey]
    delta_mills: int
    pallet_delta: int
    packaging_delta_mills: int

    @property
    def target_type(self) -> BoxTypeKey:
        return box_type_key(self.target)


@dataclass(frozen=True)
class BeamNode:
    assignment: Mapping[str, CandidateBox]
    costs: CostBreakdown
    depth: int
    last_source_types: frozenset[BoxTypeKey]
    history: tuple[tuple[tuple[int, int, int], tuple[str, ...]], ...]


@dataclass(frozen=True)
class ClusterSearchResult:
    assignment: Mapping[str, CandidateBox]
    costs: CostBreakdown
    initial_costs: CostBreakdown
    elapsed_seconds: float
    expanded_nodes: int
    generated_moves: int
    accepted_states: int
    improvements: int
    deepest_level: int
    visited_states: int
    termination: str


def _assignment_key(
    product_codes: tuple[str, ...], assignment: Mapping[str, CandidateBox]
) -> tuple[tuple[int, int, int], ...]:
    return tuple(assignment[code].internal.as_tuple() for code in product_codes)


def _candidate_rank(candidate: CandidateBox) -> tuple[object, ...]:
    return (
        candidate.internal.as_tuple(),
        -candidate.capacity_per_pallet,
        candidate.candidate_id,
    )


def facility_candidate_pool(
    data: PreparedData,
    incumbent_assignment: Mapping[str, CandidateBox],
    exact_candidates: Iterable[CandidateBox],
    freight_policy: FreightPolicy,
    *,
    destination_pool_size: int,
) -> tuple[CandidateBox, ...]:
    """Return active facilities plus globally promising inactive facilities."""

    if destination_pool_size < 1:
        raise ValueError("destination_pool_size must be positive")
    exact = tuple(exact_candidates)
    by_internal = {candidate.internal: candidate for candidate in exact}
    active = {
        by_internal[candidate.internal]
        for candidate in incumbent_assignment.values()
    }
    ranked = rank_destination_work_items(
        data,
        incumbent_assignment,
        exact,
        freight_policy,
        min_skus=1,
        min_gross_opportunity_mills=0,
        max_destinations=destination_pool_size,
    )
    ordered: list[CandidateBox] = sorted(active, key=_candidate_rank)
    seen = {candidate.internal for candidate in ordered}
    for item in ranked:
        if item.candidate.internal in seen:
            continue
        ordered.append(item.candidate)
        seen.add(item.candidate.internal)
        if len(ordered) >= len(active) + destination_pool_size:
            break
    return tuple(ordered)


def _wrap_move(
    state: IncrementalAssignmentState,
    codes: tuple[str, ...],
    target: CandidateBox,
) -> FacilityMove | None:
    if len(codes) == 1:
        exact = state.calculate_move(codes[0], target)
    else:
        exact = state.calculate_group_move(codes, target)
    if exact is None:
        return None
    return FacilityMove(
        codes=tuple(exact.codes) if hasattr(exact, "codes") else (exact.code,),
        target=target,
        source_types=frozenset(exact.source_types)
        if hasattr(exact, "source_types")
        else frozenset((exact.source_type,)),
        delta_mills=exact.total_delta_mills,
        pallet_delta=exact.pallet_delta,
        packaging_delta_mills=exact.packaging_delta_mills,
    )


def generate_facility_moves(
    state: IncrementalAssignmentState,
    products: tuple[Product, ...],
    candidates: Iterable[CandidateBox],
    *,
    min_codes: int,
    min_source_types: int,
    individual_pool_size: int,
    high_volume_pool_size: int,
    subset_beam_width: int,
    max_bundle_size: int,
    per_destination_moves: int,
    max_move_delta_mills: int,
    priority_target_types: frozenset[BoxTypeKey] = frozenset(),
    deadline: float | None = None,
) -> tuple[FacilityMove, ...]:
    """Generate low-cost multi-SKU facility moves with a subset beam.

    The subset beam is deliberately not a pair/triple enumeration.  It starts
    from the most promising individual legs but grows bundles up to
    ``max_bundle_size`` and preserves partial bundles spanning more source
    facilities, which is what exposes procurement-threshold interactions.
    """

    if min_codes < 1 or min_source_types < 1:
        raise ValueError("minimum move sizes must be positive")
    if subset_beam_width < 1 or max_bundle_size < min_codes:
        raise ValueError("invalid subset beam dimensions")
    if per_destination_moves < 1:
        raise ValueError("per_destination_moves must be positive")

    product_by_code = {product.code: product for product in products}
    all_moves: list[FacilityMove] = []
    ordered_candidates = sorted(
        candidates,
        key=lambda candidate: (
            box_type_key(candidate) not in priority_target_types,
            _candidate_rank(candidate),
        ),
    )
    for target in ordered_candidates:
        if deadline is not None and time.monotonic() >= deadline:
            break
        target_type = box_type_key(target)
        individuals: list[tuple[int, int, str]] = []
        for code in target.compatible_product_codes:
            if code not in state.assignment:
                continue
            if box_type_key(state.assignment[code]) == target_type:
                continue
            move = state.calculate_move(code, target)
            if move is None:
                continue
            individuals.append(
                (
                    move.total_delta_mills,
                    -product_by_code[code].annual_volume,
                    code,
                )
            )
        if not individuals:
            continue
        if len(
            {
                box_type_key(state.assignment[code])
                for _, _, code in individuals
            }
        ) < min_source_types:
            continue

        individuals.sort()
        code_pool = [item[2] for item in individuals[:individual_pool_size]]
        for _, _, code in sorted(individuals, key=lambda item: item[1])[
            :high_volume_pool_size
        ]:
            if code not in code_pool:
                code_pool.append(code)
        # Combination ordering is based on the ranked pool.  It provides a
        # canonical representation and avoids evaluating permutations.
        beams: list[tuple[tuple[str, ...], int]] = [((), -1)]
        destination_moves: list[FacilityMove] = []
        for bundle_size in range(1, max_bundle_size + 1):
            expanded: list[tuple[int, int, tuple[str, ...], int, FacilityMove]] = []
            for codes, last_index in beams:
                for pool_index in range(last_index + 1, len(code_pool)):
                    new_codes = (*codes, code_pool[pool_index])
                    move = _wrap_move(state, new_codes, target)
                    if move is None:
                        continue
                    # A small span bonus prevents all partial beams being
                    # monopolized by one source type before the third type can
                    # enter.  It only ranks partial bundles; exact delta ranks
                    # completed operations and all accepted states.
                    span_bonus = min(len(move.source_types), min_source_types) * 100_000
                    expanded.append(
                        (
                            move.delta_mills - span_bonus,
                            move.delta_mills,
                            new_codes,
                            pool_index,
                            move,
                        )
                    )
            if not expanded:
                break
            expanded.sort(key=lambda item: (item[0], item[1], item[2]))
            beams = [
                (item[2], item[3]) for item in expanded[:subset_beam_width]
            ]
            if bundle_size >= min_codes:
                destination_moves.extend(
                    item[4]
                    for item in expanded
                    if len(item[4].source_types) >= min_source_types
                    and item[4].delta_mills <= max_move_delta_mills
                )

        # Retain more than one bundle per facility because tier-completion
        # chains can require a slightly dearer but compositionally different
        # first leg.
        unique: dict[tuple[str, ...], FacilityMove] = {}
        for move in destination_moves:
            key = tuple(sorted(move.codes))
            old = unique.get(key)
            if old is None or move.delta_mills < old.delta_mills:
                unique[key] = move
        ranked_moves = sorted(
            unique.values(),
            key=lambda move: (
                move.delta_mills,
                move.pallet_delta,
                -len(move.source_types),
                move.codes,
            ),
        )
        all_moves.extend(ranked_moves[:per_destination_moves])

    # Priority target types are the sources evacuated by the preceding chain
    # leg.  Round-robin a small refill quota with the globally cheapest moves.
    priority = sorted(
        (move for move in all_moves if move.target_type in priority_target_types),
        key=lambda move: (move.delta_mills, move.pallet_delta, move.codes),
    )
    global_rank = sorted(
        all_moves,
        key=lambda move: (
            move.delta_mills,
            move.pallet_delta,
            move.target.internal.as_tuple(),
            move.codes,
        ),
    )
    result: list[FacilityMove] = []
    seen: set[tuple[tuple[int, int, int], tuple[str, ...]]] = set()
    for move in (*priority[: max(4, per_destination_moves * 2)], *global_rank):
        key = (move.target.internal.as_tuple(), tuple(sorted(move.codes)))
        if key in seen:
            continue
        result.append(move)
        seen.add(key)
    return tuple(result)


def tabu_beam_cluster_search(
    products: tuple[Product, ...],
    initial_assignment: Mapping[str, CandidateBox],
    candidate_pool: Iterable[CandidateBox],
    freight_policy: FreightPolicy,
    *,
    duration_seconds: float,
    beam_width: int = 6,
    max_depth: int = 10,
    moves_per_node: int = 18,
    first_min_codes: int = 4,
    first_min_source_types: int = 3,
    continuation_min_codes: int = 1,
    continuation_min_source_types: int = 1,
    individual_pool_size: int = 14,
    high_volume_pool_size: int = 4,
    subset_beam_width: int = 10,
    max_bundle_size: int = 10,
    per_destination_moves: int = 3,
    max_excursion_mills: int = 20_000_000,
    max_move_delta_mills: int = 20_000_000,
    max_pallets: int | None = None,
) -> ClusterSearchResult:
    """Run bounded deterministic beam search with assignment-level tabu."""

    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive")
    if beam_width < 1 or max_depth < 1 or moves_per_node < 1:
        raise ValueError("beam_width, max_depth and moves_per_node must be positive")
    if max_excursion_mills < 0:
        raise ValueError("max_excursion_mills cannot be negative")
    products = tuple(sorted(products, key=lambda product: product.code))
    product_codes = tuple(product.code for product in products)
    candidates = tuple(candidate_pool)
    initial_costs = evaluate_assignments(
        products, dict(initial_assignment), freight_policy
    )
    initial_node = BeamNode(
        assignment=dict(initial_assignment),
        costs=initial_costs,
        depth=0,
        last_source_types=frozenset(),
        history=(),
    )
    frontier = (initial_node,)
    best = initial_node
    visited = {_assignment_key(product_codes, initial_assignment)}
    started = time.monotonic()
    deadline = started + duration_seconds
    expanded_nodes = generated_moves = accepted_states = improvements = 0
    deepest_level = 0
    termination = "depth_limit"

    for depth in range(1, max_depth + 1):
        if time.monotonic() >= deadline:
            termination = "time_limit"
            break
        children: list[BeamNode] = []
        for node in frontier:
            if time.monotonic() >= deadline:
                termination = "time_limit"
                break
            expanded_nodes += 1
            state = IncrementalAssignmentState(
                products, dict(node.assignment), freight_policy
            )
            first_leg = node.depth == 0
            moves = generate_facility_moves(
                state,
                products,
                candidates,
                min_codes=first_min_codes if first_leg else continuation_min_codes,
                min_source_types=(
                    first_min_source_types
                    if first_leg
                    else continuation_min_source_types
                ),
                individual_pool_size=individual_pool_size,
                high_volume_pool_size=high_volume_pool_size,
                subset_beam_width=subset_beam_width,
                max_bundle_size=max_bundle_size,
                per_destination_moves=per_destination_moves,
                max_move_delta_mills=max_move_delta_mills,
                priority_target_types=node.last_source_types,
                deadline=deadline,
            )
            generated_moves += len(moves)
            # Keep a refill quota first, then cheapest alternatives.  Complete
            # assignment tabu makes explicit inverse moves harmless.
            priority = [
                move
                for move in moves
                if move.target_type in node.last_source_types
            ]
            ranked = sorted(
                moves,
                key=lambda move: (
                    move.delta_mills,
                    move.pallet_delta,
                    move.target.internal.as_tuple(),
                    move.codes,
                ),
            )
            selected: list[FacilityMove] = []
            operation_seen: set[
                tuple[tuple[int, int, int], tuple[str, ...]]
            ] = set()
            for move in (*priority[: max(2, moves_per_node // 3)], *ranked):
                operation = (
                    move.target.internal.as_tuple(),
                    tuple(sorted(move.codes)),
                )
                if operation in operation_seen or operation in node.history[-3:]:
                    continue
                selected.append(move)
                operation_seen.add(operation)
                if len(selected) >= moves_per_node:
                    break

            for move in selected:
                assignment = dict(node.assignment)
                for code in move.codes:
                    assignment[code] = move.target
                key = _assignment_key(product_codes, assignment)
                if key in visited:
                    continue
                expected_total = node.costs.total_mills + move.delta_mills
                if expected_total > initial_costs.total_mills + max_excursion_mills:
                    continue
                costs = evaluate_assignments(products, assignment, freight_policy)
                if costs.total_mills != expected_total:
                    raise AssertionError(
                        "facility move delta differs from independent evaluation: "
                        f"{expected_total} != {costs.total_mills}"
                    )
                if max_pallets is not None and costs.pallets > max_pallets:
                    continue
                visited.add(key)
                accepted_states += 1
                operation = (
                    move.target.internal.as_tuple(),
                    tuple(sorted(move.codes)),
                )
                child = BeamNode(
                    assignment=assignment,
                    costs=costs,
                    depth=depth,
                    last_source_types=move.source_types,
                    history=(*node.history[-4:], operation),
                )
                children.append(child)
                if costs.total_mills < best.costs.total_mills:
                    # A second full calculation guards both the incremental
                    # delta and the beam bookkeeping before promoting a best.
                    checked = evaluate_assignments(products, assignment, freight_policy)
                    if checked.total_mills != costs.total_mills:
                        raise AssertionError("best cluster state failed independent audit")
                    best = child
                    improvements += 1

        if not children:
            if termination != "time_limit":
                termination = "no_children"
            break
        deepest_level = depth
        # Cost-first selection with one representative per final destination
        # before filling the remaining beam slots preserves chain diversity.
        children.sort(
            key=lambda node: (
                node.costs.total_mills,
                node.costs.pallets,
                node.history[-1],
            )
        )
        diverse: list[BeamNode] = []
        destination_seen: set[tuple[int, int, int]] = set()
        for child in children:
            destination = child.history[-1][0]
            if destination in destination_seen:
                continue
            diverse.append(child)
            destination_seen.add(destination)
            if len(diverse) >= max(1, beam_width // 2):
                break
        selected_ids = {id(node) for node in diverse}
        for child in children:
            if id(child) in selected_ids:
                continue
            diverse.append(child)
            if len(diverse) >= beam_width:
                break
        frontier = tuple(diverse)

    elapsed = time.monotonic() - started
    # Never return a deterioration even though the internal beam contains it.
    if best.costs.total_mills >= initial_costs.total_mills:
        best = initial_node
    return ClusterSearchResult(
        assignment=best.assignment,
        costs=best.costs,
        initial_costs=initial_costs,
        elapsed_seconds=elapsed,
        expanded_nodes=expanded_nodes,
        generated_moves=generated_moves,
        accepted_states=accepted_states,
        improvements=improvements,
        deepest_level=deepest_level,
        visited_states=len(visited),
        termination=termination,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--warm-start", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--duration-seconds", type=float, default=600.0)
    parser.add_argument("--beam-width", type=int, default=6)
    parser.add_argument("--max-depth", type=int, default=10)
    parser.add_argument("--moves-per-node", type=int, default=18)
    parser.add_argument("--first-min-codes", type=int, default=4)
    parser.add_argument("--first-min-source-types", type=int, default=3)
    parser.add_argument("--continuation-min-codes", type=int, default=1)
    parser.add_argument("--continuation-min-source-types", type=int, default=1)
    parser.add_argument("--destination-pool-size", type=int, default=64)
    parser.add_argument("--individual-pool-size", type=int, default=14)
    parser.add_argument("--high-volume-pool-size", type=int, default=4)
    parser.add_argument("--subset-beam-width", type=int, default=10)
    parser.add_argument("--max-bundle-size", type=int, default=10)
    parser.add_argument("--per-destination-moves", type=int, default=3)
    parser.add_argument("--max-excursion-usd", type=float, default=20_000.0)
    parser.add_argument("--max-move-delta-usd", type=float, default=20_000.0)
    parser.add_argument("--max-pallets", type=int)
    parser.add_argument("--extra-region-share", type=float, default=0.0)
    return parser


def run(args: argparse.Namespace) -> dict[str, object]:
    data = load_prepared_data(args.data_dir)
    policy = FreightPolicy(extra_region_share=args.extra_region_share)
    incumbent = validate_solution_csv(args.warm_start, data, policy)
    thickness = _infer_thickness(incumbent)
    exact, exact_stats = generate_exact_candidates(
        data.products,
        thickness,
        retained_designs=_unique_internal_designs(incumbent.assignment),
    )
    pool = facility_candidate_pool(
        data,
        incumbent.assignment,
        exact,
        policy,
        destination_pool_size=args.destination_pool_size,
    )
    print(
        f"Cluster beam start: USD {incumbent.costs.total_mills / 1000:,.2f}; "
        f"{len(pool)} facilities ({len(exact):,} exact candidates)",
        flush=True,
    )
    result = tabu_beam_cluster_search(
        data.products,
        incumbent.assignment,
        pool,
        policy,
        duration_seconds=args.duration_seconds,
        beam_width=args.beam_width,
        max_depth=args.max_depth,
        moves_per_node=args.moves_per_node,
        first_min_codes=args.first_min_codes,
        first_min_source_types=args.first_min_source_types,
        continuation_min_codes=args.continuation_min_codes,
        continuation_min_source_types=args.continuation_min_source_types,
        individual_pool_size=args.individual_pool_size,
        high_volume_pool_size=args.high_volume_pool_size,
        subset_beam_width=args.subset_beam_width,
        max_bundle_size=args.max_bundle_size,
        per_destination_moves=args.per_destination_moves,
        max_excursion_mills=round(args.max_excursion_usd * 1000),
        max_move_delta_mills=round(args.max_move_delta_usd * 1000),
        max_pallets=args.max_pallets,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "asignacion_optima.csv"
    _atomic_write_assignment(output_path, data, result.assignment, policy)
    validated = validate_solution_csv(output_path, data, policy)
    if validated.costs.total_mills != result.costs.total_mills:
        raise AssertionError("written cluster solution failed CSV round-trip audit")
    payload: dict[str, object] = {
        "configuration": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "exact_candidate_stats": asdict(exact_stats),
        "facility_count": len(pool),
        "initial": _cost_payload(result.initial_costs),
        "best": _cost_payload(validated.costs),
        "saving_usd": (
            result.initial_costs.total_mills - validated.costs.total_mills
        )
        / 1000,
        "search": {
            "elapsed_seconds": result.elapsed_seconds,
            "expanded_nodes": result.expanded_nodes,
            "generated_moves": result.generated_moves,
            "accepted_states": result.accepted_states,
            "improvements": result.improvements,
            "deepest_level": result.deepest_level,
            "visited_states": result.visited_states,
            "termination": result.termination,
        },
        "output_path": str(output_path),
        "validation": {
            "valid": True,
            "rows": len(validated.assignment),
        },
    }
    _atomic_write_json(args.output_dir / "resumen_cluster_search.json", payload)
    print(
        f"Cluster beam end: USD {validated.costs.total_mills / 1000:,.2f}; "
        f"saving USD {payload['saving_usd']:,.2f}; "
        f"{result.termination}, {result.visited_states} states",
        flush=True,
    )
    return payload


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = run(args)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
