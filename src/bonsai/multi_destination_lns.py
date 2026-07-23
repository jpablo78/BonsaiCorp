"""Multi-destination exact large-neighborhood search.

The single-destination runner is intentionally narrow: each released SKU can
only stay in its incumbent design or move to one target design.  This module
combines two to eight promising exact destinations in one CP-SAT subproblem.
Each SKU sees only the destinations with which it is compatible, plus its
incumbent (added by :func:`bonsai.optimizer.solve_for_thickness`).

Neighborhoods are assembled deterministically from four complementary views:

* overlap of compatible SKUs;
* overlap of incumbent source box types (coordinated source evacuation);
* destinations with procurement-tier opportunities;
* contiguous windows of the globally ranked destination list.

Every accepted result is independently round-tripped through the submission
CSV validator before it replaces the incumbent.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from decimal import Decimal
import json
from pathlib import Path
from typing import Iterable, Mapping

from .config import FreightPolicy
from .costs import box_type_key
from .data import load_prepared_data
from .destination_lns import (
    DestinationWorkItem,
    _work_item_payload,
    rank_destination_work_items,
)
from .exact_candidates import generate_exact_candidates
from .models import CandidateBox, Dimensions, PreparedData
from .optimizer import solve_for_thickness
from .solution_validation import validate_solution_csv
from .tier_lns import (
    _atomic_write_assignment,
    _atomic_write_json,
    _cost_payload,
    _infer_thickness,
    _mills_from_usd,
    _next_snapshot_number,
    _result_payload,
    _unique_internal_designs,
)


@dataclass(frozen=True)
class MultiDestinationWorkItem:
    """One restricted subproblem containing several exact destinations."""

    neighborhood_id: str
    source_kind: str
    destinations: tuple[DestinationWorkItem, ...]
    product_codes: tuple[str, ...]
    shared_product_links: int
    shared_source_type_links: int

    @property
    def destination_count(self) -> int:
        return len(self.destinations)

    @property
    def sku_count(self) -> int:
        return len(self.product_codes)

    @property
    def gross_opportunity_mills(self) -> int:
        return sum(item.gross_opportunity_mills for item in self.destinations)

    @property
    def net_all_move_opportunity_mills(self) -> int:
        return sum(item.net_all_move_opportunity_mills for item in self.destinations)


def _destination_source_types(
    item: DestinationWorkItem,
    incumbent_assignment: Mapping[str, CandidateBox],
) -> frozenset[tuple[float, float, float, float]]:
    return frozenset(box_type_key(incumbent_assignment[code]) for code in item.product_codes)


def _pair_metrics(
    left: DestinationWorkItem,
    right: DestinationWorkItem,
    source_types: Mapping[str, frozenset[tuple[float, float, float, float]]],
) -> tuple[int, int]:
    shared_products = len(set(left.product_codes) & set(right.product_codes))
    shared_sources = len(source_types[left.destination_id] & source_types[right.destination_id])
    return shared_products, shared_sources


def _make_neighborhood(
    source_kind: str,
    sequence: int,
    destinations: Iterable[DestinationWorkItem],
    source_types: Mapping[str, frozenset[tuple[float, float, float, float]]],
    *,
    min_destination_choices_per_sku: int,
) -> MultiDestinationWorkItem | None:
    members = tuple(destinations)
    shared_products = 0
    shared_sources = 0
    for index, left in enumerate(members):
        for right in members[index + 1 :]:
            product_links, source_links = _pair_metrics(left, right, source_types)
            shared_products += product_links
            shared_sources += source_links
    choice_count_by_code: dict[str, int] = {}
    for member in members:
        for code in member.product_codes:
            choice_count_by_code[code] = choice_count_by_code.get(code, 0) + 1
    released_codes = tuple(
        sorted(
            code
            for code, choice_count in choice_count_by_code.items()
            if choice_count >= min_destination_choices_per_sku
        )
    )
    if not released_codes:
        return None
    return MultiDestinationWorkItem(
        neighborhood_id=f"multi_{source_kind}_{sequence:04d}",
        source_kind=source_kind,
        destinations=members,
        product_codes=released_codes,
        shared_product_links=shared_products,
        shared_source_type_links=shared_sources,
    )


def build_multi_destination_work_items(
    ranked_destinations: Iterable[DestinationWorkItem],
    incumbent_assignment: Mapping[str, CandidateBox],
    *,
    min_destinations: int = 2,
    max_destinations: int = 8,
    min_destination_choices_per_sku: int = 2,
    min_skus: int = 2,
    max_skus: int | None = 220,
    max_neighborhoods: int | None = 128,
) -> tuple[MultiDestinationWorkItem, ...]:
    """Combine ranked single destinations into focused multi-choice models.

    The input order is the global destination ranking.  The builder is fully
    deterministic and removes duplicate destination sets, irrespective of the
    strategy that discovered them.
    """

    if min_destinations < 2:
        raise ValueError("min_destinations must be at least 2")
    if max_destinations < min_destinations:
        raise ValueError("max_destinations cannot be smaller than min_destinations")
    if max_destinations > 8:
        raise ValueError("max_destinations cannot exceed 8")
    if min_destination_choices_per_sku < 2:
        raise ValueError("min_destination_choices_per_sku must be at least 2")
    if min_destination_choices_per_sku > max_destinations:
        raise ValueError(
            "min_destination_choices_per_sku cannot exceed max_destinations"
        )
    if min_skus < 1:
        raise ValueError("min_skus must be positive")
    if max_skus is not None and max_skus < 1:
        raise ValueError("max_skus must be positive")
    if max_skus is not None and max_skus < min_skus:
        raise ValueError("max_skus cannot be smaller than min_skus")
    if max_neighborhoods is not None and max_neighborhoods < 1:
        raise ValueError("max_neighborhoods must be positive")

    ranked = tuple(ranked_destinations)
    if len(ranked) < min_destinations:
        return ()
    source_types = {
        item.destination_id: _destination_source_types(item, incumbent_assignment)
        for item in ranked
    }
    rank_index = {item.destination_id: index for index, item in enumerate(ranked)}
    raw: list[tuple[str, tuple[DestinationWorkItem, ...]]] = []

    # Affinity stars: prioritize exact SKU overlap, then shared incumbent
    # source types.  Both expose interactions hidden from binary subproblems.
    requested_sizes = tuple(sorted({min_destinations, max_destinations}))
    for seed in ranked:
        partners = [item for item in ranked if item is not seed]
        partners.sort(
            key=lambda item: (
                -_pair_metrics(seed, item, source_types)[0],
                -_pair_metrics(seed, item, source_types)[1],
                -item.tier_crossings,
                rank_index[item.destination_id],
            )
        )
        ordered = (seed, *partners)
        for size in requested_sizes:
            raw.append(("overlap", tuple(ordered[:size])))

    # Connected components of the destination graph.  An edge represents a
    # shared movable SKU; components are split into bounded ranked chunks.
    adjacency: dict[str, set[str]] = {item.destination_id: set() for item in ranked}
    by_id = {item.destination_id: item for item in ranked}
    for index, left in enumerate(ranked):
        for right in ranked[index + 1 :]:
            if set(left.product_codes) & set(right.product_codes):
                adjacency[left.destination_id].add(right.destination_id)
                adjacency[right.destination_id].add(left.destination_id)
    unseen = set(adjacency)
    while unseen:
        root = min(unseen, key=rank_index.__getitem__)
        stack = [root]
        component_ids: set[str] = set()
        while stack:
            current = stack.pop()
            if current in component_ids:
                continue
            component_ids.add(current)
            stack.extend(adjacency[current] - component_ids)
        unseen -= component_ids
        component = tuple(
            sorted((by_id[item_id] for item_id in component_ids), key=lambda item: rank_index[item.destination_id])
        )
        if len(component) < min_destinations:
            continue
        for start in range(0, len(component), max_destinations):
            chunk = component[start : start + max_destinations]
            if len(chunk) >= min_destinations:
                raw.append(("component", chunk))

    # Tier/source bundles target simultaneous consolidation at destinations
    # while allowing the solver to account for deterioration of shared sources.
    tier_destinations = tuple(item for item in ranked if item.tier_crossings > 0)
    for seed in tier_destinations:
        partners = [item for item in tier_destinations if item is not seed]
        partners.sort(
            key=lambda item: (
                -_pair_metrics(seed, item, source_types)[1],
                -_pair_metrics(seed, item, source_types)[0],
                -item.tier_crossings,
                rank_index[item.destination_id],
            )
        )
        members = (seed, *partners[: max_destinations - 1])
        if len(members) >= min_destinations:
            raw.append(("tier_source", members))

    # Ranked windows preserve strong globally scored destinations even when
    # they have no direct graph link.  Sliding by half a window adds diversity.
    for size in requested_sizes:
        stride = max(1, size // 2)
        for start in range(0, len(ranked) - size + 1, stride):
            raw.append(("ranked_window", ranked[start : start + size]))

    unique_by_kind: dict[str, list[MultiDestinationWorkItem]] = {
        "overlap": [],
        "component": [],
        "tier_source": [],
        "ranked_window": [],
    }
    seen: set[frozenset[str]] = set()
    sequence_by_kind: dict[str, int] = {}
    for source_kind, members in raw:
        member_key = frozenset(item.destination_id for item in members)
        if len(member_key) < min_destinations or member_key in seen:
            continue
        sequence = sequence_by_kind.get(source_kind, 0)
        neighborhood = _make_neighborhood(
            source_kind,
            sequence,
            members,
            source_types,
            min_destination_choices_per_sku=min_destination_choices_per_sku,
        )
        sequence_by_kind[source_kind] = sequence + 1
        if neighborhood is None:
            continue
        if neighborhood.sku_count < min_skus:
            continue
        if max_skus is not None and neighborhood.sku_count > max_skus:
            continue
        seen.add(member_key)
        unique_by_kind[source_kind].append(neighborhood)

    # Round-robin the discovery strategies so a global cap cannot silently
    # exclude components or tier bundles merely because overlap stars were
    # generated first.
    unique: list[MultiDestinationWorkItem] = []
    for index in range(max((len(items) for items in unique_by_kind.values()), default=0)):
        for source_kind in ("overlap", "component", "tier_source", "ranked_window"):
            bucket = unique_by_kind[source_kind]
            if index < len(bucket):
                unique.append(bucket[index])
                if max_neighborhoods is not None and len(unique) >= max_neighborhoods:
                    return tuple(unique)
    return tuple(unique)


def allowed_internals_for_neighborhood(
    item: MultiDestinationWorkItem,
) -> dict[str, tuple[Dimensions, ...]]:
    """Return every compatible destination for each released SKU.

    The incumbent is intentionally absent here because the optimizer adds it
    to every supplied SKU restriction and therefore guarantees a safe fallback.
    """

    choices: dict[str, set[Dimensions]] = {code: set() for code in item.product_codes}
    for destination in item.destinations:
        for code in destination.product_codes:
            if code in choices:
                choices[code].add(destination.candidate.internal)
    return {
        code: tuple(sorted(internals, key=Dimensions.as_tuple))
        for code, internals in sorted(choices.items())
    }


def _multi_payload(item: MultiDestinationWorkItem) -> dict[str, object]:
    allowed = allowed_internals_for_neighborhood(item)
    counts = tuple(len(values) for values in allowed.values())
    return {
        "neighborhood_id": item.neighborhood_id,
        "source_kind": item.source_kind,
        "destination_count": item.destination_count,
        "destination_ids": tuple(dest.destination_id for dest in item.destinations),
        "destinations": tuple(_work_item_payload(dest) for dest in item.destinations),
        "sku_count": item.sku_count,
        "product_codes": item.product_codes,
        "min_destination_choices_per_sku": min(counts),
        "max_destination_choices_per_sku": max(counts),
        "shared_product_links": item.shared_product_links,
        "shared_source_type_links": item.shared_source_type_links,
        "gross_opportunity_usd": item.gross_opportunity_mills / 1000,
        "net_all_move_opportunity_usd": item.net_all_move_opportunity_mills / 1000,
    }


def run_multi_destination_lns(args: argparse.Namespace) -> dict[str, object]:
    if args.rounds < 1:
        raise ValueError("--rounds must be at least 1")
    if args.time_per_neighborhood <= 0:
        raise ValueError("--time-per-neighborhood must be positive")
    if args.num_search_workers < 1:
        raise ValueError("--num-search-workers must be positive")
    if args.target_mode == "hard" and args.target_total_usd is None:
        raise ValueError("--target-mode hard requires --target-total-usd")

    data: PreparedData = load_prepared_data(args.data_dir)
    policy = FreightPolicy(extra_region_share=args.extra_region_share)
    incumbent = validate_solution_csv(args.warm_start, data, policy)
    thickness = _infer_thickness(incumbent)
    target_mills = _mills_from_usd(args.target_total_usd)
    min_opportunity_mills = _mills_from_usd(args.min_gross_opportunity_usd) or 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_path = args.output_dir / "asignacion_optima.csv"
    if best_path.exists():
        existing = validate_solution_csv(best_path, data, policy)
        if _infer_thickness(existing) != thickness:
            raise ValueError("existing output uses a different global thickness")
        if existing.costs.total_mills < incumbent.costs.total_mills:
            incumbent = existing

    start_costs = incumbent.costs
    snapshot_number = _next_snapshot_number(args.output_dir)
    _atomic_write_assignment(
        args.output_dir / f"incumbent_{snapshot_number:04d}.csv",
        data,
        incumbent.assignment,
        policy,
    )
    snapshot_number += 1
    _atomic_write_assignment(best_path, data, incumbent.assignment, policy)

    exact_candidates, candidate_stats = generate_exact_candidates(
        data.products,
        thickness,
        retained_designs=_unique_internal_designs(incumbent.assignment),
    )
    summary_path = args.output_dir / "resumen_multi_destination_lns.json"
    attempts: list[dict[str, object]] = []
    improvements: list[dict[str, object]] = []
    summary: dict[str, object] = {
        "configuration": {
            "data_dir": str(args.data_dir),
            "warm_start": str(args.warm_start),
            "thickness_mm": thickness,
            "time_per_neighborhood_seconds": args.time_per_neighborhood,
            "num_search_workers": args.num_search_workers,
            "rounds": args.rounds,
            "destination_pool_size": args.destination_pool_size,
            "min_destinations_per_neighborhood": args.min_destinations_per_neighborhood,
            "max_destinations_per_neighborhood": args.max_destinations_per_neighborhood,
            "min_destination_choices_per_sku": args.min_destination_choices_per_sku,
            "max_neighborhoods": args.max_neighborhoods,
            "min_skus_per_destination": args.min_skus_per_destination,
            "min_skus_per_neighborhood": args.min_skus_per_neighborhood,
            "max_skus_per_neighborhood": args.max_skus_per_neighborhood,
            "min_gross_opportunity_usd": min_opportunity_mills / 1000,
            "target_total_usd": target_mills / 1000 if target_mills is not None else None,
            "target_mode": args.target_mode,
            "max_extra_pallets": args.max_extra_pallets,
            "random_seed": args.random_seed,
        },
        "exact_candidate_stats": asdict(candidate_stats),
        "initial": _cost_payload(start_costs),
        "attempts": attempts,
        "improvements": improvements,
    }
    termination = "round_limit"
    attempted_at_cost: set[tuple[int, frozenset[tuple[int, int, int]]]] = set()
    print(
        f"Multi-destination LNS start: USD {incumbent.costs.total_mills / 1000:,.2f}; "
        f"{len(exact_candidates):,} exact candidates",
        flush=True,
    )

    if target_mills is not None and incumbent.costs.total_mills <= target_mills:
        termination = "target_already_met"
    else:
        for round_index in range(args.rounds):
            ranked = rank_destination_work_items(
                data,
                incumbent.assignment,
                exact_candidates,
                policy,
                min_skus=args.min_skus_per_destination,
                max_skus=args.max_skus_per_neighborhood,
                min_gross_opportunity_mills=min_opportunity_mills,
                max_destinations=args.destination_pool_size,
            )
            neighborhoods = build_multi_destination_work_items(
                ranked,
                incumbent.assignment,
                min_destinations=args.min_destinations_per_neighborhood,
                max_destinations=args.max_destinations_per_neighborhood,
                min_destination_choices_per_sku=args.min_destination_choices_per_sku,
                min_skus=args.min_skus_per_neighborhood,
                max_skus=args.max_skus_per_neighborhood,
                max_neighborhoods=args.max_neighborhoods,
            )
            work_items = tuple(
                item
                for item in neighborhoods
                if (
                    incumbent.costs.total_mills,
                    frozenset(dest.candidate.internal.as_tuple() for dest in item.destinations),
                )
                not in attempted_at_cost
            )
            if not work_items:
                termination = "no_neighborhoods"
                break
            print(
                f"Round {round_index + 1}/{args.rounds}: "
                f"{len(ranked)} destinations -> {len(work_items)} neighborhoods",
                flush=True,
            )
            improved_this_round = False
            for item_index, item in enumerate(work_items):
                before_mills = incumbent.costs.total_mills
                destination_key = frozenset(
                    dest.candidate.internal.as_tuple() for dest in item.destinations
                )
                attempted_at_cost.add((before_mills, destination_key))
                seed = args.random_seed + round_index * 104_729 + item_index * 1_009
                result = solve_for_thickness(
                    data,
                    thickness,
                    policy,
                    time_limit_seconds=args.time_per_neighborhood,
                    num_search_workers=args.num_search_workers,
                    random_seed=seed,
                    initial_assignment=incumbent.assignment,
                    candidate_strategy="exact",
                    max_extra_pallets=args.max_extra_pallets,
                    target_total_mills=(target_mills if args.target_mode == "hard" else None),
                    free_product_codes=item.product_codes,
                    allowed_internals_by_product=allowed_internals_for_neighborhood(item),
                    precomputed_exact_candidates=exact_candidates,
                    precomputed_exact_candidate_stats=candidate_stats,
                )
                attempt = {
                    "round": round_index + 1,
                    "sequence": len(attempts) + 1,
                    **_multi_payload(item),
                    "before_usd": before_mills / 1000,
                    "seed": seed,
                    **_result_payload(result),
                    "accepted": False,
                }
                if result.costs.total_mills < before_mills:
                    candidate_path = args.output_dir / ".candidate_validation.csv"
                    try:
                        checked = _atomic_write_assignment(candidate_path, data, result.assignment, policy)
                        if checked.costs.total_mills != result.costs.total_mills:
                            raise RuntimeError("independent validation changed solver cost")
                    except (KeyError, RuntimeError, TypeError, ValueError) as exc:
                        attempt["validation_error"] = str(exc)
                    else:
                        incumbent = checked
                        snapshot_path = args.output_dir / f"incumbent_{snapshot_number:04d}.csv"
                        _atomic_write_assignment(snapshot_path, data, incumbent.assignment, policy)
                        _atomic_write_assignment(best_path, data, incumbent.assignment, policy)
                        snapshot_number += 1
                        improvement = {
                            "round": round_index + 1,
                            "attempt": len(attempts) + 1,
                            "neighborhood_id": item.neighborhood_id,
                            "source_kind": item.source_kind,
                            "destination_ids": tuple(dest.destination_id for dest in item.destinations),
                            "before_usd": before_mills / 1000,
                            "after_usd": incumbent.costs.total_mills / 1000,
                            "saving_usd": (before_mills - incumbent.costs.total_mills) / 1000,
                            "snapshot_path": str(snapshot_path),
                        }
                        improvements.append(improvement)
                        attempt["accepted"] = True
                        attempt["snapshot_path"] = str(snapshot_path)
                        improved_this_round = True
                        print(
                            f"  {item.neighborhood_id}: accepted "
                            f"USD {incumbent.costs.total_mills / 1000:,.2f}",
                            flush=True,
                        )
                    finally:
                        if candidate_path.exists():
                            candidate_path.unlink()
                attempts.append(attempt)
                summary["best"] = _cost_payload(incumbent.costs)
                summary["saving_usd"] = (start_costs.total_mills - incumbent.costs.total_mills) / 1000
                summary["target_met"] = (
                    incumbent.costs.total_mills <= target_mills if target_mills is not None else None
                )
                _atomic_write_json(summary_path, summary)
                if not attempt["accepted"]:
                    print(
                        f"  {item.neighborhood_id}: {result.status}, no improvement "
                        f"({item.destination_count} destinations, {item.sku_count} SKUs, "
                        f"{result.wall_time_seconds:.1f}s)",
                        flush=True,
                    )
                if target_mills is not None and incumbent.costs.total_mills <= target_mills:
                    termination = "target_met"
                    break
                if improved_this_round:
                    break
            if termination == "target_met":
                break

    summary["best"] = _cost_payload(incumbent.costs)
    summary["saving_usd"] = (start_costs.total_mills - incumbent.costs.total_mills) / 1000
    summary["target_met"] = (
        incumbent.costs.total_mills <= target_mills if target_mills is not None else None
    )
    summary["termination"] = termination
    summary["best_path"] = str(best_path)
    _atomic_write_json(summary_path, summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Exact incumbent-or-multiple-destination LNS for Bonsai Corp"
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--warm-start", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--time-per-neighborhood", type=float, default=30.0)
    parser.add_argument("--num-search-workers", type=int, default=6)
    parser.add_argument("--max-extra-pallets", type=int, default=1000)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--destination-pool-size", type=int, default=96)
    parser.add_argument("--min-destinations-per-neighborhood", type=int, default=2)
    parser.add_argument("--max-destinations-per-neighborhood", type=int, default=8)
    parser.add_argument("--min-destination-choices-per-sku", type=int, default=2)
    parser.add_argument("--max-neighborhoods", type=int, default=128)
    parser.add_argument("--min-skus-per-destination", type=int, default=1)
    parser.add_argument("--min-skus-per-neighborhood", type=int, default=2)
    parser.add_argument("--max-skus-per-neighborhood", type=int, default=220)
    parser.add_argument("--min-gross-opportunity-usd", type=Decimal, default=Decimal("1"))
    parser.add_argument("--target-total-usd", type=Decimal)
    parser.add_argument("--target-mode", choices=("stop", "hard"), default="stop")
    parser.add_argument("--extra-region-share", type=float, default=0.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run_multi_destination_lns(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
