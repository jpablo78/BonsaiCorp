"""LNS diverso de destinos triples o cuádruples con el backend SCIP reducido.

La búsqueda multidestino anterior exigía que cada SKU liberado tuviera al menos
dos destinos. Aquí se libera la unión de todos los SKU compatibles, de modo que
un tipo origen puede evacuarse repartiendo productos entre tres o cuatro
diseños receptores. Los SKU no liberados se absorben como constantes mediante
:func:`bonsai.scip_optimizer.solve_with_scip`.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from decimal import Decimal
import heapq
from itertools import combinations
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .config import FreightPolicy
from .data import load_prepared_data
from .destination_lns import DestinationWorkItem, rank_destination_work_items
from .exact_candidates import generate_exact_candidates
from .models import CandidateBox, Dimensions, PreparedData
from .pair_destination_lns import _incumbent_source_groups
from .scip_multi_destination_lns import _scip_result_payload
from .scip_optimizer import solve_with_scip
from .solution_validation import validate_solution_csv
from .tier_lns import (
    _atomic_write_assignment,
    _atomic_write_json,
    _cost_payload,
    _infer_thickness,
    _mills_from_usd,
    _next_snapshot_number,
    _unique_internal_designs,
)


DesignKey = tuple[int, int, int]
ComboKey = frozenset[DesignKey]


@dataclass(frozen=True)
class ComboDestinationWorkItem:
    combo_id: str
    destinations: tuple[DestinationWorkItem, ...]
    product_codes: tuple[str, ...]
    overlap_links: int
    shared_source_types: int
    complementary_source_types: int
    complementary_source_volume: int
    all_member_essential_source_types: int
    all_member_essential_source_volume: int

    @property
    def destination_count(self) -> int:
        return len(self.destinations)

    @property
    def combo_key(self) -> ComboKey:
        return frozenset(
            destination.candidate.internal.as_tuple()
            for destination in self.destinations
        )

    @property
    def sku_count(self) -> int:
        return len(self.product_codes)

    @property
    def tier_crossings(self) -> int:
        return sum(destination.tier_crossings for destination in self.destinations)

    @property
    def gross_opportunity_mills(self) -> int:
        return sum(
            destination.gross_opportunity_mills for destination in self.destinations
        )


def load_covered_combinations(path: Path, size: int) -> frozenset[ComboKey]:
    """Expande cada intento previo multidestino en las combinaciones que contiene."""

    if size < 2:
        raise ValueError("combination size must be at least two")
    if not path.exists():
        return frozenset()
    payload = json.loads(path.read_text(encoding="utf-8"))
    covered: set[ComboKey] = set()
    for attempt in payload.get("attempts", ()):
        geometries = {
            tuple(int(value) for value in destination["internal_mm"])
            for destination in attempt.get("destinations", ())
            if "internal_mm" in destination
        }
        for members in combinations(sorted(geometries), size):
            covered.add(frozenset(members))
    return frozenset(covered)


def combo_complementarity_metrics(
    destination_masks: Sequence[int],
    source_groups: Iterable[tuple[int, int]],
) -> tuple[int, int, int, int, int, int]:
    """Devuelve métricas de solapamiento y complementariedad exacta de cobertura.

    ``all_member_essential`` indica que quitar *cualquier* destino deja sin
    cobertura al menos un SKU del tipo origen. Por eso el tipo sólo puede
    evacuarse por completo usando todos los miembros de la combinación.
    """

    union_mask = 0
    overlap_links = 0
    for index, left_mask in enumerate(destination_masks):
        union_mask |= left_mask
        for right_mask in destination_masks[index + 1 :]:
            overlap_links += (left_mask & right_mask).bit_count()

    shared_source_types = 0
    complementary_types = 0
    complementary_volume = 0
    essential_types = 0
    essential_volume = 0
    for group_mask, group_volume in source_groups:
        touch_count = sum(bool(group_mask & mask) for mask in destination_masks)
        if touch_count >= 2:
            shared_source_types += 1
        if group_mask & ~union_mask:
            continue
        if all(group_mask & ~mask for mask in destination_masks):
            complementary_types += 1
            complementary_volume += group_volume
        if all(
            group_mask & ~_union_without_index(destination_masks, removed_index)
            for removed_index in range(len(destination_masks))
        ):
            essential_types += 1
            essential_volume += group_volume
    return (
        overlap_links,
        shared_source_types,
        complementary_types,
        complementary_volume,
        essential_types,
        essential_volume,
    )


def _union_without_index(masks: Sequence[int], removed_index: int) -> int:
    result = 0
    for index, mask in enumerate(masks):
        if index != removed_index:
            result |= mask
    return result


def _sample_destination_indices(count: int, pool_size: int) -> tuple[int, ...]:
    """Combina alto ranking con cobertura determinista de la cola larga."""

    if pool_size >= count:
        return tuple(range(count))
    head_size = min(pool_size * 2 // 3, count)
    selected = list(range(head_size))
    remaining = pool_size - head_size
    if remaining:
        tail_size = count - head_size
        selected.extend(
            head_size + min(tail_size - 1, (offset * tail_size) // remaining)
            for offset in range(remaining)
        )
    return tuple(dict.fromkeys(selected))


def _mask_context(
    destinations: Sequence[DestinationWorkItem],
    data: PreparedData,
    incumbent_assignment: Mapping[str, CandidateBox],
) -> tuple[tuple[int, ...], tuple[tuple[int, int], ...]]:
    code_index = {
        code: index
        for index, code in enumerate(sorted(product.code for product in data.products))
    }

    def make_mask(codes: Iterable[str]) -> int:
        mask = 0
        for code in codes:
            mask |= 1 << code_index[code]
        return mask

    masks = tuple(make_mask(item.product_codes) for item in destinations)
    groups = tuple(
        (make_mask(codes), volume)
        for codes, volume in _incumbent_source_groups(
            data.products, incumbent_assignment
        )
    )
    return masks, groups


def _quality(
    indices: tuple[int, ...],
    destinations: Sequence[DestinationWorkItem],
    metrics: tuple[int, int, int, int, int, int],
    sku_count: int,
) -> tuple[int, ...]:
    overlap, shared, complementary, complementary_volume, essential, essential_volume = metrics
    return (
        int(essential > 0),
        essential_volume,
        essential,
        int(complementary > 0),
        complementary_volume,
        complementary,
        shared,
        sum(destinations[index].tier_crossings for index in indices),
        overlap,
        sum(destinations[index].gross_opportunity_mills for index in indices),
        -sku_count,
        *(-index for index in indices),
    )


def _select_diverse(
    ranked_entries: Sequence[
        tuple[tuple[int, ...], tuple[int, ...], tuple[int, int, int, int, int, int]]
    ],
    *,
    count: int,
    max_occurrences_per_destination: int,
) -> tuple[
    tuple[tuple[int, ...], tuple[int, int, int, int, int, int]], ...
]:
    selected: list[
        tuple[tuple[int, ...], tuple[int, int, int, int, int, int]]
    ] = []
    occurrence: dict[int, int] = {}
    selected_keys: set[tuple[int, ...]] = set()
    for _, indices, metrics in ranked_entries:
        if any(
            occurrence.get(index, 0) >= max_occurrences_per_destination
            for index in indices
        ):
            continue
        selected.append((indices, metrics))
        selected_keys.add(indices)
        for index in indices:
            occurrence[index] = occurrence.get(index, 0) + 1
        if len(selected) >= count:
            return tuple(selected)
    # El límite promueve diversidad, pero no debe impedir la cantidad de
    # trabajo solicitada. Se completa de forma determinista desde el ranking global restante.
    for _, indices, metrics in ranked_entries:
        if indices in selected_keys:
            continue
        selected.append((indices, metrics))
        if len(selected) >= count:
            break
    return tuple(selected)


def build_ranked_triples_and_quads(
    destinations: Sequence[DestinationWorkItem],
    data: PreparedData,
    incumbent_assignment: Mapping[str, CandidateBox],
    *,
    covered_triples: Iterable[ComboKey] = (),
    covered_quads: Iterable[ComboKey] = (),
    triple_count: int = 300,
    quad_count: int = 100,
    destination_sample_size: int = 180,
    max_skus: int | None = 220,
) -> tuple[tuple[ComboDestinationWorkItem, ...], tuple[ComboDestinationWorkItem, ...]]:
    if triple_count < 1 or quad_count < 1:
        raise ValueError("triple_count and quad_count must be positive")
    if destination_sample_size < 4:
        raise ValueError("destination_sample_size must be at least four")
    covered3 = set(covered_triples)
    covered4 = set(covered_quads)
    masks, source_groups = _mask_context(destinations, data, incumbent_assignment)
    sample = _sample_destination_indices(len(destinations), destination_sample_size)

    triple_heap_size = max(3_000, triple_count * 10)
    triple_heap: list[
        tuple[
            tuple[int, ...],
            tuple[int, ...],
            tuple[int, int, int, int, int, int],
        ]
    ] = []
    for indices in combinations(sample, 3):
        key = frozenset(destinations[index].candidate.internal.as_tuple() for index in indices)
        if key in covered3:
            continue
        combo_masks = tuple(masks[index] for index in indices)
        union_mask = combo_masks[0] | combo_masks[1] | combo_masks[2]
        sku_count = union_mask.bit_count()
        if max_skus is not None and sku_count > max_skus:
            continue
        metrics = combo_complementarity_metrics(combo_masks, source_groups)
        quality = _quality(indices, destinations, metrics, sku_count)
        entry = (quality, indices, metrics)
        if len(triple_heap) < triple_heap_size:
            heapq.heappush(triple_heap, entry)
        elif quality > triple_heap[0][0]:
            heapq.heapreplace(triple_heap, entry)
    ranked_triples = sorted(triple_heap, reverse=True)
    chosen_triples = _select_diverse(
        ranked_triples,
        count=triple_count,
        max_occurrences_per_destination=max(8, triple_count * 3 // 50),
    )

    quad_heap_size = max(2_000, quad_count * 12)
    quad_heap: list[
        tuple[
            tuple[int, ...],
            tuple[int, ...],
            tuple[int, int, int, int, int, int],
        ]
    ] = []
    seen_quads: set[tuple[int, ...]] = set()
    # Se amplía un conjunto amplio de triples de alta calidad, no sólo los 300
    # seleccionados, para que los cuádruples incorporen destinos de cola y
    # coberturas origen distintas.
    for _, triple_indices, _ in ranked_triples[: max(2_000, triple_count * 5)]:
        triple_set = set(triple_indices)
        for fourth in sample:
            if fourth in triple_set:
                continue
            indices = tuple(sorted((*triple_indices, fourth)))
            if indices in seen_quads:
                continue
            seen_quads.add(indices)
            key = frozenset(
                destinations[index].candidate.internal.as_tuple() for index in indices
            )
            if key in covered4:
                continue
            combo_masks = tuple(masks[index] for index in indices)
            union_mask = 0
            for mask in combo_masks:
                union_mask |= mask
            sku_count = union_mask.bit_count()
            if max_skus is not None and sku_count > max_skus:
                continue
            metrics = combo_complementarity_metrics(combo_masks, source_groups)
            quality = _quality(indices, destinations, metrics, sku_count)
            entry = (quality, indices, metrics)
            if len(quad_heap) < quad_heap_size:
                heapq.heappush(quad_heap, entry)
            elif quality > quad_heap[0][0]:
                heapq.heapreplace(quad_heap, entry)
    ranked_quads = sorted(quad_heap, reverse=True)
    chosen_quads = _select_diverse(
        ranked_quads,
        count=quad_count,
        max_occurrences_per_destination=max(6, quad_count * 4 // 35),
    )

    def materialize(
        chosen: Sequence[
            tuple[tuple[int, ...], tuple[int, int, int, int, int, int]]
        ],
        prefix: str,
    ) -> tuple[ComboDestinationWorkItem, ...]:
        result: list[ComboDestinationWorkItem] = []
        for sequence, (indices, metrics) in enumerate(chosen):
            members = tuple(destinations[index] for index in indices)
            codes = tuple(
                sorted(set().union(*(set(member.product_codes) for member in members)))
            )
            result.append(
                ComboDestinationWorkItem(
                    combo_id=f"{prefix}_{sequence:04d}",
                    destinations=members,
                    product_codes=codes,
                    overlap_links=metrics[0],
                    shared_source_types=metrics[1],
                    complementary_source_types=metrics[2],
                    complementary_source_volume=metrics[3],
                    all_member_essential_source_types=metrics[4],
                    all_member_essential_source_volume=metrics[5],
                )
            )
        return tuple(result)

    triples = materialize(chosen_triples, "triple")
    quads = materialize(chosen_quads, "quad")
    if len(triples) < triple_count or len(quads) < quad_count:
        raise RuntimeError(
            f"insufficient combinations: {len(triples)} triples, {len(quads)} quads"
        )
    return triples, quads


def allowed_internals_for_combo(
    item: ComboDestinationWorkItem,
) -> dict[str, tuple[Dimensions, ...]]:
    choices: dict[str, set[Dimensions]] = {code: set() for code in item.product_codes}
    for destination in item.destinations:
        for code in destination.product_codes:
            choices[code].add(destination.candidate.internal)
    return {
        code: tuple(sorted(internals, key=Dimensions.as_tuple))
        for code, internals in choices.items()
    }


def _combo_payload(item: ComboDestinationWorkItem) -> dict[str, object]:
    choice_counts = tuple(map(len, allowed_internals_for_combo(item).values()))
    return {
        "combo_id": item.combo_id,
        "destination_count": item.destination_count,
        "destination_internals_mm": tuple(sorted(item.combo_key)),
        "sku_count": item.sku_count,
        "one_choice_skus": sum(value == 1 for value in choice_counts),
        "multi_choice_skus": sum(value >= 2 for value in choice_counts),
        "overlap_links": item.overlap_links,
        "shared_source_types": item.shared_source_types,
        "complementary_source_types": item.complementary_source_types,
        "complementary_source_volume": item.complementary_source_volume,
        "all_member_essential_source_types": item.all_member_essential_source_types,
        "all_member_essential_source_volume": item.all_member_essential_source_volume,
        "tier_crossings": item.tier_crossings,
        "gross_opportunity_usd": item.gross_opportunity_mills / 1000,
    }


def run_combo_destination_lns(args: argparse.Namespace) -> dict[str, object]:
    data = load_prepared_data(args.data_dir)
    policy = FreightPolicy(extra_region_share=args.extra_region_share)
    incumbent = validate_solution_csv(args.warm_start, data, policy)
    thickness = _infer_thickness(incumbent)
    if thickness != 3.0:
        raise ValueError("SCIP combo LNS supports only 3 mm")
    target_mills = _mills_from_usd(args.target_total_usd)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_path = args.output_dir / "asignacion_optima.csv"
    if best_path.exists():
        existing = validate_solution_csv(best_path, data, policy)
        if existing.costs.total_mills < incumbent.costs.total_mills:
            incumbent = existing
    start_costs = incumbent.costs
    snapshot_number = _next_snapshot_number(args.output_dir)
    _atomic_write_assignment(best_path, data, incumbent.assignment, policy)

    candidates, candidate_stats = generate_exact_candidates(
        data.products,
        thickness,
        retained_designs=_unique_internal_designs(incumbent.assignment),
    )
    destinations = rank_destination_work_items(
        data,
        incumbent.assignment,
        candidates,
        policy,
        min_skus=1,
        min_gross_opportunity_mills=0,
    )
    covered3 = load_covered_combinations(args.exclude_pool_summary, 3)
    covered4 = load_covered_combinations(args.exclude_pool_summary, 4)
    triples, quads = build_ranked_triples_and_quads(
        destinations,
        data,
        incumbent.assignment,
        covered_triples=covered3,
        covered_quads=covered4,
        triple_count=args.triple_count,
        quad_count=args.quad_count,
        destination_sample_size=args.destination_sample_size,
        max_skus=args.max_skus,
    )
    work = (*triples, *quads)

    attempts: list[dict[str, object]] = []
    improvements: list[dict[str, object]] = []
    summary: dict[str, object] = {
        "solver": "SCIP reduced triple/quad destination LNS",
        "configuration": {
            "data_dir": str(args.data_dir),
            "warm_start": str(args.warm_start),
            "exclude_pool_summary": str(args.exclude_pool_summary),
            "covered_triples": len(covered3),
            "covered_quads": len(covered4),
            "triple_count": args.triple_count,
            "quad_count": args.quad_count,
            "destination_sample_size": args.destination_sample_size,
            "time_per_combo_seconds": args.time_per_combo,
            "num_threads": 1,
            "max_skus": args.max_skus,
            "max_extra_pallets": args.max_extra_pallets,
            "target_total_usd": target_mills / 1000 if target_mills else None,
        },
        "candidate_stats": asdict(candidate_stats),
        "ranked_destination_count": len(destinations),
        "initial": _cost_payload(start_costs),
        "ranked_triples": tuple(_combo_payload(item) for item in triples),
        "ranked_quads": tuple(_combo_payload(item) for item in quads),
        "attempts": attempts,
        "improvements": improvements,
    }
    summary_path = args.output_dir / "resumen_combo_destination_lns.json"
    print(
        f"SCIP combo LNS start: USD {incumbent.costs.total_mills / 1000:,.2f}; "
        f"{len(triples)} triples + {len(quads)} quads",
        flush=True,
    )
    for sequence, item in enumerate(work, start=1):
        before_mills = incumbent.costs.total_mills
        result = solve_with_scip(
            data,
            thickness,
            policy,
            time_limit_seconds=args.time_per_combo,
            num_threads=1,
            random_seed=args.random_seed + sequence * 1_009,
            initial_assignment=incumbent.assignment,
            max_extra_pallets=args.max_extra_pallets,
            free_product_codes=item.product_codes,
            allowed_internals_by_product=allowed_internals_for_combo(item),
            precomputed_exact_candidates=candidates,
            precomputed_exact_candidate_stats=candidate_stats,
            memory_limit_mb=args.memory_limit_mb,
            scip_parameters="\n".join(args.scip_parameter) or None,
        )
        attempt = {
            "sequence": sequence,
            **_combo_payload(item),
            "before_usd": before_mills / 1000,
            **_scip_result_payload(result),
            "accepted": False,
        }
        if result.costs.total_mills < before_mills:
            candidate_path = args.output_dir / ".candidate_validation.csv"
            checked = _atomic_write_assignment(
                candidate_path, data, result.assignment, policy
            )
            candidate_path.unlink(missing_ok=True)
            if checked.costs.total_mills != result.costs.total_mills:
                raise RuntimeError("independent validation changed SCIP cost")
            incumbent = checked
            snapshot_path = args.output_dir / f"incumbent_{snapshot_number:04d}.csv"
            snapshot_number += 1
            _atomic_write_assignment(snapshot_path, data, incumbent.assignment, policy)
            _atomic_write_assignment(best_path, data, incumbent.assignment, policy)
            attempt["accepted"] = True
            attempt["snapshot_path"] = str(snapshot_path)
            improvements.append(
                {
                    "attempt": sequence,
                    "combo_id": item.combo_id,
                    "before_usd": before_mills / 1000,
                    "after_usd": incumbent.costs.total_mills / 1000,
                    "saving_usd": (before_mills - incumbent.costs.total_mills) / 1000,
                    "snapshot_path": str(snapshot_path),
                }
            )
            print(
                f"  {item.combo_id}: accepted USD {incumbent.costs.total_mills / 1000:,.2f}",
                flush=True,
            )
        else:
            print(
                f"  {item.combo_id}: {result.status}, no improvement "
                f"({item.sku_count} SKUs, {result.wall_time_seconds:.1f}s)",
                flush=True,
            )
        attempts.append(attempt)
        summary["best"] = _cost_payload(incumbent.costs)
        summary["saving_usd"] = (start_costs.total_mills - incumbent.costs.total_mills) / 1000
        _atomic_write_json(summary_path, summary)
        if target_mills is not None and incumbent.costs.total_mills <= target_mills:
            break

    summary["best"] = _cost_payload(incumbent.costs)
    summary["saving_usd"] = (start_costs.total_mills - incumbent.costs.total_mills) / 1000
    summary["target_met"] = incumbent.costs.total_mills <= target_mills if target_mills else None
    summary["best_path"] = str(best_path)
    _atomic_write_json(summary_path, summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reduced SCIP triple/quad destination LNS")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--warm-start", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--exclude-pool-summary", type=Path, required=True)
    parser.add_argument("--triple-count", type=int, default=300)
    parser.add_argument("--quad-count", type=int, default=100)
    parser.add_argument("--destination-sample-size", type=int, default=180)
    parser.add_argument("--time-per-combo", type=float, default=2.0)
    parser.add_argument("--max-skus", type=int, default=220)
    parser.add_argument("--max-extra-pallets", type=int, default=1000)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--target-total-usd", type=Decimal)
    parser.add_argument("--extra-region-share", type=float, default=0.0)
    parser.add_argument("--memory-limit-mb", type=int)
    parser.add_argument("--scip-parameter", action="append", default=[])
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run_combo_destination_lns(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
