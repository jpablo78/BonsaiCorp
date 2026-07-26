"""Búsqueda exacta enfocada sobre pares de diseños de caja destino.

Unlike :mod:`bonsai.multi_destination_lns`, this runner releases the *union*
of the SKUs compatible with the two destinations.  A released SKU may stay in
its incumbent design, move to A, move to B, or choose between A and B when it
is compatible with both.  This exposes source-type evacuations that require
two different receiving designs and were intentionally absent from the older
``min_destination_choices_per_sku=2`` neighborhoods.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from decimal import Decimal
import heapq
from itertools import combinations
import json
from pathlib import Path
from typing import Iterable, Mapping

from .config import FreightPolicy
from .costs import box_type_key
from .data import load_prepared_data
from .destination_lns import DestinationWorkItem, rank_destination_work_items
from .exact_candidates import generate_exact_candidates
from .models import CandidateBox, Dimensions, PreparedData, Product
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


DesignKey = tuple[int, int, int]
PairKey = frozenset[DesignKey]


@dataclass(frozen=True)
class PairDestinationWorkItem:
    """Un subproblema exacto con opciones incumbente, A o B."""

    pair_id: str
    left: DestinationWorkItem
    right: DestinationWorkItem
    product_codes: tuple[str, ...]
    shared_products: int
    shared_source_types: int
    complementary_source_types: int
    complementary_source_volume: int

    @property
    def pair_key(self) -> PairKey:
        return frozenset(
            (self.left.candidate.internal.as_tuple(), self.right.candidate.internal.as_tuple())
        )

    @property
    def sku_count(self) -> int:
        return len(self.product_codes)

    @property
    def tier_crossings(self) -> int:
        return self.left.tier_crossings + self.right.tier_crossings

    @property
    def gross_opportunity_mills(self) -> int:
        return self.left.gross_opportunity_mills + self.right.gross_opportunity_mills


def _incumbent_source_groups(
    products: Iterable[Product],
    incumbent_assignment: Mapping[str, CandidateBox],
) -> tuple[tuple[frozenset[str], int], ...]:
    """Devuelve membresía de SKU y demanda total para cada diseño incumbente."""

    product_by_code = {product.code: product for product in products}
    codes_by_type: dict[tuple[float, float, float, float], set[str]] = {}
    for code, candidate in incumbent_assignment.items():
        codes_by_type.setdefault(box_type_key(candidate), set()).add(code)
    return tuple(
        (
            frozenset(codes),
            sum(product_by_code[code].annual_volume for code in codes),
        )
        for _, codes in sorted(codes_by_type.items())
    )


def pair_complementarity_metrics(
    left_codes: frozenset[str],
    right_codes: frozenset[str],
    source_groups: Iterable[tuple[frozenset[str], int]],
) -> tuple[int, int, int, int]:
    """Calcula atributos de ranking de solapamiento y complementariedad de origen.

    A complementary source group is fully movable by the union of A and B but
    is not fully movable by either destination alone.  Such a group can be
    removed only in the paired model, so its count and demand are the most
    direct features for ranking work not covered by single-destination search.
    """

    union = left_codes | right_codes
    shared_products = len(left_codes & right_codes)
    shared_source_types = 0
    complementary_types = 0
    complementary_volume = 0
    for group_codes, group_volume in source_groups:
        left_touches = bool(group_codes & left_codes)
        right_touches = bool(group_codes & right_codes)
        if left_touches and right_touches:
            shared_source_types += 1
        if (
            group_codes <= union
            and not group_codes <= left_codes
            and not group_codes <= right_codes
        ):
            complementary_types += 1
            complementary_volume += group_volume
    return (
        shared_products,
        shared_source_types,
        complementary_types,
        complementary_volume,
    )


def rank_destination_pairs(
    destinations: Iterable[DestinationWorkItem],
    data: PreparedData,
    incumbent_assignment: Mapping[str, CandidateBox],
    *,
    excluded_pairs: Iterable[PairKey] = (),
    min_skus: int = 2,
    max_skus: int | None = 220,
    max_pairs: int | None = 128,
) -> tuple[PairDestinationWorkItem, ...]:
    """Ordena todos los pares destino sin limitarse a SKU solapados."""

    if min_skus < 1:
        raise ValueError("min_skus must be positive")
    if max_skus is not None and max_skus < min_skus:
        raise ValueError("max_skus cannot be smaller than min_skus")
    if max_pairs is not None and max_pairs < 1:
        raise ValueError("max_pairs must be positive")

    ranked = tuple(destinations)
    excluded = set(excluded_pairs)
    source_groups = _incumbent_source_groups(data.products, incumbent_assignment)
    code_index = {
        code: index
        for index, code in enumerate(sorted(product.code for product in data.products))
    }

    def code_mask(codes: Iterable[str]) -> int:
        mask = 0
        for code in codes:
            mask |= 1 << code_index[code]
        return mask

    destination_masks = tuple(code_mask(item.product_codes) for item in ranked)
    group_masks = tuple((code_mask(codes), volume) for codes, volume in source_groups)

    # Mantener vivas las aproximadamente 770 mil instancias de
    # PairDestinationWorkItem es más lento y consume memoria innecesariamente.
    # Un min-heap acotado retiene exactamente el top-K pedido al recorrer el
    # universo completo de pares.
    heap: list[tuple[tuple[int, ...], int, int, tuple[int, int, int, int]]] = []
    for left_index, right_index in combinations(range(len(ranked)), 2):
        left = ranked[left_index]
        right = ranked[right_index]
        key = frozenset(
            (left.candidate.internal.as_tuple(), right.candidate.internal.as_tuple())
        )
        if len(key) != 2 or key in excluded:
            continue
        left_mask = destination_masks[left_index]
        right_mask = destination_masks[right_index]
        union_mask = left_mask | right_mask
        sku_count = union_mask.bit_count()
        if sku_count < min_skus:
            continue
        if max_skus is not None and sku_count > max_skus:
            continue
        shared_products = (left_mask & right_mask).bit_count()
        shared_source_types = 0
        complementary_types = 0
        complementary_volume = 0
        for group_mask, group_volume in group_masks:
            if group_mask & left_mask and group_mask & right_mask:
                shared_source_types += 1
            if (
                group_mask & ~union_mask == 0
                and group_mask & ~left_mask != 0
                and group_mask & ~right_mask != 0
            ):
                complementary_types += 1
                complementary_volume += group_volume
        metrics = (
            shared_products,
            shared_source_types,
            complementary_types,
            complementary_volume,
        )
        quality = (
            int(complementary_types > 0),
            complementary_volume,
            complementary_types,
            shared_source_types,
            left.tier_crossings + right.tier_crossings,
            shared_products,
            left.gross_opportunity_mills + right.gross_opportunity_mills,
            -sku_count,
            -left_index,
            -right_index,
        )
        entry = (quality, left_index, right_index, metrics)
        if max_pairs is None:
            heap.append(entry)
        elif len(heap) < max_pairs:
            heapq.heappush(heap, entry)
        elif quality > heap[0][0]:
            heapq.heapreplace(heap, entry)

    selected = sorted(heap, key=lambda entry: entry[0], reverse=True)
    return tuple(
        PairDestinationWorkItem(
            pair_id=f"pair_{index:04d}",
            left=ranked[left_index],
            right=ranked[right_index],
            product_codes=tuple(
                sorted(
                    set(ranked[left_index].product_codes)
                    | set(ranked[right_index].product_codes)
                )
            ),
            shared_products=metrics[0],
            shared_source_types=metrics[1],
            complementary_source_types=metrics[2],
            complementary_source_volume=metrics[3],
        )
        for index, (_, left_index, right_index, metrics) in enumerate(selected)
    )


def allowed_internals_for_pair(
    item: PairDestinationWorkItem,
) -> dict[str, tuple[Dimensions, ...]]:
    """Asocia cada SKU liberado con su subconjunto compatible de {A, B}."""

    result: dict[str, tuple[Dimensions, ...]] = {}
    for code in item.product_codes:
        choices: list[Dimensions] = []
        if code in item.left.product_codes:
            choices.append(item.left.candidate.internal)
        if code in item.right.product_codes:
            choices.append(item.right.candidate.internal)
        result[code] = tuple(sorted(set(choices), key=Dimensions.as_tuple))
    return result


def load_excluded_pairs(
    summary_paths: Iterable[Path],
    *,
    statuses: frozenset[str] | None = None,
) -> frozenset[PairKey]:
    """Carga cada par de geometrías ya ofrecido conjuntamente en intentos previos."""

    excluded: set[PairKey] = set()
    for path in summary_paths:
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        for attempt in payload.get("attempts", ()):
            if statuses is not None and attempt.get("status") not in statuses:
                continue
            geometries = {
                tuple(int(value) for value in destination["internal_mm"])
                for destination in attempt.get("destinations", ())
                if "internal_mm" in destination
            }
            geometries.update(
                tuple(int(value) for value in internal)
                for internal in attempt.get("destination_internals_mm", ())
            )
            for left, right in combinations(sorted(geometries), 2):
                excluded.add(frozenset((left, right)))
    return frozenset(excluded)


def _pair_payload(item: PairDestinationWorkItem) -> dict[str, object]:
    choices = allowed_internals_for_pair(item)
    return {
        "pair_id": item.pair_id,
        "destination_internals_mm": tuple(sorted(item.pair_key)),
        "sku_count": item.sku_count,
        "one_choice_skus": sum(len(value) == 1 for value in choices.values()),
        "two_choice_skus": sum(len(value) == 2 for value in choices.values()),
        "shared_products": item.shared_products,
        "shared_source_types": item.shared_source_types,
        "complementary_source_types": item.complementary_source_types,
        "complementary_source_volume": item.complementary_source_volume,
        "tier_crossings": item.tier_crossings,
        "gross_opportunity_usd": item.gross_opportunity_mills / 1000,
    }


def run_pair_destination_lns(args: argparse.Namespace) -> dict[str, object]:
    data = load_prepared_data(args.data_dir)
    policy = FreightPolicy(extra_region_share=args.extra_region_share)
    incumbent = validate_solution_csv(args.warm_start, data, policy)
    thickness = _infer_thickness(incumbent)
    target_mills = _mills_from_usd(args.target_total_usd)
    excluded = set(load_excluded_pairs(args.exclude_summary))
    excluded.update(
        load_excluded_pairs(
            args.exclude_optimal_summary,
            statuses=frozenset(("OPTIMAL",)),
        )
    )

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
    pairs = rank_destination_pairs(
        destinations,
        data,
        incumbent.assignment,
        excluded_pairs=excluded,
        min_skus=args.min_skus,
        max_skus=args.max_skus,
        max_pairs=args.max_pairs,
    )

    attempts: list[dict[str, object]] = []
    improvements: list[dict[str, object]] = []
    summary: dict[str, object] = {
        "configuration": {
            "data_dir": str(args.data_dir),
            "warm_start": str(args.warm_start),
            "time_per_pair_seconds": args.time_per_pair,
            "num_search_workers": args.num_search_workers,
            "max_pairs": args.max_pairs,
            "min_skus": args.min_skus,
            "max_skus": args.max_skus,
            "max_extra_pallets": args.max_extra_pallets,
            "excluded_summary_paths": tuple(map(str, args.exclude_summary)),
            "excluded_pair_count": len(excluded),
            "exclude_optimal_summary_paths": tuple(map(str, args.exclude_optimal_summary)),
            "target_total_usd": target_mills / 1000 if target_mills else None,
        },
        "candidate_stats": asdict(candidate_stats),
        "ranked_destination_count": len(destinations),
        "initial": _cost_payload(start_costs),
        "ranked_pairs": tuple(_pair_payload(item) for item in pairs),
        "attempts": attempts,
        "improvements": improvements,
    }
    summary_path = args.output_dir / "resumen_pair_destination_lns.json"
    print(
        f"Pair LNS start: USD {incumbent.costs.total_mills / 1000:,.2f}; "
        f"{len(destinations):,} destinations, {len(excluded):,} covered pairs, "
        f"{len(pairs)} selected pairs",
        flush=True,
    )
    for index, item in enumerate(pairs):
        before_mills = incumbent.costs.total_mills
        result = solve_for_thickness(
            data,
            thickness,
            policy,
            time_limit_seconds=args.time_per_pair,
            num_search_workers=args.num_search_workers,
            random_seed=args.random_seed + index * 1_009,
            initial_assignment=incumbent.assignment,
            candidate_strategy="exact",
            max_extra_pallets=args.max_extra_pallets,
            free_product_codes=item.product_codes,
            allowed_internals_by_product=allowed_internals_for_pair(item),
            precomputed_exact_candidates=candidates,
            precomputed_exact_candidate_stats=candidate_stats,
        )
        attempt = {
            "sequence": index + 1,
            **_pair_payload(item),
            "before_usd": before_mills / 1000,
            **_result_payload(result),
            "accepted": False,
        }
        if result.costs.total_mills < before_mills:
            candidate_path = args.output_dir / ".candidate_validation.csv"
            checked = _atomic_write_assignment(candidate_path, data, result.assignment, policy)
            candidate_path.unlink(missing_ok=True)
            if checked.costs.total_mills != result.costs.total_mills:
                raise RuntimeError("independent validation changed solver cost")
            incumbent = checked
            snapshot_path = args.output_dir / f"incumbent_{snapshot_number:04d}.csv"
            snapshot_number += 1
            _atomic_write_assignment(snapshot_path, data, incumbent.assignment, policy)
            _atomic_write_assignment(best_path, data, incumbent.assignment, policy)
            attempt["accepted"] = True
            attempt["snapshot_path"] = str(snapshot_path)
            improvements.append(
                {
                    "attempt": index + 1,
                    "pair_id": item.pair_id,
                    "before_usd": before_mills / 1000,
                    "after_usd": incumbent.costs.total_mills / 1000,
                    "saving_usd": (before_mills - incumbent.costs.total_mills) / 1000,
                    "snapshot_path": str(snapshot_path),
                }
            )
            print(f"  {item.pair_id}: accepted USD {incumbent.costs.total_mills / 1000:,.2f}", flush=True)
        else:
            print(
                f"  {item.pair_id}: {result.status}, no improvement "
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
    parser = argparse.ArgumentParser(description="Exact incumbent-or-A-or-B pair LNS")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--warm-start", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--time-per-pair", type=float, default=2.0)
    parser.add_argument("--num-search-workers", type=int, default=2)
    parser.add_argument("--max-pairs", type=int, default=64)
    parser.add_argument("--min-skus", type=int, default=2)
    parser.add_argument("--max-skus", type=int, default=220)
    parser.add_argument("--max-extra-pallets", type=int, default=1000)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--target-total-usd", type=Decimal)
    parser.add_argument("--extra-region-share", type=float, default=0.0)
    parser.add_argument("--exclude-summary", type=Path, action="append", default=[])
    parser.add_argument(
        "--exclude-optimal-summary", type=Path, action="append", default=[]
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run_pair_destination_lns(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
