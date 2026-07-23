"""Plant-specific restricted large-neighborhood search.

Most challenge SKUs have demand in exactly one plant.  Releasing those SKUs
plant by plant keeps procurement interactions local while allowing a much
broader coordinated reassignment than destination-at-a-time LNS.  Candidate
choices are restricted to incumbent physical designs plus a small ranked set
of exact designs with an opportunity to cross the 100k or 500k procurement
tiers.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping

from .config import FreightPolicy
from .costs import box_type_key, freight_pallets, unit_price_mills
from .data import load_prepared_data
from .destination_lns import _representative_candidates
from .exact_candidates import generate_exact_candidates
from .models import CandidateBox, Dimensions, PLANTS, PreparedData
from .optimizer import solve_for_thickness
from .solution_validation import validate_solution_csv
from .tier_lns import (
    _atomic_write_assignment,
    _atomic_write_json,
    _cost_payload,
    _infer_thickness,
    _result_payload,
    _unique_internal_designs,
)


FOCUS_THRESHOLDS = (100_000, 500_000)


@dataclass(frozen=True)
class PlantDestination:
    plant: str
    candidate: CandidateBox
    movable_codes: tuple[str, ...]
    incumbent_volume: int
    potential_volume: int
    crossed_thresholds: tuple[int, ...]
    optimistic_procurement_saving_mills: int
    freight_delta_if_all_move_mills: int

    @property
    def score_mills(self) -> int:
        return self.optimistic_procurement_saving_mills - max(
            0, self.freight_delta_if_all_move_mills
        )


@dataclass(frozen=True)
class PlantTierTarget:
    """An incumbent destination just below a focus procurement threshold."""

    plant: str
    candidate: CandidateBox
    incumbent_volume: int
    threshold: int

    @property
    def gap(self) -> int:
        return self.threshold - self.incumbent_volume


def single_plant_codes(data: PreparedData) -> dict[str, tuple[str, ...]]:
    """Return SKUs whose non-zero demand belongs to exactly one plant."""

    result: dict[str, list[str]] = {plant: [] for plant in PLANTS}
    for product in data.products:
        active = [
            plant for plant, volume in product.annual_volume_by_plant.items() if volume
        ]
        if len(active) == 1:
            result[active[0]].append(product.code)
    return {plant: tuple(sorted(codes)) for plant, codes in result.items()}


def _volume_by_type_plant(
    data: PreparedData, assignment: Mapping[str, CandidateBox]
) -> dict[tuple[tuple[float, float, float, float], str], int]:
    volumes: dict[tuple[tuple[float, float, float, float], str], int] = {}
    for product in data.products:
        type_key = box_type_key(assignment[product.code])
        for plant, volume in product.annual_volume_by_plant.items():
            key = (type_key, plant)
            volumes[key] = volumes.get(key, 0) + volume
    return volumes


def rank_plant_destinations(
    data: PreparedData,
    incumbent_assignment: Mapping[str, CandidateBox],
    exact_candidates: Iterable[CandidateBox],
    freight_policy: FreightPolicy,
    plant: str,
    *,
    max_gap_to_focus_tier: int = 100_000,
) -> tuple[PlantDestination, ...]:
    """Rank exact designs for coordinated moves by single-plant SKUs.

    The ranking is deliberately optimistic: it ignores deterioration at donor
    types.  It is used only to choose a compact candidate universe; CP-SAT and
    the independent validator calculate the exact net cost.
    """

    if plant not in PLANTS:
        raise ValueError(f"unknown plant: {plant}")
    product_by_code = data.product_by_code
    eligible_codes = set(single_plant_codes(data)[plant])
    volumes = _volume_by_type_plant(data, incumbent_assignment)
    ranked: list[PlantDestination] = []

    for candidate in _representative_candidates(exact_candidates):
        target_key = box_type_key(candidate)
        target_volume = volumes.get((target_key, plant), 0)
        movable = tuple(
            sorted(
                code
                for code in candidate.compatible_product_codes & eligible_codes
                if incumbent_assignment[code].internal != candidate.internal
            )
        )
        if not movable:
            continue
        potential = target_volume + sum(
            product_by_code[code].annual_volume_by_plant[plant] for code in movable
        )
        crossed = tuple(
            threshold
            for threshold in FOCUS_THRESHOLDS
            if target_volume < threshold <= potential
            and threshold - target_volume <= max_gap_to_focus_tier
        )
        if not crossed:
            continue

        potential_price = unit_price_mills(candidate.thickness_mm, potential)
        procurement_saving = 0
        if target_volume:
            procurement_saving += target_volume * max(
                0,
                unit_price_mills(candidate.thickness_mm, target_volume)
                - potential_price,
            )
        freight_delta = 0
        for code in movable:
            product = product_by_code[code]
            source = incumbent_assignment[code]
            source_volume = volumes[(box_type_key(source), plant)]
            volume = product.annual_volume_by_plant[plant]
            procurement_saving += volume * max(
                0,
                unit_price_mills(candidate.thickness_mm, source_volume)
                - potential_price,
            )
            freight_delta += (
                freight_pallets(product, candidate, plant)
                - freight_pallets(product, source, plant)
            ) * freight_policy.expected_mills_per_pallet

        ranked.append(
            PlantDestination(
                plant=plant,
                candidate=candidate,
                movable_codes=movable,
                incumbent_volume=target_volume,
                potential_volume=potential,
                crossed_thresholds=crossed,
                optimistic_procurement_saving_mills=procurement_saving,
                freight_delta_if_all_move_mills=freight_delta,
            )
        )

    ranked.sort(
        key=lambda item: (
            -item.score_mills,
            -len(item.crossed_thresholds),
            -item.incumbent_volume,
            item.candidate.internal.as_tuple(),
        )
    )
    return tuple(ranked)


def allowed_plant_internals(
    data: PreparedData,
    incumbent_assignment: Mapping[str, CandidateBox],
    exact_candidates: Iterable[CandidateBox],
    plant: str,
    *,
    promising_limit: int = 16,
    include_used_designs: bool = True,
    max_gap_to_focus_tier: int = 100_000,
    freight_policy: FreightPolicy | None = None,
) -> tuple[dict[str, tuple[Dimensions, ...]], tuple[PlantDestination, ...]]:
    """Build a compact, auditable choice set for one plant."""

    policy = freight_policy or FreightPolicy()
    codes = single_plant_codes(data)[plant]
    selected: dict[Dimensions, CandidateBox] = {}
    if include_used_designs:
        for candidate in incumbent_assignment.values():
            selected[candidate.internal] = candidate

    ranked = rank_plant_destinations(
        data,
        incumbent_assignment,
        exact_candidates,
        policy,
        plant,
        max_gap_to_focus_tier=max_gap_to_focus_tier,
    )
    for opportunity in ranked[:promising_limit]:
        selected[opportunity.candidate.internal] = opportunity.candidate

    allowed: dict[str, tuple[Dimensions, ...]] = {}
    for code in codes:
        internals = tuple(
            sorted(
                (
                    internal
                    for internal, candidate in selected.items()
                    if code in candidate.compatible_product_codes
                ),
                key=Dimensions.as_tuple,
            )
        )
        if any(
            internal != incumbent_assignment[code].internal for internal in internals
        ):
            allowed[code] = internals
    return allowed, ranked


def allowed_focus_tier_internals(
    data: PreparedData,
    incumbent_assignment: Mapping[str, CandidateBox],
    exact_candidates: Iterable[CandidateBox],
    plant: str,
    *,
    max_gap_to_focus_tier: int = 100_000,
    max_targets: int = 8,
) -> tuple[dict[str, tuple[Dimensions, ...]], tuple[PlantTierTarget, ...]]:
    """Release SKUs around used designs immediately below 100k/500k.

    Unlike :func:`allowed_plant_internals`, this focused neighborhood also
    admits multi-plant SKUs when they can feed one of the selected types.  The
    one-plant SKUs remain the bulk of the neighborhood, while these bridge
    products are necessary for targets such as Monterrey 399x299x238 where
    the closest compatible donor also carries Buenos Aires demand.
    """

    if plant not in PLANTS:
        raise ValueError(f"unknown plant: {plant}")
    volumes = _volume_by_type_plant(data, incumbent_assignment)
    exact_by_internal = {
        candidate.internal: candidate
        for candidate in _representative_candidates(exact_candidates)
    }
    used_by_internal = {
        candidate.internal: candidate for candidate in incumbent_assignment.values()
    }
    targets: list[PlantTierTarget] = []
    for internal, used_candidate in used_by_internal.items():
        volume = volumes.get((box_type_key(used_candidate), plant), 0)
        if not volume:
            continue
        threshold = next(
            (
                value
                for value in FOCUS_THRESHOLDS
                if volume < value and value - volume <= max_gap_to_focus_tier
            ),
            None,
        )
        if threshold is None:
            continue
        canonical = exact_by_internal.get(internal)
        if canonical is None:
            continue
        targets.append(PlantTierTarget(plant, canonical, volume, threshold))
    targets.sort(key=lambda item: (item.gap, -item.incumbent_volume, item.candidate.internal.as_tuple()))
    selected = tuple(targets[:max_targets])

    allowed: dict[str, tuple[Dimensions, ...]] = {}
    for product in data.products:
        choices = tuple(
            sorted(
                {
                    target.candidate.internal
                    for target in selected
                    if product.code in target.candidate.compatible_product_codes
                    and target.candidate.internal
                    != incumbent_assignment[product.code].internal
                },
                key=Dimensions.as_tuple,
            )
        )
        if choices:
            allowed[product.code] = choices
    return allowed, selected


def run_plant_lns(args: argparse.Namespace) -> dict[str, object]:
    data = load_prepared_data(args.data_dir)
    policy = FreightPolicy(extra_region_share=args.extra_region_share)
    incumbent = validate_solution_csv(args.warm_start, data, policy)
    thickness = _infer_thickness(incumbent)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_path = args.output_dir / "asignacion_optima.csv"
    _atomic_write_assignment(best_path, data, incumbent.assignment, policy)

    exact_candidates, stats = generate_exact_candidates(
        data.products,
        thickness,
        retained_designs=_unique_internal_designs(incumbent.assignment),
    )
    initial_costs = incumbent.costs
    attempts: list[dict[str, object]] = []
    opportunities: dict[str, list[dict[str, object]]] = {}

    for pass_index in range(args.passes):
        improved = False
        active_plants = tuple(args.plants) if args.plants else PLANTS
        unknown_plants = set(active_plants) - set(PLANTS)
        if unknown_plants:
            raise ValueError(f"unknown plants: {sorted(unknown_plants)}")
        for plant_index, plant in enumerate(active_plants):
            if args.choice_mode == "focus":
                allowed, targets = allowed_focus_tier_internals(
                    data,
                    incumbent.assignment,
                    exact_candidates,
                    plant,
                    max_gap_to_focus_tier=args.max_gap_to_focus_tier,
                    max_targets=args.max_targets,
                )
                ranked = rank_plant_destinations(
                    data,
                    incumbent.assignment,
                    exact_candidates,
                    policy,
                    plant,
                    max_gap_to_focus_tier=args.max_gap_to_focus_tier,
                )
            else:
                allowed, ranked = allowed_plant_internals(
                    data,
                    incumbent.assignment,
                    exact_candidates,
                    plant,
                    promising_limit=args.promising_limit,
                    include_used_designs=True,
                    max_gap_to_focus_tier=args.max_gap_to_focus_tier,
                    freight_policy=policy,
                )
            opportunities[plant] = [
                {
                    "external_mm": item.candidate.external.as_tuple(),
                    "capacity_per_pallet": item.candidate.capacity_per_pallet,
                    "movable_skus": len(item.movable_codes),
                    "incumbent_volume": item.incumbent_volume,
                    "potential_volume": item.potential_volume,
                    "crossed_thresholds": item.crossed_thresholds,
                    "optimistic_procurement_saving_usd": item.optimistic_procurement_saving_mills / 1000,
                    "freight_delta_if_all_move_usd": item.freight_delta_if_all_move_mills / 1000,
                    "score_usd": item.score_mills / 1000,
                }
                for item in ranked[: args.promising_limit]
            ]
            if not allowed:
                continue

            before = incumbent.costs.total_mills
            result = solve_for_thickness(
                data,
                thickness,
                policy,
                time_limit_seconds=args.time_per_plant,
                num_search_workers=args.num_search_workers,
                random_seed=args.random_seed + pass_index * 101 + plant_index,
                initial_assignment=incumbent.assignment,
                candidate_strategy="exact",
                max_extra_pallets=args.max_extra_pallets,
                free_product_codes=allowed,
                allowed_internals_by_product=allowed,
                precomputed_exact_candidates=exact_candidates,
                precomputed_exact_candidate_stats=stats,
            )
            attempt = {
                "pass": pass_index + 1,
                "plant": plant,
                "released_skus": len(allowed),
                "choice_links": sum(len(items) for items in allowed.values()),
                "before_usd": before / 1000,
                **_result_payload(result),
                "accepted": False,
            }
            if result.costs.total_mills < before:
                candidate_path = args.output_dir / ".candidate_validation.csv"
                checked = _atomic_write_assignment(
                    candidate_path, data, result.assignment, policy
                )
                candidate_path.unlink(missing_ok=True)
                if checked.costs.total_mills != result.costs.total_mills:
                    raise RuntimeError("independent validation changed plant-LNS cost")
                incumbent = checked
                _atomic_write_assignment(best_path, data, incumbent.assignment, policy)
                attempt["accepted"] = True
                improved = True
            attempts.append(attempt)
            print(
                f"plant={plant} pass={pass_index + 1} released={len(allowed)} "
                f"status={result.status} cost={result.costs.total_mills / 1000:,.2f} "
                f"accepted={attempt['accepted']}",
                flush=True,
            )
        if not improved:
            break

    summary: dict[str, object] = {
        "configuration": vars(args) | {"output_dir": str(args.output_dir), "data_dir": str(args.data_dir), "warm_start": str(args.warm_start)},
        "candidate_stats": asdict(stats),
        "initial": _cost_payload(initial_costs),
        "best": _cost_payload(incumbent.costs),
        "saving_usd": (initial_costs.total_mills - incumbent.costs.total_mills) / 1000,
        "opportunities": opportunities,
        "attempts": attempts,
        "best_path": str(best_path),
    }
    # argparse Namespace contains Paths, which the generic writer cannot
    # serialize.  The explicit configuration above is normalized here.
    summary["configuration"] = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in summary["configuration"].items()
    }
    _atomic_write_json(args.output_dir / "resumen_plant_lns.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plant-specific exact LNS")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--warm-start", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--time-per-plant", type=float, default=20.0)
    parser.add_argument("--num-search-workers", type=int, default=6)
    parser.add_argument("--passes", type=int, default=1)
    parser.add_argument("--promising-limit", type=int, default=16)
    parser.add_argument("--choice-mode", choices=("broad", "focus"), default="focus")
    parser.add_argument("--max-targets", type=int, default=8)
    parser.add_argument("--max-gap-to-focus-tier", type=int, default=100_000)
    parser.add_argument("--max-extra-pallets", type=int, default=500)
    parser.add_argument("--random-seed", type=int, default=731)
    parser.add_argument("--extra-region-share", type=float, default=0.0)
    parser.add_argument("--plants", nargs="+", choices=PLANTS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_plant_lns(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
