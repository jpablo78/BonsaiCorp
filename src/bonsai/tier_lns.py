"""Orquestación reutilizable de búsqueda de gran vecindario centrada en tiers.

El modelo CP-SAT vive en :mod:`bonsai.optimizer`; este módulo se limita
deliberadamente a programar vecindarios, proteger la incumbente y producir
salidas auditables. Cada asignación aceptada pasa por el mismo validador CSV
usado para Kaggle antes de convertirse en la siguiente incumbente.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from decimal import Decimal, ROUND_HALF_UP
from itertools import combinations
import json
import os
from pathlib import Path
import time
from uuid import uuid4

from .config import ALLOWED_THICKNESSES, FreightPolicy
from .data import load_prepared_data
from .exact_candidates import generate_exact_candidates
from .models import CandidateBox, CostBreakdown, Dimensions, PreparedData
from .neighborhoods import Neighborhood, TierNeighborhoodPlan, build_tier_neighborhoods
from .optimizer import SolveResult, solve_for_thickness
from .reporting import write_assignment_csv, write_json
from .solution_validation import ValidationResult, validate_solution_csv


@dataclass(frozen=True)
class WorkItem:
    """Un conjunto determinista de SKU liberados en un subproblema LNS."""

    neighborhood_id: str
    source_kind: str
    product_codes: tuple[str, ...]
    member_ids: tuple[str, ...]

    @property
    def sku_count(self) -> int:
        return len(self.product_codes)


def _union_item(kind: str, members: tuple[Neighborhood, ...]) -> WorkItem:
    return WorkItem(
        neighborhood_id=f"{kind}__" + "__".join(item.neighborhood_id for item in members),
        source_kind=kind,
        product_codes=tuple(
            sorted(set().union(*(set(item.product_codes) for item in members)))
        ),
        member_ids=tuple(item.neighborhood_id for item in members),
    )


def generate_work_items(
    plan: TierNeighborhoodPlan,
    *,
    max_star_combination: int = 2,
    max_component_combination: int = 2,
    max_skus: int | None = None,
    max_neighborhoods: int | None = None,
) -> tuple[WorkItem, ...]:
    """Expande un plan de tiers en estrellas, componentes y uniones únicos.

    Los conjuntos de SKU duplicados se eliminan porque inducen el mismo modelo
    LNS. El orden de inserción prueba deliberadamente vecindarios individuales
    más pequeños y enfocados antes de sus combinaciones más amplias.
    """

    if max_star_combination < 1:
        raise ValueError("max_star_combination must be at least 1")
    if max_component_combination < 1:
        raise ValueError("max_component_combination must be at least 1")
    if max_skus is not None and max_skus < 1:
        raise ValueError("max_skus must be positive")
    if max_neighborhoods is not None and max_neighborhoods < 1:
        raise ValueError("max_neighborhoods must be positive")

    raw: list[WorkItem] = []
    raw.extend(
        WorkItem(
            neighborhood_id=item.neighborhood_id,
            source_kind="star",
            product_codes=item.product_codes,
            member_ids=(item.neighborhood_id,),
        )
        for item in plan.stars
    )
    raw.extend(
        WorkItem(
            neighborhood_id=item.neighborhood_id,
            source_kind="component",
            product_codes=item.product_codes,
            member_ids=(item.neighborhood_id,),
        )
        for item in plan.components
    )

    for size in range(2, min(max_star_combination, len(plan.stars)) + 1):
        raw.extend(
            _union_item("star_union", members)
            for members in combinations(plan.stars, size)
        )
    for size in range(2, min(max_component_combination, len(plan.components)) + 1):
        raw.extend(
            _union_item("component_union", members)
            for members in combinations(plan.components, size)
        )

    unique: list[WorkItem] = []
    seen_code_sets: set[frozenset[str]] = set()
    for item in raw:
        code_set = frozenset(item.product_codes)
        if not code_set or code_set in seen_code_sets:
            continue
        if max_skus is not None and item.sku_count > max_skus:
            continue
        seen_code_sets.add(code_set)
        unique.append(item)
        if max_neighborhoods is not None and len(unique) >= max_neighborhoods:
            break
    return tuple(unique)


def _mills_from_usd(value: Decimal | None) -> int | None:
    if value is None:
        return None
    return int((value * Decimal(1000)).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def _cost_payload(costs: CostBreakdown) -> dict[str, object]:
    return costs.as_dict()


def _result_payload(result: SolveResult) -> dict[str, object]:
    return {
        "status": result.status,
        "candidate_count": result.candidate_count,
        "candidate_strategy": result.candidate_strategy,
        "candidate_stats": asdict(result.candidate_stats) if result.candidate_stats else None,
        "solver_objective_usd": (
            result.solver_objective_mills / 1000
            if result.solver_objective_mills is not None
            else None
        ),
        "best_bound_usd": (
            result.best_bound_mills / 1000
            if result.best_bound_mills is not None
            else None
        ),
        "wall_time_seconds": result.wall_time_seconds,
        "num_conflicts": result.num_conflicts,
        "num_branches": result.num_branches,
        "selected_source": result.selected_source,
        "improved_incumbent": result.improved_incumbent,
        "target_met": result.target_met,
        "target_proven_infeasible": result.target_proven_infeasible,
        **_cost_payload(result.costs),
    }


def _atomic_write_assignment(
    path: Path,
    data: PreparedData,
    assignment: dict[str, CandidateBox],
    policy: FreightPolicy,
) -> ValidationResult:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.parent / f".{path.name}.{os.getpid()}.{uuid4().hex}.pending"
    try:
        write_assignment_csv(pending, data, assignment)
        checked = validate_solution_csv(pending, data, policy)
        os.replace(pending, path)
        return checked
    finally:
        if pending.exists():
            pending.unlink()


def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    pending = path.parent / f".{path.name}.{os.getpid()}.{uuid4().hex}.pending"
    try:
        write_json(pending, payload)
    # En Windows, una lectura concurrente puede retener brevemente un
    # identificador no compartible del destino. Se conserva el reemplazo
    # atómico, pero se tolera ese bloqueo transitorio para no abortar una
    # corrida larga.
        for attempt in range(20):
            try:
                os.replace(pending, path)
                break
            except PermissionError:
                if attempt == 19:
                    raise
                time.sleep(0.05)
    finally:
        if pending.exists():
            pending.unlink()


def _infer_thickness(validation: ValidationResult) -> float:
    thicknesses = {box.thickness_mm for box in validation.assignment.values()}
    if len(thicknesses) != 1:
        raise ValueError("incumbent must use one global thickness")
    thickness = next(iter(thicknesses))
    if thickness not in ALLOWED_THICKNESSES:
        raise ValueError(f"unsupported incumbent thickness: {thickness}")
    return thickness


def _unique_internal_designs(
    assignment: dict[str, CandidateBox],
) -> tuple[Dimensions, ...]:
    return tuple(
        sorted(
            {box.internal for box in assignment.values()},
            key=lambda dimensions: dimensions.as_tuple(),
        )
    )


def _next_snapshot_number(output_dir: Path) -> int:
    numbers: list[int] = []
    for path in output_dir.glob("incumbent_*.csv"):
        suffix = path.stem.removeprefix("incumbent_")
        if suffix.isdigit():
            numbers.append(int(suffix))
    return max(numbers, default=-1) + 1


def run_tier_lns(args: argparse.Namespace) -> dict[str, object]:
    """Ejecuta LNS iterativo por tiers y devuelve el resumen auditable serializable en JSON."""

    if args.rounds < 1:
        raise ValueError("--rounds must be at least 1")
    if args.time_per_neighborhood <= 0:
        raise ValueError("--time-per-neighborhood must be positive")
    if args.num_search_workers < 1:
        raise ValueError("--num-search-workers must be positive")
    if args.target_mode == "hard" and args.target_total_usd is None:
        raise ValueError("--target-mode hard requires --target-total-usd")

    data = load_prepared_data(args.data_dir)
    policy = FreightPolicy(extra_region_share=args.extra_region_share)
    warm = validate_solution_csv(args.warm_start, data, policy)
    thickness = _infer_thickness(warm)
    target_mills = _mills_from_usd(args.target_total_usd)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_path = args.output_dir / "asignacion_optima.csv"
    initial_source = str(args.warm_start)
    incumbent = warm
    existing: ValidationResult | None = None
    if best_path.exists():
        existing = validate_solution_csv(best_path, data, policy)
        if _infer_thickness(existing) != thickness:
            raise ValueError("existing output uses a different global thickness")
        if existing.costs.total_mills < incumbent.costs.total_mills:
            incumbent = existing
            initial_source = str(best_path)

    start_costs = incumbent.costs
    snapshot_number = _next_snapshot_number(args.output_dir)
    start_snapshot = args.output_dir / f"incumbent_{snapshot_number:04d}.csv"
    checked_start = _atomic_write_assignment(
        start_snapshot, data, incumbent.assignment, policy
    )
    if checked_start.costs.total_mills != incumbent.costs.total_mills:
        raise RuntimeError("initial snapshot validation changed incumbent cost")
    if existing is None or incumbent.costs.total_mills < existing.costs.total_mills:
        _atomic_write_assignment(best_path, data, incumbent.assignment, policy)
    snapshot_number += 1

    exact_candidates, candidate_stats = generate_exact_candidates(
        data.products,
        thickness,
        retained_designs=_unique_internal_designs(incumbent.assignment),
    )
    print(
        f"LNS start: USD {incumbent.costs.total_mills / 1000:,.2f}; "
        f"{len(exact_candidates):,} exact candidates",
        flush=True,
    )
    attempts: list[dict[str, object]] = []
    improvements: list[dict[str, object]] = []
    summary_path = args.output_dir / "resumen_tier_lns.json"
    termination = "round_limit"

    summary: dict[str, object] = {
        "configuration": {
            "data_dir": str(args.data_dir),
            "warm_start": str(args.warm_start),
            "initial_source": initial_source,
            "thickness_mm": thickness,
            "time_per_neighborhood_seconds": args.time_per_neighborhood,
            "num_search_workers": args.num_search_workers,
            "max_extra_pallets": args.max_extra_pallets,
            "rounds": args.rounds,
            "target_total_usd": (
                target_mills / 1000 if target_mills is not None else None
            ),
            "target_mode": args.target_mode,
            "random_seed": args.random_seed,
        },
        "exact_candidate_stats": asdict(candidate_stats),
        "initial": _cost_payload(start_costs),
        "attempts": attempts,
        "improvements": improvements,
    }

    if target_mills is not None and incumbent.costs.total_mills <= target_mills:
        termination = "target_already_met"
    else:
        for round_index in range(args.rounds):
            plan = build_tier_neighborhoods(
                data,
                incumbent.assignment,
                exact_candidates,
                max_gap_units=args.max_gap_units,
                max_gap_ratio=args.max_gap_ratio,
                require_reachable=not args.include_unreachable,
                max_targets=args.max_targets,
                max_source_groups=args.max_source_groups,
                max_skus=args.max_star_skus,
            )
            work_items = generate_work_items(
                plan,
                max_star_combination=args.max_star_combination,
                max_component_combination=args.max_component_combination,
                max_skus=args.max_combined_skus,
                max_neighborhoods=args.max_neighborhoods,
            )
            if not work_items:
                termination = "no_neighborhoods"
                break

            print(
                f"Round {round_index + 1}/{args.rounds}: "
                f"{len(plan.targets)} targets, {len(work_items)} neighborhoods",
                flush=True,
            )

            improved_this_round = False
            for item_index, item in enumerate(work_items):
                before_mills = incumbent.costs.total_mills
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
                    target_total_mills=(
                        target_mills if args.target_mode == "hard" else None
                    ),
                    free_product_codes=item.product_codes,
                    precomputed_exact_candidates=exact_candidates,
                    precomputed_exact_candidate_stats=candidate_stats,
                )
                attempt: dict[str, object] = {
                    "round": round_index + 1,
                    "sequence": len(attempts) + 1,
                    "neighborhood_id": item.neighborhood_id,
                    "source_kind": item.source_kind,
                    "member_ids": item.member_ids,
                    "sku_count": item.sku_count,
                    "seed": seed,
                    "before_usd": before_mills / 1000,
                    **_result_payload(result),
                    "accepted": False,
                }

                if result.costs.total_mills < before_mills:
                    candidate_path = args.output_dir / ".candidate_validation.csv"
                    try:
                        checked = _atomic_write_assignment(
                            candidate_path, data, result.assignment, policy
                        )
                        if checked.costs.total_mills != result.costs.total_mills:
                            raise RuntimeError(
                                "independent validation does not match solver cost"
                            )
                    except (KeyError, RuntimeError, TypeError, ValueError) as exc:
                        attempt["validation_error"] = str(exc)
                    else:
                        incumbent = checked
                        snapshot_path = (
                            args.output_dir / f"incumbent_{snapshot_number:04d}.csv"
                        )
                        _atomic_write_assignment(
                            snapshot_path, data, incumbent.assignment, policy
                        )
                        _atomic_write_assignment(
                            best_path, data, incumbent.assignment, policy
                        )
                        snapshot_number += 1
                        improvement = {
                            "round": round_index + 1,
                            "attempt": len(attempts) + 1,
                            "neighborhood_id": item.neighborhood_id,
                            "sku_count": item.sku_count,
                            "before_usd": before_mills / 1000,
                            "after_usd": incumbent.costs.total_mills / 1000,
                            "saving_usd": (
                                before_mills - incumbent.costs.total_mills
                            ) / 1000,
                            "snapshot_path": str(snapshot_path),
                        }
                        improvements.append(improvement)
                        attempt["accepted"] = True
                        attempt["snapshot_path"] = str(snapshot_path)
                        improved_this_round = True
                        print(
                            f"  {item.neighborhood_id}: accepted USD "
                            f"{incumbent.costs.total_mills / 1000:,.2f} "
                            f"(saved USD {improvement['saving_usd']:,.2f})",
                            flush=True,
                        )
                    finally:
                        if candidate_path.exists():
                            candidate_path.unlink()

                attempts.append(attempt)
                summary["best"] = _cost_payload(incumbent.costs)
                summary["target_met"] = (
                    incumbent.costs.total_mills <= target_mills
                    if target_mills is not None
                    else None
                )
                _atomic_write_json(summary_path, summary)

                if not attempt["accepted"]:
                    print(
                        f"  {item.neighborhood_id}: {result.status}, no improvement "
                        f"({item.sku_count} SKUs, {result.wall_time_seconds:.1f}s)",
                        flush=True,
                    )

                if target_mills is not None and incumbent.costs.total_mills <= target_mills:
                    termination = "target_met"
                    break
                if improved_this_round:
    # Cambiaron los grupos de tipos de la incumbente; se reconstruye el plan
    # de tiers antes de liberar otro vecindario completo de grupos origen.
                    break

            if termination == "target_met":
                break

    summary["best"] = _cost_payload(incumbent.costs)
    summary["saving_usd"] = (
        start_costs.total_mills - incumbent.costs.total_mills
    ) / 1000
    summary["target_met"] = (
        incumbent.costs.total_mills <= target_mills
        if target_mills is not None
        else None
    )
    summary["termination"] = termination
    summary["best_path"] = str(best_path)
    _atomic_write_json(summary_path, summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Iterative exact tier-focused LNS for Bonsai Corp"
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--warm-start", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--time-per-neighborhood", type=float, default=60.0)
    parser.add_argument("--num-search-workers", type=int, default=6)
    parser.add_argument("--max-extra-pallets", type=int, default=500)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--target-total-usd", type=Decimal)
    parser.add_argument(
        "--target-mode",
        choices=("stop", "hard"),
        default="stop",
        help=(
            "stop: optimize each neighborhood and stop at the target; "
            "hard: ask every subproblem only for a solution at or below the target"
        ),
    )
    parser.add_argument("--extra-region-share", type=float, default=0.0)
    parser.add_argument("--max-gap-units", type=int, default=10_000)
    parser.add_argument("--max-gap-ratio", type=float)
    parser.add_argument("--max-targets", type=int, default=12)
    parser.add_argument("--max-source-groups", type=int, default=8)
    parser.add_argument("--max-star-skus", type=int, default=220)
    parser.add_argument("--max-combined-skus", type=int, default=220)
    parser.add_argument("--max-star-combination", type=int, default=2)
    parser.add_argument("--max-component-combination", type=int, default=2)
    parser.add_argument("--max-neighborhoods", type=int, default=64)
    parser.add_argument(
        "--include-unreachable",
        action="store_true",
        help="also schedule targets whose directly compatible donors do not fill the gap",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run_tier_lns(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through wrapper
    raise SystemExit(main())
