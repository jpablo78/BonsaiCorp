"""Interfaz de línea de comandos para el flujo de optimización de Bonsai."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from decimal import Decimal, ROUND_HALF_UP
import json
import os
from pathlib import Path
from uuid import uuid4

from .baseline import standardized_baseline
from .bounds import thickness_cost_lower_bound
from .config import ALLOWED_THICKNESSES, FreightPolicy
from .costs import evaluate_assignments, freight_pallets
from .data import audit_dataset, export_cleaned_sources, load_prepared_data
from .exact_candidates import generate_exact_candidates
from .heuristics import greedy_cover_assignment
from .lifecycle import (
    candidate_payload,
    evaluate_existing_type_assignment,
    infer_decimal_places,
    load_new_product,
    recalculated_catalog,
    write_incremental_assignment,
)
from .models import PLANTS
from .optimizer import SolveResult, solve_for_thickness
from .plateau import diversify_assignment
from .reporting import scenario_payload, write_assignment_csv, write_json
from .solution_validation import validate_solution_csv


def _freight_policy(args: argparse.Namespace) -> FreightPolicy:
    return FreightPolicy(extra_region_share=args.extra_region_share)


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument(
        "--extra-region-share",
        type=float,
        default=0.0,
        help="future scenario input: fraction of pallets charged at USD 500 instead of USD 150",
    )


def command_validate_data(args: argparse.Namespace) -> int:
    audit = audit_dataset(args.data_dir)
    print(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def command_clean_data(args: argparse.Namespace) -> int:
    payload = export_cleaned_sources(args.data_dir, args.output_dir)
    write_json(args.output_dir / "cleaning_report.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def command_baseline(args: argparse.Namespace) -> int:
    data = load_prepared_data(args.data_dir)
    policy = _freight_policy(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scenarios = []
    for thickness in ALLOWED_THICKNESSES:
        _, costs = standardized_baseline(data, thickness, policy)
        payload = scenario_payload("sin_consolidacion", thickness, costs)
        scenarios.append(payload)
        write_json(args.output_dir / f"baseline_{thickness:g}mm.json", payload)
    write_json(args.output_dir / "baselines.json", {"scenarios": scenarios})
    print(json.dumps({"scenarios": scenarios}, ensure_ascii=False, indent=2))
    return 0


def command_greedy(args: argparse.Namespace) -> int:
    data = load_prepared_data(args.data_dir)
    policy = _freight_policy(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for thickness in ALLOWED_THICKNESSES:
        assignment = greedy_cover_assignment(data, thickness)
        costs = evaluate_assignments(data.products, assignment, policy)
        path = args.output_dir / f"asignacion_greedy_{thickness:g}mm.csv"
        write_assignment_csv(path, data, assignment)
        results.append({"path": str(path), **scenario_payload("greedy", thickness, costs)})
    write_json(args.output_dir / "greedy.json", {"scenarios": results})
    print(json.dumps({"scenarios": results}, ensure_ascii=False, indent=2))
    return 0


def command_lower_bounds(args: argparse.Namespace) -> int:
    data = load_prepared_data(args.data_dir)
    policy = _freight_policy(args)
    bounds = [
        thickness_cost_lower_bound(data, thickness, policy).as_dict()
        for thickness in ALLOWED_THICKNESSES
    ]
    payload: dict[str, object] = {"thickness_lower_bounds": bounds}
    if args.incumbent is not None:
        incumbent = validate_solution_csv(args.incumbent, data, policy)
        payload["incumbent"] = incumbent.costs.as_dict()
        payload["thicknesses_proven_unable_to_beat_incumbent"] = [
            bound["thickness_mm"]
            for bound in bounds
            if bound["total_lower_bound_usd"]
            >= incumbent.costs.total_mills / 1000
        ]
    if args.output_path is not None:
        write_json(args.output_path, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def command_optimize(args: argparse.Namespace) -> int:
    if args.restarts < 1:
        raise ValueError("--restarts must be at least 1")
    if args.time_limit_seconds <= 0:
        raise ValueError("--time-limit-seconds must be positive")
    if (
        args.candidate_strategy != "exact"
        and (
            args.preserve_individual_max_capacity
            or args.max_extra_pallets is not None
            or args.plateau_walk_steps > 0
        )
    ):
        raise ValueError("pallet-capacity restrictions and plateau walks require exact candidates")
    if args.plateau_walk_steps < 0:
        raise ValueError("--plateau-walk-steps cannot be negative")
    if args.plateau_walk_steps and args.preserve_individual_max_capacity:
        raise ValueError("plateau walks cannot be combined with fixed individual capacity")
    data = load_prepared_data(args.data_dir)
    policy = _freight_policy(args)
    target_total_mills = (
        int(
            (args.target_total_usd * Decimal(1000)).quantize(
                Decimal("1"), rounding=ROUND_HALF_UP
            )
        )
        if args.target_total_usd is not None
        else None
    )
    initial_assignment = None
    if args.warm_start is not None:
        warm_start = validate_solution_csv(args.warm_start, data, policy)
        initial_assignment = warm_start.assignment
        if args.thickness_mm is None:
            raise ValueError("--warm-start requires --thickness-mm")

    solver_kwargs = {
        "num_search_workers": args.num_search_workers,
        "pair_profile_limit": args.pair_profile_limit,
        "max_extra_pair_designs": args.max_extra_pair_designs,
        "pallet_variant_profile_limit": args.pallet_variant_profile_limit,
        "max_pallet_variants_per_profile": args.max_pallet_variants_per_profile,
        "warm_start_variant_profile_limit": args.warm_start_variant_profile_limit,
        "warm_start_compromise_group_limit": args.warm_start_compromise_group_limit,
        "max_compromise_variants_per_group": args.max_compromise_variants_per_group,
        "candidate_strategy": args.candidate_strategy,
        "preserve_individual_max_capacity": args.preserve_individual_max_capacity,
        "max_extra_pallets": args.max_extra_pallets,
        "target_total_mills": target_total_mills,
    }
    if initial_assignment is not None:
        warm_thickness = next(iter(initial_assignment.values())).thickness_mm
        if warm_thickness != args.thickness_mm:
            raise ValueError(
                f"warm start uses {warm_thickness:g} mm, not requested {args.thickness_mm:g} mm"
            )

    plateau_candidates = None
    plateau_max_pallets = None
    plateau_walk_payloads: list[dict[str, object]] = []
    if args.plateau_walk_steps:
        if initial_assignment is None or args.thickness_mm is None:
            raise ValueError("plateau walks require --warm-start and --thickness-mm")
        plateau_candidates, _ = generate_exact_candidates(
            data.products,
            args.thickness_mm,
            retained_designs=tuple(
                sorted(
                    {candidate.internal for candidate in initial_assignment.values()},
                    key=lambda dimensions: dimensions.as_tuple(),
                )
            ),
        )
        if args.max_extra_pallets is not None:
            minimum_pallets = sum(
                min(
                    sum(freight_pallets(product, candidate, plant) for plant in PLANTS)
                    for candidate in plateau_candidates
                    if product.code in candidate.compatible_product_codes
                )
                for product in data.products
            )
            plateau_max_pallets = minimum_pallets + args.max_extra_pallets

    thicknesses = ALLOWED_THICKNESSES if args.thickness_mm is None else (args.thickness_mm,)
    run_results: list[SolveResult] = []
    final_results: list[SolveResult] = []
    seconds_per_restart = args.time_limit_seconds / args.restarts
    for thickness in thicknesses:
        incumbent = initial_assignment if thickness == args.thickness_mm else None
        thickness_runs: list[SolveResult] = []
        for restart_index in range(args.restarts):
            if plateau_candidates is not None and incumbent is not None:
                walk = diversify_assignment(
                    data.products,
                    incumbent,
                    plateau_candidates,
                    policy,
                    max_steps=args.plateau_walk_steps,
                    random_seed=args.random_seed + 104_729 * restart_index,
                    max_pallets=plateau_max_pallets,
                )
                incumbent = walk.assignment
                plateau_walk_payloads.append(
                    {
                        "restart_index": restart_index,
                        "random_seed": args.random_seed + 104_729 * restart_index,
                        "walk_steps": walk.walk_steps,
                        "visited_states": walk.visited_states,
                        "distance_from_start": walk.distance_from_start,
                        "moves_in_selected_path": len(walk.moves),
                        **walk.final_costs.as_dict(),
                    }
                )
            result = solve_for_thickness(
                data,
                thickness,
                policy,
                time_limit_seconds=seconds_per_restart,
                random_seed=args.random_seed + 1009 * restart_index,
                initial_assignment=incumbent,
                **solver_kwargs,
            )
            thickness_runs.append(result)
            run_results.append(result)
            incumbent = result.assignment
        final_results.append(
            min(thickness_runs, key=lambda result: result.costs.total_mills)
        )
    all_results = tuple(final_results)
    best = min(all_results, key=lambda result: result.costs.total_mills)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    submission_path = args.output_dir / "asignacion_optima.csv"
    latest_run_path = args.output_dir / "asignacion_ultima_corrida.csv"
    temporary_suffix = f"{os.getpid()}.{uuid4().hex}.pending.csv"
    pending_path = args.output_dir / f".asignacion_optima.{temporary_suffix}"
    write_assignment_csv(pending_path, data, best.assignment)
    independently_checked = validate_solution_csv(pending_path, data, policy)
    if independently_checked.costs.total_mills != best.costs.total_mills:
        raise RuntimeError("independent output validation does not match optimizer cost")

    latest_pending_path = args.output_dir / f".asignacion_ultima_corrida.{temporary_suffix}"
    write_assignment_csv(latest_pending_path, data, best.assignment)
    latest_checked = validate_solution_csv(latest_pending_path, data, policy)
    if latest_checked.costs.total_mills != best.costs.total_mills:
        raise RuntimeError("latest-run output validation does not match optimizer cost")
    os.replace(latest_pending_path, latest_run_path)

    output_action = "created"
    existing_costs = None
    existing_validation_error = None
    if submission_path.exists():
        try:
            existing = validate_solution_csv(submission_path, data, policy)
        except (KeyError, TypeError, ValueError) as exc:
            existing_validation_error = str(exc)
            pending_path.unlink()
            output_action = "kept_invalid_existing_manual_review"
        else:
            existing_costs = existing.costs
            if existing.costs.total_mills <= independently_checked.costs.total_mills:
                pending_path.unlink()
                output_action = "kept_existing_not_worse"
            else:
                os.replace(pending_path, submission_path)
                output_action = "replaced_with_improvement"
    else:
        os.replace(pending_path, submission_path)
    selected_output_path = (
        latest_run_path
        if output_action == "kept_invalid_existing_manual_review"
        else submission_path
    )
    final_output = validate_solution_csv(selected_output_path, data, policy)
    final_output_thickness = next(
        iter(final_output.assignment.values())
    ).thickness_mm

    def result_payload(result: SolveResult) -> dict[str, object]:
        return {
            "status": result.status,
            "candidate_count": result.candidate_count,
            "candidate_strategy": result.candidate_strategy,
            "candidate_stats": (
                asdict(result.candidate_stats) if result.candidate_stats is not None else None
            ),
            "solver_objective_usd": (
                result.solver_objective_mills / 1000
                if result.solver_objective_mills is not None
                else None
            ),
            "candidate_universe_best_bound_usd": (
                result.best_bound_mills / 1000
                if result.best_bound_mills is not None
                else None
            ),
            "candidate_universe_gap_percent": (
                result.candidate_universe_relative_gap * 100
                if result.candidate_universe_relative_gap is not None
                else None
            ),
            "wall_time_seconds": result.wall_time_seconds,
            "num_conflicts": result.num_conflicts,
            "num_branches": result.num_branches,
            "random_seed": result.random_seed,
            "incumbent_usd": result.incumbent_mills / 1000,
            "improved_incumbent": result.improved_incumbent,
            "selected_source": result.selected_source,
            "minimum_possible_pallets": result.minimum_possible_pallets,
            "max_extra_pallets": result.max_extra_pallets,
            "target_total_usd": (
                result.target_total_mills / 1000
                if result.target_total_mills is not None
                else None
            ),
            "target_met": result.target_met,
            "target_proven_infeasible": result.target_proven_infeasible,
            **scenario_payload("cp_sat", result.thickness_mm, result.costs),
        }

    result_payloads = [
        result_payload(result)
        for result in all_results
    ]
    payload = {
        "assumptions": {
            "demand_source": "operaciones_planta.csv",
            "extra_region_share": policy.extra_region_share,
            "perimeter": "external",
            "external_dimension_unit": "integer_mm",
            "product_model": "faq_10_volume_preserving_fit_with_positive_per_axis_headspace_cap",
            "target_total_usd": (
                target_total_mills / 1000 if target_total_mills is not None else None
            ),
        },
        "selected_thickness_mm": final_output_thickness,
        "submission_path": str(submission_path),
        "latest_run_path": str(latest_run_path),
        "selected_output_path": str(selected_output_path),
        "output_action": output_action,
        "previous_output_costs": existing_costs.as_dict() if existing_costs is not None else None,
        "existing_output_validation_error": existing_validation_error,
        "selected": scenario_payload(
            "best_validated_output", final_output_thickness, final_output.costs
        ),
        "run_selected": scenario_payload("cp_sat", best.thickness_mm, best.costs),
        "all_thickness_results": result_payloads,
        "restart_results": [result_payload(result) for result in run_results],
        "plateau_walks": plateau_walk_payloads,
    }
    write_json(args.output_dir / "resumen_optimizacion.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def command_validate_solution(args: argparse.Namespace) -> int:
    data = load_prepared_data(args.data_dir)
    result = validate_solution_csv(args.solution_path, data, _freight_policy(args))
    payload = result.costs.as_dict()
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def _cost_delta_payload(before, after) -> dict[str, object]:
    """Resume el efecto económico entre dos evaluaciones independientes."""

    return {
        "packaging_usd": (after.packaging_mills - before.packaging_mills) / 1000,
        "freight_usd": (after.freight_mills - before.freight_mills) / 1000,
        "total_usd": (after.total_mills - before.total_mills) / 1000,
        "pallets": after.pallets - before.pallets,
        "types": after.types - before.types,
    }


def command_recalculate_demand(args: argparse.Namespace) -> int:
    """Reevalúa una asignación fija contra un pronóstico de demanda nuevo."""

    policy = _freight_policy(args)
    current_data = load_prepared_data(args.data_dir)
    projected_data = load_prepared_data(
        args.data_dir, operations_override=args.operaciones_override
    )
    current = recalculated_catalog(args.solution, current_data, policy)
    projected = recalculated_catalog(args.solution, projected_data, policy)
    payload = {
        "mode": "recalcular_demanda_catalogo_fijo",
        "solution_path": str(args.solution),
        "operations_override": str(args.operaciones_override),
        "current_costs": current.costs.as_dict(),
        "projected_costs": projected.costs.as_dict(),
        "delta_projected_minus_current": _cost_delta_payload(
            current.costs, projected.costs
        ),
        "assignment_changes": 0,
        "new_box_types": 0,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "resultado_recalculo.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def command_incremental_onboarding(args: argparse.Namespace) -> int:
    """Asigna un SKU nuevo únicamente a tipos que ya están activos."""

    policy = _freight_policy(args)
    data = load_prepared_data(
        args.data_dir, operations_override=args.operaciones_override
    )
    product = load_new_product(args.new_product, set(data.product_by_code))
    decision = evaluate_existing_type_assignment(data, args.solution, product, policy)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    payload: dict[str, object] = {
        "mode": "alta_incremental_solo_tipos_existentes",
        "solution_path": str(args.solution),
        "operations_override": (
            str(args.operaciones_override)
            if args.operaciones_override is not None
            else None
        ),
        "new_product_path": str(args.new_product),
        "sku": product.code,
        "active_types_evaluated": decision.active_types_evaluated,
        "feasible_active_types": decision.feasible_active_types,
        "cost_before_usd": decision.baseline.costs.total_mills / 1000,
    }
    if decision.selected_candidate is None:
        payload.update(
            {
                "decision": "requiere_nuevo_diseno",
                "motivo": (
                    "Ninguno de los tipos de caja vigentes cumple simultáneamente "
                    "las restricciones de dimensiones, headspace, ECT y palletización."
                ),
                "tiers_afectados": [],
            }
        )
    else:
        output_path = args.output_dir / "asignacion_incremental.csv"
        checked = write_incremental_assignment(
            output_path,
            decision,
            decimal_places=infer_decimal_places(args.solution),
            freight_policy=policy,
        )
        payload.update(
            {
                "decision": "usar_tipo_existente",
                "tipo_elegido": candidate_payload(decision.selected_candidate),
                "cost_after_usd": checked.costs.total_mills / 1000,
                "costo_incremental_usd": (
                    checked.costs.total_mills - decision.baseline.costs.total_mills
                )
                / 1000,
                "tiers_afectados": list(decision.tier_changes),
                "output_csv": str(output_path),
            }
        )
    write_json(args.output_dir / "decision_alta_incremental.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bonsai Corp packaging optimization")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_data = subparsers.add_parser("validate-data", help="audit source CSVs")
    _add_common_arguments(validate_data)
    validate_data.set_defaults(handler=command_validate_data)

    clean_data = subparsers.add_parser(
        "clean-data", help="write normalized, non-destructive copies of source CSVs"
    )
    clean_data.add_argument("--data-dir", type=Path, default=Path("."))
    clean_data.add_argument("--output-dir", type=Path, default=Path("output/cleaned_data"))
    clean_data.set_defaults(handler=command_clean_data)

    baseline = subparsers.add_parser("baseline", help="compute feasible no-consolidation baselines")
    _add_common_arguments(baseline)
    baseline.add_argument("--output-dir", type=Path, default=Path("output"))
    baseline.set_defaults(handler=command_baseline)

    greedy = subparsers.add_parser("greedy", help="generate greedy feasible assignments")
    _add_common_arguments(greedy)
    greedy.add_argument("--output-dir", type=Path, default=Path("output"))
    greedy.set_defaults(handler=command_greedy)

    lower_bounds = subparsers.add_parser(
        "lower-bounds",
        help="compute rigorous cost lower bounds for all allowed thicknesses",
    )
    _add_common_arguments(lower_bounds)
    lower_bounds.add_argument(
        "--incumbent",
        type=Path,
        help="optional validated CSV used to eliminate thicknesses by bound",
    )
    lower_bounds.add_argument("--output-path", type=Path)
    lower_bounds.set_defaults(handler=command_lower_bounds)

    optimize = subparsers.add_parser("optimize", help="run one CP-SAT model per allowed thickness")
    _add_common_arguments(optimize)
    optimize.add_argument("--output-dir", type=Path, default=Path("output"))
    optimize.add_argument(
        "--time-limit-seconds",
        type=float,
        default=300.0,
        help="total CP-SAT time budget per thickness, split across restarts",
    )
    optimize.add_argument("--num-search-workers", type=int, default=8)
    optimize.add_argument(
        "--restarts",
        type=int,
        default=1,
        help="sequential seeded runs; each inherits the best protected incumbent",
    )
    optimize.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="CP-SAT search seed; use a different value for an independent restart",
    )
    optimize.add_argument(
        "--thickness-mm",
        type=float,
        choices=ALLOWED_THICKNESSES,
        help="solve only one permitted global thickness",
    )
    optimize.add_argument(
        "--warm-start",
        type=Path,
        help="validated solution CSV used as an incumbent hint for the same thickness",
    )
    optimize.add_argument(
        "--candidate-strategy",
        choices=("exact", "heuristic"),
        default="exact",
        help="exact enumerates the full documented integer-mm domain",
    )
    optimize.add_argument(
        "--preserve-individual-max-capacity",
        action="store_true",
        help=(
            "solve procurement exactly while fixing every SKU at its maximum feasible "
            "boxes-per-pallet capacity"
        ),
    )
    optimize.add_argument(
        "--max-extra-pallets",
        type=int,
        help=(
            "allow at most this many pallets above the per-SKU global minimum; "
            "use 0 to optimize discounts at the exact minimum pallet count"
        ),
    )
    optimize.add_argument(
        "--target-total-usd",
        type=Decimal,
        help=(
            "solve a pure feasibility model requiring total cost at or below this value; "
            "the protected incumbent is returned when the target is not reached"
        ),
    )
    optimize.add_argument(
        "--plateau-walk-steps",
        type=int,
        default=0,
        help=(
            "before each restart, diversify the warm start through this many "
            "exact zero-cost single-SKU moves"
        ),
    )
    optimize.add_argument(
        "--pair-profile-limit",
        type=int,
        default=0,
        help="opt-in: number of high-volume profiles used to form pair-max designs",
    )
    optimize.add_argument(
        "--max-extra-pair-designs",
        type=int,
        default=0,
        help="opt-in cap for pair-max candidate designs",
    )
    optimize.add_argument(
        "--pallet-variant-profile-limit",
        type=int,
        default=90,
        help="number of high-volume profiles used to create pallet-aligned variants",
    )
    optimize.add_argument(
        "--max-pallet-variants-per-profile",
        type=int,
        default=18,
        help="cap on pallet-aligned variants retained per source profile",
    )
    optimize.add_argument(
        "--warm-start-variant-profile-limit",
        type=int,
        default=60,
        help="number of high-volume incumbent designs expanded around pallet steps",
    )
    optimize.add_argument(
        "--warm-start-compromise-group-limit",
        type=int,
        default=40,
        help="number of incumbent SKU groups used to build common compromise boxes",
    )
    optimize.add_argument(
        "--max-compromise-variants-per-group",
        type=int,
        default=18,
        help="cap on common pallet-aligned designs retained per incumbent SKU group",
    )
    optimize.set_defaults(handler=command_optimize)

    validate_solution = subparsers.add_parser(
        "validate-solution", help="validate a Kaggle-format solution CSV independently"
    )
    _add_common_arguments(validate_solution)
    validate_solution.add_argument("solution_path", type=Path)
    validate_solution.set_defaults(handler=command_validate_solution)

    recalculate_demand = subparsers.add_parser(
        "recalcular-demanda",
        help="reevaluate a fixed validated catalog against projected demand",
    )
    _add_common_arguments(recalculate_demand)
    recalculate_demand.add_argument(
        "--solution", type=Path, required=True, help="validated catalog assignment CSV"
    )
    recalculate_demand.add_argument(
        "--operaciones-override",
        type=Path,
        required=True,
        help="projected operations CSV with the same schema as operaciones_planta.csv",
    )
    recalculate_demand.add_argument(
        "--output-dir", type=Path, required=True, help="directory for resultado_recalculo.json"
    )
    recalculate_demand.set_defaults(handler=command_recalculate_demand)

    onboarding = subparsers.add_parser(
        "alta-incremental",
        help="assign one new SKU exclusively to active existing box types",
    )
    _add_common_arguments(onboarding)
    onboarding.add_argument(
        "--solution", type=Path, required=True, help="validated catalog assignment CSV"
    )
    onboarding.add_argument(
        "--nuevo-producto",
        dest="new_product",
        type=Path,
        required=True,
        help="single-row CSV following the documented new-product contract",
    )
    onboarding.add_argument(
        "--operaciones-override",
        type=Path,
        help="optional projected demand for existing SKUs; defaults to operaciones_planta.csv",
    )
    onboarding.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="directory for the incremental decision and, if feasible, assignment CSV",
    )
    onboarding.set_defaults(handler=command_incremental_onboarding)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.handler(args)
