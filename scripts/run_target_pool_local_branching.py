"""Ramificación local exacta sobre un conjunto compacto y auditable de diseños objetivo.

The pool contains the incumbent, each SKU's best exact one-SKU alternatives,
and active physical box types that are closest to a documented Procurement
discount threshold.  Those signals only restrict a search neighbourhood; SCIP
still applies the full exact cost model and a CSV round-trip validates output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from bonsai.annealing import IncrementalAssignmentState
from bonsai.config import FreightPolicy, MILLS_PER_USD
from bonsai.data import load_prepared_data
from bonsai.exact_candidates import generate_exact_candidates
from bonsai.models import Dimensions
from bonsai.procurement_lns import rank_procurement_exposures
from bonsai.reporting import write_assignment_csv, write_json
from bonsai.scip_optimizer import solve_with_scip
from bonsai.solution_validation import validate_solution_csv


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--warm-start", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--time-limit-seconds", type=float, default=3_600.0)
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument("--random-seed", type=int, default=20260727)
    parser.add_argument("--target-usd", type=float, default=188_078_500.0)
    parser.add_argument("--max-changed-products", type=int, default=40)
    parser.add_argument("--min-changed-products", type=int)
    parser.add_argument("--single-targets-per-sku", type=int, default=8)
    parser.add_argument("--priority-exposure-count", type=int, default=24)
    parser.add_argument("--memory-limit-mb", type=int, default=4_000)
    parser.add_argument("--scip-parameter", action="append", default=[])
    return parser


def _build_allowed_pool(
    data,
    assignment,
    candidates,
    policy: FreightPolicy,
    *,
    single_targets_per_sku: int,
    priority_exposure_count: int,
) -> tuple[dict[str, frozenset[Dimensions]], dict[str, object]]:
    if single_targets_per_sku < 0 or priority_exposure_count < 0:
        raise ValueError("pool sizes cannot be negative")
    state = IncrementalAssignmentState(data.products, assignment, policy)
    by_code: dict[str, list] = {product.code: [] for product in data.products}
    for candidate in candidates:
        for code in candidate.compatible_product_codes:
            by_code[code].append(candidate)

# Los deltas incrementales exactos ordenan alternativas, incluido su efecto
# actual en Procurement en todas las plantas. No aproximan lo que optimiza
# posteriormente el MIP.
    allowed: dict[str, set[Dimensions]] = {
        product.code: {assignment[product.code].internal} for product in data.products
    }
    for product in data.products:
        alternatives = []
        for candidate in by_code[product.code]:
            move = state.calculate_move(product.code, candidate)
            if move is not None:
                alternatives.append(
                    (move.total_delta_mills, move.pallet_delta, candidate.internal)
                )
        alternatives.sort(key=lambda item: (item[0], item[1], item[2].as_tuple()))
        allowed[product.code].update(
            internal for _, _, internal in alternatives[:single_targets_per_sku]
        )

    exposures = rank_procurement_exposures(data.products, assignment, candidates)
    priority_internals: list[Dimensions] = []
    seen: set[Dimensions] = set()
    for exposure in exposures:
        if exposure.internal in seen:
            continue
        priority_internals.append(exposure.internal)
        seen.add(exposure.internal)
        if len(priority_internals) >= priority_exposure_count:
            break
    candidate_by_internal = {candidate.internal: candidate for candidate in candidates}
    for internal in priority_internals:
        candidate = candidate_by_internal[internal]
        for code in candidate.compatible_product_codes:
            allowed[code].add(internal)

    payload = {
        "single_targets_per_sku": single_targets_per_sku,
        "priority_exposure_count": priority_exposure_count,
        "priority_internals": [internal.as_tuple() for internal in priority_internals],
        "target_count_by_sku": {code: len(items) for code, items in allowed.items()},
        "average_targets_per_sku": sum(map(len, allowed.values())) / len(allowed),
    }
    return {code: frozenset(items) for code, items in allowed.items()}, payload


def run(args: argparse.Namespace) -> dict[str, object]:
    data = load_prepared_data(args.data_dir)
    policy = FreightPolicy()
    warm = validate_solution_csv(args.warm_start, data, policy)
    candidates, candidate_stats = generate_exact_candidates(
        data.products,
        3.0,
        retained_designs=tuple({box.internal for box in warm.assignment.values()}),
    )
    allowed, pool_payload = _build_allowed_pool(
        data,
        warm.assignment,
        candidates,
        policy,
        single_targets_per_sku=args.single_targets_per_sku,
        priority_exposure_count=args.priority_exposure_count,
    )
    target_mills = round(args.target_usd * MILLS_PER_USD)
    print(
        f"Target-pool local branch: {sum(map(len, allowed.values())):,} arcs; "
        f"target <= USD {target_mills / 1000:,.2f}",
        flush=True,
    )
    result = solve_with_scip(
        data,
        3.0,
        policy,
        time_limit_seconds=args.time_limit_seconds,
        num_threads=args.num_threads,
        random_seed=args.random_seed,
        initial_assignment=warm.assignment,
        target_total_mills=target_mills,
        min_changed_products=args.min_changed_products,
        max_changed_products=args.max_changed_products,
        allowed_internals_by_product=allowed,
        precomputed_exact_candidates=candidates,
        precomputed_exact_candidate_stats=candidate_stats,
        memory_limit_mb=args.memory_limit_mb,
        scip_parameters="\n".join(args.scip_parameter) or None,
        progress_callback=lambda message: print(f"[pool] {message}", flush=True),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "asignacion_optima.csv"
    write_assignment_csv(output_path, data, result.assignment)
    checked = validate_solution_csv(output_path, data, policy)
    if checked.costs.total_mills != result.costs.total_mills:
        raise AssertionError("CSV round trip changed the independently evaluated cost")
    payload: dict[str, object] = {
        "configuration": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "candidate_stats": asdict(candidate_stats),
        "pool": pool_payload,
        "warm_start": warm.costs.as_dict(),
        "result": {
            "status": result.status,
            "selected_source": result.selected_source,
            "costs": checked.costs.as_dict(),
            "target_met": checked.costs.total_mills <= target_mills,
            "changed_product_count": result.changed_product_count,
            "nodes": result.nodes,
            "solve_time_seconds": result.solve_time_seconds,
            "output_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest().upper(),
        },
    }
    write_json(args.output_dir / "resumen_target_pool_local_branching.json", payload)
    return payload


if __name__ == "__main__":
    print(json.dumps(run(_parser().parse_args()), ensure_ascii=False, indent=2))
