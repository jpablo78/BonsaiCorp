"""Ejecuta una secuencia guiada por LP de maestros SCIP exactos restringidos."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time
from uuid import uuid4

from bonsai.config import FreightPolicy
from bonsai.costs import evaluate_assignments
from bonsai.data import load_prepared_data
from bonsai.exact_candidates import generate_exact_candidates
from bonsai.lp_pool import build_lp_candidate_pools, round_lp_assignment
from bonsai.reporting import write_assignment_csv, write_json
from bonsai.scip_optimizer import solve_with_scip
from bonsai.solution_validation import validate_solution_csv


def _write_checked(path, data, policy, assignment):
    pending = path.parent / f".{path.name}.{os.getpid()}.{uuid4().hex}.pending"
    write_assignment_csv(pending, data, assignment)
    checked = validate_solution_csv(pending, data, policy)
    os.replace(pending, path)
    return checked


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LP-guided SCIP pool heuristic")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--warm-start", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--total-time-seconds", type=float, default=600)
    parser.add_argument("--lp-time-seconds", type=float, default=90)
    parser.add_argument("--max-extra-pallets", type=int, default=5000)
    parser.add_argument("--pool-sizes", type=int, nargs="+", default=[4, 8, 12, 20, 32])
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument("--random-seed", type=int, default=4049)
    parser.add_argument("--memory-limit-mb", type=int)
    return parser


def run(args: argparse.Namespace) -> dict[str, object]:
    started = time.perf_counter()
    data = load_prepared_data(args.data_dir)
    policy = FreightPolicy()
    initial = validate_solution_csv(args.warm_start, data, policy)
    incumbent = initial
    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_path = args.output_dir / "asignacion_optima.csv"
    _write_checked(best_path, data, policy, incumbent.assignment)

    retained = tuple({box.internal for box in incumbent.assignment.values()})
    candidates, candidate_stats = generate_exact_candidates(
        data.products, 3.0, retained_designs=retained
    )
    print(f"Exact universe: {len(candidates):,} candidates", flush=True)
    lp = solve_with_scip(
        data,
        3.0,
        policy,
        time_limit_seconds=min(args.lp_time_seconds, args.total_time_seconds),
        num_threads=1,
        random_seed=args.random_seed,
        initial_assignment=incumbent.assignment,
        max_extra_pallets=args.max_extra_pallets,
        precomputed_exact_candidates=candidates,
        precomputed_exact_candidate_stats=candidate_stats,
        relax_integrality=True,
        memory_limit_mb=args.memory_limit_mb,
        progress_callback=lambda message: print(message, flush=True),
    )
    if lp.status not in {"OPTIMAL", "FEASIBLE"} or not lp.assignment_arc_values:
        raise RuntimeError(f"LP relaxation did not return an arc solution: {lp.status}")
    print(
        f"LP: USD {lp.solver_objective_mills / 1000:,.2f}; "
        f"{sum(value > 1e-7 for value in lp.assignment_arc_values.values()):,} positive arcs",
        flush=True,
    )

    attempts: list[dict[str, object]] = []
    rounded_assignment = round_lp_assignment(
        data.products, candidates, incumbent.assignment, lp.assignment_arc_values
    )
    rounded_costs = evaluate_assignments(data.products, rounded_assignment, policy)
    rounded_budget_ok = (
        lp.minimum_possible_pallets is None
        or rounded_costs.pallets
        <= lp.minimum_possible_pallets + args.max_extra_pallets
    )
    attempts.append(
        {
            "kind": "independent_lp_rounding",
            "costs": rounded_costs.as_dict(),
            "pallet_budget_ok": rounded_budget_ok,
            "accepted": False,
        }
    )
    if rounded_budget_ok and rounded_costs.total_mills < incumbent.costs.total_mills:
        checked = _write_checked(best_path, data, policy, rounded_assignment)
        if checked.costs.total_mills != rounded_costs.total_mills:
            raise RuntimeError("rounded assignment cost changed after CSV validation")
        incumbent = checked
        attempts[-1]["accepted"] = True
        print(f"Rounded LP accepted: USD {incumbent.costs.total_mills / 1000:,.2f}", flush=True)

    for attempt_index, pool_size in enumerate(args.pool_sizes):
        elapsed = time.perf_counter() - started
        remaining = args.total_time_seconds - elapsed
        remaining_attempts = len(args.pool_sizes) - attempt_index
        if remaining < 2:
            break
        pools, stats = build_lp_candidate_pools(
            data.products,
            candidates,
            incumbent.assignment,
            lp.assignment_arc_values,
            lp.assignment_arc_reduced_costs_mills,
            pool_size=pool_size,
        )
        solve_seconds = max(1.0, remaining / remaining_attempts)
        before = incumbent.costs.total_mills
        print(
            f"Pool {pool_size}: {stats.total_arcs:,} arcs, "
            f"SCIP {solve_seconds:.1f}s",
            flush=True,
        )
        result = solve_with_scip(
            data,
            3.0,
            policy,
            time_limit_seconds=solve_seconds,
            num_threads=args.num_threads,
            random_seed=args.random_seed + attempt_index * 1009,
            initial_assignment=incumbent.assignment,
            max_extra_pallets=args.max_extra_pallets,
            allowed_internals_by_product=pools,
            precomputed_exact_candidates=candidates,
            precomputed_exact_candidate_stats=candidate_stats,
            memory_limit_mb=args.memory_limit_mb,
            progress_callback=lambda message: print(message, flush=True),
        )
        accepted = result.costs.total_mills < before
        if accepted:
            checked = _write_checked(best_path, data, policy, result.assignment)
            if checked.costs.total_mills != result.costs.total_mills:
                raise RuntimeError("MIP pool cost changed after CSV validation")
            incumbent = checked
            snapshot = args.output_dir / f"incumbent_pool_{pool_size}.csv"
            _write_checked(snapshot, data, policy, incumbent.assignment)
            print(f"Accepted: USD {incumbent.costs.total_mills / 1000:,.2f}", flush=True)
        attempts.append(
            {
                "kind": "restricted_mip",
                "pool_size": pool_size,
                "pool_stats": vars(stats),
                "status": result.status,
                "before_usd": before / 1000,
                "after_usd": result.costs.total_mills / 1000,
                "accepted": accepted,
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
                "solve_time_seconds": result.solve_time_seconds,
            }
        )

    checked = validate_solution_csv(best_path, data, policy)
    payload = {
        "method": "full LP relaxation -> consolidation-aware per-SKU pools -> exact SCIP MIPs",
        "initial_costs": initial.costs.as_dict(),
        "best_costs": checked.costs.as_dict(),
        "saving_usd": (initial.costs.total_mills - checked.costs.total_mills) / 1000,
        "best_path": str(best_path),
        "best_sha256": hashlib.sha256(best_path.read_bytes()).hexdigest().upper(),
        "independently_validated": True,
        "lp": {
            "status": lp.status,
            "objective_usd": (
                lp.solver_objective_mills / 1000
                if lp.solver_objective_mills is not None
                else None
            ),
            "arc_count": len(lp.assignment_arc_values),
            "positive_arc_count": sum(
                value > 1e-7 for value in lp.assignment_arc_values.values()
            ),
            "solve_time_seconds": lp.solve_time_seconds,
        },
        "max_extra_pallets": args.max_extra_pallets,
        "attempts": attempts,
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_json(args.output_dir / "resumen_lp_pool.json", payload)
    return payload


def main(argv=None):
    args = build_parser().parse_args(argv)
    print(json.dumps(run(args), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
