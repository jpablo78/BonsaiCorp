"""Parallel, exact SCIP neighbourhood batches for the Bonsai incumbent.

One SCIP neighbourhood often cannot keep every CPU core busy because its tree
is small.  This runner solves independent plant combinations concurrently,
each from the same validated incumbent, then promotes only the independently
validated best result before starting the next batch.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import multiprocessing
import os
from pathlib import Path
import time
from uuid import uuid4

from bonsai.config import FreightPolicy
from bonsai.data import load_prepared_data
from bonsai.exact_candidates import generate_exact_candidates
from bonsai.models import CandidateBox, PLANTS, PreparedData
from bonsai.reporting import write_assignment_csv, write_json
from bonsai.scip_optimizer import solve_with_scip
from bonsai.solution_validation import validate_solution_csv


TARGET_TOTAL_MILLS = 188_079_000_000

PAIRS = (
    ("curitiba", "monterrey"),
    ("curitiba", "bakersfield"),
    ("buenos_aires", "santiago"),
    ("buenos_aires", "monterrey"),
    ("buenos_aires", "bakersfield"),
    ("santiago", "monterrey"),
    ("santiago", "bakersfield"),
    ("monterrey", "bakersfield"),
    ("buenos_aires", "curitiba"),
    ("curitiba", "santiago"),
)
TRIPLES = (
    ("curitiba", "monterrey", "bakersfield"),
    ("buenos_aires", "curitiba", "monterrey"),
    ("buenos_aires", "curitiba", "bakersfield"),
    ("curitiba", "santiago", "monterrey"),
    ("curitiba", "santiago", "bakersfield"),
    ("buenos_aires", "monterrey", "bakersfield"),
    ("buenos_aires", "santiago", "monterrey"),
    ("buenos_aires", "santiago", "bakersfield"),
    ("santiago", "monterrey", "bakersfield"),
    ("buenos_aires", "curitiba", "santiago"),
)
QUADS = (
    ("buenos_aires", "curitiba", "monterrey", "bakersfield"),
    ("buenos_aires", "curitiba", "santiago", "monterrey"),
    ("buenos_aires", "curitiba", "santiago", "bakersfield"),
    ("buenos_aires", "santiago", "monterrey", "bakersfield"),
    ("curitiba", "santiago", "monterrey", "bakersfield"),
)


def _slug(plants: tuple[str, ...]) -> str:
    return "_".join(plants)


def _chunks(items: tuple[tuple[str, ...], ...], size: int):
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _free_codes(data: PreparedData, plants: tuple[str, ...]) -> frozenset[str]:
    return frozenset(
        product.code
        for product in data.products
        if any(product.annual_volume_by_plant[plant] > 0 for plant in plants)
    )


def _worker(
    data_dir: str,
    assignment: dict[str, CandidateBox],
    plants: tuple[str, ...],
    limit_seconds: float,
    num_threads: int,
    memory_limit_mb: int,
    random_seed: int,
    scip_parameters: str | None,
):
    """Worker payload intentionally contains no output path or file writes."""

    data = load_prepared_data(Path(data_dir))
    candidates, stats = generate_exact_candidates(
        data.products,
        3.0,
        retained_designs=tuple({box.internal for box in assignment.values()}),
    )
    result = solve_with_scip(
        data,
        3.0,
        FreightPolicy(),
        time_limit_seconds=limit_seconds,
        num_threads=num_threads,
        random_seed=random_seed,
        initial_assignment=assignment,
        free_product_codes=_free_codes(data, plants),
        precomputed_exact_candidates=candidates,
        precomputed_exact_candidate_stats=stats,
        max_extra_pallets=5_000,
        memory_limit_mb=memory_limit_mb,
        scip_parameters=scip_parameters,
        progress_callback=None,
    )
    return {
        "plants": plants,
        "assignment": result.assignment,
        "status": result.status,
        "nodes": result.nodes,
        "solve_time_seconds": result.solve_time_seconds,
        "best_bound_mills": result.best_bound_mills,
        "free_product_count": result.fixed_product_count,
    }


def _write_checked(
    path: Path,
    data: PreparedData,
    policy: FreightPolicy,
    assignment: dict[str, CandidateBox],
):
    pending = path.parent / f".{path.name}.{os.getpid()}.{uuid4().hex}.pending"
    write_assignment_csv(pending, data, assignment)
    checked = validate_solution_csv(pending, data, policy)
    os.replace(pending, path)
    return checked


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Parallel exact Bonsai neighbourhood batches")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--warm-start", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--total-time-seconds", type=float, default=28_800.0)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--threads-per-worker", type=int, default=2)
    parser.add_argument("--memory-limit-per-worker-mb", type=int, default=3_500)
    parser.add_argument("--random-seed", type=int, default=20260722)
    parser.add_argument(
        "--scip-parameter",
        action="append",
        default=[],
        help="raw SCIP parameter line; may be repeated",
    )
    parser.add_argument(
        "--start-stage",
        choices=("pairs_a", "triples_a", "quads", "global_a", "pairs_b", "triples_b", "global_b"),
        help="skip earlier stages when running a focused portfolio",
    )
    parser.add_argument(
        "--end-stage",
        choices=("pairs_a", "triples_a", "quads", "global_a", "pairs_b", "triples_b", "global_b"),
        help="stop after the named stage when running a focused portfolio",
    )
    parser.add_argument("--stop-at-target", action="store_true")
    return parser


def _configuration(args: argparse.Namespace) -> dict[str, object]:
    return {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}


def _stages() -> tuple[tuple[str, tuple[tuple[str, ...], ...], float], ...]:
    """Ordered batches; theoretical duration exceeds 8h, then total limit stops."""

    return (
        ("pairs_a", PAIRS, 900.0),
        ("triples_a", TRIPLES, 1200.0),
        ("quads", QUADS, 1800.0),
        ("global_a", (PLANTS, PLANTS, PLANTS), 5400.0),
        ("pairs_b", PAIRS, 900.0),
        ("triples_b", TRIPLES, 1200.0),
        ("global_b", (PLANTS, PLANTS, PLANTS), 5400.0),
    )


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.total_time_seconds <= 0 or args.workers < 1 or args.threads_per_worker < 1:
        raise ValueError("time, workers and threads must be positive")
    started = time.perf_counter()
    data = load_prepared_data(args.data_dir)
    policy = FreightPolicy()
    incumbent = validate_solution_csv(args.warm_start, data, policy)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_path = args.output_dir / "asignacion_optima.csv"
    _write_checked(best_path, data, policy, incumbent.assignment)
    records: list[dict[str, object]] = []
    step = 0
    started_stage = args.start_stage is None
    scip_parameters = "\n".join(args.scip_parameter) or None

    context = multiprocessing.get_context("spawn")
    for stage_name, groups, per_problem_seconds in _stages():
        if not started_stage:
            if stage_name != args.start_stage:
                continue
            started_stage = True
        for batch_index, batch in enumerate(_chunks(groups, args.workers), start=1):
            remaining = args.total_time_seconds - (time.perf_counter() - started)
            if remaining <= 2.0:
                break
            limit = min(per_problem_seconds, remaining)
            before = incumbent.costs.total_mills
            print(f"[{stage_name}:{batch_index}] starting {len(batch)} workers at {limit:.0f}s", flush=True)
            answers: list[dict[str, object]] = []
            with ProcessPoolExecutor(max_workers=len(batch), mp_context=context) as pool:
                futures = [
                    pool.submit(
                        _worker,
                        str(args.data_dir),
                        incumbent.assignment,
                        tuple(group),
                        limit,
                        args.threads_per_worker,
                        args.memory_limit_per_worker_mb,
                        args.random_seed + step + offset,
                        scip_parameters,
                    )
                    for offset, group in enumerate(batch, start=1)
                ]
                for future in as_completed(futures):
                    answers.append(future.result())
            candidates: list[tuple[dict[str, object], object, Path]] = []
            for answer in answers:
                step += 1
                plants = tuple(answer["plants"])
                path = args.output_dir / "snapshots" / f"{step:03d}_{stage_name}_{_slug(plants)}" / "asignacion_optima.csv"
                checked = _write_checked(path, data, policy, answer["assignment"])
                if checked.costs.total_mills > before:
                    raise RuntimeError("worker returned a worse-than-incumbent assignment")
                record = {
                    "step": step,
                    "stage": stage_name,
                    "plants": list(plants),
                    "before_total_usd": before / 1000,
                    "after_total_usd": checked.costs.total_mills / 1000,
                    "saving_usd": (before - checked.costs.total_mills) / 1000,
                    "status": answer["status"],
                    "nodes": answer["nodes"],
                    "solve_time_seconds": answer["solve_time_seconds"],
                    "best_bound_usd": (
                        answer["best_bound_mills"] / 1000
                        if answer["best_bound_mills"] is not None
                        else None
                    ),
                    "snapshot": str(path),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest().upper(),
                }
                records.append(record)
                candidates.append((record, checked, path))
            winner = min(candidates, key=lambda item: item[1].costs.total_mills)
            if winner[1].costs.total_mills < incumbent.costs.total_mills:
                incumbent = winner[1]
                _write_checked(best_path, data, policy, incumbent.assignment)
                print(f"[{stage_name}:{batch_index}] improved to {incumbent.costs.total_mills / 1000:,.2f}", flush=True)
            else:
                print(f"[{stage_name}:{batch_index}] no improvement", flush=True)
            write_json(
                args.output_dir / "resumen_parallel.json",
                {
                    "configuration": _configuration(args),
                    "elapsed_seconds": time.perf_counter() - started,
                    "records": records,
                    "best": incumbent.costs.as_dict(),
                    "best_path": str(best_path),
                    "best_sha256": hashlib.sha256(best_path.read_bytes()).hexdigest().upper(),
                    "target_total_usd": TARGET_TOTAL_MILLS / 1000,
                    "target_met": incumbent.costs.total_mills <= TARGET_TOTAL_MILLS,
                },
            )
            if args.stop_at_target and incumbent.costs.total_mills <= TARGET_TOTAL_MILLS:
                return {"target_met": True, "best": incumbent.costs.as_dict(), "steps": step}
        if args.end_stage == stage_name:
            break
        else:
            continue
        break

    final = validate_solution_csv(best_path, data, policy)
    return {
        "target_met": final.costs.total_mills <= TARGET_TOTAL_MILLS,
        "best": final.costs.as_dict(),
        "best_path": str(best_path),
        "steps": step,
        "elapsed_seconds": time.perf_counter() - started,
    }


if __name__ == "__main__":
    print(run(_parser().parse_args()))
