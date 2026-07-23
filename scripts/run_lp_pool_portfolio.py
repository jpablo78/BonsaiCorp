"""Run several LP-guided exact-pool searches concurrently and validate them."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import multiprocessing
import os
from types import SimpleNamespace
from uuid import uuid4

from bonsai.config import FreightPolicy
from bonsai.data import load_prepared_data
from bonsai.reporting import write_assignment_csv, write_json
from bonsai.solution_validation import validate_solution_csv


PORTFOLIO = (
    (20260724, (2, 3, 4, 6, 8, 12, 16, 24)),
    (20260725, (3, 5, 7, 11, 17, 27, 43)),
    (20260726, (4, 8, 16, 32, 48, 64)),
)


def _worker(
    data_dir: str,
    warm_start: str,
    output_dir: str,
    total_time_seconds: float,
    seed: int,
    pool_sizes: tuple[int, ...],
):
    # Imported in the spawned child so Windows does not execute the runner at
    # module-import time in the parent process.
    import sys

    sys.path.insert(0, str(Path(__file__).parent))
    from run_lp_pool import run

    return run(
        SimpleNamespace(
            data_dir=Path(data_dir),
            warm_start=Path(warm_start),
            output_dir=Path(output_dir),
            total_time_seconds=total_time_seconds,
            lp_time_seconds=120.0,
            max_extra_pallets=5_000,
            pool_sizes=list(pool_sizes),
            num_threads=2,
            random_seed=seed,
            memory_limit_mb=3_500,
        )
    )


def _write_checked(path: Path, data, policy, assignment):
    pending = path.parent / f".{path.name}.{os.getpid()}.{uuid4().hex}.pending"
    write_assignment_csv(pending, data, assignment)
    checked = validate_solution_csv(pending, data, policy)
    os.replace(pending, path)
    return checked


def main() -> int:
    parser = argparse.ArgumentParser(description="Parallel LP-pool portfolio")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--warm-start", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--total-time-seconds", type=float, default=5_400.0)
    args = parser.parse_args()

    data = load_prepared_data(args.data_dir)
    policy = FreightPolicy()
    baseline = validate_solution_csv(args.warm_start, data, policy)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_path = args.output_dir / "asignacion_optima.csv"
    _write_checked(best_path, data, policy, baseline.assignment)

    context = multiprocessing.get_context("spawn")
    results = []
    with ProcessPoolExecutor(max_workers=3, mp_context=context) as pool:
        futures = {
            pool.submit(
                _worker,
                str(args.data_dir),
                str(args.warm_start),
                str(args.output_dir / f"portfolio_{index}"),
                args.total_time_seconds,
                seed,
                pools,
            ): (index, seed, pools)
            for index, (seed, pools) in enumerate(PORTFOLIO, start=1)
        }
        for future in as_completed(futures):
            index, seed, pools = futures[future]
            payload = future.result()
            output = args.output_dir / f"portfolio_{index}" / "asignacion_optima.csv"
            checked = validate_solution_csv(output, data, policy)
            results.append(
                {
                    "portfolio": index,
                    "seed": seed,
                    "pool_sizes": list(pools),
                    "costs": checked.costs.as_dict(),
                    "saving_usd": (baseline.costs.total_mills - checked.costs.total_mills) / 1000,
                    "path": str(output),
                    "payload": payload,
                }
            )
    winner = min(results, key=lambda item: item["costs"]["total_usd"])
    winner_checked = validate_solution_csv(winner["path"], data, policy)
    if winner_checked.costs.total_mills < baseline.costs.total_mills:
        _write_checked(best_path, data, policy, winner_checked.assignment)
    final = validate_solution_csv(best_path, data, policy)
    write_json(
        args.output_dir / "resumen_lp_pool_portfolio.json",
        {
            "baseline": baseline.costs.as_dict(),
            "results": results,
            "best": final.costs.as_dict(),
            "best_path": str(best_path),
            "independently_validated": True,
        },
    )
    print(final.costs.as_dict())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
