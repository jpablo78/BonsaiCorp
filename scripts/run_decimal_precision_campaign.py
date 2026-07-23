"""Run an auditable, hard-capped overnight decimal-precision campaign."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class Phase:
    name: str
    decimal_places: int
    seconds: int
    mip_focus: int
    cuts: int
    seed: int


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument(
        "--initial-warm-start",
        type=Path,
        default=Path(
            "output_gurobi_tenth_mm_diagnostic_5m/asignacion_tenth_mm_DIAGNOSTIC.csv"
        ),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("output_decimal_precision_campaign_8h")
    )
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--campaign-hours", type=float, default=8.0)
    return parser


def _write_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def _load_json(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _configured_phases() -> list[Phase]:
    return [
        Phase("01_tenth_deep", 1, 3600, 1, 2, 2026072401),
        Phase("02_hundredth_deep", 2, 10800, 1, 2, 2026072402),
        Phase("03_thousandth_deep", 3, 10800, 1, 2, 2026072403),
        Phase("04_thousandth_bound", 3, 2700, 2, 2, 2026072404),
    ]


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.campaign_hours <= 0 or args.campaign_hours > 8:
        raise ValueError("campaign-hours must be in (0, 8]")
    if args.threads < 1:
        raise ValueError("threads must be positive")
    if not args.initial_warm_start.exists():
        raise FileNotFoundError(args.initial_warm_start)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    status_path = args.output_dir / "campaign_status.json"
    runner = Path(__file__).with_name("run_gurobi_tenth_mm_diagnostic.py")
    started = time.time()
    deadline = started + args.campaign_hours * 3600
    warm_start = args.initial_warm_start
    best_total_usd: float | None = None
    completed: list[dict[str, object]] = []
    planned = _configured_phases()
    status: dict[str, object] = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "campaign_hours": args.campaign_hours,
        "hard_deadline_epoch": deadline,
        "threads": args.threads,
        "initial_warm_start": str(args.initial_warm_start),
        "planned_phases": [asdict(phase) for phase in planned],
        "completed": completed,
        "termination": "running",
        "current_phase": None,
        "best_output": str(warm_start),
        "best_total_usd": best_total_usd,
    }
    _write_json(status_path, status)

    phase_queue = list(planned)
    diversification_index = 0
    while phase_queue or deadline - time.time() > 300:
        remaining = deadline - time.time()
        if remaining <= 180:
            break
        if phase_queue:
            phase = phase_queue.pop(0)
        else:
            diversification_index += 1
            phase = Phase(
                f"05_thousandth_diversified_{diversification_index:02d}",
                3,
                3600,
                1,
                1 if diversification_index % 2 else 2,
                2026072500 + diversification_index,
            )
        # Reserve three minutes for CSV validation, JSON writes and clean exit.
        solve_seconds = max(1, min(phase.seconds, int(remaining - 180)))
        phase_dir = args.output_dir / phase.name
        phase_dir.mkdir(parents=True, exist_ok=True)
        stdout_path = phase_dir / "runner_stdout.log"
        stderr_path = phase_dir / "runner_stderr.log"
        command = [
            sys.executable,
            str(runner),
            "--data-dir",
            str(args.data_dir),
            "--warm-start",
            str(warm_start),
            "--output-dir",
            str(phase_dir),
            "--time-limit-seconds",
            str(solve_seconds),
            "--threads",
            str(args.threads),
            "--decimal-places",
            str(phase.decimal_places),
            "--mip-focus",
            str(phase.mip_focus),
            "--cuts",
            str(phase.cuts),
            "--seed",
            str(phase.seed),
        ]
        status["current_phase"] = {
            **asdict(phase),
            "actual_solve_seconds": solve_seconds,
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "warm_start": str(warm_start),
        }
        _write_json(status_path, status)

        phase_started = time.time()
        return_code: int | None = None
        error: str | None = None
        try:
            with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
                "w", encoding="utf-8"
            ) as stderr:
                result = subprocess.run(
                    command,
                    cwd=Path.cwd(),
                    stdout=stdout,
                    stderr=stderr,
                    timeout=solve_seconds + 150,
                    check=False,
                )
                return_code = result.returncode
        except subprocess.TimeoutExpired:
            error = "runner_timeout_after_solver_allowance"

        summary_path = phase_dir / "resumen_decimal.json"
        summary = _load_json(summary_path)
        output_path = phase_dir / "asignacion_decimal.csv"
        if summary is not None and output_path.exists():
            costs = summary.get("costs")
            total_usd = (
                float(costs["total_usd"])
                if isinstance(costs, dict) and "total_usd" in costs
                else None
            )
            if total_usd is not None and (
                best_total_usd is None or total_usd <= best_total_usd + 1e-9
            ):
                best_total_usd = total_usd
                warm_start = output_path
        phase_record: dict[str, object] = {
            **asdict(phase),
            "actual_solve_seconds": solve_seconds,
            "wall_time_seconds": time.time() - phase_started,
            "return_code": return_code,
            "error": error,
            "summary": summary,
            "output": str(output_path) if output_path.exists() else None,
        }
        completed.append(phase_record)
        status["completed"] = completed
        status["current_phase"] = None
        status["best_output"] = str(warm_start)
        status["best_total_usd"] = best_total_usd
        status["elapsed_seconds"] = time.time() - started
        _write_json(status_path, status)

        if (
            summary is not None
            and summary.get("optimal") is True
            and phase.decimal_places == 3
        ):
            status["termination"] = "optimal_at_0.001_mm"
            break

    if status["termination"] == "running":
        status["termination"] = "hard_campaign_budget_reached"
    status["finished_utc"] = datetime.now(timezone.utc).isoformat()
    status["elapsed_seconds"] = time.time() - started
    status["best_output"] = str(warm_start)
    status["best_total_usd"] = best_total_usd
    status["current_phase"] = None
    _write_json(status_path, status)
    return status


if __name__ == "__main__":
    print(json.dumps(run(_parser().parse_args()), ensure_ascii=False, indent=2))
