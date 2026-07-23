"""Run the bounded 12-hour Gurobi campaign for the strict 3-mm Bonsai model.

The campaign never extends itself beyond its configured duration.  It writes a
durable status JSON after every phase so it remains auditable and can be
resumed manually after an interruption.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class Phase:
    name: str
    seconds: int
    mip_focus: int
    seed: int
    target_constraint: bool = False
    best_obj_stop: bool = False
    heuristics: float | None = None
    no_rel_heur_time: float = 0.0
    cuts: int = -1
    solution_limit: int | None = None


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument(
        "--warm-start",
        type=Path,
        default=Path("output_lp_pool_after_ba_15m/asignacion_optima.csv"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("output_gurobi_exact_campaign")
    )
    parser.add_argument("--target-usd", type=float, default=188_079_300.0)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument(
        "--campaign-hours",
        type=float,
        default=12.0,
        help="hard upper bound; this script never auto-extends it",
    )
    return parser


def _phase_plan(hours: float) -> tuple[Phase, ...]:
    # The standard campaign totals 12 h.  When a shorter duration is requested,
    # preserve phase order and truncate the final phase rather than extending.
    standard = (
        Phase(
            "01_incumbent_hunt",
            2 * 3600,
            mip_focus=1,
            seed=2026072201,
            best_obj_stop=True,
            heuristics=0.20,
            no_rel_heur_time=120.0,
        ),
        Phase(
            "02_target_feasibility",
            2 * 3600,
            mip_focus=1,
            seed=2026072202,
            target_constraint=True,
            heuristics=0.25,
            no_rel_heur_time=180.0,
            solution_limit=1,
        ),
        Phase(
            "03_bound_push",
            3 * 3600,
            mip_focus=3,
            seed=2026072203,
            cuts=2,
        ),
        Phase(
            "04_diversified_primal",
            2 * 3600,
            mip_focus=1,
            seed=2026072204,
            best_obj_stop=True,
            heuristics=0.35,
            no_rel_heur_time=300.0,
            cuts=1,
        ),
        Phase(
            "05_exact_cleanup",
            3 * 3600,
            mip_focus=2,
            seed=2026072205,
            cuts=2,
        ),
    )
    remaining = round(hours * 3600)
    planned: list[Phase] = []
    for phase in standard:
        if remaining <= 0:
            break
        seconds = min(phase.seconds, remaining)
        planned.append(
            Phase(
                **{**asdict(phase), "seconds": seconds},
            )
        )
        remaining -= seconds
    return tuple(planned)


def _write_status(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _load_summary(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _target_reached(summary: dict[str, object], target_usd: float) -> bool:
    costs = summary.get("costs")
    return isinstance(costs, dict) and float(costs["total_usd"]) <= target_usd


def _target_ruled_out(summary: dict[str, object], target_usd: float) -> bool:
    # A valid MIP lower bound above the target proves no unseen solution can
    # reach it within the exact strict model.  Keep a one-dollar margin for
    # floating-point reporting of the transformed objective.
    bound = summary.get("best_bound_usd")
    return bound is not None and float(bound) > target_usd + 1.0


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.campaign_hours <= 0 or args.campaign_hours > 12.0:
        raise ValueError("campaign-hours must be in (0, 12]; extensions require explicit approval")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runner = Path(__file__).with_name("run_gurobi_proto_neighborhood.py")
    status_path = args.output_dir / "campaign_status.json"
    phases = _phase_plan(args.campaign_hours)
    status: dict[str, object] = {
        "target_usd": args.target_usd,
        "campaign_hours": args.campaign_hours,
        "threads": args.threads,
        "phases": [asdict(phase) for phase in phases],
        "completed": [],
        "termination": "running",
    }
    _write_status(status_path, status)

    for phase in phases:
        phase_dir = args.output_dir / phase.name
        phase_dir.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            str(runner),
            "--data-dir",
            str(args.data_dir),
            "--warm-start",
            str(args.warm_start),
            "--output-dir",
            str(phase_dir),
            "--free-products",
            "427",
            "--time-limit-seconds",
            str(phase.seconds),
            "--threads",
            str(args.threads),
            "--mip-focus",
            str(phase.mip_focus),
            "--mip-gap",
            "0",
            "--mip-gap-abs",
            "0",
            "--cuts",
            str(phase.cuts),
            "--seed",
            str(phase.seed),
            "--log-file",
            str(phase_dir / "gurobi.log"),
        ]
        if phase.target_constraint:
            command.extend(("--target-usd", str(args.target_usd)))
        if phase.best_obj_stop:
            command.extend(("--best-obj-stop-usd", str(args.target_usd)))
        if phase.heuristics is not None:
            command.extend(("--heuristics", str(phase.heuristics)))
        if phase.no_rel_heur_time:
            command.extend(("--no-rel-heur-time", str(phase.no_rel_heur_time)))
        if phase.solution_limit is not None:
            command.extend(("--solution-limit", str(phase.solution_limit)))

        started = time.time()
        with (phase_dir / "runner_stdout.log").open("w", encoding="utf-8") as output:
            completed = subprocess.run(command, stdout=output, stderr=subprocess.STDOUT)
        summary = _load_summary(phase_dir / "resumen_gurobi_proto.json")
        result: dict[str, object] = {
            "name": phase.name,
            "returncode": completed.returncode,
            "elapsed_seconds": time.time() - started,
            "summary": summary,
        }
        completed_phases = status["completed"]
        assert isinstance(completed_phases, list)
        completed_phases.append(result)

        if completed.returncode != 0:
            status["termination"] = "phase_error"
            status["failed_phase"] = phase.name
            _write_status(status_path, status)
            return status
        if summary is None:
            status["termination"] = "missing_phase_summary"
            status["failed_phase"] = phase.name
            _write_status(status_path, status)
            return status
        if _target_reached(summary, args.target_usd):
            status["termination"] = "target_reached"
            status["successful_phase"] = phase.name
            _write_status(status_path, status)
            return status
        if _target_ruled_out(summary, args.target_usd):
            status["termination"] = "target_ruled_out_by_bound"
            status["proof_phase"] = phase.name
            _write_status(status_path, status)
            return status
        _write_status(status_path, status)

    status["termination"] = "campaign_time_budget_exhausted"
    _write_status(status_path, status)
    return status


if __name__ == "__main__":
    print(json.dumps(run(_parser().parse_args()), ensure_ascii=False, indent=2))
