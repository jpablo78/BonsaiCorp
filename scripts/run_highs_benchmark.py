"""Run the exported exact Bonsai master with the HiGHS MIP backend.

The production master is built by :mod:`bonsai.scip_optimizer` because it
already contains the independently reviewed linearisation of procurement
tiers.  OR-Tools exports that exact model as LP and HiGHS imports it without
translation.  The interchange artifact is the lossless OR-Tools MPModelProto,
rather than LP or MPS, because their text writers round large integer demand
coefficients.  This script then reconstructs the protected incumbent as a
complete MIP start, including the volume and tier auxiliary columns.

The resulting CSV is always independently checked with the project evaluator;
if HiGHS has not produced a strictly cheaper feasible solution, the supplied
warm start is retained.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re
import time

import numpy as np

from bonsai.config import DISCOUNT_TIERS, FreightPolicy
from bonsai.data import load_prepared_data
from bonsai.exact_candidates import generate_exact_candidates
from bonsai.reporting import write_assignment_csv, write_json
from bonsai.scip_optimizer import solve_with_scip
from bonsai.solution_validation import validate_solution_csv


_ASSIGNMENT_NAME = re.compile(r"^x_p(?P<product>\d+)_c(?P<candidate>\d+)$")
_VOLUME_NAME = re.compile(r"^volume_c(?P<candidate>\d+)_(?P<plant>.+)$")
_REACHED_NAME = re.compile(
    r"^reached_c(?P<candidate>\d+)_(?P<plant>.+)_t(?P<tier>\d+)$"
)
_DISCOUNTED_VOLUME_NAME = re.compile(
    r"^discounted_volume_c(?P<candidate>\d+)_(?P<plant>.+)_t(?P<tier>\d+)$"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark the exact Bonsai MIP model with HiGHS"
    )
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument("--warm-start", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--model-path",
        type=Path,
        help="Existing lossless .pb export. If omitted, export the master before solving.",
    )
    parser.add_argument("--time-limit-seconds", type=float, default=300.0)
    parser.add_argument("--num-threads", type=int, default=6)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--max-extra-pallets", type=int, default=5000)
    parser.add_argument(
        "--mip-heuristic-effort",
        type=float,
        default=0.30,
        help="HiGHS heuristic effort in [0, 1] (default 0.30).",
    )
    parser.add_argument(
        "--export-time-limit-seconds",
        type=float,
        default=0.1,
        help="SCIP time only used when an LP export must be built.",
    )
    parser.add_argument(
        "--solver-output",
        action="store_true",
        help="Show HiGHS diagnostic output (useful to inspect a MIP start).",
    )
    return parser


def _retained_internals(assignment: dict[str, object]) -> tuple[object, ...]:
    return tuple(
        sorted(
            {box.internal for box in assignment.values()},
            key=lambda dimensions: dimensions.as_tuple(),
        )
    )


def _complete_mip_start(
    column_names: list[str],
    data: object,
    candidates: tuple[object, ...],
    assignment: dict[str, object],
) -> np.ndarray:
    """Return an exact MIP start matching the names from the exported master."""

    candidate_by_internal = {
        candidate.internal: candidate_index
        for candidate_index, candidate in enumerate(candidates)
    }
    selected_candidate_by_product = {
        product_index: candidate_by_internal[assignment[product.code].internal]
        for product_index, product in enumerate(data.products)
    }
    volume_by_candidate_plant: dict[tuple[int, str], int] = {}
    for product_index, product in enumerate(data.products):
        candidate_index = selected_candidate_by_product[product_index]
        for plant, demand in product.annual_volume_by_plant.items():
            volume_by_candidate_plant[(candidate_index, plant)] = (
                volume_by_candidate_plant.get((candidate_index, plant), 0) + demand
            )

    values = np.zeros(len(column_names), dtype=np.float64)
    for column_index, name in enumerate(column_names):
        if name == "Constant":
            values[column_index] = 1.0
            continue
        match = _ASSIGNMENT_NAME.match(name)
        if match is not None:
            product_index = int(match["product"])
            candidate_index = int(match["candidate"])
            values[column_index] = float(
                selected_candidate_by_product[product_index] == candidate_index
            )
            continue
        match = _VOLUME_NAME.match(name)
        if match is not None:
            values[column_index] = volume_by_candidate_plant.get(
                (int(match["candidate"]), match["plant"]), 0
            )
            continue
        match = _REACHED_NAME.match(name)
        if match is not None:
            volume = volume_by_candidate_plant.get(
                (int(match["candidate"]), match["plant"]), 0
            )
            threshold = DISCOUNT_TIERS[int(match["tier"])].lower_inclusive
            values[column_index] = float(volume >= threshold)
            continue
        match = _DISCOUNTED_VOLUME_NAME.match(name)
        if match is not None:
            volume = volume_by_candidate_plant.get(
                (int(match["candidate"]), match["plant"]), 0
            )
            threshold = DISCOUNT_TIERS[int(match["tier"])].lower_inclusive
            values[column_index] = float(volume if volume >= threshold else 0)
            continue
        raise RuntimeError(f"unrecognised exported LP column: {name!r}")
    return values


def _assignment_from_highs_solution(
    column_names: list[str],
    column_values: list[float],
    data: object,
    candidates: tuple[object, ...],
    fallback: dict[str, object],
) -> dict[str, object]:
    """Decode assignment columns, retaining fixed products from ``fallback``."""

    chosen: dict[int, tuple[float, int]] = {}
    for name, value in zip(column_names, column_values):
        match = _ASSIGNMENT_NAME.match(name)
        if match is None:
            continue
        product_index = int(match["product"])
        candidate_index = int(match["candidate"])
        prior = chosen.get(product_index)
        if prior is None or value > prior[0]:
            chosen[product_index] = (value, candidate_index)

    result = dict(fallback)
    for product_index, product in enumerate(data.products):
        if product_index not in chosen:
            continue
        value, candidate_index = chosen[product_index]
        if value < 0.999:
            raise RuntimeError(
                f"HiGHS solution is not integral for {product.code}: best x={value}"
            )
        result[product.code] = candidates[candidate_index]
    return result


def _info_value(info: object, name: str) -> object | None:
    return getattr(info, name, None)


def _highs_lp_from_proto(model_proto: object, highspy: object) -> object:
    """Convert an OR-Tools MPModelProto to HiGHS without textual rounding."""

    variable_count = len(model_proto.variable)
    row_count = len(model_proto.constraint)
    entries_by_column: list[list[tuple[int, float]]] = [
        [] for _ in range(variable_count)
    ]
    for row_index, constraint in enumerate(model_proto.constraint):
        for column_index, coefficient in zip(
            constraint.var_index, constraint.coefficient
        ):
            entries_by_column[column_index].append((row_index, coefficient))

    start = np.empty(variable_count + 1, dtype=np.int64)
    start[0] = 0
    for column_index, entries in enumerate(entries_by_column):
        start[column_index + 1] = start[column_index] + len(entries)
    nonzero_count = int(start[-1])
    indices = np.empty(nonzero_count, dtype=np.int32)
    coefficients = np.empty(nonzero_count, dtype=np.float64)
    cursor = 0
    for entries in entries_by_column:
        for row_index, coefficient in entries:
            indices[cursor] = row_index
            coefficients[cursor] = coefficient
            cursor += 1

    lp = highspy.HighsLp()
    lp.model_name_ = model_proto.name or "bonsai_exact_master"
    lp.num_col_ = variable_count
    lp.num_row_ = row_count
    lp.col_cost_ = np.asarray(
        [variable.objective_coefficient for variable in model_proto.variable],
        dtype=np.float64,
    )
    lp.col_lower_ = np.asarray(
        [variable.lower_bound for variable in model_proto.variable], dtype=np.float64
    )
    lp.col_upper_ = np.asarray(
        [variable.upper_bound for variable in model_proto.variable], dtype=np.float64
    )
    lp.col_names_ = [variable.name for variable in model_proto.variable]
    lp.integrality_ = [
        highspy.HighsVarType.kInteger
        if variable.is_integer
        else highspy.HighsVarType.kContinuous
        for variable in model_proto.variable
    ]
    lp.row_lower_ = np.asarray(
        [constraint.lower_bound for constraint in model_proto.constraint],
        dtype=np.float64,
    )
    lp.row_upper_ = np.asarray(
        [constraint.upper_bound for constraint in model_proto.constraint],
        dtype=np.float64,
    )
    lp.row_names_ = [constraint.name for constraint in model_proto.constraint]
    lp.offset_ = model_proto.objective_offset
    matrix = highspy.HighsSparseMatrix()
    matrix.format_ = highspy.MatrixFormat.kColwise
    matrix.num_col_ = variable_count
    matrix.num_row_ = row_count
    matrix.start_ = start
    matrix.index_ = indices
    matrix.value_ = coefficients
    lp.a_matrix_ = matrix
    return lp


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.time_limit_seconds <= 0:
        raise ValueError("--time-limit-seconds must be positive")
    if args.num_threads < 1:
        raise ValueError("--num-threads must be positive")
    if not 0.0 <= args.mip_heuristic_effort <= 1.0:
        raise ValueError("--mip-heuristic-effort must be in [0, 1]")

    try:
        import highspy
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise RuntimeError("HiGHS is not installed; install highspy first") from exc

    data = load_prepared_data(args.data_dir)
    policy = FreightPolicy()
    warm = validate_solution_csv(args.warm_start, data, policy)
    retained = _retained_internals(warm.assignment)
    candidates, _ = generate_exact_candidates(
        data.products, 3.0, retained_designs=retained
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.model_path or args.output_dir / "master_highs.pb"
    export_result: dict[str, object] | None = None
    if not model_path.exists():
        exported = solve_with_scip(
            data,
            3.0,
            policy,
            time_limit_seconds=args.export_time_limit_seconds,
            num_threads=1,
            random_seed=args.random_seed,
            initial_assignment=warm.assignment,
            max_extra_pallets=args.max_extra_pallets,
            export_model_path=model_path,
        )
        export_result = {
            "status": exported.status,
            "wall_time_seconds": exported.wall_time_seconds,
            "model_build_time_seconds": exported.model_build_time_seconds,
        }

    highs = highspy.Highs()
    highs.setOptionValue("output_flag", args.solver_output)
    highs.setOptionValue("threads", args.num_threads)
    highs.setOptionValue("random_seed", args.random_seed)
    highs.setOptionValue("mip_heuristic_effort", args.mip_heuristic_effort)
    if model_path.suffix.lower() == ".pb":
        from ortools.linear_solver import linear_solver_pb2

        model_proto = linear_solver_pb2.MPModelProto()
        model_proto.ParseFromString(model_path.read_bytes())
        read_status = highs.passModel(_highs_lp_from_proto(model_proto, highspy))
    else:
        read_status = highs.readModel(str(model_path))
    if read_status != highspy.HighsStatus.kOk:
        raise RuntimeError(f"HiGHS could not import {model_path}: {read_status}")
    # Set this only after parsing: HiGHS applies ``time_limit`` to model input
    # as well as to optimization, and the full exact MPS takes a few seconds
    # to parse on its own.
    highs.setOptionValue("time_limit", args.time_limit_seconds)

    lp = highs.getLp()
    start = highspy.HighsSolution()
    start.value_valid = True
    start.col_value = _complete_mip_start(
        lp.col_names_, data, candidates, warm.assignment
    )
    start_status = highs.setSolution(start)

    solve_started_at = time.perf_counter()
    run_status = highs.run()
    solve_time_seconds = time.perf_counter() - solve_started_at
    model_status = highs.modelStatusToString(highs.getModelStatus())
    solution = highs.getSolution()
    solver_assignment = None
    solver_costs = None
    decode_error = None
    if solution.value_valid:
        try:
            solver_assignment = _assignment_from_highs_solution(
                lp.col_names_, solution.col_value, data, candidates, warm.assignment
            )
            solver_costs = validate_solution_csv(
                _write_temp_assignment(args.output_dir, data, solver_assignment),
                data,
                policy,
            ).costs
        except (RuntimeError, ValueError) as exc:
            decode_error = str(exc)

    selected_assignment = warm.assignment
    selected_costs = warm.costs
    selected_source = "warm_start"
    if solver_costs is not None and solver_costs.total_mills < warm.costs.total_mills:
        selected_assignment = solver_assignment
        selected_costs = solver_costs
        selected_source = "highs"

    output_path = args.output_dir / "asignacion_optima.csv"
    write_assignment_csv(output_path, data, selected_assignment)
    checked = validate_solution_csv(output_path, data, policy)
    if checked.costs.total_mills != selected_costs.total_mills:
        raise RuntimeError("HiGHS output did not match independent validation")
    output_digest = hashlib.sha256(output_path.read_bytes()).hexdigest().upper()
    info = highs.getInfo()
    payload: dict[str, object] = {
        "solver": f"HiGHS via highspy {highspy.HIGHS_VERSION_MAJOR}.{highspy.HIGHS_VERSION_MINOR}.{highspy.HIGHS_VERSION_PATCH}",
        "model_path": str(model_path),
        "model_columns": lp.num_col_,
        "model_rows": lp.num_row_,
        "read_status": str(read_status),
        "mip_start_status": str(start_status),
        "run_status": str(run_status),
        "model_status": model_status,
        "solution_value_valid": bool(solution.value_valid),
        "decode_error": decode_error,
        "selected_source": selected_source,
        "warm_start_costs": warm.costs.as_dict(),
        "highs_solution_costs": solver_costs.as_dict() if solver_costs else None,
        "costs": checked.costs.as_dict(),
        "output_path": str(output_path),
        "output_sha256": output_digest,
        "independently_validated": True,
        "time_limit_seconds": args.time_limit_seconds,
        "num_threads": args.num_threads,
        "random_seed": args.random_seed,
        "mip_heuristic_effort": args.mip_heuristic_effort,
        "solve_time_seconds": solve_time_seconds,
        "export_result": export_result,
        "highs_info": {
            name: _info_value(info, name)
            for name in (
                "mip_dual_bound",
                "mip_gap",
                "mip_node_count",
                "mip_primal_bound",
                "objective_function_value",
                "run_time",
            )
        },
    }
    write_json(args.output_dir / "resumen_highs.json", payload)
    return payload


def _write_temp_assignment(
    output_dir: Path, data: object, assignment: dict[str, object]
) -> Path:
    """Write a short-lived CSV solely to invoke the public independent parser."""

    path = output_dir / ".highs_candidate_check.csv"
    write_assignment_csv(path, data, assignment)
    return path


def main() -> int:
    args = build_parser().parse_args()
    print(json.dumps(run(args), ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
