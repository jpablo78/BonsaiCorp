"""First CPLEX smoke test on a real restricted Bonsai MIP.

This deliberately uses an MPS interchange file only to validate that CPLEX
can consume a real neighbourhood of the current formulation.  It is not the
production backend: large-coefficient text MPS needs independent cost checks,
which this script performs before reporting its answer.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import cplex

from bonsai.config import FreightPolicy
from bonsai.costs import evaluate_assignments
from bonsai.data import load_prepared_data
from bonsai.exact_candidates import generate_exact_candidates
from bonsai.reporting import write_assignment_csv, write_json
from bonsai.scip_optimizer import solve_with_scip
from bonsai.solution_validation import validate_solution_csv


_ASSIGNMENT_NAME = re.compile(r"^x_p(?P<product>\d+)_c(?P<candidate>\d+)$")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--warm-start", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--free-products", type=int, default=20)
    parser.add_argument("--time-limit-seconds", type=float, default=60.0)
    parser.add_argument("--threads", type=int, default=6)
    return parser


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.free_products < 1 or args.time_limit_seconds <= 0:
        raise ValueError("free-products and time-limit-seconds must be positive")
    data = load_prepared_data(args.data_dir)
    policy = FreightPolicy()
    warm = validate_solution_csv(args.warm_start, data, policy)
    candidates, stats = generate_exact_candidates(
        data.products,
        3.0,
        retained_designs=tuple({box.internal for box in warm.assignment.values()}),
    )
    free_codes = frozenset(
        product.code for product in sorted(data.products, key=lambda item: item.code)[: args.free_products]
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / "neighborhood.mps"

    # Export happens before the short SCIP call.  The SCIP result itself is
    # intentionally ignored: CPLEX is the engine under test.
    solve_with_scip(
        data,
        3.0,
        policy,
        time_limit_seconds=1.0,
        num_threads=1,
        initial_assignment=warm.assignment,
        free_product_codes=free_codes,
        precomputed_exact_candidates=candidates,
        precomputed_exact_candidate_stats=stats,
        export_model_path=model_path,
    )

    model = cplex.Cplex(str(model_path))
    model.parameters.timelimit.set(args.time_limit_seconds)
    model.parameters.threads.set(args.threads)
    model.set_log_stream(None)
    model.set_results_stream(None)
    model.set_warning_stream(None)
    model.solve()
    status = model.solution.get_status_string()
    if not model.solution.is_primal_feasible():
        raise RuntimeError(f"CPLEX returned no primal solution: {status}")

    assignment = dict(warm.assignment)
    for name, value in zip(
        model.variables.get_names(), model.solution.get_values(), strict=True
    ):
        match = _ASSIGNMENT_NAME.match(name)
        if match is not None and value > 0.5:
            product_index = int(match.group("product"))
            candidate_index = int(match.group("candidate"))
            assignment[data.products[product_index].code] = candidates[candidate_index]

    costs = evaluate_assignments(data.products, assignment, policy)
    output_path = args.output_dir / "asignacion_optima.csv"
    write_assignment_csv(output_path, data, assignment)
    checked = validate_solution_csv(output_path, data, policy)
    if checked.costs.total_mills != costs.total_mills:
        raise AssertionError("CPLEX assignment failed the CSV round-trip audit")
    payload = {
        "cplex_status": status,
        "cplex_objective": model.solution.get_objective_value(),
        "independent_costs": checked.costs.as_dict(),
        "warm_costs": warm.costs.as_dict(),
        "free_products": sorted(free_codes),
        "mps_path": str(model_path),
        "mps_bytes": model_path.stat().st_size,
    }
    write_json(args.output_dir / "resumen_cplex_smoke.json", payload)
    return payload


if __name__ == "__main__":
    print(json.dumps(run(_parser().parse_args()), ensure_ascii=False, indent=2))
