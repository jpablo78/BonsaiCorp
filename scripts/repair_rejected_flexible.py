"""Conserva sólo filas alineadas con Kaggle de una solución de layout flexible rechazada."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bonsai.config import FreightPolicy
from bonsai.data import load_prepared_data
from bonsai.geometry import product_fits_candidate
from bonsai.reporting import write_assignment_csv
from bonsai.solution_validation import validate_solution_csv


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--rejected", type=Path, required=True)
    parser.add_argument("--incumbent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def run(args: argparse.Namespace) -> dict[str, object]:
    data = load_prepared_data(args.data_dir)
    policy = FreightPolicy()
    rejected = validate_solution_csv(
        args.rejected, data, policy, flexible_layout=True
    )
    incumbent = validate_solution_csv(args.incumbent, data, policy)

    assignment = dict(rejected.assignment)
    reverted = []
    for product in data.products:
        candidate = assignment[product.code]
        if not product_fits_candidate(
            product, candidate.internal, candidate.thickness_mm
        ):
            assignment[product.code] = incumbent.assignment[product.code]
            reverted.append(product.code)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_assignment_csv(args.output, data, assignment)
    checked = validate_solution_csv(args.output, data, policy)
    changed_vs_incumbent = sum(
        checked.assignment[product.code].external
        != incumbent.assignment[product.code].external
        for product in data.products
    )
    return {
        "reverted_rows": len(reverted),
        "retained_changes_vs_incumbent": changed_vs_incumbent,
        "repaired_total_usd": checked.costs.total_mills / 1000,
        "incumbent_total_usd": incumbent.costs.total_mills / 1000,
        "difference_vs_incumbent_usd": (
            checked.costs.total_mills - incumbent.costs.total_mills
        ) / 1000,
    }


if __name__ == "__main__":
    print(json.dumps(run(_parser().parse_args()), ensure_ascii=False, indent=2))
