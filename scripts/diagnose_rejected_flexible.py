"""Diagnose why a flexible-layout solution fails the Kaggle-aligned oracle."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from bonsai.config import FreightPolicy
from bonsai.data import load_prepared_data, parse_number
from bonsai.geometry import (
    boxes_per_pallet,
    compression_feasible,
    external_from_internal,
    faq_reconciled_headspace_feasible,
    flexible_volume_headspace_feasible,
    respects_dimension_adjustment,
)
from bonsai.models import Dimensions
from bonsai.solution_validation import validate_solution_csv


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--solution", type=Path, required=True)
    parser.add_argument("--incumbent", type=Path, required=True)
    return parser


def run(args: argparse.Namespace) -> dict[str, object]:
    data = load_prepared_data(args.data_dir)
    policy = FreightPolicy()
    flexible = validate_solution_csv(
        args.solution, data, policy, flexible_layout=True
    )
    incumbent = validate_solution_csv(args.incumbent, data, policy)

    rows = {}
    with args.solution.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            rows[row["codigo_producto"].strip()] = row

    counts: Counter[str] = Counter()
    examples: list[dict[str, object]] = []
    invalid_codes: list[str] = []
    for product in data.products:
        row = rows[product.code]
        thickness = parse_number(row["caja_grosor_mm"])
        external = Dimensions(
            parse_number(row["caja_exterior_largo"]),
            parse_number(row["caja_exterior_ancho"]),
            parse_number(row["caja_exterior_alto"]),
        )
        internal = Dimensions(
            external.length - 2 * thickness,
            external.width - 2 * thickness,
            external.height - 2 * thickness,
        )
        rebuilt_external = external_from_internal(internal, thickness)
        checks = {
            "dimension_adjustment": respects_dimension_adjustment(product, internal),
            "flexible_volume_headspace": flexible_volume_headspace_feasible(
                product.product_volume_mm3, internal, thickness
            ),
            "faq_axis_headspace": faq_reconciled_headspace_feasible(
                product, internal, thickness
            ),
            "pallet": boxes_per_pallet(rebuilt_external) > 0,
            "compression": compression_feasible(
                product, rebuilt_external, thickness
            ),
        }
        failed = [name for name, ok in checks.items() if not ok]
        if failed:
            invalid_codes.append(product.code)
            for name in failed:
                counts[name] += 1
            if len(examples) < 8:
                examples.append(
                    {
                        "codigo_producto": product.code,
                        "original_internal": product.current_internal.as_tuple(),
                        "submitted_internal": internal.as_tuple(),
                        "failed": failed,
                    }
                )

    changed_codes = [
        product.code
        for product in data.products
        if flexible.assignment[product.code].external
        != incumbent.assignment[product.code].external
    ]
    return {
        "rows": len(data.products),
        "changed_vs_accepted_incumbent": len(changed_codes),
        "invalid_under_kaggle_aligned_oracle": len(invalid_codes),
        "invalid_changed_rows": len(set(invalid_codes) & set(changed_codes)),
        "failures_by_rule": dict(counts),
        "examples": examples,
        "flexible_total_usd": flexible.costs.total_mills / 1000,
        "accepted_incumbent_total_usd": incumbent.costs.total_mills / 1000,
    }


if __name__ == "__main__":
    print(json.dumps(run(_parser().parse_args()), ensure_ascii=False, indent=2))
