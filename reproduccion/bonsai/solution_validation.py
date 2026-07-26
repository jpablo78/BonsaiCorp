"""Validación independiente de un CSV de asignación con formato Kaggle."""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from .config import ALLOWED_THICKNESSES, FreightPolicy
from .costs import evaluate_assignments
from .data import parse_number
from .geometry import (
    boxes_per_pallet,
    external_from_internal,
    flexible_product_fits_candidate,
    product_fits_candidate,
)
from .models import CandidateBox, CostBreakdown, Dimensions, PreparedData


REQUIRED_OUTPUT_COLUMNS = (
    "codigo_producto",
    "caja_grosor_mm",
    "caja_exterior_largo",
    "caja_exterior_ancho",
    "caja_exterior_alto",
)


@dataclass(frozen=True)
class ValidationResult:
    assignment: dict[str, CandidateBox]
    costs: CostBreakdown


def _integer_mm(value: str) -> int:
    parsed = parse_number(value)
    rounded = round(parsed)
    if abs(parsed - rounded) > 1e-9:
        raise ValueError(f"external dimensions must be integer mm, got {value!r}")
    return int(rounded)


def validate_output_rows(
    rows: list[dict[str, str]],
    data: PreparedData,
    freight_policy: FreightPolicy,
    *,
    flexible_layout: bool = False,
) -> ValidationResult:
    expected_codes = {product.code for product in data.products}
    provided_codes = [row["codigo_producto"].strip() for row in rows]
    if len(provided_codes) != len(set(provided_codes)):
        raise ValueError("output contains duplicate codigo_producto values")
    if set(provided_codes) != expected_codes:
        raise ValueError(
            "output SKU set differs from catalog: "
            f"missing={sorted(expected_codes - set(provided_codes))[:5]}, "
            f"unknown={sorted(set(provided_codes) - expected_codes)[:5]}"
        )

    product_by_code = data.product_by_code
    rows_by_design: dict[tuple[float, int, int, int], list[str]] = defaultdict(list)
    thicknesses: set[float] = set()
    for row in rows:
        thickness = parse_number(row["caja_grosor_mm"])
        if thickness not in ALLOWED_THICKNESSES:
            raise ValueError(f"unsupported thickness: {thickness}")
        design = (
            thickness,
            _integer_mm(row["caja_exterior_largo"]),
            _integer_mm(row["caja_exterior_ancho"]),
            _integer_mm(row["caja_exterior_alto"]),
        )
        rows_by_design[design].append(row["codigo_producto"].strip())
        thicknesses.add(thickness)
    if len(thicknesses) != 1:
        raise ValueError("all output rows must use one global carton thickness")

    assignment: dict[str, CandidateBox] = {}
    for ordinal, (design, codes) in enumerate(sorted(rows_by_design.items())):
        thickness, length, width, height = design
        external = Dimensions(length, width, height)
        internal = Dimensions(length - 2 * thickness, width - 2 * thickness, height - 2 * thickness)
        if min(internal.as_tuple()) <= 0:
            raise ValueError(f"design {design} has non-positive internal dimensions")
        # También confirma que la convención es exterior = interior + 2t.
        external_from_internal(internal, thickness)
        candidate = CandidateBox(
            candidate_id=f"output_{ordinal:05d}",
            thickness_mm=thickness,
            internal=internal,
            external=external,
            capacity_per_pallet=boxes_per_pallet(external),
            compatible_product_codes=frozenset(codes),
        )
        if candidate.capacity_per_pallet < 1:
            raise ValueError(f"design {design} cannot be stacked on the pallet")
        for code in codes:
            product = product_by_code[code]
            fit_oracle = (
                flexible_product_fits_candidate if flexible_layout else product_fits_candidate
            )
            if not fit_oracle(product, internal, thickness):
                raise ValueError(f"design {design} is infeasible for SKU {code}")
            assignment[code] = candidate
    return ValidationResult(assignment, evaluate_assignments(data.products, assignment, freight_policy))


def validate_solution_csv(
    solution_path: str | Path,
    data: PreparedData,
    freight_policy: FreightPolicy,
    *,
    flexible_layout: bool = False,
) -> ValidationResult:
    path = Path(solution_path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != REQUIRED_OUTPUT_COLUMNS:
            raise ValueError(
                f"output columns must be exactly {REQUIRED_OUTPUT_COLUMNS}; got {reader.fieldnames}"
            )
        return validate_output_rows(
            list(reader), data, freight_policy, flexible_layout=flexible_layout
        )
