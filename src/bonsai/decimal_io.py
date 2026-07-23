"""CSV input/output and validation for decimal-mm box dimensions.

The competition accepted decimal external dimensions.  This module keeps that
format independent from any particular solver so the same assignment can be
validated, evaluated and written by OR-Tools/SCIP, HiGHS or a commercial
backend.
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

from .config import ALLOWED_THICKNESSES, FreightPolicy
from .costs import evaluate_assignments
from .data import parse_number
from .decimal_candidates import decimal_external_from_internal, decimal_product_fits_candidate
from .geometry import boxes_per_pallet
from .models import CandidateBox, Dimensions, PreparedData
from .solution_validation import REQUIRED_OUTPUT_COLUMNS, ValidationResult


def write_decimal_assignment_csv(
    path: str | Path,
    data: PreparedData,
    assignment: dict[str, CandidateBox],
    *,
    decimal_places: int,
) -> None:
    """Write one assignment row per SKU without truncating decimal dimensions."""

    if decimal_places < 1 or decimal_places > 6:
        raise ValueError("decimal_places must be between 1 and 6")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REQUIRED_OUTPUT_COLUMNS)
        writer.writeheader()
        for product in data.products:
            box = assignment[product.code]
            writer.writerow(
                {
                    "codigo_producto": product.code,
                    "caja_grosor_mm": f"{box.thickness_mm:g}",
                    "caja_exterior_largo": f"{box.external.length:.{decimal_places}f}",
                    "caja_exterior_ancho": f"{box.external.width:.{decimal_places}f}",
                    "caja_exterior_alto": f"{box.external.height:.{decimal_places}f}",
                }
            )


def validate_decimal_solution_csv(
    path: str | Path,
    data: PreparedData,
    freight_policy: FreightPolicy,
    *,
    required_thickness_mm: float | None = None,
) -> ValidationResult:
    """Validate a Kaggle-schema CSV whose exterior dimensions may be decimal.

    This applies the same strict fit, headspace, ECT, pallet and cost rules as
    the integer submission validator, while preserving decimal dimensions.
    """

    source = Path(path)
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != REQUIRED_OUTPUT_COLUMNS:
            raise ValueError("decimal output columns differ from the Kaggle schema")
        rows = list(reader)
    if not rows:
        raise ValueError("decimal output has no assignment rows")

    codes = [row["codigo_producto"].strip() for row in rows]
    expected_codes = {product.code for product in data.products}
    if len(codes) != len(set(codes)):
        raise ValueError("decimal output contains duplicate codigo_producto values")
    if set(codes) != expected_codes:
        raise ValueError("decimal output product set differs from catalog")

    designs: dict[tuple[float, float, float, float], list[str]] = defaultdict(list)
    for row in rows:
        thickness = parse_number(row["caja_grosor_mm"])
        if thickness not in ALLOWED_THICKNESSES:
            raise ValueError(f"unsupported thickness: {thickness}")
        design = (
            thickness,
            parse_number(row["caja_exterior_largo"]),
            parse_number(row["caja_exterior_ancho"]),
            parse_number(row["caja_exterior_alto"]),
        )
        designs[design].append(row["codigo_producto"].strip())

    thicknesses = {design[0] for design in designs}
    if len(thicknesses) != 1:
        raise ValueError("all output rows must use one global carton thickness")
    if required_thickness_mm is not None and thicknesses != {required_thickness_mm}:
        raise ValueError(
            f"decimal output must use global {required_thickness_mm:g}-mm thickness"
        )

    products = data.product_by_code
    assignment: dict[str, CandidateBox] = {}
    for ordinal, (design, design_codes) in enumerate(sorted(designs.items())):
        thickness, length, width, height = design
        external = Dimensions(length, width, height)
        internal = Dimensions(
            round(length - 2 * thickness, 6),
            round(width - 2 * thickness, 6),
            round(height - 2 * thickness, 6),
        )
        if min(internal.as_tuple()) <= 0:
            raise ValueError(f"decimal design {design} has non-positive internal dimensions")
        rebuilt = decimal_external_from_internal(internal, thickness)
        if not all(
            math.isclose(left, right, abs_tol=1e-6)
            for left, right in zip(rebuilt.as_tuple(), external.as_tuple())
        ):
            raise ValueError(f"decimal external/internal round trip failed for {design}")
        candidate = CandidateBox(
            candidate_id=f"decimal_output_{ordinal:05d}",
            thickness_mm=thickness,
            internal=internal,
            external=external,
            capacity_per_pallet=boxes_per_pallet(external),
            compatible_product_codes=frozenset(design_codes),
        )
        if candidate.capacity_per_pallet < 1:
            raise ValueError(f"decimal design {design} cannot be stacked on the pallet")
        for code in design_codes:
            if not decimal_product_fits_candidate(products[code], internal, thickness):
                raise ValueError(f"decimal design {design} is infeasible for {code}")
            assignment[code] = candidate

    return ValidationResult(
        assignment=assignment,
        costs=evaluate_assignments(data.products, assignment, freight_policy),
    )
