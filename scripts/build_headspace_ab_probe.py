"""Construye una entrega A/B para aislar la interpretación de headspace.

La prueba parte de un CSV ya aceptado y modifica exclusivamente BR0004. La
nueva caja respeta volumen, ±10%, pallet, ECT y la formulación de producto
reconfigurable; falla deliberadamente la regla híbrida vigente. Un score cero
del archivo resultante aislaría esa diferencia de interpretación.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from bonsai.config import FreightPolicy
from bonsai.costs import evaluate_assignments
from bonsai.data import load_prepared_data
from bonsai.decimal_io import validate_decimal_solution_csv
from bonsai.geometry import (
    boxes_per_pallet,
    compression_feasible,
    flexible_product_fits_candidate,
    headspace_maxima,
    product_fits_candidate,
    respects_dimension_adjustment,
)
from bonsai.models import CandidateBox, Dimensions


SKU = "BR0004"
THICKNESS_MM = 3.0
EXTERNAL = Dimensions(399.0, 300.0, 249.0)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument(
        "--baseline", type=Path, default=Path("baseline/asignacion_0_1mm.csv")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("output_headspace_ab_probe")
    )
    return parser


def run(args: argparse.Namespace) -> dict[str, object]:
    data = load_prepared_data(args.data_dir)
    policy = FreightPolicy()
    baseline = validate_decimal_solution_csv(
        args.baseline, data, policy, required_thickness_mm=THICKNESS_MM
    )
    product = data.product_by_code[SKU]
    internal = Dimensions(
        EXTERNAL.length - 2 * THICKNESS_MM,
        EXTERNAL.width - 2 * THICKNESS_MM,
        EXTERNAL.height - 2 * THICKNESS_MM,
    )
    candidate = CandidateBox(
        candidate_id="headspace_ab_probe",
        thickness_mm=THICKNESS_MM,
        internal=internal,
        external=EXTERNAL,
        capacity_per_pallet=boxes_per_pallet(EXTERNAL),
        compatible_product_codes=frozenset({SKU}),
    )

    if not respects_dimension_adjustment(product, internal):
        raise AssertionError("la caja A/B no cumple la tolerancia ±10%")
    if not flexible_product_fits_candidate(product, internal, THICKNESS_MM):
        raise AssertionError("la caja A/B no cumple la formulación flexible")
    if product_fits_candidate(product, internal, THICKNESS_MM):
        raise AssertionError("la caja A/B debe diferir de la regla híbrida")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "asignacion_headspace_ab_BR0004.csv"
    with args.baseline.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = reader.fieldnames
    if fieldnames is None:
        raise ValueError("baseline sin encabezados")

    previous_row: dict[str, str] | None = None
    for row in rows:
        if row["codigo_producto"].strip() == SKU:
            previous_row = dict(row)
            row.update(
                {
                    "caja_grosor_mm": "3",
                    "caja_exterior_largo": "399",
                    "caja_exterior_ancho": "300",
                    "caja_exterior_alto": "249",
                }
            )
            break
    if previous_row is None:
        raise ValueError(f"{SKU} no existe en la baseline")
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    assignment = dict(baseline.assignment)
    assignment[SKU] = candidate
    costs = evaluate_assignments(data.products, assignment, policy)

    # Una construcción explícita que demuestra que existe una forma del
    # producto con el mismo volumen y headspace dentro de los tres topes.
    witness_product = Dimensions(380.0, 294.0, product.product_volume_mm3 / (380.0 * 294.0))
    witness_headspace = Dimensions(
        internal.length - witness_product.length,
        internal.width - witness_product.width,
        internal.height - witness_product.height,
    )
    witness_maxima = headspace_maxima(internal, THICKNESS_MM)
    if abs(witness_product.volume_mm3 - product.product_volume_mm3) > 1e-6:
        raise AssertionError("el testigo no conserva el volumen del producto")
    if any(
        headspace < -1e-8 or headspace > maximum + 1e-8
        for headspace, maximum in zip(witness_headspace.as_tuple(), witness_maxima.as_tuple())
    ):
        raise AssertionError("el headspace del testigo excede un tope por eje")
    report = {
        "purpose": "A/B: misma baseline, una sola fila flexible-only",
        "baseline_path": str(args.baseline),
        "baseline_sha256": _sha256(args.baseline),
        "output_path": str(output_path),
        "output_sha256": _sha256(output_path),
        "rows": len(rows),
        "changed_sku": SKU,
        "previous_row": previous_row,
        "probe_row": next(row for row in rows if row["codigo_producto"] == SKU),
        "probe_internal_mm": internal.as_tuple(),
        "checks": {
            "global_thickness_3mm": True,
            "dimension_adjustment_pm10": True,
            "pallet_capacity": candidate.capacity_per_pallet,
            "compression": compression_feasible(product, EXTERNAL, THICKNESS_MM),
            "flexible_volume_headspace": True,
            "hybrid_current_rule": False,
        },
        "witness_reconfigured_product_mm": witness_product.as_tuple(),
        "witness_headspace_mm": witness_headspace.as_tuple(),
        "witness_headspace_maxima_mm": witness_maxima.as_tuple(),
        "cost_after_single_change_usd": costs.total_mills / 1000,
    }
    report_path = args.output_dir / "reporte_headspace_ab_BR0004.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(run(_parser().parse_args()), ensure_ascii=False, indent=2))
