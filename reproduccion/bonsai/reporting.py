"""Salidas estables para entrega Kaggle y reporte de escenarios."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from .models import CandidateBox, CostBreakdown, PreparedData
from .solution_validation import REQUIRED_OUTPUT_COLUMNS


def write_assignment_csv(
    output_path: str | Path,
    data: PreparedData,
    assignment: dict[str, CandidateBox],
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REQUIRED_OUTPUT_COLUMNS)
        writer.writeheader()
        for product in data.products:
            box = assignment[product.code]
            writer.writerow(
                {
                    "codigo_producto": product.code,
                    "caja_grosor_mm": f"{box.thickness_mm:g}",
                    "caja_exterior_largo": int(box.external.length),
                    "caja_exterior_ancho": int(box.external.width),
                    "caja_exterior_alto": int(box.external.height),
                }
            )


def write_json(output_path: str | Path, payload: dict[str, object]) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)


def scenario_payload(name: str, thickness_mm: float, costs: CostBreakdown) -> dict[str, object]:
    return {"scenario": name, "thickness_mm": thickness_mm, **costs.as_dict()}
