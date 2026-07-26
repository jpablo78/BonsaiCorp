"""Genera las métricas y tablas principales a partir del CSV final validado."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from bonsai.config import FreightPolicy
from bonsai.data import load_prepared_data, read_csv_path
from bonsai.decimal_io import validate_decimal_solution_csv
from bonsai.models import PLANTS


def _historical_costs(data_dir: Path) -> tuple[float, float]:
    """Reconstituye packaging y flete históricos desde Operaciones."""

    rows = read_csv_path(data_dir / "operaciones_planta.csv")
    packaging = sum(float(row["costo_total"]) for row in rows)
    freight = sum(float(row["costo_pallets_total"]) for row in rows)
    return packaging, freight


def main() -> int:
    parser = argparse.ArgumentParser(description="Genera tablas ejecutivas de la solución.")
    parser.add_argument("--data-dir", type=Path, default=Path("datos"))
    parser.add_argument("--solution", type=Path, default=Path("asignacion_final.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("salida"))
    args = parser.parse_args()
    data = load_prepared_data(args.data_dir)
    checked = validate_decimal_solution_csv(args.solution, data, FreightPolicy())
    costs = checked.costs
    historical_packaging, historical_freight = _historical_costs(args.data_dir)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)

    rows = [
        ["Escenario", "Cartón_USD", "Flete_USD", "Total_USD", "Pallets", "Tipos"],
        ["Histórico", f"{historical_packaging:.2f}", f"{historical_freight:.2f}", f"{historical_packaging + historical_freight:.2f}", "", ""],
        ["Catálogo final", f"{costs.packaging_mills / 1000:.2f}", f"{costs.freight_mills / 1000:.2f}", f"{costs.total_mills / 1000:.2f}", str(costs.pallets), str(costs.types)],
    ]
    with (output / "tabla_costos.csv").open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerows(rows)
    with (output / "tabla_utilizacion_por_planta.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Planta", "Utilizacion_pallet_pct"])
        for plant in PLANTS:
            writer.writerow([plant, f"{100 * costs.pallet_utilization_by_plant[plant]:.4f}"])

    historical_total = historical_packaging + historical_freight
    payload = {
        "sku": len(checked.assignment),
        "tipos_caja": costs.types,
        "pallets": costs.pallets,
        "carton_usd": costs.packaging_mills / 1000,
        "flete_usd": costs.freight_mills / 1000,
        "costo_total_usd": costs.total_mills / 1000,
        "costo_historico_usd": historical_total,
        "ahorro_usd": historical_total - costs.total_mills / 1000,
        "ahorro_pct": 100 * (historical_total - costs.total_mills / 1000) / historical_total,
        "utilizacion_pallet_por_planta": costs.pallet_utilization_by_plant,
    }
    (output / "metricas_finales.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
