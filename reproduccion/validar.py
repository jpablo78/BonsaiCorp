"""Valida de forma independiente el CSV final contra los cuatro CSV fuente."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bonsai.config import FreightPolicy
from bonsai.data import load_prepared_data
from bonsai.decimal_io import validate_decimal_solution_csv


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Valida la factibilidad y los costos del catálogo final de Bonsai."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("datos"),
        help="carpeta con los cuatro CSV fuente",
    )
    parser.add_argument("--solution", type=Path, default=Path("asignacion_final.csv"))
    args = parser.parse_args()
    data = load_prepared_data(args.data_dir)
    checked = validate_decimal_solution_csv(args.solution, data, FreightPolicy())
    print(json.dumps({"valida": True, "sku": len(checked.assignment), "costos": checked.costs.as_dict()}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
