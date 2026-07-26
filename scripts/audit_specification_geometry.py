"""Audita el formato crudo de grosor de cartón y la coherencia geométrica histórica de cajas.

Es deliberadamente de sólo lectura. Las dimensiones exteriores históricas no se
usan para construir cajas propuestas, pero sus discrepancias son evidencia útil
de calidad de datos y no deben reinterpretarse silenciosamente como dimensiones de producto.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

from bonsai.data import parse_number


AXES = ("largo", "ancho", "alto")


def number_text(value: float) -> str:
    return f"{value:g}"


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        return list(csv.DictReader(source))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument("--examples", type=int, default=12)
    args = parser.parse_args()
    root = args.data_dir

    specs = load_rows(root / "especificaciones_cajas.csv")
    products = load_rows(root / "catalogo_productos.csv")
    operations = load_rows(root / "operaciones_planta.csv")

    product_ids_by_box: dict[str, list[str]] = defaultdict(list)
    for row in products:
        product_ids_by_box[row["caja_tipo_id"]].append(row["codigo_producto"])

    demand_by_product: Counter[str] = Counter()
    for row in operations:
        product_id = row.get("codigo_producto")
        if not product_id:
            continue
# Esto refleja la fuente de demanda autorizada. Las demás columnas numéricas
# no son relevantes para la auditoría geométrica.
        for candidate in ("volumen_producto_total", "cantidad_cajas", "cantidad", "demanda"):
            if row.get(candidate, "").strip():
                try:
                    demand_by_product[product_id] += parse_number(row[candidate])
                except ValueError:
                    pass
                break

    raw_thicknesses: Counter[str] = Counter()
    canonical_thicknesses: Counter[str] = Counter()
    mismatch_by_axis: Counter[str] = Counter()
    residual_patterns: dict[str, Counter[str]] = {axis: Counter() for axis in AXES}
    mismatches: list[dict[str, object]] = []

    for row in specs:
        raw = row["caja_grosor_mm"]
        thickness = parse_number(raw)
        raw_thicknesses[raw] += 1
        canonical_thicknesses[number_text(thickness)] += 1
        residuals: dict[str, float] = {}
        for axis in AXES:
            internal = parse_number(row[f"caja_interior_{axis}"])
            external = parse_number(row[f"caja_exterior_{axis}"])
            residual = external - (internal + 2 * thickness)
            residuals[axis] = residual
            if abs(residual) > 1e-9:
                mismatch_by_axis[axis] += 1
                residual_patterns[axis][number_text(residual)] += 1
        if any(abs(value) > 1e-9 for value in residuals.values()):
            product_ids = product_ids_by_box.get(row["caja_tipo_id"], [])
            demand = sum(demand_by_product[product_id] for product_id in product_ids)
            mismatches.append(
                {
                    "box": row["caja_tipo_id"],
                    "raw": raw,
                    "thickness": thickness,
                    "internal": tuple(parse_number(row[f"caja_interior_{axis}"]) for axis in AXES),
                    "external": tuple(parse_number(row[f"caja_exterior_{axis}"]) for axis in AXES),
                    "residuals": residuals,
                    "products": product_ids,
                    "demand": demand,
                }
            )

    print(f"specification_rows={len(specs)}")
    print(f"raw_thickness_encodings={len(raw_thicknesses)}")
    print("raw_thickness_variants:")
    for raw, count in sorted(raw_thicknesses.items(), key=lambda item: (parse_number(item[0]), item[0])):
        print(f"  {raw!r}: {count} -> {number_text(parse_number(raw))} mm")
    print("canonical_thicknesses:")
    for thickness, count in sorted(canonical_thicknesses.items(), key=lambda item: float(item[0])):
        print(f"  {thickness} mm: {count}")

    print(f"geometry_mismatch_any_axis={len(mismatches)}")
    for axis in AXES:
        print(f"geometry_mismatch_{axis}={mismatch_by_axis[axis]}")
        patterns = ", ".join(
            f"{residual} mm ({count})"
            for residual, count in residual_patterns[axis].most_common(8)
        ) or "none"
        print(f"  residual_patterns_{axis}: {patterns}")

    affected_products = {product_id for item in mismatches for product_id in item["products"]}
    affected_demand = sum(demand_by_product[product_id] for product_id in affected_products)
    print(f"affected_catalog_products={len(affected_products)}")
    print(f"affected_operations_demand={number_text(affected_demand)}")
    print("examples:")
    for item in mismatches[: args.examples]:
        residual_text = ", ".join(
            f"{axis}={number_text(value)}"
            for axis, value in item["residuals"].items()
            if abs(value) > 1e-9
        )
        product_text = ",".join(item["products"][:5]) or "(no catalog product)"
        print(
            "  "
            f"box={item['box']} raw_thickness={item['raw']!r} "
            f"internal={item['internal']} external={item['external']} "
            f"external-(internal+2t): {residual_text}; "
            f"products={product_text}; operations_demand={number_text(item['demand'])}"
        )


if __name__ == "__main__":
    main()
