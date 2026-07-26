"""Carga, normalización y preparación auditable de los CSV de la competencia."""

from __future__ import annotations

import csv
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from .models import CurrentBoxSpec, Dimensions, PLANTS, PreparedData, Product


REQUIRED_FILES = (
    "catalogo_productos.csv",
    "especificaciones_cajas.csv",
    "operaciones_planta.csv",
    "procurement_cajas.csv",
)


class DataValidationError(ValueError):
    """Se lanza cuando una relación de entrada no puede interpretarse con seguridad."""


def parse_number(value: str | int | float) -> float:
    """Interpreta los formatos decimales inconsistentes de los CSV entregados."""

    text = str(value).strip().lower().replace("mm", "").replace(" ", "")
    if not text or text == "error":
        raise ValueError(f"not a numeric value: {value!r}")
    return float(text.replace(",", "."))


def parse_int(value: str | int | float) -> int:
    number = parse_number(value)
    rounded = round(number)
    if not math.isclose(number, rounded, abs_tol=1e-9):
        raise ValueError(f"expected an integer quantity, got {value!r}")
    return int(rounded)


def read_csv(data_dir: Path, filename: str) -> list[dict[str, str]]:
    path = data_dir / filename
    return read_csv_path(path)


def read_csv_path(path: str | Path) -> list[dict[str, str]]:
    """Lee un CSV UTF-8 con BOM opcional desde una ruta explícita."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"missing source file: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def canonical_decimal(value: str | int | float) -> str:
    """Formatea un número normalizado sin ceros finales innecesarios."""

    return f"{parse_number(value):g}"


def export_cleaned_sources(data_dir: str | Path, output_dir: str | Path) -> dict[str, object]:
    """Escribe una copia normalizada y no destructiva de los cuatro CSV fuente.

    Los insumos originales no se modifican. El artefacto explícito permite
    revisar la limpieza crítica de grosor: variantes de coma/punto decimal y
    sufijos opcionales `mm` se convierten a cadenas numéricas canónicas. Las
    inconsistencias deliberadas de Procurement se preservan, no se imputan.
    """

    source_root, clean_root = Path(data_dir), Path(output_dir)
    clean_root.mkdir(parents=True, exist_ok=True)
    changed_thickness_rows = 0
    normalized_thicknesses: Counter[str] = Counter()
    for filename in REQUIRED_FILES:
        rows = read_csv(source_root, filename)
        if not rows:
            raise DataValidationError(f"source file has no rows: {filename}")
        fieldnames = list(rows[0])
        if filename == "especificaciones_cajas.csv":
            for row in rows:
                before = row["caja_grosor_mm"]
                after = canonical_decimal(before)
                changed_thickness_rows += int(before.strip() != after)
                row["caja_grosor_mm"] = after
                normalized_thicknesses[after] += 1
        with (clean_root / filename).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    return {
        "output_dir": str(clean_root),
        "files_written": list(REQUIRED_FILES),
        "thickness_rows_normalized": changed_thickness_rows,
        "normalized_thickness_counts": dict(sorted(normalized_thicknesses.items(), key=lambda item: float(item[0]))),
        "preserved_by_design": [
            "procurement_cajas.csv ERROR values and inconsistent historical volumes",
            "especificaciones_cajas.csv missing cantidad_cajas_total values",
        ],
    }


def _index_unique(rows: Iterable[dict[str, str]], key: str, source: str) -> dict[str, dict[str, str]]:
    indexed: dict[str, dict[str, str]] = {}
    for row in rows:
        value = row[key].strip()
        if not value:
            raise DataValidationError(f"{source}.{key} has an empty key")
        if value in indexed:
            raise DataValidationError(f"{source}.{key} is not unique: {value}")
        indexed[value] = row
    return indexed


def _dimensions(row: dict[str, str], prefix: str) -> Dimensions:
    return Dimensions(
        parse_number(row[f"{prefix}_largo"]),
        parse_number(row[f"{prefix}_ancho"]),
        parse_number(row[f"{prefix}_alto"]),
    )


def load_prepared_data(
    data_dir: str | Path,
    *,
    infer_internal_from_external: bool = False,
    operations_override: str | Path | None = None,
) -> PreparedData:
    """Crea registros listos para optimizar, usando Operaciones como demanda.

    ``infer_internal_from_external`` es una convención alternativa sólo para
    diagnóstico. Deriva cada dimensión interna histórica como la exterior
    menos dos veces el grosor histórico. Estas dimensiones de referencia
    pueden ser fraccionarias; las dimensiones propuestas de salida permanecen
    en milímetros enteros. La convención documentada/predeterminada sigue
    siendo las columnas explícitas ``caja_interior_*``.
    """

    root = Path(data_dir)
    for filename in REQUIRED_FILES:
        if not (root / filename).exists():
            raise FileNotFoundError(f"missing required source file: {root / filename}")

    catalog = _index_unique(read_csv(root, "catalogo_productos.csv"), "codigo_producto", "catalogo")
    specs = _index_unique(
        read_csv(root, "especificaciones_cajas.csv"), "caja_tipo_id", "especificaciones"
    )
    original_operation_rows = read_csv(root, "operaciones_planta.csv")
    operation_rows = (
        read_csv_path(operations_override)
        if operations_override is not None
        else original_operation_rows
    )
    if not operation_rows:
        raise DataValidationError("operations source has no rows")
    expected_operation_columns = set(original_operation_rows[0])
    supplied_operation_columns = set(operation_rows[0])
    if supplied_operation_columns != expected_operation_columns:
        missing = sorted(expected_operation_columns - supplied_operation_columns)
        unexpected = sorted(supplied_operation_columns - expected_operation_columns)
        raise DataValidationError(
            "operations override columns differ from operaciones_planta.csv: "
            f"missing={missing}, unexpected={unexpected}"
        )
    operations = _index_unique(operation_rows, "codigo_producto", "operaciones")
    procurement = _index_unique(
        read_csv(root, "procurement_cajas.csv"), "caja_tipo_id", "procurement"
    )

    catalog_codes, operation_codes = set(catalog), set(operations)
    if catalog_codes != operation_codes:
        raise DataValidationError(
            "catalogo and operaciones do not contain the same SKU keys: "
            f"catalogo_only={sorted(catalog_codes - operation_codes)[:5]}, "
            f"operaciones_only={sorted(operation_codes - catalog_codes)[:5]}"
        )

    current_boxes: dict[str, CurrentBoxSpec] = {}
    for box_id, row in specs.items():
        if box_id not in procurement:
            raise DataValidationError(f"box type {box_id} is absent from procurement")
        thickness = parse_number(row["caja_grosor_mm"])
        external = _dimensions(row, "caja_exterior")
        internal = (
            Dimensions(
                # Conserva decimales genuinos de origen (por ejemplo 394.6 mm),
                # eliminando a la vez artefactos de punto flotante binario como
                # 254.99999999999997.
                round(external.length - 2 * thickness, 6),
                round(external.width - 2 * thickness, 6),
                round(external.height - 2 * thickness, 6),
            )
            if infer_internal_from_external
            else _dimensions(row, "caja_interior")
        )
        current_boxes[box_id] = CurrentBoxSpec(
            box_type_id=box_id,
            thickness_mm=thickness,
            internal=internal,
            external=external,
        )

    products: list[Product] = []
    for code, row in catalog.items():
        box_type_id = row["caja_tipo_id"].strip()
        if box_type_id not in current_boxes:
            raise DataValidationError(f"SKU {code} references absent box type {box_type_id}")
        operation = operations[code]
        demand = {
            plant: parse_int(operation[f"volumen_producto_planta_{plant}"])
            for plant in PLANTS
        }
        if sum(demand.values()) != parse_int(operation["volumen_producto_total"]):
            raise DataValidationError(f"plant volumes do not sum to total volume for {code}")
        products.append(
            Product(
                code=code,
                current_box_type_id=box_type_id,
                current_internal=current_boxes[box_type_id].internal,
                net_weight_kg=parse_number(row["peso_neto_caja"]),
                annual_volume_by_plant=demand,
            )
        )
    return PreparedData(tuple(sorted(products, key=lambda product: product.code)), current_boxes)


def _tier_factor(volume: int) -> float:
    if volume < 20_000:
        return 1.10
    if volume < 50_000:
        return 1.00
    if volume < 100_000:
        return 0.90
    if volume < 500_000:
        return 0.80
    return 0.70


def audit_dataset(data_dir: str | Path) -> dict[str, object]:
    """Devuelve una auditoría legible por máquina, sin decidir ninguna limpieza."""

    root = Path(data_dir)
    catalog_rows = read_csv(root, "catalogo_productos.csv")
    spec_rows = read_csv(root, "especificaciones_cajas.csv")
    operation_rows = read_csv(root, "operaciones_planta.csv")
    procurement_rows = read_csv(root, "procurement_cajas.csv")
    catalog = _index_unique(catalog_rows, "codigo_producto", "catalogo")
    specs = _index_unique(spec_rows, "caja_tipo_id", "especificaciones")
    operations = _index_unique(operation_rows, "codigo_producto", "operaciones")
    procurement = _index_unique(procurement_rows, "caja_tipo_id", "procurement")

    missing_total = sum(not row["cantidad_cajas_total"].strip() for row in spec_rows)
    exterior_length_mismatch = 0
    for row in spec_rows:
        thickness = parse_number(row["caja_grosor_mm"])
        if not math.isclose(
            parse_number(row["caja_exterior_largo"]),
            parse_number(row["caja_interior_largo"]) + 2 * thickness,
            abs_tol=1e-8,
        ):
            exterior_length_mismatch += 1

    type_by_sku = {code: row["caja_tipo_id"] for code, row in catalog.items()}
    operation_demand: dict[str, dict[str, int]] = {
        box_id: {plant: 0 for plant in PLANTS} for box_id in specs
    }
    for code, row in operations.items():
        for plant in PLANTS:
            operation_demand[type_by_sku[code]][plant] += parse_int(
                row[f"volumen_producto_planta_{plant}"]
            )

    procurement_volume_mismatches = 0
    procurement_volume_operations_total = 0
    procurement_volume_total = 0
    base_cost_operations_mismatches = 0
    discount_matches_operations_tier = 0
    discount_observations = 0
    error_unit_prices = 0
    for box_id, row in procurement.items():
        operation_volume_total = sum(operation_demand[box_id].values())
        expected_base_total = operation_volume_total * parse_number(row["costo_unitario_base"])
        if not math.isclose(
            expected_base_total, parse_number(row["costo_total_base"]), abs_tol=1e-6
        ):
            base_cost_operations_mismatches += 1
        for plant in PLANTS:
            proc_volume = parse_int(row[f"volumen_tipo_planta_{plant}"])
            ops_volume = operation_demand[box_id][plant]
            procurement_volume_total += proc_volume
            procurement_volume_operations_total += ops_volume
            procurement_volume_mismatches += int(proc_volume != ops_volume)
            raw_discount = row[f"descuento_planta_{plant}"].replace("%", "").replace(" ", "")
            discount_factor = 1 + parse_number(raw_discount) / 100
            discount_matches_operations_tier += int(
                math.isclose(discount_factor, _tier_factor(ops_volume), abs_tol=1e-9)
            )
            discount_observations += 1
            error_unit_prices += int(row[f"costo_unitario_planta_{plant}"].strip().upper() == "ERROR")

    package_fields_missing = sum(
        not row["cantidad_paquetes"].strip() or not row["peso_neto_paquete"].strip()
        for row in catalog_rows
    )
    return {
        "rows": {
            "catalogo_productos": len(catalog_rows),
            "especificaciones_cajas": len(spec_rows),
            "operaciones_planta": len(operation_rows),
            "procurement_cajas": len(procurement_rows),
        },
        "keys": {
            "catalogo_and_operaciones_same_skus": set(catalog) == set(operations),
            "catalog_box_types": len(set(type_by_sku.values())),
            "spec_and_procurement_same_types": set(specs) == set(procurement),
        },
        "data_quality": {
            "missing_cantidad_cajas_total": missing_total,
            "external_length_not_inner_plus_two_thickness": exterior_length_mismatch,
            "catalog_rows_missing_package_fields": package_fields_missing,
            "procurement_error_unit_price_cells": error_unit_prices,
        },
        "procurement_vs_operations": {
            "mismatching_type_plant_volumes": procurement_volume_mismatches,
            "operations_volume": procurement_volume_operations_total,
            "procurement_volume": procurement_volume_total,
            "costo_total_base_not_equal_operations_volume_times_base": base_cost_operations_mismatches,
            "discounts_matching_operations_tier": discount_matches_operations_tier,
            "discount_observations": discount_observations,
        },
    }
