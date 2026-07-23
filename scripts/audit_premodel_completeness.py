"""Independent audit of the strict integer-mm candidate premodel.

The audit does not solve or modify the optimization instance.  It checks that:

* the pallet axis convention reproduces historical layouts when those fields
  are present;
* accelerated compatibility masks equal the independent geometry oracle in
  both directions for every unpruned representative;
* capacity stored in every candidate is recomputed correctly; and
* every pruned representative has a retained candidate with at least as much
  capacity and a compatibility superset.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from bonsai.data import load_prepared_data, parse_number
from bonsai.exact_candidates import generate_exact_candidates
from bonsai.geometry import boxes_per_pallet, external_from_internal, product_fits_candidate


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        return list(csv.DictReader(source))


def _optional_integer(value: str) -> int | None:
    text = value.strip()
    if not text:
        return None
    return round(parse_number(text))


def _is_superset(left: frozenset[str], right: frozenset[str]) -> bool:
    return right <= left


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument("--thickness-mm", type=float, default=3.0, choices=(3.0, 4.5, 5.0))
    args = parser.parse_args()

    data = load_prepared_data(args.data_dir)
    specs = _rows(args.data_dir / "especificaciones_cajas.csv")

    historical_checked = 0
    historical_mismatches: list[dict[str, object]] = []
    for row in specs:
        supplied = {
            "alto": _optional_integer(row["cantidad_cajas_alto"]),
            "largo": _optional_integer(row["cantidad_cajas_largo"]),
            "ancho": _optional_integer(row["cantidad_cajas_ancho"]),
            "total": _optional_integer(row["cantidad_cajas_total"]),
        }
        if any(value is None for value in supplied.values()):
            continue
        external_length = parse_number(row["caja_exterior_largo"])
        external_width = parse_number(row["caja_exterior_ancho"])
        external_height = parse_number(row["caja_exterior_alto"])
        calculated = {
            "alto": int(1800 // external_height),
            # Required source convention: box long side along pallet short side.
            "largo": int(800 // external_length),
            "ancho": int(1200 // external_width),
        }
        calculated["total"] = (
            calculated["alto"] * calculated["largo"] * calculated["ancho"]
        )
        historical_checked += 1
        if calculated != supplied:
            historical_mismatches.append(
                {
                    "caja_tipo_id": row["caja_tipo_id"],
                    "supplied": supplied,
                    "calculated": calculated,
                }
            )

    raw, raw_stats = generate_exact_candidates(
        data.products, args.thickness_mm, prune_dominated=False
    )
    retained, retained_stats = generate_exact_candidates(
        data.products, args.thickness_mm, prune_dominated=True
    )

    capacity_mismatches: list[tuple[tuple[float, float, float], int, int]] = []
    compatibility_mismatches: list[dict[str, object]] = []
    for candidate in raw:
        recalculated_capacity = boxes_per_pallet(
            external_from_internal(candidate.internal, args.thickness_mm)
        )
        if recalculated_capacity != candidate.capacity_per_pallet:
            capacity_mismatches.append(
                (
                    candidate.internal.as_tuple(),
                    candidate.capacity_per_pallet,
                    recalculated_capacity,
                )
            )
        oracle_codes = frozenset(
            product.code
            for product in data.products
            if product_fits_candidate(product, candidate.internal, args.thickness_mm)
        )
        if oracle_codes != candidate.compatible_product_codes:
            compatibility_mismatches.append(
                {
                    "internal": candidate.internal.as_tuple(),
                    "missing_from_accelerated": sorted(
                        oracle_codes - candidate.compatible_product_codes
                    ),
                    "extra_in_accelerated": sorted(
                        candidate.compatible_product_codes - oracle_codes
                    ),
                }
            )

    retained_by_capacity = sorted(
        retained,
        key=lambda candidate: (-candidate.capacity_per_pallet, candidate.internal.as_tuple()),
    )
    retained_internals = {candidate.internal for candidate in retained}
    pruned_without_witness: list[dict[str, object]] = []
    for candidate in raw:
        if candidate.internal in retained_internals:
            continue
        witness = next(
            (
                other
                for other in retained_by_capacity
                if other.capacity_per_pallet >= candidate.capacity_per_pallet
                and _is_superset(
                    other.compatible_product_codes,
                    candidate.compatible_product_codes,
                )
            ),
            None,
        )
        if witness is None:
            pruned_without_witness.append(
                {
                    "internal": candidate.internal.as_tuple(),
                    "capacity": candidate.capacity_per_pallet,
                    "compatible_products": len(candidate.compatible_product_codes),
                }
            )

    payload = {
        "thickness_mm": args.thickness_mm,
        "historical_layout_rows_checked": historical_checked,
        "historical_layout_mismatches": historical_mismatches,
        "raw_candidate_stats": raw_stats.__dict__,
        "retained_candidate_stats": retained_stats.__dict__,
        "raw_candidates": len(raw),
        "retained_candidates": len(retained),
        "capacity_mismatches": capacity_mismatches,
        "compatibility_mismatches": compatibility_mismatches,
        "pruned_without_valid_retained_witness": pruned_without_witness,
        "passed": not (
            historical_mismatches
            or capacity_mismatches
            or compatibility_mismatches
            or pruned_without_witness
        ),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
