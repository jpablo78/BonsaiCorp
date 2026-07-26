"""Extensiones acotadas para recalcular demanda y dar de alta un SKU.

Estos flujos no modifican el optimizador ni reasignan los SKU vigentes. Sirven
para evaluar un catálogo validado con un forecast nuevo y, en el segundo caso,
para asignar un producto nuevo exclusivamente a tipos de caja ya activos.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from .config import FreightPolicy
from .costs import box_type_key, evaluate_assignments, tier_index
from .data import DataValidationError, parse_int, parse_number
from .decimal_candidates import decimal_product_fits_candidate
from .decimal_io import validate_decimal_solution_csv, write_decimal_assignment_csv
from .models import CandidateBox, Dimensions, PLANTS, PreparedData, Product
from .solution_validation import ValidationResult


NEW_PRODUCT_COLUMNS = (
    "codigo_producto",
    "referencia_interna_largo_mm",
    "referencia_interna_ancho_mm",
    "referencia_interna_alto_mm",
    "peso_neto_caja_kg",
    *(f"volumen_producto_planta_{plant}" for plant in PLANTS),
)


@dataclass(frozen=True)
class IncrementalDecision:
    """Resultado de evaluar cajas activas para un único SKU nuevo."""

    product: Product
    data_with_product: PreparedData
    selected_assignment: dict[str, CandidateBox] | None
    selected_candidate: CandidateBox | None
    baseline: ValidationResult
    resulting_costs_usd: float | None
    tier_changes: tuple[dict[str, object], ...]
    active_types_evaluated: int
    feasible_active_types: int

    @property
    def uses_existing_type(self) -> bool:
        return self.selected_candidate is not None


def validate_decimal_solution(
    solution_path: str | Path,
    data: PreparedData,
    freight_policy: FreightPolicy,
) -> ValidationResult:
    """Valida soluciones operativas enteras o decimales con una sola semántica.

    El validador decimal acepta también valores enteros. Esto evita que los
    flujos de ciclo de vida tengan una ruta distinta para la solución de 0,1 mm.
    """

    return validate_decimal_solution_csv(solution_path, data, freight_policy)


def load_new_product(path: str | Path, existing_codes: set[str]) -> Product:
    """Carga el contrato mínimo de un nuevo SKU y valida sus datos operativos."""

    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"missing new-product file: {source}")
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != NEW_PRODUCT_COLUMNS:
            raise DataValidationError(
                "new-product columns must be exactly "
                f"{NEW_PRODUCT_COLUMNS}; got {reader.fieldnames}"
            )
        rows = list(reader)
    if len(rows) != 1:
        raise DataValidationError("new-product file must contain exactly one row")

    row = rows[0]
    code = row["codigo_producto"].strip()
    if not code:
        raise DataValidationError("new product codigo_producto is empty")
    if code in existing_codes:
        raise DataValidationError(f"new product code already exists: {code}")
    internal = Dimensions(
        parse_number(row["referencia_interna_largo_mm"]),
        parse_number(row["referencia_interna_ancho_mm"]),
        parse_number(row["referencia_interna_alto_mm"]),
    )
    weight = parse_number(row["peso_neto_caja_kg"])
    if min(internal.as_tuple()) <= 0 or weight <= 0:
        raise DataValidationError("new product dimensions and weight must be positive")
    demand = {
        plant: parse_int(row[f"volumen_producto_planta_{plant}"])
        for plant in PLANTS
    }
    if any(volume < 0 for volume in demand.values()):
        raise DataValidationError("new product projected demand cannot be negative")
    if not sum(demand.values()):
        raise DataValidationError("new product projected annual demand must be positive")
    return Product(
        code=code,
        current_box_type_id="new_product_reference",
        current_internal=internal,
        net_weight_kg=weight,
        annual_volume_by_plant=demand,
    )


def data_with_new_product(data: PreparedData, product: Product) -> PreparedData:
    """Devuelve un catálogo aumentado sin alterar los productos existentes."""

    if product.code in data.product_by_code:
        raise DataValidationError(f"new product code already exists: {product.code}")
    products = tuple(sorted((*data.products, product), key=lambda item: item.code))
    return PreparedData(products=products, current_boxes=data.current_boxes)


def recalculated_catalog(
    solution_path: str | Path,
    data: PreparedData,
    freight_policy: FreightPolicy,
) -> ValidationResult:
    """Recalcula costos del catálogo fijo con la demanda contenida en ``data``."""

    return validate_decimal_solution(solution_path, data, freight_policy)


def _active_designs(assignment: dict[str, CandidateBox]) -> tuple[CandidateBox, ...]:
    by_type: dict[tuple[float, float, float, float], CandidateBox] = {}
    for candidate in assignment.values():
        by_type.setdefault(box_type_key(candidate), candidate)
    return tuple(
        candidate
        for _, candidate in sorted(
            by_type.items(), key=lambda item: item[0]
        )
    )


def _candidate_for_new_product(candidate: CandidateBox, code: str) -> CandidateBox:
    """Replica un diseño físico activo con compatibilidad limitada al SKU nuevo."""

    return CandidateBox(
        candidate_id=f"incremental_{candidate.candidate_id}_{code}",
        thickness_mm=candidate.thickness_mm,
        internal=candidate.internal,
        external=candidate.external,
        capacity_per_pallet=candidate.capacity_per_pallet,
        compatible_product_codes=frozenset({code}),
    )


def _tier_changes(
    data: PreparedData,
    assignment: dict[str, CandidateBox],
    product: Product,
    selected: CandidateBox,
) -> tuple[dict[str, object], ...]:
    selected_key = box_type_key(selected)
    volumes = {plant: 0 for plant in PLANTS}
    for existing in data.products:
        if box_type_key(assignment[existing.code]) != selected_key:
            continue
        for plant, volume in existing.annual_volume_by_plant.items():
            volumes[plant] += volume

    changed: list[dict[str, object]] = []
    for plant in PLANTS:
        before = volumes[plant]
        after = before + product.annual_volume_by_plant[plant]
        if not after:
            continue
        before_tier = tier_index(before) + 1 if before else None
        after_tier = tier_index(after) + 1
        if before_tier != after_tier:
            changed.append(
                {
                    "plant": plant,
                    "annual_volume_before": before,
                    "annual_volume_after": after,
                    "tier_before": before_tier,
                    "tier_after": after_tier,
                }
            )
    return tuple(changed)


def evaluate_existing_type_assignment(
    data: PreparedData,
    solution_path: str | Path,
    new_product: Product,
    freight_policy: FreightPolicy,
) -> IncrementalDecision:
    """Elige la caja activa factible de menor costo incremental.

    Las asignaciones existentes no cambian. Para cada tipo activo se recalcula
    el costo de toda la red, pues agregar volumen puede modificar los tiers de
    Procurement de los SKU ya asignados a ese mismo diseño.
    """

    baseline = validate_decimal_solution(solution_path, data, freight_policy)
    active = _active_designs(baseline.assignment)
    augmented_data = data_with_new_product(data, new_product)
    alternatives: list[tuple[int, CandidateBox, dict[str, CandidateBox]]] = []
    feasible = 0
    for active_candidate in active:
        if not decimal_product_fits_candidate(
            new_product, active_candidate.internal, active_candidate.thickness_mm
        ):
            continue
        feasible += 1
        new_candidate = _candidate_for_new_product(active_candidate, new_product.code)
        assignment = {**baseline.assignment, new_product.code: new_candidate}
        costs = evaluate_assignments(augmented_data.products, assignment, freight_policy)
        alternatives.append((costs.total_mills, new_candidate, assignment))

    if not alternatives:
        return IncrementalDecision(
            product=new_product,
            data_with_product=augmented_data,
            selected_assignment=None,
            selected_candidate=None,
            baseline=baseline,
            resulting_costs_usd=None,
            tier_changes=(),
            active_types_evaluated=len(active),
            feasible_active_types=0,
        )

    _, selected, assignment = min(
        alternatives,
        key=lambda item: (item[0], item[1].external.as_tuple()),
    )
    costs = evaluate_assignments(augmented_data.products, assignment, freight_policy)
    return IncrementalDecision(
        product=new_product,
        data_with_product=augmented_data,
        selected_assignment=assignment,
        selected_candidate=selected,
        baseline=baseline,
        resulting_costs_usd=costs.total_mills / 1000,
        tier_changes=_tier_changes(data, baseline.assignment, new_product, selected),
        active_types_evaluated=len(active),
        feasible_active_types=feasible,
    )


def infer_decimal_places(solution_path: str | Path) -> int:
    """Preserva la precisión exterior declarada por una solución existente."""

    with Path(solution_path).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    places = 0
    for row in rows:
        for column in (
            "caja_exterior_largo",
            "caja_exterior_ancho",
            "caja_exterior_alto",
        ):
            value = row[column].strip().replace(",", ".")
            places = max(places, len(value.partition(".")[2]))
    return min(max(places, 1), 6)


def write_incremental_assignment(
    path: str | Path,
    decision: IncrementalDecision,
    *,
    decimal_places: int,
    freight_policy: FreightPolicy,
) -> ValidationResult:
    """Escribe y vuelve a validar el artefacto operativo de 428 filas."""

    if decision.selected_assignment is None:
        raise ValueError("cannot write an incremental assignment without a feasible active type")
    write_decimal_assignment_csv(
        path,
        decision.data_with_product,
        decision.selected_assignment,
        decimal_places=decimal_places,
    )
    checked = validate_decimal_solution_csv(path, decision.data_with_product, freight_policy)
    expected = evaluate_assignments(
        decision.data_with_product.products,
        decision.selected_assignment,
        freight_policy,
    )
    if checked.costs.total_mills != expected.total_mills:
        raise AssertionError("incremental CSV round trip changed total cost")
    return checked


def candidate_payload(candidate: CandidateBox) -> dict[str, object]:
    """Representación estable de un tipo físico para JSON de operación."""

    return {
        "thickness_mm": candidate.thickness_mm,
        "external_mm": {
            "length": candidate.external.length,
            "width": candidate.external.width,
            "height": candidate.external.height,
        },
        "internal_mm": {
            "length": candidate.internal.length,
            "width": candidate.internal.width,
            "height": candidate.internal.height,
        },
        "capacity_per_pallet": candidate.capacity_per_pallet,
    }
