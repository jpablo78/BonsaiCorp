from pathlib import Path

from bonsai.config import FreightPolicy
from bonsai.decimal_candidates import decimal_external_from_internal
from bonsai.decimal_io import write_decimal_assignment_csv
from bonsai.lifecycle import (
    data_with_new_product,
    evaluate_existing_type_assignment,
    infer_decimal_places,
    write_incremental_assignment,
)
from bonsai.models import CandidateBox, Dimensions, PLANTS, PreparedData, Product


def _product(code: str, internal: Dimensions, demand: int) -> Product:
    return Product(
        code=code,
        current_box_type_id=f"ref_{code}",
        current_internal=internal,
        net_weight_kg=1.0,
        annual_volume_by_plant={
            plant: (demand if plant == PLANTS[0] else 0) for plant in PLANTS
        },
    )


def _candidate(code: str, internal: Dimensions, compatible: set[str]) -> CandidateBox:
    return CandidateBox(
        candidate_id=code,
        thickness_mm=3.0,
        internal=internal,
        external=decimal_external_from_internal(internal, 3.0),
        capacity_per_pallet=1,
        compatible_product_codes=frozenset(compatible),
    )


def _base_solution(tmp_path: Path) -> tuple[PreparedData, Path]:
    small = _product("OLD_SMALL", Dimensions(100.0, 100.0, 100.0), 30_000)
    large = _product("OLD_LARGE", Dimensions(200.0, 200.0, 200.0), 10_000)
    small_box = _candidate("small", small.current_internal, {small.code})
    large_box = _candidate("large", large.current_internal, {large.code})
    data = PreparedData(products=(large, small), current_boxes={})
    path = tmp_path / "base.csv"
    write_decimal_assignment_csv(
        path,
        data,
        {small.code: small_box, large.code: large_box},
        decimal_places=1,
    )
    return data, path


def test_incremental_assignment_selects_feasible_active_type(tmp_path: Path) -> None:
    data, solution = _base_solution(tmp_path)
    new = _product("NEW001", Dimensions(100.0, 100.0, 100.0), 1_000)

    decision = evaluate_existing_type_assignment(data, solution, new, FreightPolicy())

    assert decision.uses_existing_type
    assert decision.selected_candidate is not None
    assert decision.selected_candidate.internal == Dimensions(100.0, 100.0, 100.0)
    assert decision.active_types_evaluated == 2
    assert decision.feasible_active_types == 1

    output = tmp_path / "asignacion_incremental.csv"
    checked = write_incremental_assignment(
        output, decision, decimal_places=infer_decimal_places(solution), freight_policy=FreightPolicy()
    )
    assert set(checked.assignment) == {"OLD_SMALL", "OLD_LARGE", "NEW001"}


def test_incremental_assignment_justifies_new_design_when_no_active_type_fits(tmp_path: Path) -> None:
    data, solution = _base_solution(tmp_path)
    new = _product("NEW001", Dimensions(400.0, 400.0, 400.0), 1_000)

    decision = evaluate_existing_type_assignment(data, solution, new, FreightPolicy())

    assert not decision.uses_existing_type
    assert decision.selected_assignment is None
    assert decision.active_types_evaluated == 2
    assert decision.feasible_active_types == 0


def test_augmented_data_preserves_existing_catalog_and_adds_one_sku(tmp_path: Path) -> None:
    data, _ = _base_solution(tmp_path)
    new = _product("NEW001", Dimensions(100.0, 100.0, 100.0), 1_000)

    augmented = data_with_new_product(data, new)

    assert [product.code for product in augmented.products] == [
        "NEW001",
        "OLD_LARGE",
        "OLD_SMALL",
    ]
