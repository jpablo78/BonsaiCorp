from pathlib import Path

from bonsai.config import FreightPolicy
from bonsai.decimal_candidates import decimal_external_from_internal
from bonsai.decimal_io import validate_decimal_solution_csv, write_decimal_assignment_csv
from bonsai.models import CandidateBox, Dimensions, PLANTS, PreparedData, Product


def test_decimal_csv_round_trip_preserves_fractional_dimension(tmp_path: Path) -> None:
    product = Product(
        code="NEW001",
        current_box_type_id="reference",
        current_internal=Dimensions(100.0, 100.0, 100.0),
        net_weight_kg=1.0,
        annual_volume_by_plant={plant: (1 if plant == PLANTS[0] else 0) for plant in PLANTS},
    )
    internal = Dimensions(100.1, 100.0, 100.0)
    candidate = CandidateBox(
        candidate_id="decimal",
        thickness_mm=3.0,
        internal=internal,
        external=decimal_external_from_internal(internal, 3.0),
        capacity_per_pallet=1,
        compatible_product_codes=frozenset({product.code}),
    )
    data = PreparedData(products=(product,), current_boxes={})
    path = tmp_path / "assignment.csv"

    write_decimal_assignment_csv(path, data, {product.code: candidate}, decimal_places=1)
    checked = validate_decimal_solution_csv(
        path, data, FreightPolicy(), required_thickness_mm=3.0
    )

    assert checked.assignment[product.code].external.length == 106.1
