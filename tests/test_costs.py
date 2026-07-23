import unittest

from bonsai.config import FreightPolicy
from bonsai.costs import evaluate_assignments, unit_price_mills
from bonsai.models import CandidateBox, Dimensions, PLANTS, Product


class CostTierTests(unittest.TestCase):
    def test_price_tiers_are_exact_in_mills(self) -> None:
        self.assertEqual(unit_price_mills(3.0, 19_999), 660)
        self.assertEqual(unit_price_mills(3.0, 20_000), 600)
        self.assertEqual(unit_price_mills(3.0, 49_999), 600)
        self.assertEqual(unit_price_mills(3.0, 50_000), 540)
        self.assertEqual(unit_price_mills(3.0, 99_999), 540)
        self.assertEqual(unit_price_mills(3.0, 100_000), 480)
        self.assertEqual(unit_price_mills(3.0, 499_999), 480)
        self.assertEqual(unit_price_mills(3.0, 500_000), 420)
        self.assertEqual(unit_price_mills(4.5, 20_000), 650)
        self.assertEqual(unit_price_mills(5.0, 50_000), 630)
        self.assertEqual(unit_price_mills(4.5, 500_000), 455)


class PhysicalBoxIdentityTests(unittest.TestCase):
    @staticmethod
    def _product(code: str) -> Product:
        return Product(
            code=code,
            current_box_type_id=f"current_{code}",
            current_internal=Dimensions(100, 100, 100),
            net_weight_kg=1,
            annual_volume_by_plant={
                plant: 10_000 if plant == "buenos_aires" else 0 for plant in PLANTS
            },
        )

    @staticmethod
    def _candidate(
        candidate_id: str, code: str, external: Dimensions
    ) -> CandidateBox:
        return CandidateBox(
            candidate_id=candidate_id,
            thickness_mm=3.0,
            internal=Dimensions(
                external.length - 6,
                external.width - 6,
                external.height - 6,
            ),
            external=external,
            capacity_per_pallet=10,
            compatible_product_codes=frozenset({code}),
        )

    def test_equal_geometry_with_different_ids_is_one_commercial_type(self) -> None:
        products = (self._product("P1"), self._product("P2"))
        shared_geometry = Dimensions(106, 106, 106)
        assignment = {
            "P1": self._candidate("local_a", "P1", shared_geometry),
            "P2": self._candidate("local_b", "P2", shared_geometry),
        }
        costs = evaluate_assignments(products, assignment, FreightPolicy())
        self.assertEqual(costs.types, 1)
        self.assertEqual(costs.packaging_mills, 20_000 * 600)

    def test_different_geometry_with_same_id_remains_two_types(self) -> None:
        products = (self._product("P1"), self._product("P2"))
        assignment = {
            "P1": self._candidate("colliding_id", "P1", Dimensions(106, 106, 106)),
            "P2": self._candidate("colliding_id", "P2", Dimensions(107, 106, 106)),
        }
        costs = evaluate_assignments(products, assignment, FreightPolicy())
        self.assertEqual(costs.types, 2)
        self.assertEqual(costs.packaging_mills, 20_000 * 660)


if __name__ == "__main__":
    unittest.main()
