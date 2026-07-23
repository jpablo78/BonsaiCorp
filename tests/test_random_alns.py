import random
import unittest

from bonsai.config import FreightPolicy
from bonsai.costs import box_type_key, evaluate_assignments
from bonsai.models import CandidateBox, Dimensions, PLANTS, Product
from bonsai.random_alns import build_random_neighborhood, ruin_assignment


class RandomAlnsTests(unittest.TestCase):
    @staticmethod
    def _product(code: str, volume: int = 10_000) -> Product:
        return Product(
            code=code,
            current_box_type_id=f"current_{code}",
            current_internal=Dimensions(100, 100, 100),
            net_weight_kg=1,
            annual_volume_by_plant={
                plant: volume if plant == "buenos_aires" else 0
                for plant in PLANTS
            },
        )

    @staticmethod
    def _box(index: int, codes: set[str], capacity: int = 100) -> CandidateBox:
        external = Dimensions(106 + index, 106, 106)
        return CandidateBox(
            candidate_id=f"c{index}",
            thickness_mm=3.0,
            internal=Dimensions(100 + index, 100, 100),
            external=external,
            capacity_per_pallet=capacity,
            compatible_product_codes=frozenset(codes),
        )

    def _fixture(self):
        products = tuple(self._product(f"P{i}") for i in range(8))
        codes = {product.code for product in products}
        incumbent_boxes = tuple(self._box(i, codes) for i in range(8))
        destinations = tuple(self._box(20 + i, codes) for i in range(12))
        assignment = {
            product.code: incumbent_boxes[index]
            for index, product in enumerate(products)
        }
        return products, assignment, incumbent_boxes + destinations

    def test_neighborhood_releases_only_complete_source_groups(self) -> None:
        products, assignment, candidates = self._fixture()
        item = build_random_neighborhood(
            products,
            assignment,
            candidates,
            random.Random(17),
            min_source_types=5,
            max_source_types=5,
            min_destinations=5,
            max_destinations=7,
            max_skus=8,
        )
        groups = {}
        for code, candidate in assignment.items():
            groups.setdefault(box_type_key(candidate), set()).add(code)
        self.assertEqual(len(item.source_types), 5)
        self.assertGreaterEqual(len(item.destinations), 5)
        self.assertLessEqual(len(item.destinations), 7)
        self.assertEqual(
            set(item.product_codes),
            set().union(*(groups[key] for key in item.source_types)),
        )
        self.assertFalse(item.uncovered_product_codes)

    def test_neighborhood_is_reproducible_for_seed(self) -> None:
        products, assignment, candidates = self._fixture()
        kwargs = dict(
            min_source_types=5,
            max_source_types=7,
            min_destinations=5,
            max_destinations=10,
            max_skus=8,
        )
        first = build_random_neighborhood(
            products, assignment, candidates, random.Random(99), **kwargs
        )
        second = build_random_neighborhood(
            products, assignment, candidates, random.Random(99), **kwargs
        )
        self.assertEqual(first, second)

    def test_ruin_respects_exact_cost_and_pallet_caps(self) -> None:
        products, assignment, candidates = self._fixture()
        item = build_random_neighborhood(
            products,
            assignment,
            candidates,
            random.Random(5),
            min_source_types=5,
            max_source_types=5,
            min_destinations=5,
            max_destinations=5,
            max_skus=8,
        )
        initial = evaluate_assignments(products, assignment, FreightPolicy())
        result = ruin_assignment(
            products,
            assignment,
            item,
            FreightPolicy(),
            random.Random(6),
            move_fraction=0.5,
            max_total_mills=initial.total_mills + 100_000_000,
            max_pallets=initial.pallets + 1_000,
        )
        checked = evaluate_assignments(products, result.assignment, FreightPolicy())
        self.assertEqual(result.costs.total_mills, checked.total_mills)
        self.assertGreater(result.applied_moves, 0)
        self.assertLessEqual(result.costs.pallets, initial.pallets + 1_000)


if __name__ == "__main__":
    unittest.main()
