import unittest

from bonsai.config import FreightPolicy
from bonsai.destination_lns import rank_destination_work_items
from bonsai.models import CandidateBox, Dimensions, PLANTS, PreparedData, Product
from bonsai.multi_destination_lns import (
    allowed_internals_for_neighborhood,
    build_multi_destination_work_items,
)


def _volumes(buenos_aires: int) -> dict[str, int]:
    return {plant: buenos_aires if plant == "buenos_aires" else 0 for plant in PLANTS}


def _product(code: str, dimension: int, volume: int) -> Product:
    return Product(
        code=code,
        current_box_type_id=f"box_{code}",
        current_internal=Dimensions(dimension, 100, 100),
        net_weight_kg=1.0,
        annual_volume_by_plant=_volumes(volume),
    )


def _candidate(identifier: str, dimension: int, compatible: tuple[str, ...]) -> CandidateBox:
    return CandidateBox(
        candidate_id=identifier,
        thickness_mm=3.0,
        internal=Dimensions(dimension, 100, 100),
        external=Dimensions(dimension + 6, 106, 106),
        capacity_per_pallet=20,
        compatible_product_codes=frozenset(compatible),
    )


class MultiDestinationNeighborhoodTests(unittest.TestCase):
    def setUp(self) -> None:
        products = (
            _product("A", 100, 19_000),
            _product("B", 120, 2_000),
            _product("C", 130, 2_000),
            _product("D", 140, 2_000),
        )
        self.data = PreparedData(products=products, current_boxes={})
        self.incumbent = {
            code: _candidate(f"inc_{code}", dimension, (code,))
            for code, dimension in (("A", 100), ("B", 120), ("C", 130), ("D", 140))
        }
        candidates = (
            _candidate("dest_1", 101, ("A", "B", "C")),
            _candidate("dest_2", 102, ("B", "C", "D")),
            _candidate("dest_3", 103, ("C", "D")),
        )
        self.ranked = rank_destination_work_items(
            self.data,
            self.incumbent,
            candidates,
            FreightPolicy(),
            min_gross_opportunity_mills=0,
        )

    def test_build_is_deterministic_bounded_and_unique(self) -> None:
        forward = build_multi_destination_work_items(
            self.ranked,
            self.incumbent,
            min_destinations=2,
            max_destinations=3,
            max_neighborhoods=20,
        )
        again = build_multi_destination_work_items(
            self.ranked,
            self.incumbent,
            min_destinations=2,
            max_destinations=3,
            max_neighborhoods=20,
        )

        self.assertEqual(forward, again)
        self.assertTrue(forward)
        destination_sets = [
            frozenset(item.destination_id for item in neighborhood.destinations)
            for neighborhood in forward
        ]
        self.assertEqual(len(destination_sets), len(set(destination_sets)))
        self.assertTrue(all(2 <= item.destination_count <= 3 for item in forward))
        self.assertTrue(
            all(
                len(internals) >= 2
                for item in forward
                for internals in allowed_internals_for_neighborhood(item).values()
            )
        )

    def test_each_sku_gets_only_compatible_destinations(self) -> None:
        neighborhood = next(
            item
            for item in build_multi_destination_work_items(
                self.ranked,
                self.incumbent,
                min_destinations=3,
                max_destinations=3,
            )
            if item.destination_count == 3
        )

        allowed = allowed_internals_for_neighborhood(neighborhood)

        for code, internals in allowed.items():
            expected = {
                destination.candidate.internal
                for destination in neighborhood.destinations
                if code in destination.product_codes
            }
            self.assertEqual(set(internals), expected)
            self.assertNotIn(self.incumbent[code].internal, internals)
        self.assertGreaterEqual(max(map(len, allowed.values())), 2)
        self.assertGreaterEqual(min(map(len, allowed.values())), 2)

    def test_rejects_invalid_destination_bounds_and_honors_sku_cap(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least 2"):
            build_multi_destination_work_items(
                self.ranked, self.incumbent, min_destinations=1
            )
        with self.assertRaisesRegex(ValueError, "exceed 8"):
            build_multi_destination_work_items(
                self.ranked, self.incumbent, max_destinations=9
            )
        with self.assertRaisesRegex(ValueError, "choices_per_sku"):
            build_multi_destination_work_items(
                self.ranked,
                self.incumbent,
                min_destinations=2,
                max_destinations=3,
                min_destination_choices_per_sku=4,
            )
        capped = build_multi_destination_work_items(
            self.ranked,
            self.incumbent,
            min_destinations=2,
            max_destinations=3,
            max_skus=1,
            min_skus=1,
        )
        self.assertTrue(all(item.sku_count <= 1 for item in capped))


if __name__ == "__main__":
    unittest.main()
