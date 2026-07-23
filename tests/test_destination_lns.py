import unittest

from bonsai.config import FreightPolicy
from bonsai.destination_lns import (
    allowed_internals_for_destination,
    rank_destination_work_items,
)
from bonsai.models import CandidateBox, Dimensions, PLANTS, PreparedData, Product


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


def _candidate(
    identifier: str,
    dimension: int,
    capacity: int,
    compatible: tuple[str, ...],
) -> CandidateBox:
    return CandidateBox(
        candidate_id=identifier,
        thickness_mm=3.0,
        internal=Dimensions(dimension, 100, 100),
        external=Dimensions(dimension + 6, 106, 106),
        capacity_per_pallet=capacity,
        compatible_product_codes=frozenset(compatible),
    )


class DestinationRankingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.products = (
            _product("A", 100, 19_000),
            _product("B", 120, 2_000),
            _product("C", 130, 2_000),
        )
        self.data = PreparedData(products=self.products, current_boxes={})
        self.incumbent = {
            "A": _candidate("inc_A", 100, 20, ("A",)),
            "B": _candidate("inc_B", 120, 10, ("B",)),
            "C": _candidate("inc_C", 130, 10, ("C",)),
        }
        self.tier_destination = _candidate("tier", 100, 20, ("A", "B"))
        self.freight_destination = _candidate("freight", 110, 15, ("B",))

    def test_ranking_is_deterministic_and_rewards_tier_plus_freight(self) -> None:
        policy = FreightPolicy()
        forward = rank_destination_work_items(
            self.data,
            self.incumbent,
            (self.freight_destination, self.tier_destination),
            policy,
        )
        reverse = rank_destination_work_items(
            self.data,
            self.incumbent,
            (self.tier_destination, self.freight_destination),
            policy,
        )

        self.assertEqual(
            [item.candidate.internal for item in forward],
            [item.candidate.internal for item in reverse],
        )
        self.assertEqual(forward[0].candidate.internal, Dimensions(100, 100, 100))
        self.assertEqual(forward[0].product_codes, ("B",))
        self.assertEqual(forward[0].tier_crossings, 1)
        self.assertGreater(forward[0].procurement_opportunity_mills, 0)
        self.assertGreater(forward[0].freight_saving_opportunity_mills, 0)

    def test_work_item_builds_binary_destination_mapping(self) -> None:
        item = rank_destination_work_items(
            self.data,
            self.incumbent,
            (self.tier_destination,),
            FreightPolicy(),
        )[0]

        allowed = allowed_internals_for_destination(item)

        self.assertEqual(allowed, {"B": (Dimensions(100, 100, 100),)})
        self.assertNotIn("A", allowed)  # A already uses the target.

    def test_filters_size_opportunity_and_invalid_limits(self) -> None:
        self.assertEqual(
            rank_destination_work_items(
                self.data,
                self.incumbent,
                (self.tier_destination,),
                FreightPolicy(),
                min_skus=2,
                max_skus=2,
            ),
            (),
        )
        with self.assertRaisesRegex(ValueError, "max_skus"):
            rank_destination_work_items(
                self.data,
                self.incumbent,
                (),
                FreightPolicy(),
                min_skus=2,
                max_skus=1,
            )
        with self.assertRaisesRegex(ValueError, "max_destinations"):
            rank_destination_work_items(
                self.data,
                self.incumbent,
                (),
                FreightPolicy(),
                max_destinations=0,
            )


if __name__ == "__main__":
    unittest.main()
