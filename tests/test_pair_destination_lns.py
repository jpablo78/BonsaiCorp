import json
from pathlib import Path
import tempfile
import unittest

from bonsai.config import FreightPolicy
from bonsai.destination_lns import rank_destination_work_items
from bonsai.models import CandidateBox, Dimensions, PLANTS, PreparedData, Product
from bonsai.pair_destination_lns import (
    allowed_internals_for_pair,
    load_excluded_pairs,
    pair_complementarity_metrics,
    rank_destination_pairs,
)


def _volumes(value: int) -> dict[str, int]:
    return {plant: value if plant == "buenos_aires" else 0 for plant in PLANTS}


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


class PairDestinationLnsTests(unittest.TestCase):
    def setUp(self) -> None:
        products = (
            _product("A", 100, 10_000),
            _product("B", 100, 8_000),
            _product("C", 130, 3_000),
        )
        self.data = PreparedData(products=products, current_boxes={})
        source_ab = _candidate("source_ab", 100, ("A", "B"))
        self.incumbent = {
            "A": source_ab,
            "B": source_ab,
            "C": _candidate("source_c", 130, ("C",)),
        }
        candidates = (
            _candidate("dest_a", 101, ("A",)),
            _candidate("dest_b", 102, ("B", "C")),
            _candidate("dest_ac", 103, ("A", "C")),
        )
        self.destinations = rank_destination_work_items(
            self.data,
            self.incumbent,
            candidates,
            FreightPolicy(),
            min_skus=1,
            min_gross_opportunity_mills=0,
        )

    def test_complementarity_delta_detects_pair_only_source_evacuation(self) -> None:
        groups = ((frozenset(("A", "B")), 18_000), (frozenset(("C",)), 3_000))
        metrics = pair_complementarity_metrics(
            frozenset(("A",)), frozenset(("B", "C")), groups
        )
        self.assertEqual(metrics, (0, 1, 1, 18_000))

    def test_ranking_prefers_pair_that_completes_source_and_releases_union(self) -> None:
        ranked = rank_destination_pairs(
            self.destinations,
            self.data,
            self.incumbent,
            max_pairs=3,
        )
        self.assertTrue(ranked)
        best = ranked[0]
        self.assertEqual(best.complementary_source_types, 1)
        self.assertEqual(best.complementary_source_volume, 18_000)
        allowed = allowed_internals_for_pair(best)
        self.assertEqual(set(allowed), set(best.product_codes))
        self.assertTrue(any(len(choices) == 1 for choices in allowed.values()))

    def test_excluded_summary_expands_prior_multidestination_set_to_pairs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "summary.json"
            path.write_text(
                json.dumps(
                    {
                        "attempts": [
                            {
                                "destinations": [
                                    {"internal_mm": [1, 2, 3]},
                                    {"internal_mm": [4, 5, 6]},
                                    {"internal_mm": [7, 8, 9]},
                                ]
                            },
                            {
                                "destination_internals_mm": [
                                    [10, 11, 12],
                                    [13, 14, 15],
                                ],
                                "status": "FEASIBLE",
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )
            excluded = load_excluded_pairs((path,))
        self.assertEqual(len(excluded), 4)
        self.assertIn(frozenset(((1, 2, 3), (7, 8, 9))), excluded)
        self.assertIn(frozenset(((10, 11, 12), (13, 14, 15))), excluded)
        optimal_only = load_excluded_pairs((path,), statuses=frozenset(("OPTIMAL",)))
        self.assertEqual(optimal_only, frozenset())


if __name__ == "__main__":
    unittest.main()
