import unittest

from bonsai.models import CandidateBox, Dimensions
from bonsai.neighborhoods import Neighborhood, TierNeighborhoodPlan
from bonsai.tier_lns import generate_work_items


def _candidate(length: int) -> CandidateBox:
    return CandidateBox(
        candidate_id=str(length),
        thickness_mm=3.0,
        internal=Dimensions(length - 6, 100, 100),
        external=Dimensions(length, 106, 106),
        capacity_per_pallet=100,
        compatible_product_codes=frozenset(),
    )


def _neighborhood(identifier: str, codes: tuple[str, ...], kind: str) -> Neighborhood:
    return Neighborhood(
        neighborhood_id=identifier,
        kind=kind,
        product_codes=codes,
        source_types=((3.0, len(codes), 1, 1),),
        targets=(),
        selected_donor_volume=0,
        reaches_primary_target=False,
    )


class TierLnsWorkItemTests(unittest.TestCase):
    def test_generates_unique_stars_components_and_unions(self) -> None:
        stars = (
            _neighborhood("star_0000", ("A", "B"), "star"),
            _neighborhood("star_0001", ("B", "C"), "star"),
            _neighborhood("star_0002", ("D",), "star"),
        )
        # This component duplicates the union of the first two stars.
        components = (
            _neighborhood("component_0000", ("A", "B", "C"), "component"),
            _neighborhood("component_0001", ("D",), "component"),
        )
        plan = TierNeighborhoodPlan(targets=(), stars=stars, components=components)

        items = generate_work_items(
            plan,
            max_star_combination=2,
            max_component_combination=2,
        )

        code_sets = [frozenset(item.product_codes) for item in items]
        self.assertEqual(len(code_sets), len(set(code_sets)))
        self.assertIn(frozenset(("A", "B", "C")), code_sets)
        self.assertIn(frozenset(("A", "B", "D")), code_sets)
        self.assertIn(frozenset(("A", "B", "C", "D")), code_sets)

    def test_caps_neighborhood_size_and_count(self) -> None:
        plan = TierNeighborhoodPlan(
            targets=(),
            stars=(
                _neighborhood("star_0000", ("A",), "star"),
                _neighborhood("star_0001", ("B",), "star"),
                _neighborhood("star_0002", ("C",), "star"),
            ),
            components=(),
        )

        items = generate_work_items(
            plan,
            max_star_combination=3,
            max_component_combination=1,
            max_skus=2,
            max_neighborhoods=4,
        )

        self.assertEqual(len(items), 4)
        self.assertTrue(all(item.sku_count <= 2 for item in items))

    def test_rejects_invalid_limits(self) -> None:
        plan = TierNeighborhoodPlan(targets=(), stars=(), components=())
        with self.assertRaisesRegex(ValueError, "max_star_combination"):
            generate_work_items(plan, max_star_combination=0)
        with self.assertRaisesRegex(ValueError, "max_neighborhoods"):
            generate_work_items(plan, max_neighborhoods=0)


if __name__ == "__main__":
    unittest.main()
