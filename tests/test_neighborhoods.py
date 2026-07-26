import unittest

from bonsai.costs import box_type_key
from bonsai.models import CandidateBox, Dimensions, PLANTS, PreparedData, Product
from bonsai.neighborhoods import (
    build_component_neighborhoods,
    build_star_neighborhoods,
    build_tier_neighborhoods,
    identify_tier_targets,
)


def _demand(buenos_aires: int) -> dict[str, int]:
    return {
        plant: buenos_aires if plant == "buenos_aires" else 0
        for plant in PLANTS
    }


def _product(code: str, volume: int) -> Product:
    return Product(
        code=code,
        current_box_type_id=f"current_{code}",
        current_internal=Dimensions(100, 100, 100),
        net_weight_kg=1,
        annual_volume_by_plant=_demand(volume),
    )


def _candidate(
    candidate_id: str,
    external_length: int,
    compatible_codes: set[str],
) -> CandidateBox:
    external = Dimensions(external_length, 106, 106)
    return CandidateBox(
        candidate_id=candidate_id,
        thickness_mm=3.0,
        internal=Dimensions(external_length - 6, 100, 100),
        external=external,
        capacity_per_pallet=1_000,
        compatible_product_codes=frozenset(compatible_codes),
    )


class TierNeighborhoodTests(unittest.TestCase):
    def _single_target_case(self):
        receiver = _product("R", 97_000)
        donor = _product("D", 3_500)
        sibling = _product("S", 100)
        products = (receiver, donor, sibling)
        incumbent_a = _candidate("inc_a", 106, {"R"})
        incumbent_b = _candidate("inc_b", 107, {"D", "S"})
        exact_a = _candidate("exact_a", 106, {"R", "D"})
        exact_b = _candidate("exact_b", 107, {"D", "S"})
        assignment = {"R": incumbent_a, "D": incumbent_b, "S": incumbent_b}
        data = PreparedData(products=products, current_boxes={})
        return data, assignment, (exact_a, exact_b), exact_a, exact_b

    def test_identifies_next_tier_gap_and_complete_donor_group(self) -> None:
        data, assignment, candidates, exact_a, exact_b = self._single_target_case()

        targets = identify_tier_targets(
            data,
            assignment,
            candidates,
            plants=("buenos_aires",),
            max_gap_units=5_000,
            require_reachable=True,
        )

        self.assertEqual(len(targets), 1)
        target = targets[0]
        self.assertEqual(target.box_type, box_type_key(exact_a))
        self.assertEqual(target.current_volume, 97_000)
        self.assertEqual(target.current_tier_name, "tier_3")
        self.assertEqual(target.current_factor_percent, 90)
        self.assertEqual(target.next_tier_name, "tier_4")
        self.assertEqual(target.next_factor_percent, 80)
        self.assertEqual(target.next_threshold, 100_000)
        self.assertEqual(target.gap_units, 3_000)
        self.assertAlmostEqual(target.gap_ratio, 0.03)
        self.assertEqual(target.discount_per_unit_mills, 60)
        self.assertEqual(target.incumbent_discount_value_mills, 5_820_000)
        self.assertEqual(target.threshold_discount_value_mills, 6_000_000)
        self.assertEqual(target.eligible_donor_volume, 3_500)
        self.assertTrue(target.reachable_from_eligible_donors)
        self.assertAlmostEqual(target.donor_coverage_ratio, 3_500 / 3_000)

        self.assertEqual(len(target.donor_groups), 1)
        donor_group = target.donor_groups[0]
        self.assertEqual(donor_group.source_type, box_type_key(exact_b))
        self.assertEqual(donor_group.source_codes, ("D", "S"))
        self.assertEqual(donor_group.eligible_codes, ("D",))
        self.assertEqual(donor_group.eligible_volume_at_target_plant, 3_500)
        self.assertEqual(donor_group.source_volume_at_target_plant, 3_600)

    def test_star_includes_receiver_and_whole_source_groups(self) -> None:
        data, assignment, candidates, exact_a, exact_b = self._single_target_case()
        targets = identify_tier_targets(
            data,
            assignment,
            candidates,
            plants=("buenos_aires",),
            max_gap_units=5_000,
            require_reachable=True,
        )

        stars = build_star_neighborhoods(targets)

        self.assertEqual(len(stars), 1)
        star = stars[0]
        self.assertEqual(star.neighborhood_id, "star_0000")
        self.assertEqual(star.product_codes, ("D", "R", "S"))
        self.assertEqual(
            star.source_types,
            tuple(sorted((box_type_key(exact_a), box_type_key(exact_b)))),
        )
        self.assertEqual(star.selected_donor_volume, 3_500)
        self.assertTrue(star.reaches_primary_target)

    def test_sku_cap_never_splits_an_incumbent_source_group(self) -> None:
        data, assignment, candidates, _, _ = self._single_target_case()
        targets = identify_tier_targets(
            data,
            assignment,
            candidates,
            plants=("buenos_aires",),
            max_gap_units=5_000,
        )

    # R encaja, pero agregar sólo D dividiría el grupo completo (D, S).
        stars = build_star_neighborhoods(targets, max_skus=2)

        self.assertEqual(stars, ())

    def test_overlapping_stars_form_one_connected_component(self) -> None:
        a = _product("A", 19_000)
        b = _product("B", 19_000)
        c1 = _product("C1", 1_000)
        c2 = _product("C2", 1_000)
        data = PreparedData(products=(a, b, c1, c2), current_boxes={})
        incumbent_a = _candidate("inc_a", 106, {"A"})
        incumbent_b = _candidate("inc_b", 107, {"B"})
        incumbent_c = _candidate("inc_c", 108, {"C1", "C2"})
        assignment = {
            "A": incumbent_a,
            "B": incumbent_b,
            "C1": incumbent_c,
            "C2": incumbent_c,
        }
        exact_a = _candidate("exact_a", 106, {"A", "C1"})
        exact_b = _candidate("exact_b", 107, {"B", "C2"})
        exact_c = _candidate("exact_c", 108, {"C1", "C2"})

        plan = build_tier_neighborhoods(
            data,
            assignment,
            (exact_a, exact_b, exact_c),
            plants=("buenos_aires",),
            max_gap_units=1_000,
            require_reachable=True,
        )

        self.assertEqual(len(plan.targets), 2)
        self.assertEqual(len(plan.stars), 2)
        self.assertEqual(len(plan.components), 1)
        component = plan.components[0]
        self.assertEqual(component.neighborhood_id, "component_0000")
        self.assertEqual(component.product_codes, ("A", "B", "C1", "C2"))
        self.assertEqual(component.target_count, 2)
        self.assertTrue(component.reaches_primary_target)

        reversed_components = build_component_neighborhoods(reversed(plan.stars))
        self.assertEqual(reversed_components, plan.components)

    def test_results_are_independent_from_input_order(self) -> None:
        data, assignment, candidates, _, _ = self._single_target_case()
        forward = identify_tier_targets(
            data,
            assignment,
            candidates,
            plants=("buenos_aires",),
            max_gap_units=5_000,
        )
        reversed_data = PreparedData(
            products=tuple(reversed(data.products)), current_boxes={}
        )
        reversed_assignment = dict(reversed(tuple(assignment.items())))
        backward = identify_tier_targets(
            reversed_data,
            reversed_assignment,
            reversed(candidates),
            plants=("buenos_aires",),
            max_gap_units=5_000,
        )

        self.assertEqual(backward, forward)

    def test_missing_incumbent_design_is_rejected(self) -> None:
        data, assignment, candidates, _, _ = self._single_target_case()

        with self.assertRaisesRegex(ValueError, "incumbent physical types"):
            identify_tier_targets(data, assignment, candidates[:1])


if __name__ == "__main__":
    unittest.main()
