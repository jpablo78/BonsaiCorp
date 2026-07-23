import itertools
import unittest

from bonsai.exact_candidates import exact_axis_bounds, generate_exact_candidates
from bonsai.decimal_candidates import (
    decimal_product_fits_candidate,
    generate_tenth_mm_candidates,
)
from bonsai.geometry import boxes_per_pallet, external_from_internal, product_fits_candidate
from bonsai.models import Dimensions, PLANTS, Product


class ExactBoundsTests(unittest.TestCase):
    def test_integer_bounds_include_positive_headspace_cap(self) -> None:
        self.assertEqual(exact_axis_bounds(100, 3.0), (90, 106))
        self.assertEqual(exact_axis_bounds(100, 4.5), (90, 108))
        self.assertEqual(exact_axis_bounds(100, 5.0), (90, 110))

    def test_fractional_source_dimensions_keep_integer_output_grid(self) -> None:
        self.assertEqual(exact_axis_bounds(394.6, 3.0), (356, 419))

    def test_generated_universe_contains_the_true_individual_capacity_maximum(self) -> None:
        product = Product(
            code="P1",
            current_box_type_id="B1",
            current_internal=Dimensions(100, 100, 100),
            net_weight_kg=1,
            annual_volume_by_plant={plant: 1 for plant in PLANTS},
        )
        candidates, stats = generate_exact_candidates((product,), 3.0)
        candidate_maximum = max(candidate.capacity_per_pallet for candidate in candidates)

        bounds = [range(lower, upper + 1) for lower, upper in (
            exact_axis_bounds(100, 3.0),
            exact_axis_bounds(100, 3.0),
            exact_axis_bounds(100, 3.0),
        )]
        brute_force_maximum = max(
            boxes_per_pallet(external_from_internal(internal, 3.0))
            for values in itertools.product(*bounds)
            for internal in (Dimensions(*values),)
            if product_fits_candidate(product, internal, 3.0)
        )
        self.assertEqual(candidate_maximum, brute_force_maximum)
        self.assertGreater(stats.feasible_signatures, 0)
        self.assertLessEqual(stats.nondominated_signatures, stats.feasible_signatures)

    def test_tenth_mm_candidates_match_decimal_oracle(self) -> None:
        product = Product(
            code="P1",
            current_box_type_id="B1",
            current_internal=Dimensions(100, 100, 100),
            net_weight_kg=1,
            annual_volume_by_plant={plant: 1 for plant in PLANTS},
        )
        candidates, stats = generate_tenth_mm_candidates((product,), 3.0)
        self.assertGreater(stats.feasible_signatures, 0)
        self.assertTrue(
            all(
                decimal_product_fits_candidate(product, candidate.internal, 3.0)
                for candidate in candidates
            )
        )
        self.assertTrue(
            any(
                any(not float(value).is_integer() for value in candidate.internal.as_tuple())
                for candidate in candidates
            )
        )


if __name__ == "__main__":
    unittest.main()
