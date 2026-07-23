import unittest

from bonsai.candidates import (
    _group_compromise_variants,
    _pallet_aligned_variants,
    generate_candidates,
)
from bonsai.geometry import (
    documented_axis_headspace_feasible,
    faq_reconciled_headspace_feasible,
    flexible_volume_headspace_feasible,
    flexible_product_fits_candidate,
    product_fits_candidate,
)
from bonsai.models import Dimensions, Product


class FlexibleHeadspaceTests(unittest.TestCase):
    def test_volume_can_be_redistributed_across_axes_with_zero_headspace(self) -> None:
        product_volume = 100 * 100 * 100
        self.assertTrue(
            flexible_volume_headspace_feasible(
                product_volume, Dimensions(100, 80, 125), 3.0
            )
        )

    def test_headspace_can_be_allocated_to_one_axis(self) -> None:
        product_volume = 100 * 100 * 100
        self.assertTrue(
            flexible_volume_headspace_feasible(product_volume, Dimensions(104, 100, 100), 3.0)
        )

    def test_excessive_headspace_in_all_axes_is_infeasible(self) -> None:
        product_volume = 100 * 100 * 100
        self.assertFalse(
            flexible_volume_headspace_feasible(product_volume, Dimensions(110, 110, 110), 3.0)
        )

    def test_flexible_fit_allows_compensating_axis_changes(self) -> None:
        product = Product(
            code="P1",
            current_box_type_id="B1",
            current_internal=Dimensions(100, 100, 100),
            net_weight_kg=1,
            annual_volume_by_plant={plant: 1 for plant in ("buenos_aires", "curitiba", "santiago", "monterrey", "bakersfield")},
        )
        candidate = Dimensions(90, 102, 109)
        self.assertTrue(flexible_product_fits_candidate(product, candidate, 3.0))
        self.assertFalse(faq_reconciled_headspace_feasible(product, candidate, 3.0))


class DocumentedAxisHeadspaceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.product = Product(
            code="P1",
            current_box_type_id="B1",
            current_internal=Dimensions(100, 100, 100),
            net_weight_kg=1,
            annual_volume_by_plant={
                "buenos_aires": 1,
                "curitiba": 0,
                "santiago": 0,
                "monterrey": 0,
                "bakersfield": 0,
            },
        )

    def test_current_dimensions_have_zero_headspace(self) -> None:
        self.assertTrue(
            documented_axis_headspace_feasible(self.product, Dimensions(100, 100, 100), 3.0)
        )

    def test_shrinking_an_axis_is_infeasible(self) -> None:
        self.assertFalse(
            documented_axis_headspace_feasible(self.product, Dimensions(99, 105, 105), 3.0)
        )

    def test_positive_headspace_above_cap_is_infeasible(self) -> None:
        self.assertFalse(
            documented_axis_headspace_feasible(self.product, Dimensions(107, 100, 100), 3.0)
        )


class FaqReconciledHeadspaceTests(DocumentedAxisHeadspaceTests):
    def test_shrinking_an_axis_is_allowed_when_volume_is_sufficient(self) -> None:
        self.assertTrue(
            faq_reconciled_headspace_feasible(self.product, Dimensions(99, 105, 105), 3.0)
        )

    def test_positive_headspace_above_cap_is_still_infeasible(self) -> None:
        self.assertFalse(
            faq_reconciled_headspace_feasible(self.product, Dimensions(107, 100, 100), 3.0)
        )

class PalletAlignedCandidatesTests(unittest.TestCase):
    def test_variants_preserve_or_expand_product_volume(self) -> None:
        profile = Dimensions(393, 255, 250)
        variants = _pallet_aligned_variants(profile, 3.0, limit=18)
        self.assertTrue(variants)
        self.assertTrue(
            all(
                flexible_volume_headspace_feasible(profile.volume_mm3, variant, 3.0)
                for variant in variants
            )
        )

    def test_seed_profile_is_accepted_for_pallet_variants(self) -> None:
        product = Product(
            code="P1",
            current_box_type_id="B1",
            current_internal=Dimensions(100, 100, 100),
            net_weight_kg=1,
            annual_volume_by_plant={
                "buenos_aires": 1,
                "curitiba": 0,
                "santiago": 0,
                "monterrey": 0,
                "bakersfield": 0,
            },
        )
        candidates = generate_candidates(
            (product,),
            3.0,
            pallet_variant_profile_limit=0,
            seed_pallet_variant_profiles=(Dimensions(100, 100, 100),),
        )
        self.assertTrue(candidates)

    def test_compromise_variants_fit_all_products_in_group(self) -> None:
        products = (
            Product(
                code="P1",
                current_box_type_id="B1",
                current_internal=Dimensions(100, 100, 100),
                net_weight_kg=1,
                annual_volume_by_plant={plant: 1 for plant in ("buenos_aires", "curitiba", "santiago", "monterrey", "bakersfield")},
            ),
            Product(
                code="P2",
                current_box_type_id="B2",
                current_internal=Dimensions(102, 98, 100),
                net_weight_kg=1,
                annual_volume_by_plant={plant: 1 for plant in ("buenos_aires", "curitiba", "santiago", "monterrey", "bakersfield")},
            ),
        )
        variants = _group_compromise_variants(products, 3.0, limit=18)
        self.assertTrue(variants)
        self.assertTrue(
            all(
                all(product_fits_candidate(product, variant, 3.0) for product in products)
                for variant in variants
            )
        )

if __name__ == "__main__":
    unittest.main()
