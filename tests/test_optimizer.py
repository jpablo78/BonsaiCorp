import unittest
from unittest.mock import patch

from bonsai.config import FreightPolicy
from bonsai.costs import evaluate_assignments
from bonsai.geometry import boxes_per_pallet, external_from_internal
from bonsai.models import CandidateBox, Dimensions, PLANTS, PreparedData, Product
from bonsai.optimizer import solve_for_thickness


def _small_data(volume: int = 20_000) -> PreparedData:
    product = Product(
        code="P1",
        current_box_type_id="B1",
        current_internal=Dimensions(100, 100, 100),
        net_weight_kg=1,
        annual_volume_by_plant={
            plant: volume if plant == "buenos_aires" else 0 for plant in PLANTS
        },
    )
    return PreparedData(products=(product,), current_boxes={})


def _current_assignment(data: PreparedData) -> dict[str, CandidateBox]:
    product = data.products[0]
    external = external_from_internal(product.current_internal, 3.0)
    candidate = CandidateBox(
        candidate_id="incumbent",
        thickness_mm=3.0,
        internal=product.current_internal,
        external=external,
        capacity_per_pallet=boxes_per_pallet(external),
        compatible_product_codes=frozenset({product.code}),
    )
    return {product.code: candidate}


def _two_product_data(volume: int = 100) -> PreparedData:
    products = tuple(
        Product(
            code=code,
            current_box_type_id=f"B{index}",
            current_internal=internal,
            net_weight_kg=1,
            annual_volume_by_plant={
                plant: volume if plant == "buenos_aires" else 0 for plant in PLANTS
            },
        )
        for index, (code, internal) in enumerate(
            (
                ("P1", Dimensions(100, 100, 100)),
                ("P2", Dimensions(120, 100, 100)),
            ),
            start=1,
        )
    )
    return PreparedData(products=products, current_boxes={})


def _candidate(
    candidate_id: str,
    internal: Dimensions,
    capacity: int,
    compatible_codes: set[str],
) -> CandidateBox:
    return CandidateBox(
        candidate_id=candidate_id,
        thickness_mm=3.0,
        internal=internal,
        external=external_from_internal(internal, 3.0),
        capacity_per_pallet=capacity,
        compatible_product_codes=frozenset(compatible_codes),
    )


class OptimizerIncumbentTests(unittest.TestCase):
    def test_lns_can_reuse_a_precomputed_exact_universe(self) -> None:
        data = _small_data()
        incumbent = _current_assignment(data)
        universe = tuple(incumbent.values())
        with patch(
            "bonsai.optimizer.generate_exact_candidates",
            side_effect=AssertionError("exact grid should be reused"),
        ):
            result = solve_for_thickness(
                data,
                3.0,
                FreightPolicy(),
                time_limit_seconds=0.000001,
                num_search_workers=1,
                initial_assignment=incumbent,
                candidate_strategy="exact",
                free_product_codes={"P1"},
                precomputed_exact_candidates=universe,
            )

        self.assertEqual(result.costs.total_mills, result.incumbent_mills)

    def test_near_zero_time_never_regresses_warm_start(self) -> None:
        data = _small_data()
        policy = FreightPolicy()
        incumbent = _current_assignment(data)
        incumbent_cost = evaluate_assignments(data.products, incumbent, policy)
        result = solve_for_thickness(
            data,
            3.0,
            policy,
            time_limit_seconds=0.000001,
            num_search_workers=1,
            initial_assignment=incumbent,
            candidate_strategy="exact",
        )
        self.assertLessEqual(result.costs.total_mills, incumbent_cost.total_mills)
        self.assertEqual(result.incumbent_mills, incumbent_cost.total_mills)

    def test_zero_extra_pallets_builds_a_valid_fallback_without_warm_start(self) -> None:
        data = _small_data()
        result = solve_for_thickness(
            data,
            3.0,
            FreightPolicy(),
            time_limit_seconds=0.000001,
            num_search_workers=1,
            candidate_strategy="exact",
            max_extra_pallets=0,
        )
        self.assertEqual(result.costs.pallets, result.minimum_possible_pallets)

    def test_capacity_restrictions_reject_incomplete_candidate_universe(self) -> None:
        with self.assertRaisesRegex(ValueError, "complete exact candidate strategy"):
            solve_for_thickness(
                _small_data(),
                3.0,
                FreightPolicy(),
                candidate_strategy="heuristic",
                max_extra_pallets=0,
            )

    def test_cumulative_discount_model_matches_independent_costs_at_all_tiers(self) -> None:
        policy = FreightPolicy()
        for volume in (19_999, 20_000, 49_999, 50_000, 99_999, 100_000, 499_999, 500_000):
            with self.subTest(volume=volume):
                data = _small_data(volume)
                incumbent = _current_assignment(data)
                expected = evaluate_assignments(data.products, incumbent, policy)
                forced_candidate = incumbent["P1"]
                with patch(
                    "bonsai.optimizer.generate_exact_candidates",
                    return_value=((forced_candidate,), None),
                ):
                    result = solve_for_thickness(
                        data,
                        3.0,
                        policy,
                        time_limit_seconds=2,
                        num_search_workers=1,
                        initial_assignment=incumbent,
                        candidate_strategy="exact",
                    )
                self.assertEqual(result.status, "OPTIMAL")
                self.assertEqual(result.solver_objective_mills, expected.total_mills)
                self.assertEqual(result.costs.total_mills, expected.total_mills)

    def test_infeasible_cost_target_returns_protected_incumbent(self) -> None:
        data = _small_data(20_000)
        policy = FreightPolicy()
        incumbent = _current_assignment(data)
        expected = evaluate_assignments(data.products, incumbent, policy)
        with patch(
            "bonsai.optimizer.generate_exact_candidates",
            return_value=((incumbent["P1"],), None),
        ):
            result = solve_for_thickness(
                data,
                3.0,
                policy,
                time_limit_seconds=2,
                num_search_workers=1,
                initial_assignment=incumbent,
                candidate_strategy="exact",
                target_total_mills=expected.total_mills - 1,
            )
        self.assertEqual(result.status, "INFEASIBLE")
        self.assertEqual(result.selected_source, "incumbent")
        self.assertFalse(result.improved_incumbent)
        self.assertEqual(result.costs.total_mills, expected.total_mills)
        self.assertIsNone(result.solver_objective_mills)
        self.assertIsNone(result.best_bound_mills)

    def test_feasible_cost_target_reports_independently_evaluated_total(self) -> None:
        data = _small_data(20_000)
        policy = FreightPolicy()
        incumbent = _current_assignment(data)
        expected = evaluate_assignments(data.products, incumbent, policy)
        with patch(
            "bonsai.optimizer.generate_exact_candidates",
            return_value=((incumbent["P1"],), None),
        ):
            result = solve_for_thickness(
                data,
                3.0,
                policy,
                time_limit_seconds=2,
                num_search_workers=1,
                initial_assignment=incumbent,
                candidate_strategy="exact",
                target_total_mills=expected.total_mills,
            )
        self.assertEqual(result.status, "OPTIMAL")
        self.assertEqual(result.solver_objective_mills, expected.total_mills)
        self.assertEqual(result.costs.total_mills, expected.total_mills)
        self.assertIsNone(result.best_bound_mills)

    def test_negative_cost_target_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "target_total_mills cannot be negative"):
            solve_for_thickness(
                _small_data(),
                3.0,
                FreightPolicy(),
                target_total_mills=-1,
            )

    def test_lns_neighbourhood_requires_an_incumbent(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires an initial assignment"):
            solve_for_thickness(
                _small_data(),
                3.0,
                FreightPolicy(),
                free_product_codes={"P1"},
            )

    def test_lns_neighbourhood_rejects_unknown_product_codes(self) -> None:
        data = _small_data()
        with self.assertRaisesRegex(ValueError, "unknown products.*NOT_A_PRODUCT"):
            solve_for_thickness(
                data,
                3.0,
                FreightPolicy(),
                initial_assignment=_current_assignment(data),
                free_product_codes={"NOT_A_PRODUCT"},
            )

    def test_lns_fixes_other_products_and_filters_irrelevant_candidates(self) -> None:
        data = _two_product_data()
        incumbent_p1 = _candidate(
            "incumbent_p1", Dimensions(100, 100, 100), 10, {"P1"}
        )
        incumbent_p2 = _candidate(
            "incumbent_p2", Dimensions(120, 100, 100), 10, {"P2"}
        )
        better_for_free = _candidate(
            "better_for_free", Dimensions(101, 100, 100), 100, {"P1", "P2"}
        )
        irrelevant_for_fixed = _candidate(
            "irrelevant_for_fixed", Dimensions(121, 100, 100), 100, {"P2"}
        )
        incumbent = {"P1": incumbent_p1, "P2": incumbent_p2}
        candidates = (
            incumbent_p1,
            incumbent_p2,
            better_for_free,
            irrelevant_for_fixed,
        )

        with patch(
            "bonsai.optimizer.generate_exact_candidates",
            return_value=(candidates, None),
        ):
            result = solve_for_thickness(
                data,
                3.0,
                FreightPolicy(),
                time_limit_seconds=2,
                num_search_workers=1,
                initial_assignment=incumbent,
                candidate_strategy="exact",
                free_product_codes={"P1"},
            )

        self.assertEqual(result.status, "OPTIMAL")
        self.assertEqual(result.candidate_count, 3)
        self.assertEqual(result.assignment["P2"].internal, incumbent_p2.internal)
        self.assertEqual(result.assignment["P1"].internal, better_for_free.internal)

    def test_lns_keeps_global_pallet_budget_baseline_and_target_fallback(self) -> None:
        data = _two_product_data()
        incumbent_p1 = _candidate(
            "incumbent_p1", Dimensions(100, 100, 100), 10, {"P1"}
        )
        incumbent_p2 = _candidate(
            "incumbent_p2", Dimensions(120, 100, 100), 10, {"P2"}
        )
        best_p1 = _candidate("best_p1", Dimensions(101, 100, 100), 100, {"P1"})
        best_p2 = _candidate("best_p2", Dimensions(121, 100, 100), 100, {"P2"})
        incumbent = {"P1": incumbent_p1, "P2": incumbent_p2}
        candidates = (incumbent_p1, incumbent_p2, best_p1, best_p2)
        incumbent_cost = evaluate_assignments(data.products, incumbent, FreightPolicy())

        with patch(
            "bonsai.optimizer.generate_exact_candidates",
            return_value=(candidates, None),
        ):
            result = solve_for_thickness(
                data,
                3.0,
                FreightPolicy(),
                time_limit_seconds=2,
                num_search_workers=1,
                initial_assignment=incumbent,
                candidate_strategy="exact",
                max_extra_pallets=18,
                target_total_mills=incumbent_cost.total_mills - 1,
                free_product_codes=set(),
            )

        self.assertEqual(result.minimum_possible_pallets, 2)
        self.assertEqual(result.candidate_count, 2)
        self.assertEqual(result.status, "INFEASIBLE")
        self.assertEqual(result.selected_source, "incumbent")
        self.assertEqual(result.costs.total_mills, incumbent_cost.total_mills)

    def test_allowed_internals_requires_an_incumbent(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires an initial assignment"):
            solve_for_thickness(
                _small_data(),
                3.0,
                FreightPolicy(),
                allowed_internals_by_product={
                    "P1": {Dimensions(101, 100, 100)}
                },
            )

    def test_allowed_internals_rejects_unknown_products_and_invalid_values(self) -> None:
        data = _small_data()
        incumbent = _current_assignment(data)
        with self.assertRaisesRegex(ValueError, "unknown products.*NOT_A_PRODUCT"):
            solve_for_thickness(
                data,
                3.0,
                FreightPolicy(),
                initial_assignment=incumbent,
                allowed_internals_by_product={
                    "NOT_A_PRODUCT": {Dimensions(101, 100, 100)}
                },
            )
        with self.assertRaisesRegex(TypeError, "must contain Dimensions"):
            solve_for_thickness(
                data,
                3.0,
                FreightPolicy(),
                initial_assignment=incumbent,
                allowed_internals_by_product={"P1": {(101, 100, 100)}},
            )

    def test_allowed_internals_builds_binary_stay_or_target_neighbourhood(self) -> None:
        data = _two_product_data()
        incumbent_p1 = _candidate(
            "incumbent_p1", Dimensions(100, 100, 100), 10, {"P1"}
        )
        incumbent_p2 = _candidate(
            "incumbent_p2", Dimensions(120, 100, 100), 10, {"P2"}
        )
        target = _candidate(
            "target", Dimensions(101, 100, 100), 100, {"P1"}
        )
        tempting_but_forbidden = _candidate(
            "tempting_but_forbidden", Dimensions(102, 100, 100), 200, {"P1"}
        )
        incumbent = {"P1": incumbent_p1, "P2": incumbent_p2}
        candidates = (incumbent_p1, incumbent_p2, target, tempting_but_forbidden)

        with patch(
            "bonsai.optimizer.generate_exact_candidates",
            return_value=(candidates, None),
        ):
            result = solve_for_thickness(
                data,
                3.0,
                FreightPolicy(),
                time_limit_seconds=2,
                num_search_workers=1,
                initial_assignment=incumbent,
                candidate_strategy="exact",
                free_product_codes={"P1"},
                # The incumbent is intentionally omitted: the optimizer must
                # add it automatically to protect the fallback.
                allowed_internals_by_product={"P1": {target.internal}},
            )

        self.assertEqual(result.status, "OPTIMAL")
        self.assertEqual(result.candidate_count, 3)
        self.assertEqual(result.assignment["P1"].internal, target.internal)
        self.assertEqual(result.assignment["P2"].internal, incumbent_p2.internal)

    def test_allowed_internal_must_be_compatible_with_its_product(self) -> None:
        data = _two_product_data()
        incumbent_p1 = _candidate(
            "incumbent_p1", Dimensions(100, 100, 100), 10, {"P1"}
        )
        incumbent_p2 = _candidate(
            "incumbent_p2", Dimensions(120, 100, 100), 10, {"P2"}
        )
        incompatible_for_p1 = _candidate(
            "only_p2", Dimensions(121, 100, 100), 100, {"P2"}
        )
        incumbent = {"P1": incumbent_p1, "P2": incumbent_p2}

        with patch(
            "bonsai.optimizer.generate_exact_candidates",
            return_value=((incumbent_p1, incumbent_p2, incompatible_for_p1), None),
        ):
            with self.assertRaisesRegex(
                ValueError, "allowed internal geometries are infeasible for P1"
            ):
                solve_for_thickness(
                    data,
                    3.0,
                    FreightPolicy(),
                    initial_assignment=incumbent,
                    candidate_strategy="exact",
                    free_product_codes={"P1"},
                    allowed_internals_by_product={
                        "P1": {incompatible_for_p1.internal}
                    },
                )


if __name__ == "__main__":
    unittest.main()
