import unittest

from bonsai.config import FreightPolicy
from bonsai.costs import evaluate_assignments
from bonsai.geometry import external_from_internal
from bonsai.models import CandidateBox, Dimensions, PLANTS, PreparedData, Product
from bonsai.scip_optimizer import solve_with_scip


def _product(code: str, internal: Dimensions, volume: int) -> Product:
    return Product(
        code=code,
        current_box_type_id=f"BOX_{code}",
        current_internal=internal,
        net_weight_kg=1,
        annual_volume_by_plant={
            plant: volume if plant == "buenos_aires" else 0 for plant in PLANTS
        },
    )


def _candidate(
    candidate_id: str,
    internal: Dimensions,
    capacity: int,
    codes: set[str],
) -> CandidateBox:
    return CandidateBox(
        candidate_id=candidate_id,
        thickness_mm=3.0,
        internal=internal,
        external=external_from_internal(internal, 3.0),
        capacity_per_pallet=capacity,
        compatible_product_codes=frozenset(codes),
    )


class ScipOptimizerTests(unittest.TestCase):
    def test_lp_relaxation_exposes_assignment_arc_values(self) -> None:
        product = _product("P1", Dimensions(100, 100, 100), 20_000)
        data = PreparedData((product,), {})
        first = _candidate("first", Dimensions(100, 100, 100), 100, {"P1"})
        second = _candidate("second", Dimensions(101, 100, 100), 100, {"P1"})
        incumbent = {"P1": first}

        result = solve_with_scip(
            data,
            3.0,
            FreightPolicy(),
            time_limit_seconds=2,
            num_threads=1,
            initial_assignment=incumbent,
            precomputed_exact_candidates=(first, second),
            relax_integrality=True,
        )

        self.assertTrue(result.relaxation)
        self.assertEqual(result.selected_source, "lp_relaxation_incumbent")
        self.assertEqual(result.assignment, incumbent)
        self.assertAlmostEqual(sum(result.assignment_arc_values.values()), 1.0)
        self.assertEqual(
            set(result.assignment_arc_values),
            {("P1", first.internal), ("P1", second.internal)},
        )
        self.assertIsNotNone(result.solver_objective_mills)

    def test_cumulative_threshold_cost_matches_independent_evaluator(self) -> None:
        policy = FreightPolicy()
        for volume in (
            19_999,
            20_000,
            49_999,
            50_000,
            99_999,
            100_000,
            499_999,
            500_000,
        ):
            with self.subTest(volume=volume):
                product = _product("P1", Dimensions(100, 100, 100), volume)
                data = PreparedData((product,), {})
                candidate = _candidate(
                    "only", product.current_internal, 100, {product.code}
                )
                incumbent = {product.code: candidate}
                expected = evaluate_assignments(data.products, incumbent, policy)
                result = solve_with_scip(
                    data,
                    3.0,
                    policy,
                    time_limit_seconds=2,
                    num_threads=1,
                    initial_assignment=incumbent,
                    precomputed_exact_candidates=(candidate,),
                )
                self.assertEqual(result.status, "OPTIMAL")
                self.assertEqual(result.solver_objective_mills, expected.total_mills)
                self.assertEqual(result.costs.total_mills, expected.total_mills)
                self.assertEqual(result.threshold_variable_count, 0)

    def test_scip_finds_tier_consolidation_and_protects_incumbent(self) -> None:
        products = (
            _product("P1", Dimensions(100, 100, 100), 10_000),
            _product("P2", Dimensions(120, 100, 100), 10_000),
        )
        data = PreparedData(products, {})
        first = _candidate("first", Dimensions(100, 100, 100), 100, {"P1"})
        second = _candidate("second", Dimensions(120, 100, 100), 100, {"P2"})
        shared = _candidate("shared", Dimensions(110, 100, 100), 100, {"P1", "P2"})
        incumbent = {"P1": first, "P2": second}
        incumbent_cost = evaluate_assignments(products, incumbent, FreightPolicy())

        result = solve_with_scip(
            data,
            3.0,
            FreightPolicy(),
            time_limit_seconds=2,
            num_threads=1,
            initial_assignment=incumbent,
            precomputed_exact_candidates=(first, shared, second),
        )

        self.assertEqual(result.status, "OPTIMAL")
        self.assertTrue(result.improved_incumbent)
        self.assertLess(result.costs.total_mills, incumbent_cost.total_mills)
        self.assertEqual(result.assignment["P1"].internal, shared.internal)
        self.assertEqual(result.assignment["P2"].internal, shared.internal)
        self.assertEqual(
            result.solver_objective_mills,
            evaluate_assignments(products, result.assignment, FreightPolicy()).total_mills,
        )

    def test_pallet_budget_is_measured_from_global_minimum(self) -> None:
        product = _product("P1", Dimensions(100, 100, 100), 100)
        data = PreparedData((product,), {})
        minimum = _candidate("minimum", Dimensions(100, 100, 100), 100, {"P1"})
        too_many = _candidate("too_many", Dimensions(101, 100, 100), 10, {"P1"})

        result = solve_with_scip(
            data,
            3.0,
            FreightPolicy(),
            time_limit_seconds=2,
            num_threads=1,
            initial_assignment={"P1": minimum},
            max_extra_pallets=0,
            precomputed_exact_candidates=(minimum, too_many),
        )

        self.assertEqual(result.minimum_possible_pallets, 1)
        self.assertEqual(result.costs.pallets, 1)
        self.assertEqual(result.assignment_variable_count, 0)
        self.assertEqual(result.fixed_product_count, 1)

    def test_objective_filter_prunes_arc_that_cannot_match_incumbent(self) -> None:
        product = _product("P1", Dimensions(100, 100, 100), 100)
        data = PreparedData((product,), {})
        incumbent_box = _candidate(
            "incumbent", Dimensions(100, 100, 100), 100, {"P1"}
        )
        expensive_box = _candidate(
            "expensive", Dimensions(101, 100, 100), 1, {"P1"}
        )

        result = solve_with_scip(
            data,
            3.0,
            FreightPolicy(),
            time_limit_seconds=2,
            num_threads=1,
            initial_assignment={"P1": incumbent_box},
            precomputed_exact_candidates=(incumbent_box, expensive_box),
        )

        self.assertEqual(result.pruned_assignment_count, 1)
        self.assertEqual(result.fixed_product_count, 1)
        self.assertEqual(result.assignment_variable_count, 0)
        self.assertEqual(result.objective_scale_mills, 60)

    def test_single_positive_arc_procurement_is_linearized_directly(self) -> None:
        product = _product("P1", Dimensions(100, 100, 100), 50_000)
        data = PreparedData((product,), {})
        first = _candidate("first", Dimensions(100, 100, 100), 100, {"P1"})
        second = _candidate("second", Dimensions(101, 100, 100), 100, {"P1"})
        incumbent = {"P1": first}

        result = solve_with_scip(
            data,
            3.0,
            FreightPolicy(),
            time_limit_seconds=2,
            num_threads=1,
            initial_assignment=incumbent,
            precomputed_exact_candidates=(first, second),
        )

        self.assertEqual(result.assignment_variable_count, 2)
        self.assertEqual(result.threshold_variable_count, 0)
        self.assertEqual(
            result.solver_objective_mills,
            evaluate_assignments(data.products, result.assignment, FreightPolicy()).total_mills,
        )

    def test_infeasible_target_returns_protected_incumbent(self) -> None:
        product = _product("P1", Dimensions(100, 100, 100), 20_000)
        data = PreparedData((product,), {})
        candidate = _candidate("only", product.current_internal, 100, {"P1"})
        incumbent = {"P1": candidate}
        incumbent_cost = evaluate_assignments(data.products, incumbent, FreightPolicy())

        result = solve_with_scip(
            data,
            3.0,
            FreightPolicy(),
            time_limit_seconds=2,
            num_threads=1,
            initial_assignment=incumbent,
            target_total_mills=incumbent_cost.total_mills - 1,
            precomputed_exact_candidates=(candidate,),
        )

        self.assertEqual(result.status, "INFEASIBLE")
        self.assertEqual(result.selected_source, "incumbent")
        self.assertEqual(result.costs.total_mills, incumbent_cost.total_mills)
        self.assertFalse(result.target_met)

    def test_local_branching_bounds_changed_products(self) -> None:
        products = (
            _product("P1", Dimensions(100, 100, 100), 10_000),
            _product("P2", Dimensions(120, 100, 100), 10_000),
        )
        data = PreparedData(products, {})
        first = _candidate("first", Dimensions(100, 100, 100), 100, {"P1"})
        second = _candidate("second", Dimensions(120, 100, 100), 100, {"P2"})
        shared = _candidate("shared", Dimensions(110, 100, 100), 100, {"P1", "P2"})
        incumbent = {"P1": first, "P2": second}

        blocked = solve_with_scip(
            data,
            3.0,
            FreightPolicy(),
            time_limit_seconds=2,
            num_threads=1,
            initial_assignment=incumbent,
            max_changed_products=1,
            precomputed_exact_candidates=(first, shared, second),
        )
        self.assertFalse(blocked.improved_incumbent)
        self.assertEqual(blocked.changed_product_count, 0)

        admitted = solve_with_scip(
            data,
            3.0,
            FreightPolicy(),
            time_limit_seconds=2,
            num_threads=1,
            initial_assignment=incumbent,
            min_changed_products=2,
            max_changed_products=2,
            precomputed_exact_candidates=(first, shared, second),
        )
        self.assertTrue(admitted.improved_incumbent)
        self.assertEqual(admitted.changed_product_count, 2)

    def test_restricted_lns_eliminates_fixed_sku_and_absorbs_its_costs(self) -> None:
        products = (
            _product("P1", Dimensions(100, 100, 100), 10_000),
            _product("P2", Dimensions(120, 100, 100), 10_000),
        )
        data = PreparedData(products, {})
        first = _candidate("first", Dimensions(100, 100, 100), 100, {"P1"})
        fixed = _candidate("fixed", Dimensions(120, 100, 100), 100, {"P1", "P2"})
        incumbent = {"P1": first, "P2": fixed}

        result = solve_with_scip(
            data,
            3.0,
            FreightPolicy(),
            time_limit_seconds=2,
            num_threads=1,
            initial_assignment=incumbent,
            free_product_codes={"P1"},
            allowed_internals_by_product={"P1": {fixed.internal}},
            precomputed_exact_candidates=(first, fixed),
        )

        self.assertEqual(result.status, "OPTIMAL")
        self.assertTrue(result.improved_incumbent)
        self.assertEqual(result.assignment["P1"].internal, fixed.internal)
        self.assertEqual(result.assignment["P2"].internal, fixed.internal)
        # P2 is a true fixed constant: no assignment variable is created for it.
        self.assertEqual(result.assignment_variable_count, 2)
        self.assertEqual(result.fixed_product_count, 1)
        self.assertEqual(
            result.solver_objective_mills,
            evaluate_assignments(products, result.assignment, FreightPolicy()).total_mills,
        )

    def test_restricted_lns_allowed_set_keeps_incumbent_and_excludes_other_arcs(self) -> None:
        product = _product("P1", Dimensions(100, 100, 100), 10_000)
        data = PreparedData((product,), {})
        incumbent_box = _candidate(
            "incumbent", Dimensions(100, 100, 100), 10, {"P1"}
        )
        target = _candidate("target", Dimensions(101, 100, 100), 100, {"P1"})
        forbidden = _candidate("forbidden", Dimensions(102, 100, 100), 200, {"P1"})

        result = solve_with_scip(
            data,
            3.0,
            FreightPolicy(),
            time_limit_seconds=2,
            num_threads=1,
            initial_assignment={"P1": incumbent_box},
            free_product_codes={"P1"},
            # The incumbent is deliberately omitted and must be added safely.
            allowed_internals_by_product={"P1": {target.internal}},
            precomputed_exact_candidates=(incumbent_box, target, forbidden),
        )

        self.assertEqual(result.status, "OPTIMAL")
        self.assertEqual(result.assignment_variable_count, 2)
        self.assertEqual(result.assignment["P1"].internal, target.internal)

    def test_restricted_lns_validates_codes_and_allowed_geometry_types(self) -> None:
        product = _product("P1", Dimensions(100, 100, 100), 100)
        data = PreparedData((product,), {})
        incumbent_box = _candidate("only", product.current_internal, 100, {"P1"})
        incumbent = {"P1": incumbent_box}
        with self.assertRaisesRegex(ValueError, "unknown products.*UNKNOWN"):
            solve_with_scip(
                data,
                3.0,
                FreightPolicy(),
                initial_assignment=incumbent,
                free_product_codes={"UNKNOWN"},
                precomputed_exact_candidates=(incumbent_box,),
            )
        with self.assertRaisesRegex(TypeError, "must contain Dimensions"):
            solve_with_scip(
                data,
                3.0,
                FreightPolicy(),
                initial_assignment=incumbent,
                allowed_internals_by_product={"P1": {(101, 100, 100)}},
                precomputed_exact_candidates=(incumbent_box,),
            )


if __name__ == "__main__":
    unittest.main()
