import random
import unittest

from bonsai.annealing import (
    IncrementalAssignmentState,
    build_targets_by_code,
    guided_single_proposal,
    minimum_pallets_for_candidates,
    simulated_annealing,
)
from bonsai.config import FreightPolicy
from bonsai.costs import evaluate_assignments
from bonsai.models import CandidateBox, Dimensions, PLANTS, Product


class IncrementalAnnealingTests(unittest.TestCase):
    @staticmethod
    def _product(code: str, buenos_aires: int, santiago: int = 0) -> Product:
        return Product(
            code=code,
            current_box_type_id=f"current_{code}",
            current_internal=Dimensions(100, 100, 100),
            net_weight_kg=1,
            annual_volume_by_plant={
                plant: (
                    buenos_aires
                    if plant == "buenos_aires"
                    else santiago if plant == "santiago" else 0
                )
                for plant in PLANTS
            },
        )

    @staticmethod
    def _candidate(
        candidate_id: str,
        external_length: int,
        capacity: int,
        compatible_codes: set[str],
    ) -> CandidateBox:
        return CandidateBox(
            candidate_id=candidate_id,
            thickness_mm=3.0,
            internal=Dimensions(external_length - 6, 100, 100),
            external=Dimensions(external_length, 106, 106),
            capacity_per_pallet=capacity,
            compatible_product_codes=frozenset(compatible_codes),
        )

    def test_incremental_deltas_match_full_evaluation_across_tiers(self) -> None:
        products = (
            self._product("P1", 15_000, 5_000),
            self._product("P2", 35_000, 15_000),
            self._product("P3", 60_000, 25_000),
        )
        all_codes = {product.code for product in products}
        box_a = self._candidate("a", 106, 100, all_codes)
        box_b = self._candidate("b", 107, 80, all_codes)
        box_c = self._candidate("c", 108, 120, all_codes)
        assignment = {"P1": box_a, "P2": box_a, "P3": box_b}
        policy = FreightPolicy()
        state = IncrementalAssignmentState(products, assignment, policy)

        for code, target in (("P1", box_b), ("P3", box_c), ("P2", box_c)):
            before = evaluate_assignments(products, state.assignment, policy)
            move = state.calculate_move(code, target)
            self.assertIsNotNone(move)
            changed = dict(state.assignment)
            changed[code] = target
            after = evaluate_assignments(products, changed, policy)
            self.assertEqual(
                move.packaging_delta_mills,
                after.packaging_mills - before.packaging_mills,
            )
            self.assertEqual(
                move.freight_delta_mills, after.freight_mills - before.freight_mills
            )
            self.assertEqual(move.pallet_delta, after.pallets - before.pallets)
            self.assertEqual(move.type_delta, after.types - before.types)
            state.apply(move)
            checked = state.validate()
            self.assertEqual(checked.total_mills, after.total_mills)

    def test_stale_move_is_rejected(self) -> None:
        products = (self._product("P1", 10_000),)
        box_a = self._candidate("a", 106, 100, {"P1"})
        box_b = self._candidate("b", 107, 100, {"P1"})
        state = IncrementalAssignmentState(
            products, {"P1": box_a}, FreightPolicy()
        )
        move = state.calculate_move("P1", box_b)
        state.apply(move)
        with self.assertRaisesRegex(ValueError, "stale annealing move"):
            state.apply(move)

    def test_group_delta_matches_full_evaluation_at_discount_threshold(self) -> None:
        products = (
            self._product("P1", 11_000),
            self._product("P2", 9_000),
            self._product("P3", 15_000),
        )
        all_codes = {product.code for product in products}
        box_a = self._candidate("a", 106, 100, all_codes)
        box_b = self._candidate("b", 107, 100, all_codes)
        assignment = {"P1": box_a, "P2": box_a, "P3": box_b}
        policy = FreightPolicy()
        state = IncrementalAssignmentState(products, assignment, policy)
        before = evaluate_assignments(products, assignment, policy)

        move = state.calculate_group_move(("P1", "P2"), box_b)
        self.assertIsNotNone(move)
        changed = dict(assignment)
        changed["P1"] = box_b
        changed["P2"] = box_b
        after = evaluate_assignments(products, changed, policy)
        self.assertEqual(
            move.packaging_delta_mills,
            after.packaging_mills - before.packaging_mills,
        )
        self.assertEqual(move.total_delta_mills, after.total_mills - before.total_mills)
        self.assertEqual(move.type_delta, after.types - before.types)
        state.apply_group(move)
        self.assertEqual(state.validate().total_mills, after.total_mills)

    def test_guided_proposal_prefers_low_exact_delta(self) -> None:
        products = (self._product("P1", 10_000),)
        box_a = self._candidate("a", 106, 100, {"P1"})
        cheap = self._candidate("cheap", 107, 100, {"P1"})
        costly = self._candidate("costly", 108, 50, {"P1"})
        state = IncrementalAssignmentState(
            products, {"P1": box_a}, FreightPolicy()
        )
        move = guided_single_proposal(
            state,
            {"P1": (box_a, cheap, costly)},
            ("P1",),
            random.Random(17),
            sample_size=100,
            used_target_probability=0,
            greediness=1,
        )
        self.assertEqual(move.target_candidate, cheap)

    def test_target_precomputation_deduplicates_physical_boxes(self) -> None:
        products = (self._product("P1", 10_000), self._product("P2", 10_000))
        box_a = self._candidate("a", 106, 100, {"P1", "P2"})
        box_b = self._candidate("b", 107, 100, {"P1", "P2"})
        duplicate_b = self._candidate("duplicate_b", 107, 100, {"P1", "P2"})
        assignment = {"P1": box_a, "P2": box_a}

        targets = build_targets_by_code(
            products,
            assignment,
            (box_b, duplicate_b),
            free_product_codes={"P1"},
        )

        self.assertEqual(set(targets), {"P1"})
        self.assertEqual(len(targets["P1"]), 2)
        self.assertEqual(
            minimum_pallets_for_candidates(products, targets, assignment), 200
        )

    def test_annealing_accepts_worse_states_but_returns_best_validated(self) -> None:
        products = (
            self._product("P1", 10_000),
            self._product("P2", 10_000),
        )
        all_codes = {"P1", "P2"}
        box_a = self._candidate("a", 106, 100, all_codes)
        box_b = self._candidate("b", 107, 90, all_codes)
        assignment = {"P1": box_a, "P2": box_a}
        initial = evaluate_assignments(products, assignment, FreightPolicy())

        result = simulated_annealing(
            products,
            assignment,
            (box_a, box_b),
            FreightPolicy(),
            max_steps=200,
            random_seed=7,
            initial_temperature_usd=1_000_000,
            final_temperature_usd=1_000_000,
            validation_interval=5,
        )

        self.assertGreater(result.accepted_worse_moves, 0)
        self.assertLessEqual(result.costs.total_mills, initial.total_mills)
        self.assertEqual(
            result.costs.total_mills,
            evaluate_assignments(products, result.assignment, FreightPolicy()).total_mills,
        )

    def test_pallet_budget_blocks_excess(self) -> None:
        products = (self._product("P1", 10_000),)
        fast = self._candidate("fast", 106, 100, {"P1"})
        slow = self._candidate("slow", 107, 50, {"P1"})
        result = simulated_annealing(
            products,
            {"P1": fast},
            (fast, slow),
            FreightPolicy(),
            max_steps=100,
            random_seed=5,
            initial_temperature_usd=1_000_000,
            final_temperature_usd=1_000_000,
            max_extra_pallets=0,
        )
        self.assertEqual(result.max_pallets, 100)
        self.assertEqual(result.current_costs.pallets, 100)
        self.assertEqual(result.accepted_moves, 0)

    def test_lns_pallet_budget_uses_global_candidate_minimum(self) -> None:
        products = (self._product("P1", 10_000), self._product("P2", 10_000))
        all_codes = {"P1", "P2"}
        incumbent = self._candidate("incumbent", 106, 50, all_codes)
        fast = self._candidate("fast", 107, 100, all_codes)
        result = simulated_annealing(
            products,
            {"P1": incumbent, "P2": fast},
            (incumbent, fast),
            FreightPolicy(),
            max_steps=0,
            free_product_codes={"P1"},
            max_extra_pallets=100,
        )
        # Both SKUs have a 100-pallet global minimum.  P2 is fixed during the
        # walk, but its incumbent does not loosen the optimizer-style budget.
        self.assertEqual(result.minimum_pallets, 200)
        self.assertEqual(result.max_pallets, 300)

    def test_restarts_preserve_exact_best_and_are_counted(self) -> None:
        products = (self._product("P1", 10_000),)
        box_a = self._candidate("a", 106, 100, {"P1"})
        box_b = self._candidate("b", 107, 90, {"P1"})
        result = simulated_annealing(
            products,
            {"P1": box_a},
            (box_a, box_b),
            FreightPolicy(),
            max_steps=21,
            random_seed=11,
            initial_temperature_usd=1_000_000,
            final_temperature_usd=1_000_000,
            proposal_strategy="uniform",
            restart_interval_steps=5,
        )
        self.assertEqual(result.restarts, 4)
        self.assertEqual(
            result.costs.total_mills,
            evaluate_assignments(
                products, result.assignment, FreightPolicy()
            ).total_mills,
        )


if __name__ == "__main__":
    unittest.main()
