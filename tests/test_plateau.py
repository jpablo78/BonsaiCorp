import unittest

from bonsai.config import FreightPolicy
from bonsai.costs import box_type_key, evaluate_assignments
from bonsai.models import CandidateBox, Dimensions, PLANTS, Product
from bonsai.plateau import (
    apply_plateau_move,
    diversify_assignment,
    enumerate_zero_cost_moves,
)


class PlateauDiversificationTests(unittest.TestCase):
    @staticmethod
    def _product(code: str) -> Product:
        return Product(
            code=code,
            current_box_type_id=f"current_{code}",
            current_internal=Dimensions(100, 100, 100),
            net_weight_kg=1,
            annual_volume_by_plant={
                plant: (
                    10_000
                    if plant == "buenos_aires"
                    else 20_000 if plant == "santiago" else 0
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
        external = Dimensions(external_length, 106, 106)
        return CandidateBox(
            candidate_id=candidate_id,
            thickness_mm=3.0,
            internal=Dimensions(external_length - 6, 100, 100),
            external=external,
            capacity_per_pallet=capacity,
            compatible_product_codes=frozenset(compatible_codes),
        )

    def _zero_cost_case(self):
        products = (self._product("P1"), self._product("P2"))
        source = self._candidate("source_a", 106, 1_000, {"P1"})
    # El candidato incumbente tiene deliberadamente un conjunto local de
    # compatibilidad; exact_b representa la misma caja física en el universo.
        incumbent_b = self._candidate("incumbent_b", 107, 800, {"P2"})
        exact_b = self._candidate("exact_b", 107, 800, {"P1", "P2"})
        assignment = {"P1": source, "P2": incumbent_b}
        return products, assignment, (source, exact_b), exact_b

    def test_zero_delta_uses_physical_identity_plant_tiers_and_freight(self) -> None:
        products, assignment, candidates, exact_b = self._zero_cost_case()
        policy = FreightPolicy()

        moves = enumerate_zero_cost_moves(products, assignment, candidates, policy)

        self.assertEqual(len(moves), 1)
        move = moves[0]
        self.assertEqual(move.code, "P1")
        self.assertEqual(move.target_candidate, exact_b)
        self.assertEqual(move.packaging_delta_mills, -1_200_000)
        self.assertEqual(move.pallet_delta, 8)
        self.assertEqual(move.freight_delta_mills, 1_200_000)
        self.assertEqual(move.total_delta_mills, 0)
        self.assertEqual(move.type_delta, -1)

        changed = apply_plateau_move(assignment, move)
        before = evaluate_assignments(products, assignment, policy)
        after = evaluate_assignments(products, changed, policy)
        self.assertEqual(after.total_mills, before.total_mills)
        self.assertEqual(
            after.packaging_mills - before.packaging_mills,
            move.packaging_delta_mills,
        )
        self.assertEqual(after.freight_mills - before.freight_mills, move.freight_delta_mills)
        self.assertEqual(after.pallets - before.pallets, move.pallet_delta)
        self.assertEqual(after.types - before.types, move.type_delta)
        self.assertEqual(box_type_key(changed["P1"]), box_type_key(changed["P2"]))

    def test_nonzero_total_move_is_not_enumerated(self) -> None:
        products = (self._product("P1"), self._product("P2"))
        source = self._candidate("source_a", 106, 1_000, {"P1"})
        target = self._candidate("target_b", 107, 700, {"P1", "P2"})
        incumbent_b = self._candidate("incumbent_b", 107, 700, {"P2"})
        assignment = {"P1": source, "P2": incumbent_b}

        moves = enumerate_zero_cost_moves(
            products, assignment, (source, target), FreightPolicy()
        )

        self.assertEqual(moves, ())

    def test_random_walk_returns_farthest_reproducible_equal_cost_state(self) -> None:
        products, assignment, candidates, _ = self._zero_cost_case()
        policy = FreightPolicy()

        first = diversify_assignment(
            products,
            assignment,
            candidates,
            policy,
            max_steps=10,
            random_seed=17,
        )
        second = diversify_assignment(
            products,
            assignment,
            candidates,
            policy,
            max_steps=10,
            random_seed=17,
        )

        self.assertEqual(first.distance_from_start, 1)
        self.assertEqual(first.walk_steps, 1)
        self.assertEqual(first.visited_states, 2)
        self.assertEqual(len(first.moves), 1)
        self.assertEqual(first.final_costs.total_mills, first.initial_costs.total_mills)
        self.assertNotEqual(
            box_type_key(first.assignment["P1"]), box_type_key(assignment["P1"])
        )
        self.assertEqual(
            {code: box.candidate_id for code, box in first.assignment.items()},
            {code: box.candidate_id for code, box in second.assignment.items()},
        )

    def test_same_geometry_candidate_id_change_is_not_a_move(self) -> None:
        product = self._product("P1")
        incumbent = self._candidate("incumbent", 106, 1_000, {"P1"})
        duplicate = self._candidate("duplicate", 106, 1_000, {"P1"})

        moves = enumerate_zero_cost_moves(
            (product,), {"P1": incumbent}, (duplicate,), FreightPolicy()
        )

        self.assertEqual(moves, ())

    def test_random_walk_respects_pallet_cap(self) -> None:
        products, assignment, candidates, _ = self._zero_cost_case()
        initial = evaluate_assignments(products, assignment, FreightPolicy())
        result = diversify_assignment(
            products,
            assignment,
            candidates,
            FreightPolicy(),
            max_steps=10,
            random_seed=17,
            max_pallets=initial.pallets,
        )
        self.assertEqual(result.distance_from_start, 0)
        self.assertEqual(result.final_costs.pallets, initial.pallets)

    def test_apply_rejects_a_stale_move(self) -> None:
        products, assignment, candidates, _ = self._zero_cost_case()
        move = enumerate_zero_cost_moves(
            products, assignment, candidates, FreightPolicy()
        )[0]
        changed = apply_plateau_move(assignment, move)

        with self.assertRaisesRegex(ValueError, "stale plateau move"):
            apply_plateau_move(changed, move)


if __name__ == "__main__":
    unittest.main()
