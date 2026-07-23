import unittest

from bonsai.config import FreightPolicy
from bonsai.geometry import external_from_internal
from bonsai.lp_pool import build_lp_candidate_pools, round_lp_assignment
from bonsai.models import CandidateBox, Dimensions, PLANTS, Product


def _product(code):
    return Product(
        code=code,
        current_box_type_id=code,
        current_internal=Dimensions(100, 100, 100),
        net_weight_kg=1,
        annual_volume_by_plant={plant: 100 for plant in PLANTS},
    )


def _candidate(name, internal, codes):
    return CandidateBox(
        candidate_id=name,
        thickness_mm=3.0,
        internal=internal,
        external=external_from_internal(internal, 3.0),
        capacity_per_pallet=10,
        compatible_product_codes=frozenset(codes),
    )


class LpPoolTests(unittest.TestCase):
    def test_pool_keeps_incumbent_positive_and_shared_signal(self):
        products = (_product("P1"), _product("P2"))
        old1 = _candidate("old1", Dimensions(100, 100, 100), {"P1"})
        old2 = _candidate("old2", Dimensions(101, 100, 100), {"P2"})
        shared = _candidate("shared", Dimensions(102, 100, 100), {"P1", "P2"})
        candidates = (old1, old2, shared)
        incumbent = {"P1": old1, "P2": old2}
        values = {
            ("P1", old1.internal): 0.4,
            ("P1", shared.internal): 0.6,
            ("P2", old2.internal): 0.9,
            ("P2", shared.internal): 0.1,
        }
        pools, stats = build_lp_candidate_pools(
            products, candidates, incumbent, values, {}, pool_size=2
        )
        self.assertEqual(pools["P1"], {old1.internal, shared.internal})
        self.assertEqual(pools["P2"], {old2.internal, shared.internal})
        self.assertEqual(stats.total_arcs, 4)
        self.assertEqual(stats.positive_lp_arcs, 4)

    def test_rounding_uses_largest_arc_and_retains_missing_row(self):
        products = (_product("P1"), _product("P2"))
        old1 = _candidate("old1", Dimensions(100, 100, 100), {"P1"})
        old2 = _candidate("old2", Dimensions(101, 100, 100), {"P2"})
        new1 = _candidate("new1", Dimensions(102, 100, 100), {"P1"})
        incumbent = {"P1": old1, "P2": old2}
        rounded = round_lp_assignment(
            products,
            (old1, old2, new1),
            incumbent,
            {("P1", old1.internal): 0.2, ("P1", new1.internal): 0.8},
        )
        self.assertEqual(rounded["P1"], new1)
        self.assertEqual(rounded["P2"], old2)


if __name__ == "__main__":
    unittest.main()
