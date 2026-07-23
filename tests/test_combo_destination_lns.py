import json
from pathlib import Path
import tempfile
import unittest

from bonsai.combo_destination_lns import (
    combo_complementarity_metrics,
    load_covered_combinations,
)


class ComboDestinationTests(unittest.TestCase):
    def test_metrics_identify_source_requiring_all_three_destinations(self) -> None:
        masks = (0b001, 0b010, 0b100)
        metrics = combo_complementarity_metrics(masks, ((0b111, 25_000),))
        self.assertEqual(metrics, (0, 1, 1, 25_000, 1, 25_000))

    def test_source_covered_by_pair_is_not_all_member_essential(self) -> None:
        masks = (0b001, 0b110, 0b1000)
        metrics = combo_complementarity_metrics(masks, ((0b111, 25_000),))
        self.assertEqual(metrics[2:4], (1, 25_000))
        self.assertEqual(metrics[4:], (0, 0))

    def test_prior_eight_destination_attempt_expands_to_contained_combos(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "summary.json"
            path.write_text(
                json.dumps(
                    {
                        "attempts": [
                            {
                                "destinations": [
                                    {"internal_mm": [index, index + 1, index + 2]}
                                    for index in range(8)
                                ]
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            triples = load_covered_combinations(path, 3)
            quads = load_covered_combinations(path, 4)
        self.assertEqual(len(triples), 56)
        self.assertEqual(len(quads), 70)


if __name__ == "__main__":
    unittest.main()
