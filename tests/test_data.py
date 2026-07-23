import unittest

from bonsai.data import canonical_decimal, parse_number


class NumericNormalizationTests(unittest.TestCase):
    def test_thickness_variants_normalize_to_one_value(self) -> None:
        values = ["2,5", "2.5", "2.50", "2.5 mm", "4.1mm"]
        self.assertEqual([canonical_decimal(value) for value in values], ["2.5", "2.5", "2.5", "2.5", "4.1"])
        self.assertEqual(parse_number("4,6 mm"), 4.6)


if __name__ == "__main__":
    unittest.main()
