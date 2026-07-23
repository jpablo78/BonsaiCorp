from bonsai.models import Dimensions, PreparedData, Product
from bonsai.plant_lns import PlantTierTarget, single_plant_codes


def test_single_plant_codes_excludes_multi_plant_and_empty() -> None:
    products = (
        Product("A", "T", Dimensions(1, 1, 1), 1, {"buenos_aires": 4, "curitiba": 0, "santiago": 0, "monterrey": 0, "bakersfield": 0}),
        Product("B", "T", Dimensions(1, 1, 1), 1, {"buenos_aires": 1, "curitiba": 2, "santiago": 0, "monterrey": 0, "bakersfield": 0}),
        Product("C", "T", Dimensions(1, 1, 1), 1, {"buenos_aires": 0, "curitiba": 0, "santiago": 0, "monterrey": 0, "bakersfield": 0}),
    )
    result = single_plant_codes(PreparedData(products, {}))
    assert result["buenos_aires"] == ("A",)
    assert all("B" not in codes and "C" not in codes for codes in result.values())


def test_plant_tier_target_gap() -> None:
    target = PlantTierTarget("curitiba", None, 91_234, 100_000)  # type: ignore[arg-type]
    assert target.gap == 8_766
