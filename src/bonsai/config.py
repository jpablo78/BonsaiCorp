"""Centralized assumptions and commercial constants."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final


ALLOWED_THICKNESSES: Final[tuple[float, ...]] = (3.0, 4.5, 5.0)
BASE_PRICE_USD: Final[dict[float, float]] = {3.0: 0.60, 4.5: 0.65, 5.0: 0.70}
ECT_N_PER_M: Final[dict[float, int]] = {3.0: 1000, 4.5: 1400, 5.0: 1650}
GRAVITY_M_PER_S2: Final[float] = 9.81

PALLET_LENGTH_MM: Final[int] = 1200
PALLET_WIDTH_MM: Final[int] = 800
PALLET_MAX_HEIGHT_MM: Final[int] = 1800

MILLS_PER_USD: Final[int] = 1000


@dataclass(frozen=True)
class DiscountTier:
    """One exclusive annual-volume tier for one type at one plant."""

    name: str
    lower_inclusive: int
    upper_inclusive: int | None
    factor_percent: int

    def contains(self, volume: int) -> bool:
        return volume >= self.lower_inclusive and (
            self.upper_inclusive is None or volume <= self.upper_inclusive
        )


DISCOUNT_TIERS: Final[tuple[DiscountTier, ...]] = (
    DiscountTier("tier_1", 1, 19_999, 110),
    DiscountTier("tier_2", 20_000, 49_999, 100),
    DiscountTier("tier_3", 50_000, 99_999, 90),
    DiscountTier("tier_4", 100_000, 499_999, 80),
    DiscountTier("tier_5", 500_000, None, 70),
)


@dataclass(frozen=True)
class FreightPolicy:
    """Freight policy extensible to an extra-region shipment share.

    The base challenge data has no destination matrix, so the approved scenario
    uses an extra-region share of zero.  When a defensible share becomes
    available, set it between zero and one; the expected pallet rate becomes
    the weighted average of intra- and extra-region rates.
    """

    intra_region_usd_per_pallet: float = 150.0
    extra_region_usd_per_pallet: float = 500.0
    extra_region_share: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.extra_region_share <= 1.0:
            raise ValueError("extra_region_share must be between 0 and 1")

    @property
    def expected_usd_per_pallet(self) -> float:
        return (
            self.intra_region_usd_per_pallet * (1.0 - self.extra_region_share)
            + self.extra_region_usd_per_pallet * self.extra_region_share
        )

    @property
    def expected_mills_per_pallet(self) -> int:
        return round(self.expected_usd_per_pallet * MILLS_PER_USD)
