"""Objetos de dominio usados en preparación, optimización y validación."""

from __future__ import annotations

from dataclasses import dataclass, field


PLANTS = ("buenos_aires", "curitiba", "santiago", "monterrey", "bakersfield")


@dataclass(frozen=True)
class Dimensions:
    """Largo, ancho y alto en milímetros."""

    length: float
    width: float
    height: float

    @property
    def volume_mm3(self) -> float:
        return self.length * self.width * self.height

    def as_tuple(self) -> tuple[int, int, int]:
        return (self.length, self.width, self.height)


@dataclass(frozen=True)
class Product:
    code: str
    current_box_type_id: str
    current_internal: Dimensions
    net_weight_kg: float
    annual_volume_by_plant: dict[str, int]

    @property
    def product_volume_mm3(self) -> int:
        return self.current_internal.volume_mm3

    @property
    def annual_volume(self) -> int:
        return sum(self.annual_volume_by_plant.values())


@dataclass(frozen=True)
class CurrentBoxSpec:
    box_type_id: str
    thickness_mm: float
    internal: Dimensions
    external: Dimensions


@dataclass(frozen=True)
class CandidateBox:
    """Un diseño discreto de milímetros enteros considerado por el optimizador."""

    candidate_id: str
    thickness_mm: float
    internal: Dimensions
    external: Dimensions
    capacity_per_pallet: int
    compatible_product_codes: frozenset[str]


@dataclass(frozen=True)
class AssignmentRow:
    code: str
    thickness_mm: float
    external: Dimensions


@dataclass
class CostBreakdown:
    packaging_mills: int = 0
    freight_mills: int = 0
    pallets: int = 0
    types: int = 0
    pallet_utilization_by_plant: dict[str, float] = field(default_factory=dict)

    @property
    def total_mills(self) -> int:
        return self.packaging_mills + self.freight_mills

    def as_dict(self) -> dict[str, object]:
        return {
            "packaging_usd": self.packaging_mills / 1000,
            "freight_usd": self.freight_mills / 1000,
            "total_usd": self.total_mills / 1000,
            "pallets": self.pallets,
            "types": self.types,
            "pallet_utilization_by_plant": self.pallet_utilization_by_plant,
        }


@dataclass(frozen=True)
class PreparedData:
    products: tuple[Product, ...]
    current_boxes: dict[str, CurrentBoxSpec]

    @property
    def product_by_code(self) -> dict[str, Product]:
        return {product.code: product for product in self.products}
