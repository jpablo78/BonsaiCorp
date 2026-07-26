"""Constructor voraz determinista para escenarios factibles de comparación rápida."""

from __future__ import annotations

from .candidates import generate_candidates
from .models import CandidateBox, PreparedData


def greedy_cover_assignment(
    data: PreparedData,
    thickness_mm: float,
    *,
    pair_profile_limit: int = 90,
    max_extra_pair_designs: int = 1_000,
) -> dict[str, CandidateBox]:
    """Cubre productos compatibles de forma voraz y prioriza el volumen anual.

    Es un inicio cálido o referencia factible; no reemplaza al modelo CP-SAT
    porque no optimiza globalmente las interacciones entre tiers.
    """

    candidates = generate_candidates(
        data.products,
        thickness_mm,
        pair_profile_limit=pair_profile_limit,
        max_extra_pair_designs=max_extra_pair_designs,
    )
    product_by_code = data.product_by_code
    unassigned = {product.code for product in data.products}
    assignment: dict[str, CandidateBox] = {}
    while unassigned:
        def score(candidate: CandidateBox) -> tuple[int, int, tuple[float, float, float]]:
            covered = unassigned & candidate.compatible_product_codes
            demand = sum(product_by_code[code].annual_volume for code in covered)
            return (demand, len(covered), tuple(-value for value in candidate.external.as_tuple()))

        selected = max(candidates, key=score)
        covered = unassigned & selected.compatible_product_codes
        if not covered:
            raise RuntimeError("greedy candidate universe cannot cover remaining products")
        for code in covered:
            assignment[code] = selected
        unassigned -= covered
    return assignment
