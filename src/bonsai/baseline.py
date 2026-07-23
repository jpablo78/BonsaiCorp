"""Feasible reference scenarios with no discretionary box consolidation."""

from __future__ import annotations

from .candidates import generate_candidates
from .config import FreightPolicy
from .costs import evaluate_assignments
from .models import CandidateBox, CostBreakdown, PreparedData


def no_consolidation_assignment(
    data: PreparedData, thickness_mm: float
) -> dict[str, CandidateBox]:
    """Keep each SKU at its current internal profile under one allowed thickness."""

    candidates = generate_candidates(data.products, thickness_mm, pair_profile_limit=0)
    by_internal = {candidate.internal: candidate for candidate in candidates}
    assignment: dict[str, CandidateBox] = {}
    for product in data.products:
        candidate = by_internal.get(product.current_internal)
        if candidate is None or product.code not in candidate.compatible_product_codes:
            raise ValueError(f"no standardized baseline candidate for {product.code}")
        assignment[product.code] = candidate
    return assignment


def standardized_baseline(
    data: PreparedData, thickness_mm: float, freight_policy: FreightPolicy
) -> tuple[dict[str, CandidateBox], CostBreakdown]:
    assignment = no_consolidation_assignment(data, thickness_mm)
    return assignment, evaluate_assignments(data.products, assignment, freight_policy)
