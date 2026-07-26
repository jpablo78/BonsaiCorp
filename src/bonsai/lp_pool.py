"""Pools de candidatos guiados por LP para el problema maestro exacto con SCIP.

La relajación LP completa se usa sólo como heurística de búsqueda. Cada solución
candidata se resuelve luego como MIP entero y se verifica con el mismo
evaluador independiente usado por el resto del proyecto.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Mapping, Sequence

from .models import CandidateBox, Dimensions, Product


ArcKey = tuple[str, Dimensions]


@dataclass(frozen=True)
class LpPoolStats:
    pool_size_limit: int
    total_arcs: int
    minimum_sku_pool_size: int
    maximum_sku_pool_size: int
    positive_lp_arcs: int
    activated_designs: int


def build_lp_candidate_pools(
    products: Sequence[Product],
    candidates: Sequence[CandidateBox],
    incumbent: Mapping[str, CandidateBox],
    arc_values: Mapping[ArcKey, float],
    reduced_costs_mills: Mapping[ArcKey, float],
    *,
    pool_size: int,
    value_epsilon: float = 1e-7,
) -> tuple[dict[str, frozenset[Dimensions]], LpPoolStats]:
    """Construye pools pequeños, conscientes de consolidación, desde una solución LP completa.

    Se priorizan arcos positivos para el propio SKU. Los espacios restantes
    alternan entre diseños activados por otros SKU, señal útil de consolidación
    de Procurement, y arcos de bajo costo reducido para ese SKU. La incumbente
    se conserva incondicionalmente.
    """

    if pool_size < 1:
        raise ValueError("pool_size must be at least one")
    product_codes = {product.code for product in products}
    if set(incumbent) != product_codes:
        raise ValueError("incumbent must contain exactly one box per product")

    candidate_by_internal = {candidate.internal: candidate for candidate in candidates}
    global_mass: dict[Dimensions, float] = defaultdict(float)
    global_positive_users: dict[Dimensions, int] = defaultdict(int)
    for (code, internal), value in arc_values.items():
        if code not in product_codes or internal not in candidate_by_internal:
            continue
        if value > value_epsilon:
            global_mass[internal] += value
            global_positive_users[internal] += 1

    arcs_by_code: dict[str, list[tuple[Dimensions, float, float]]] = defaultdict(list)
    for (code, internal), value in arc_values.items():
        if code not in product_codes or internal not in candidate_by_internal:
            continue
        reduced = reduced_costs_mills.get((code, internal), 0.0)
        arcs_by_code[code].append((internal, value, reduced))

    pools: dict[str, frozenset[Dimensions]] = {}
    for product in products:
        code = product.code
        row = arcs_by_code.get(code, [])
        chosen: list[Dimensions] = []
        seen: set[Dimensions] = set()

        def add(internal: Dimensions) -> None:
            if len(chosen) < pool_size and internal not in seen:
                seen.add(internal)
                chosen.append(internal)

        add(incumbent[code].internal)

        positive = sorted(
            (item for item in row if item[1] > value_epsilon),
            key=lambda item: (
                -item[1],
                item[2],
                -global_mass.get(item[0], 0.0),
                item[0].as_tuple(),
            ),
        )
        for internal, _, _ in positive:
            add(internal)

        activated = sorted(
            row,
            key=lambda item: (
                -global_positive_users.get(item[0], 0),
                -global_mass.get(item[0], 0.0),
                item[2],
                -item[1],
                item[0].as_tuple(),
            ),
        )
        reduced = sorted(
            row,
            key=lambda item: (
                item[2],
                -item[1],
                -global_mass.get(item[0], 0.0),
                item[0].as_tuple(),
            ),
        )
    # La intercalación conserva ambas piezas de información LP cuando el conjunto
    # solicitado es muy pequeño.
        for index in range(max(len(activated), len(reduced))):
            if index < len(activated):
                add(activated[index][0])
            if index < len(reduced):
                add(reduced[index][0])
            if len(chosen) >= pool_size:
                break
        pools[code] = frozenset(chosen)

    sizes = [len(pool) for pool in pools.values()]
    return pools, LpPoolStats(
        pool_size_limit=pool_size,
        total_arcs=sum(sizes),
        minimum_sku_pool_size=min(sizes),
        maximum_sku_pool_size=max(sizes),
        positive_lp_arcs=sum(value > value_epsilon for value in arc_values.values()),
        activated_designs=len(global_mass),
    )


def round_lp_assignment(
    products: Sequence[Product],
    candidates: Sequence[CandidateBox],
    incumbent: Mapping[str, CandidateBox],
    arc_values: Mapping[ArcKey, float],
) -> dict[str, CandidateBox]:
    """Redondea independientemente cada fila de asignación a su mayor arco LP."""

    candidate_by_internal = {candidate.internal: candidate for candidate in candidates}
    rounded = dict(incumbent)
    best: dict[str, tuple[float, Dimensions]] = {}
    for (code, internal), value in arc_values.items():
        if internal not in candidate_by_internal:
            continue
        key = (value, tuple(-axis for axis in internal.as_tuple()))
        previous = best.get(code)
        if previous is None or key > (
            previous[0],
            tuple(-axis for axis in previous[1].as_tuple()),
        ):
            best[code] = (value, internal)
    for product in products:
        if product.code in best:
            rounded[product.code] = candidate_by_internal[best[product.code][1]]
    return rounded
