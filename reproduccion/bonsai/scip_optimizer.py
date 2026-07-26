"""Modelo maestro MIP alternativo con OR-Tools MPSolver/SCIP.

Comparte universo de candidatos y evaluador independiente de costos con
CP-SAT. Procurement usa umbrales acumulados de descuento all-units: un binario
indica si se alcanza el umbral y un auxiliar entero toma el volumen completo de
caja/planta (cero si no se alcanza). Esto mantiene el modelo lineal y coincide
exactamente con ``costs.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import time
from typing import Callable, Collection, Mapping

from .config import DISCOUNT_TIERS, FreightPolicy
from .costs import evaluate_assignments, freight_pallets, unit_price_mills
from .exact_candidates import ExactCandidateStats, generate_exact_candidates
from .models import CandidateBox, CostBreakdown, Dimensions, PLANTS, PreparedData


@dataclass(frozen=True)
class ScipSolveResult:
    thickness_mm: float
    assignment: dict[str, CandidateBox]
    costs: CostBreakdown
    status: str
    candidate_count: int
    assignment_variable_count: int
    threshold_variable_count: int
    candidate_stats: ExactCandidateStats | None
    solver_objective_mills: int | None
    best_bound_mills: float | None
    candidate_universe_relative_gap: float | None
    wall_time_seconds: float
    nodes: int
    incumbent_mills: int
    improved_incumbent: bool
    selected_source: str
    minimum_possible_pallets: int | None
    max_extra_pallets: int | None
    target_total_mills: int | None
    target_met: bool | None
    max_changed_products: int | None
    min_changed_products: int | None
    changed_product_count: int
    fixed_product_count: int
    pruned_assignment_count: int
    pallet_pruned_assignment_count: int
    objective_pruned_assignment_count: int
    objective_filter_enabled: bool
    objective_scale_mills: int
    preparation_time_seconds: float
    model_build_time_seconds: float
    solve_time_seconds: float
    relaxation: bool
    assignment_arc_values: dict[tuple[str, Dimensions], float]
    assignment_arc_reduced_costs_mills: dict[tuple[str, Dimensions], float]


def _import_mpsolver():
    try:
        from ortools.linear_solver import pywraplp
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "OR-Tools with SCIP is required. Install the project dependencies."
        ) from exc
    return pywraplp


def _status_name(pywraplp, status_code: int) -> str:
    names = {
        pywraplp.Solver.OPTIMAL: "OPTIMAL",
        pywraplp.Solver.FEASIBLE: "FEASIBLE",
        pywraplp.Solver.INFEASIBLE: "INFEASIBLE",
        pywraplp.Solver.UNBOUNDED: "UNBOUNDED",
        pywraplp.Solver.ABNORMAL: "ABNORMAL",
        pywraplp.Solver.MODEL_INVALID: "MODEL_INVALID",
        pywraplp.Solver.NOT_SOLVED: "NOT_SOLVED",
    }
    return names.get(status_code, f"STATUS_{status_code}")


def solve_with_scip(
    data: PreparedData,
    thickness_mm: float,
    freight_policy: FreightPolicy,
    *,
    time_limit_seconds: float = 300.0,
    num_threads: int = 2,
    random_seed: int = 42,
    initial_assignment: dict[str, CandidateBox] | None = None,
    max_extra_pallets: int | None = None,
    target_total_mills: int | None = None,
    enable_objective_filter: bool = True,
    max_changed_products: int | None = None,
    min_changed_products: int | None = None,
    free_product_codes: Collection[str] | None = None,
    allowed_internals_by_product: Mapping[str, Collection[Dimensions]] | None = None,
    precomputed_exact_candidates: tuple[CandidateBox, ...] | None = None,
    precomputed_exact_candidate_stats: ExactCandidateStats | None = None,
    export_model_path: str | Path | None = None,
    enable_solver_output: bool = False,
    memory_limit_mb: int | None = None,
    scip_parameters: str | None = None,
    progress_callback: Callable[[str], None] | None = None,
    relax_integrality: bool = False,
) -> ScipSolveResult:
    """Resuelve con SCIP el problema maestro entero completo en milímetros.

    Un warm start también es una incumbente protegida: el MIP se acota por su
    costo evaluado independientemente y nunca devuelve una asignación peor.
    ``max_extra_pallets`` se mide desde el mínimo global por SKU sobre el
    universo exacto completo, como en CP-SAT.

    ``free_product_codes`` y ``allowed_internals_by_product`` definen un
    vecindario grande restringido con la misma semántica de
    :func:`bonsai.optimizer.solve_for_thickness`. Los SKU fuera del conjunto
    libre se eliminan del MIP: su flete incumbente es constante y su demanda se
    incorpora al volumen fijo de cada tier de caja/planta. La geometría
    incumbente siempre se agrega a los conjuntos permitidos.

    ``max_changed_products`` y ``min_changed_products`` son controles de
    búsqueda, no restricciones comerciales. Acotan la distancia de Hamming
    respecto de ``initial_assignment`` para local branching exacto.
    """

    started_at = time.perf_counter()

    def report(message: str) -> None:
        if progress_callback is not None:
            progress_callback(message)

    if time_limit_seconds <= 0:
        raise ValueError("time_limit_seconds must be positive")
    if num_threads < 1:
        raise ValueError("num_threads must be at least 1")
    if max_extra_pallets is not None and max_extra_pallets < 0:
        raise ValueError("max_extra_pallets cannot be negative")
    if target_total_mills is not None and target_total_mills < 0:
        raise ValueError("target_total_mills cannot be negative")
    if max_changed_products is not None and max_changed_products < 0:
        raise ValueError("max_changed_products cannot be negative")
    if min_changed_products is not None and min_changed_products < 0:
        raise ValueError("min_changed_products cannot be negative")
    if (
        max_changed_products is not None
        and min_changed_products is not None
        and min_changed_products > max_changed_products
    ):
        raise ValueError("min_changed_products cannot exceed max_changed_products")
    if memory_limit_mb is not None and memory_limit_mb <= 0:
        raise ValueError("memory_limit_mb must be positive")
    if precomputed_exact_candidate_stats is not None and precomputed_exact_candidates is None:
        raise ValueError("precomputed stats require precomputed exact candidates")

    expected_codes = {product.code for product in data.products}
    if free_product_codes is not None and initial_assignment is None:
        raise ValueError("free_product_codes requires an initial assignment")
    if allowed_internals_by_product is not None and initial_assignment is None:
        raise ValueError(
            "allowed_internals_by_product requires an initial assignment"
        )
    if (
        (max_changed_products is not None or min_changed_products is not None)
        and initial_assignment is None
    ):
        raise ValueError("local branching requires an initial assignment")
    active_free_product_codes = (
        expected_codes if free_product_codes is None else set(free_product_codes)
    )
    unknown_free_codes = active_free_product_codes - expected_codes
    if unknown_free_codes:
        raise ValueError(
            "free_product_codes contains unknown products: "
            f"{sorted(unknown_free_codes)}"
        )
    supplied_allowed_internals = (
        {} if allowed_internals_by_product is None else dict(allowed_internals_by_product)
    )
    unknown_allowed_codes = set(supplied_allowed_internals) - expected_codes
    if unknown_allowed_codes:
        raise ValueError(
            "allowed_internals_by_product contains unknown products: "
            f"{sorted(unknown_allowed_codes)}"
        )
    if initial_assignment is not None:
        if set(initial_assignment) != expected_codes:
            raise ValueError("initial assignment must contain exactly one row for every product")
        initial_thicknesses = {
            candidate.thickness_mm for candidate in initial_assignment.values()
        }
        if initial_thicknesses != {thickness_mm}:
            raise ValueError(
                f"warm-start thicknesses {sorted(initial_thicknesses)} do not match "
                f"requested thickness {thickness_mm:g} mm"
            )

    active_allowed_internals: dict[str, frozenset[Dimensions]] = {}
    for code, internals in supplied_allowed_internals.items():
        requested = tuple(internals)
        invalid = [internal for internal in requested if not isinstance(internal, Dimensions)]
        if invalid:
            raise TypeError(
                "allowed_internals_by_product values must contain Dimensions; "
                f"invalid values for {code}: {invalid!r}"
            )
        active_allowed_internals[code] = frozenset(
            (*requested, initial_assignment[code].internal)
        )

    retained = tuple(
        sorted(
            {
                candidate.internal
                for candidate in initial_assignment.values()
            }
            if initial_assignment is not None
            else {product.current_internal for product in data.products},
            key=Dimensions.as_tuple,
        )
    )
    retained = tuple(
        sorted(
            set(retained)
            | {
                internal
                for internals in active_allowed_internals.values()
                for internal in internals
            },
            key=Dimensions.as_tuple,
        )
    )
    if precomputed_exact_candidates is None:
        report("SCIP preparation: generating exact integer-mm candidates")
        candidates, candidate_stats = generate_exact_candidates(
            data.products, thickness_mm, retained_designs=retained
        )
    else:
        report("SCIP preparation: using precomputed exact candidates")
        candidates = precomputed_exact_candidates
        candidate_stats = precomputed_exact_candidate_stats
        if not candidates:
            raise ValueError("precomputed exact candidate universe cannot be empty")
        if {candidate.thickness_mm for candidate in candidates} != {thickness_mm}:
            raise ValueError("precomputed candidates do not match requested thickness")
    if len({candidate.internal for candidate in candidates}) != len(candidates):
        raise ValueError(
            "exact candidate universe must have one canonical row per internal geometry"
        )
    compatible_internals_by_code: dict[str, set[Dimensions]] = {
        code: set() for code in active_allowed_internals
    }
    for candidate in candidates:
        for code in candidate.compatible_product_codes & active_allowed_internals.keys():
            compatible_internals_by_code[code].add(candidate.internal)
    for code, allowed_internals in active_allowed_internals.items():
        missing_internals = allowed_internals - compatible_internals_by_code[code]
        if missing_internals:
            raise ValueError(
                f"allowed internal geometries are infeasible for {code}: "
                f"{sorted(internal.as_tuple() for internal in missing_internals)}"
            )
    report(
        f"SCIP preparation: {len(data.products)} products, "
        f"{len(candidates)} exact candidates"
    )

    pallet_count: dict[tuple[int, int], int] = {}
    candidate_indices_by_product: list[list[int]] = []
    minimum_pallets_by_product: list[int] = []
    for product_index, product in enumerate(data.products):
        compatible: list[int] = []
        counts: list[int] = []
        for candidate_index, candidate in enumerate(candidates):
            if product.code not in candidate.compatible_product_codes:
                continue
            count = sum(
                freight_pallets(product, candidate, plant) for plant in PLANTS
            )
            pallet_count[(product_index, candidate_index)] = count
            compatible.append(candidate_index)
            counts.append(count)
        if not compatible:
            raise RuntimeError(f"no exact candidate covers {product.code}")
        candidate_indices_by_product.append(compatible)
        minimum_pallets_by_product.append(min(counts))

    raw_assignment_count = sum(map(len, candidate_indices_by_product))

    minimum_possible_pallets = (
        sum(minimum_pallets_by_product) if max_extra_pallets is not None else None
    )
    if max_extra_pallets is not None:
        limit = minimum_possible_pallets + max_extra_pallets
        for product_index, candidate_indices in enumerate(candidate_indices_by_product):
            candidate_indices_by_product[product_index] = [
                candidate_index
                for candidate_index in candidate_indices
                if pallet_count[(product_index, candidate_index)]
                <= minimum_pallets_by_product[product_index] + max_extra_pallets
            ]
            if not candidate_indices_by_product[product_index]:
                raise RuntimeError(
                    f"pallet restriction leaves no candidate for "
                    f"{data.products[product_index].code}"
                )
    else:
        limit = None

    assignment_count_after_pallet_filter = sum(map(len, candidate_indices_by_product))
    pallet_pruned_assignment_count = (
        raw_assignment_count - assignment_count_after_pallet_filter
    )

    candidate_by_internal = {
        candidate.internal: candidate_index
        for candidate_index, candidate in enumerate(candidates)
    }
    incumbent_indices: list[int] = []
    incumbent_assignment: dict[str, CandidateBox] = {}
    for product_index, product in enumerate(data.products):
        if initial_assignment is not None:
            internal = initial_assignment[product.code].internal
            if internal not in candidate_by_internal:
                raise ValueError(
                    f"warm-start design {internal.as_tuple()} is absent from candidate universe"
                )
            candidate_index = candidate_by_internal[internal]
            if candidate_index not in candidate_indices_by_product[product_index]:
                raise ValueError(
                    f"warm-start design for {product.code} violates compatibility or pallet budget"
                )
        else:
            candidate_index = min(
                candidate_indices_by_product[product_index],
                key=lambda index: (
                    pallet_count[(product_index, index)]
                    if max_extra_pallets is not None
                    else int(candidates[index].internal != product.current_internal),
                    candidates[index].internal.as_tuple(),
                ),
            )
        incumbent_indices.append(candidate_index)
        incumbent_assignment[product.code] = candidates[candidate_index]

    incumbent_costs = evaluate_assignments(
        data.products, incumbent_assignment, freight_policy
    )
    if limit is not None and incumbent_costs.pallets > limit:
        raise ValueError(
            f"initial assignment exceeds pallet budget: {incumbent_costs.pallets} > {limit}"
        )

    # Las restricciones LNS se aplican sólo después de calcular los mínimos
    # globales de pallets por SKU. Así ``max_extra_pallets`` coincide con el
    # modelo maestro completo. Los productos no libres pasan a filas unitarias y
    # se absorben luego como constantes, sin variables de asignación.
    for product_index, product in enumerate(data.products):
        if product.code not in active_free_product_codes:
            candidate_indices_by_product[product_index] = [
                incumbent_indices[product_index]
            ]
            continue
        if product.code in active_allowed_internals:
            allowed = active_allowed_internals[product.code]
            candidate_indices_by_product[product_index] = [
                candidate_index
                for candidate_index in candidate_indices_by_product[product_index]
                if candidates[candidate_index].internal in allowed
            ]
            if not candidate_indices_by_product[product_index]:
                raise RuntimeError(
                    f"allowed geometry restriction leaves no candidate for {product.code}"
                )

    # Filtrado seguro tipo costo reducido. Si p se asigna a c, cada unidad de una
    # planta paga al menos el precio alcanzable si *todos* los productos hoy
    # compatibles usan c allí. Al sumar esa cota de packaging al flete exacto se
    # obtiene una cota inferior separable para cada arco de asignación. Un arco
    # cuya contribución más las mejores cotas de los demás productos supera la
    # incumbente protegida nunca puede pertenecer a una solución mejoradora.
    # Repetir tras eliminaciones ajusta los volúmenes máximos candidatos y la
    # cota, sin cambiar el conjunto factible mejorador.
    for _ in range(4 if enable_objective_filter else 0):
        maximum_volume_by_candidate_plant = [
            {plant: 0 for plant in PLANTS} for _ in candidates
        ]
        for product_index, candidate_indices in enumerate(candidate_indices_by_product):
            product = data.products[product_index]
            for candidate_index in candidate_indices:
                by_plant = maximum_volume_by_candidate_plant[candidate_index]
                for plant in PLANTS:
                    by_plant[plant] += product.annual_volume_by_plant[plant]

        arc_lower_bound: dict[tuple[int, int], int] = {}
        minimum_arc_lower_bound: list[int] = []
        for product_index, candidate_indices in enumerate(candidate_indices_by_product):
            product = data.products[product_index]
            row_bounds: list[int] = []
            for candidate_index in candidate_indices:
                packaging_lower_bound = 0
                maximum_by_plant = maximum_volume_by_candidate_plant[candidate_index]
                for plant in PLANTS:
                    product_volume = product.annual_volume_by_plant[plant]
                    if product_volume:
                        packaging_lower_bound += product_volume * unit_price_mills(
                            thickness_mm, maximum_by_plant[plant]
                        )
                bound = (
                    pallet_count[(product_index, candidate_index)]
                    * freight_policy.expected_mills_per_pallet
                    + packaging_lower_bound
                )
                arc_lower_bound[(product_index, candidate_index)] = bound
                row_bounds.append(bound)
            minimum_arc_lower_bound.append(min(row_bounds))
        global_assignment_lower_bound = sum(minimum_arc_lower_bound)

        changed = False
        for product_index, candidate_indices in enumerate(candidate_indices_by_product):
            other_products_lower_bound = (
                global_assignment_lower_bound - minimum_arc_lower_bound[product_index]
            )
            filtered = [
                candidate_index
                for candidate_index in candidate_indices
                if arc_lower_bound[(product_index, candidate_index)]
                + other_products_lower_bound
                <= incumbent_costs.total_mills
            ]
            if not filtered:
                raise RuntimeError(
                    f"objective filtering leaves no candidate for "
                    f"{data.products[product_index].code}"
                )
            if len(filtered) != len(candidate_indices):
                candidate_indices_by_product[product_index] = filtered
                changed = True
        if not changed:
            break

    final_assignment_count = sum(map(len, candidate_indices_by_product))
    objective_pruned_assignment_count = (
        assignment_count_after_pallet_filter - final_assignment_count
    )
    pruned_assignment_count = raw_assignment_count - final_assignment_count
    fixed_candidate_by_product = {
        product_index: candidate_indices[0]
        for product_index, candidate_indices in enumerate(candidate_indices_by_product)
        if len(candidate_indices) == 1
    }
    report(
        f"SCIP preparation: {sum(map(len, candidate_indices_by_product)):,} arcs "
        f"({pruned_assignment_count:,} pruned), "
        f"{len(fixed_candidate_by_product)} fixed products"
    )

    preparation_time_seconds = time.perf_counter() - started_at
    model_build_started_at = time.perf_counter()

    pywraplp = _import_mpsolver()
    solver_backend = "GLOP" if relax_integrality else "SCIP"
    solver = pywraplp.Solver.CreateSolver(solver_backend)
    if solver is None:  # pragma: no cover - depends on binary distribution
        raise RuntimeError("this OR-Tools installation does not include SCIP")
    solver.SetTimeLimit(round(time_limit_seconds * 1000))
    if not relax_integrality and not solver.SetNumThreads(num_threads):
        raise RuntimeError(f"SCIP rejected num_threads={num_threads}")
    if enable_solver_output:
        solver.EnableOutput()
    if not relax_integrality:
        parameter_lines = [
            f"randomization/randomseedshift = {random_seed}",
    # SCIP almacena filas LP como números de doble precisión. Los coeficientes
    # del objetivo se escalan más abajo, pero una tolerancia de factibilidad
    # estricta sigue siendo útil en un límite objetivo duro.
            "numerics/feastol = 1e-9",
        ]
        if memory_limit_mb is not None:
            parameter_lines.append(f"limits/memory = {memory_limit_mb}")
        if scip_parameters:
            parameter_lines.append(scip_parameters.strip())
        parameter_status = solver.SetSolverSpecificParametersAsString(
            "\n".join(parameter_lines) + "\n"
        )
        if not parameter_status:
            raise RuntimeError("SCIP rejected one or more solver-specific parameters")

    infinity = solver.infinity()
    assignment_variables: dict[tuple[int, int], object] = {}
    variables_by_candidate: list[list[tuple[int, object]]] = [
        [] for _ in candidates
    ]
    fixed_products_by_candidate: list[list[int]] = [[] for _ in candidates]
    for product_index, candidate_indices in enumerate(candidate_indices_by_product):
        if product_index in fixed_candidate_by_product:
            fixed_products_by_candidate[candidate_indices[0]].append(product_index)
            continue
        row = solver.RowConstraint(1.0, 1.0, f"one_candidate_p{product_index}")
        for candidate_index in candidate_indices:
            variable = (
                solver.NumVar(0.0, 1.0, f"x_p{product_index}_c{candidate_index}")
                if relax_integrality
                else solver.BoolVar(f"x_p{product_index}_c{candidate_index}")
            )
            assignment_variables[(product_index, candidate_index)] = variable
            variables_by_candidate[candidate_index].append((product_index, variable))
            row.SetCoefficient(variable, 1.0)

    # Un vecindario de branching local sólo se expresa en filas que pueden
    # variar. Las filas unitarias conservan su geometría incumbente (el arco
    # incumbente se preservó explícitamente arriba), por lo que su contribución
    # a la distancia de Hamming siempre es cero y puede eliminarse.
    variable_product_indices = {
        product_index for product_index, _ in assignment_variables
    }
    if min_changed_products is not None and min_changed_products > len(
        variable_product_indices
    ):
    # Se mantiene exacto el vecindario solicitado: esta fila imposible permite
    # que SCIP informe infactibilidad en vez de ampliarlo silenciosamente.
        solver.RowConstraint(1.0, 0.0, "local_branching_min_changes_infeasible")
    if max_changed_products is not None:
        required_same = max(0, len(variable_product_indices) - max_changed_products)
        local_max = solver.RowConstraint(
            float(required_same), infinity, "local_branching_max_changes"
        )
        for product_index in variable_product_indices:
            incumbent_index = incumbent_indices[product_index]
            local_max.SetCoefficient(
                assignment_variables[(product_index, incumbent_index)], 1.0
            )
    if min_changed_products is not None:
        allowed_same = len(variable_product_indices) - min_changed_products
        local_min = solver.RowConstraint(
            -infinity, float(allowed_same), "local_branching_min_changes"
        )
        for product_index in variable_product_indices:
            incumbent_index = incumbent_indices[product_index]
            local_min.SetCoefficient(
                assignment_variables[(product_index, incumbent_index)], 1.0
            )

    if limit is not None:
        fixed_pallets = sum(
            pallet_count[(product_index, candidate_index)]
            for product_index, candidate_index in fixed_candidate_by_product.items()
        )
        pallet_constraint = solver.RowConstraint(
            -infinity, limit - fixed_pallets, "pallet_budget"
        )
        for key, variable in assignment_variables.items():
            pallet_constraint.SetCoefficient(variable, pallet_count[key])

    first_tier_unit_mills = unit_price_mills(
        thickness_mm, DISCOUNT_TIERS[0].lower_inclusive
    )
    tier_prices_mills = [
        unit_price_mills(thickness_mm, tier.lower_inclusive)
        for tier in DISCOUNT_TIERS
    ]
    objective_scale_mills = freight_policy.expected_mills_per_pallet
    objective_scale_mills = math.gcd(
        objective_scale_mills, first_tier_unit_mills
    )
    for previous_price, tier_price in zip(
        tier_prices_mills, tier_prices_mills[1:]
    ):
        objective_scale_mills = math.gcd(
            objective_scale_mills, previous_price - tier_price
        )
    objective_scale_mills = max(objective_scale_mills, 1)

    objective = solver.Objective()
    objective.SetMinimization()
    base_packaging_mills = first_tier_unit_mills * sum(
        product.annual_volume for product in data.products
    )
    fixed_freight_mills = sum(
        pallet_count[(product_index, candidate_index)]
        * freight_policy.expected_mills_per_pallet
        for product_index, candidate_index in fixed_candidate_by_product.items()
    )
    # Se pliega como constante Procurement de pares candidato/planta cuyo
    # volumen no puede cambiar en este modelo restringido. Antes, estos tipos
    # sólo fijos recibían una variable de volumen y hasta ocho auxiliares de
    # umbral. Un LNS corto con pocos arcos realmente libres podía cargar cientos
    # de variables enteras inútiles. ``base_packaging_mills`` ya valora toda la
    # demanda al tier 1; aquí sólo se suma el delta exacto de descuento sobre todo el volumen.
    fixed_packaging_adjustment_mills = 0
    for candidate_index, candidate_assignments in enumerate(variables_by_candidate):
        fixed_product_indices = fixed_products_by_candidate[candidate_index]
        if not fixed_product_indices:
            continue
        for plant in PLANTS:
            has_variable_volume = any(
                data.products[product_index].annual_volume_by_plant[plant]
                for product_index, _ in candidate_assignments
            )
            if has_variable_volume:
                continue
            fixed_volume = sum(
                data.products[product_index].annual_volume_by_plant[plant]
                for product_index in fixed_product_indices
            )
            if fixed_volume:
                fixed_packaging_adjustment_mills += fixed_volume * (
                    unit_price_mills(thickness_mm, fixed_volume)
                    - first_tier_unit_mills
                )
    fixed_objective_mills = (
        base_packaging_mills
        + fixed_freight_mills
        + fixed_packaging_adjustment_mills
    )
    if fixed_objective_mills % objective_scale_mills:
        raise AssertionError("objective scale must exactly divide the fixed cost")
    fixed_objective_scaled = fixed_objective_mills // objective_scale_mills
    objective.SetOffset(fixed_objective_scaled)

    objective_cap_mills = min(
        incumbent_costs.total_mills,
        target_total_mills
        if target_total_mills is not None
        else incumbent_costs.total_mills,
    )
    objective_cap = solver.RowConstraint(
        -infinity,
        objective_cap_mills // objective_scale_mills - fixed_objective_scaled,
        "objective_cap_without_constant",
    )
    objective_coefficient_by_assignment: dict[tuple[int, int], int] = {}
    for (product_index, candidate_index), variable in assignment_variables.items():
        freight_mills = (
            pallet_count[(product_index, candidate_index)]
            * freight_policy.expected_mills_per_pallet
        )
        if freight_mills % objective_scale_mills:
            raise AssertionError("objective scale must divide every freight cost")
        freight_scaled = freight_mills // objective_scale_mills
        objective_coefficient_by_assignment[(product_index, candidate_index)] = (
            freight_scaled
        )
        objective.SetCoefficient(variable, freight_scaled)
        objective_cap.SetCoefficient(variable, freight_scaled)

    threshold_variables: list[object] = []
    auxiliary_hint_values: list[float] = []
    first_threshold = DISCOUNT_TIERS[1].lower_inclusive
    for candidate_index, candidate_assignments in enumerate(variables_by_candidate):
        fixed_product_indices = fixed_products_by_candidate[candidate_index]
        if not candidate_assignments and not fixed_product_indices:
            continue
        for plant in PLANTS:
            fixed_volume = sum(
                data.products[product_index].annual_volume_by_plant[plant]
                for product_index in fixed_product_indices
            )
            maximum_volume = fixed_volume + sum(
                data.products[product_index].annual_volume_by_plant[plant]
                for product_index, _ in candidate_assignments
            )
            if maximum_volume < first_threshold:
                continue
            positive_assignments = [
                (product_index, variable)
                for product_index, variable in candidate_assignments
                if data.products[product_index].annual_volume_by_plant[plant]
            ]
            if not positive_assignments:
    # Su delta exacto de Procurement se plegó en la constante del objetivo
    # anterior; no hay decisión para este tipo/planta.
                continue
    # Sin demanda fija y con una sola decisión de volumen positivo, el costo de
    # Procurement de tipo/planta tiene exactamente dos estados: cero o el
    # volumen completo del SKU. Se coloca ese delta exacto directamente sobre x
    # y se evita una variable de volumen más hasta ocho auxiliares de tier.
            if fixed_volume == 0 and len(positive_assignments) == 1:
                product_index, variable = positive_assignments[0]
                product_volume = data.products[
                    product_index
                ].annual_volume_by_plant[plant]
                packaging_adjustment_mills = product_volume * (
                    unit_price_mills(thickness_mm, product_volume)
                    - first_tier_unit_mills
                )
                if packaging_adjustment_mills % objective_scale_mills:
                    raise AssertionError(
                        "objective scale must divide direct procurement costs"
                    )
                key = (product_index, candidate_index)
                new_coefficient = (
                    objective_coefficient_by_assignment[key]
                    + packaging_adjustment_mills // objective_scale_mills
                )
                objective_coefficient_by_assignment[key] = new_coefficient
                objective.SetCoefficient(variable, new_coefficient)
                objective_cap.SetCoefficient(variable, new_coefficient)
                continue
            volume = (
                solver.NumVar(
                    fixed_volume,
                    maximum_volume,
                    f"volume_c{candidate_index}_{plant}",
                )
                if relax_integrality
                else solver.IntVar(
                    fixed_volume,
                    maximum_volume,
                    f"volume_c{candidate_index}_{plant}",
                )
            )
            volume_definition = solver.RowConstraint(
                -fixed_volume,
                -fixed_volume,
                f"define_volume_c{candidate_index}_{plant}",
            )
            volume_definition.SetCoefficient(volume, -1.0)
            for product_index, variable in candidate_assignments:
                product_volume = data.products[
                    product_index
                ].annual_volume_by_plant[plant]
                if product_volume:
                    volume_definition.SetCoefficient(variable, product_volume)
            incumbent_volume = fixed_volume + sum(
                data.products[product_index].annual_volume_by_plant[plant]
                for product_index, _ in candidate_assignments
                if incumbent_indices[product_index] == candidate_index
            )
            threshold_variables.append(volume)
            auxiliary_hint_values.append(float(incumbent_volume))

            previous_price = first_tier_unit_mills
            previous_reached = None
            for tier_index, tier in enumerate(DISCOUNT_TIERS[1:], start=1):
                threshold = tier.lower_inclusive
                if threshold > maximum_volume:
                    break
                reached = (
                    solver.NumVar(
                        0.0,
                        1.0,
                        f"reached_c{candidate_index}_{plant}_t{tier_index}",
                    )
                    if relax_integrality
                    else solver.BoolVar(
                        f"reached_c{candidate_index}_{plant}_t{tier_index}"
                    )
                )
                discounted_volume = (
                    solver.NumVar(
                        0.0,
                        maximum_volume,
                        f"discounted_volume_c{candidate_index}_{plant}_t{tier_index}",
                    )
                    if relax_integrality
                    else solver.IntVar(
                        0.0,
                        maximum_volume,
                        f"discounted_volume_c{candidate_index}_{plant}_t{tier_index}",
                    )
                )

    # reached=1 si y sólo si volumen >= umbral.
                lower = solver.RowConstraint(0.0, infinity)
                lower.SetCoefficient(volume, 1.0)
                lower.SetCoefficient(reached, -threshold)
                upper = solver.RowConstraint(-infinity, threshold - 1.0)
                upper.SetCoefficient(volume, 1.0)
                upper.SetCoefficient(
                    reached, -(maximum_volume - threshold + 1)
                )
                if previous_reached is not None:
                    nested = solver.RowConstraint(0.0, infinity)
                    nested.SetCoefficient(previous_reached, 1.0)
                    nested.SetCoefficient(reached, -1.0)

    # discounted_volume = volumen cuando se alcanza el umbral; de lo contrario, cero.
                at_most_volume = solver.RowConstraint(-infinity, 0.0)
                at_most_volume.SetCoefficient(discounted_volume, 1.0)
                at_most_volume.SetCoefficient(volume, -1.0)
                at_most_if_reached = solver.RowConstraint(-infinity, 0.0)
                at_most_if_reached.SetCoefficient(discounted_volume, 1.0)
                at_most_if_reached.SetCoefficient(reached, -maximum_volume)
                at_least_if_reached = solver.RowConstraint(-maximum_volume, infinity)
                at_least_if_reached.SetCoefficient(discounted_volume, 1.0)
                at_least_if_reached.SetCoefficient(volume, -1.0)
                at_least_if_reached.SetCoefficient(reached, -maximum_volume)

                tier_price = unit_price_mills(thickness_mm, threshold)
                discount_coefficient = -(previous_price - tier_price)
                if discount_coefficient % objective_scale_mills:
                    raise AssertionError(
                        "objective scale must divide every procurement discount"
                    )
                discount_scaled = discount_coefficient // objective_scale_mills
                objective.SetCoefficient(discounted_volume, discount_scaled)
                objective_cap.SetCoefficient(discounted_volume, discount_scaled)
                threshold_variables.extend((reached, discounted_volume))
                was_reached = incumbent_volume >= threshold
                auxiliary_hint_values.extend(
                    (float(was_reached), float(incumbent_volume if was_reached else 0))
                )
                previous_price = tier_price
                previous_reached = reached

    hint_variables: list[object] = []
    hint_values: list[float] = []
    for (product_index, candidate_index), variable in assignment_variables.items():
        hint_variables.append(variable)
        hint_values.append(float(candidate_index == incumbent_indices[product_index]))
    hint_variables.extend(threshold_variables)
    hint_values.extend(auxiliary_hint_values)
    solver.SetHint(hint_variables, hint_values)

    if export_model_path is not None:
        model_path = Path(export_model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
    # Los escritores de texto LP y MPS de OR-Tools imprimen coeficientes grandes
    # con pocos dígitos significativos. Sirven para inspección manual, pero un
    # benchmark entre solvers necesita un formato sin pérdidas. Por eso ``.pb``
    # serializa el MPModelProto nativo, cuyos dobles representan exactamente los
    # coeficientes enteros de esta magnitud.
        if model_path.suffix.lower() == ".pb":
            from ortools.linear_solver import linear_solver_pb2

            model_proto = linear_solver_pb2.MPModelProto()
            solver.ExportModelToProto(model_proto)
            model_path.write_bytes(model_proto.SerializeToString())
        elif model_path.suffix.lower() == ".mps":
            exported_model = solver.ExportModelAsMpsFormat(False, False)
            model_path.write_text(exported_model, encoding="utf-8")
        else:
            exported_model = solver.ExportModelAsLpFormat(False)
            model_path.write_text(exported_model, encoding="utf-8")

    model_build_time_seconds = time.perf_counter() - model_build_started_at
    report(
        f"SCIP model: {len(assignment_variables):,} assignment variables, "
        f"{len(threshold_variables):,} procurement auxiliaries, "
        f"built in {model_build_time_seconds:.1f}s; solving"
    )
    solve_started_at = time.perf_counter()
    status_code = solver.Solve()
    solve_time_seconds = time.perf_counter() - solve_started_at
    status = _status_name(pywraplp, status_code)
    report(f"SCIP solve finished with {status} in {solve_time_seconds:.1f}s")
    if status in {"UNBOUNDED", "ABNORMAL", "MODEL_INVALID"}:
        raise RuntimeError(f"SCIP failed with status {status}")
    if status == "INFEASIBLE" and target_total_mills is None:
        raise RuntimeError("SCIP reported the incumbent-bounded model infeasible")

    assignment_arc_values: dict[tuple[str, Dimensions], float] = {}
    assignment_arc_reduced_costs_mills: dict[tuple[str, Dimensions], float] = {}
    if status in {"OPTIMAL", "FEASIBLE"}:
        for (product_index, candidate_index), variable in assignment_variables.items():
            key = (
                data.products[product_index].code,
                candidates[candidate_index].internal,
            )
            assignment_arc_values[key] = variable.solution_value()
            if relax_integrality:
                assignment_arc_reduced_costs_mills[key] = (
                    variable.reduced_cost() * objective_scale_mills
                )

    solver_assignment: dict[str, CandidateBox] | None = None
    solver_costs: CostBreakdown | None = None
    solver_objective_mills: int | None = None
    if status in {"OPTIMAL", "FEASIBLE"} and not relax_integrality:
        solver_assignment = {}
        for product_index, product in enumerate(data.products):
            if product_index in fixed_candidate_by_product:
                selected = fixed_candidate_by_product[product_index]
            else:
                selected = max(
                    candidate_indices_by_product[product_index],
                    key=lambda candidate_index: assignment_variables[
                        (product_index, candidate_index)
                    ].solution_value(),
                )
                if assignment_variables[(product_index, selected)].solution_value() < 0.5:
                    raise RuntimeError(
                        f"SCIP returned no integral selection for {product.code}"
                    )
            solver_assignment[product.code] = candidates[selected]
        solver_costs = evaluate_assignments(
            data.products, solver_assignment, freight_policy
        )
        solver_objective_mills = round(objective.Value() * objective_scale_mills)
        if solver_objective_mills != solver_costs.total_mills:
            raise RuntimeError(
                "SCIP objective does not match independent cost evaluation: "
                f"solver={solver_objective_mills}, evaluated={solver_costs.total_mills}"
            )

    if relax_integrality:
        selected_assignment = incumbent_assignment
        selected_costs = incumbent_costs
        selected_source = "lp_relaxation_incumbent"
        improved_incumbent = False
        solver_objective_mills = (
            round(objective.Value() * objective_scale_mills)
            if status in {"OPTIMAL", "FEASIBLE"}
            else None
        )
    elif solver_costs is not None and solver_costs.total_mills < incumbent_costs.total_mills:
        selected_assignment = solver_assignment
        selected_costs = solver_costs
        selected_source = "solver"
        improved_incumbent = True
    else:
        selected_assignment = incumbent_assignment
        selected_costs = incumbent_costs
        selected_source = "incumbent"
        improved_incumbent = False

    changed_product_count = sum(
        selected_assignment[product.code].internal
        != incumbent_assignment[product.code].internal
        for product in data.products
    )

    raw_best_bound = (
        float(objective.Value()) * objective_scale_mills
        if relax_integrality and status in {"OPTIMAL", "FEASIBLE"}
        else (
            float(objective.BestBound()) * objective_scale_mills
            if status in {"OPTIMAL", "FEASIBLE", "NOT_SOLVED"}
            else None
        )
    )
    # Antes de que SCIP procese la relajación raíz, MPSolver puede exponer como
    # cota su centinela interno de infinito negativo. Los costos no son
    # negativos, por lo que una cota negativa o no finita no es certificable.
    best_bound = (
        raw_best_bound
        if raw_best_bound is not None
        and math.isfinite(raw_best_bound)
        and raw_best_bound >= 0
        else None
    )
    relative_gap = (
        max(
            0.0,
            (selected_costs.total_mills - best_bound)
            / max(abs(selected_costs.total_mills), 1),
        )
        if best_bound is not None
        else None
    )
    return ScipSolveResult(
        thickness_mm=thickness_mm,
        assignment=selected_assignment,
        costs=selected_costs,
        status=status,
        candidate_count=sum(
            bool(variable_items or fixed_items)
            for variable_items, fixed_items in zip(
                variables_by_candidate, fixed_products_by_candidate
            )
        ),
        assignment_variable_count=len(assignment_variables),
        threshold_variable_count=len(threshold_variables),
        candidate_stats=candidate_stats,
        solver_objective_mills=solver_objective_mills,
        best_bound_mills=best_bound,
        candidate_universe_relative_gap=relative_gap,
        wall_time_seconds=solver.WallTime() / 1000.0,
        nodes=0 if relax_integrality else solver.nodes(),
        incumbent_mills=incumbent_costs.total_mills,
        improved_incumbent=improved_incumbent,
        selected_source=selected_source,
        minimum_possible_pallets=minimum_possible_pallets,
        max_extra_pallets=max_extra_pallets,
        target_total_mills=target_total_mills,
        target_met=(
            selected_costs.total_mills <= target_total_mills
            if target_total_mills is not None
            else None
        ),
        max_changed_products=max_changed_products,
        min_changed_products=min_changed_products,
        changed_product_count=changed_product_count,
        fixed_product_count=len(fixed_candidate_by_product),
        pruned_assignment_count=pruned_assignment_count,
        pallet_pruned_assignment_count=pallet_pruned_assignment_count,
        objective_pruned_assignment_count=objective_pruned_assignment_count,
        objective_filter_enabled=enable_objective_filter,
        objective_scale_mills=objective_scale_mills,
        preparation_time_seconds=preparation_time_seconds,
        model_build_time_seconds=model_build_time_seconds,
        solve_time_seconds=solve_time_seconds,
        relaxation=relax_integrality,
        assignment_arc_values=assignment_arc_values,
        assignment_arc_reduced_costs_mills=assignment_arc_reduced_costs_mills,
    )
