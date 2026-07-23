"""OR-Tools CP-SAT model for the discrete candidate-box master problem."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Collection, Mapping
from dataclasses import dataclass

from .candidates import generate_candidates
from .config import DISCOUNT_TIERS, FreightPolicy
from .costs import (
    evaluate_assignments,
    freight_pallets,
    unit_price_mills,
)
from .exact_candidates import ExactCandidateStats, generate_exact_candidates
from .models import CandidateBox, CostBreakdown, Dimensions, PLANTS, PreparedData, Product


@dataclass(frozen=True)
class SolveResult:
    thickness_mm: float
    assignment: dict[str, CandidateBox]
    costs: CostBreakdown
    status: str
    candidate_count: int
    candidate_strategy: str
    candidate_stats: ExactCandidateStats | None
    solver_objective_mills: int | None
    best_bound_mills: float | None
    candidate_universe_relative_gap: float | None
    wall_time_seconds: float
    num_conflicts: int
    num_branches: int
    random_seed: int
    incumbent_mills: int
    improved_incumbent: bool
    selected_source: str
    minimum_possible_pallets: int | None
    max_extra_pallets: int | None
    target_total_mills: int | None
    target_met: bool | None
    target_proven_infeasible: bool | None


def _import_cp_sat():
    try:
        from ortools.sat.python import cp_model
    except ImportError as exc:  # pragma: no cover - depends on local environment
        raise RuntimeError(
            "OR-Tools is required. Install project dependencies with "
            "`py -3 -m pip install -e .`."
        ) from exc
    return cp_model


def solve_for_thickness(
    data: PreparedData,
    thickness_mm: float,
    freight_policy: FreightPolicy,
    *,
    time_limit_seconds: float = 300.0,
    num_search_workers: int = 8,
    pair_profile_limit: int = 90,
    max_extra_pair_designs: int = 1_000,
    pallet_variant_profile_limit: int = 90,
    max_pallet_variants_per_profile: int = 18,
    warm_start_variant_profile_limit: int = 60,
    warm_start_compromise_group_limit: int = 40,
    max_compromise_variants_per_group: int = 18,
    random_seed: int = 42,
    initial_assignment: dict[str, CandidateBox] | None = None,
    candidate_strategy: str = "exact",
    preserve_individual_max_capacity: bool = False,
    max_extra_pallets: int | None = None,
    target_total_mills: int | None = None,
    free_product_codes: Collection[str] | None = None,
    allowed_internals_by_product: Mapping[str, Collection[Dimensions]] | None = None,
    precomputed_exact_candidates: tuple[CandidateBox, ...] | None = None,
    precomputed_exact_candidate_stats: ExactCandidateStats | None = None,
    cp_model_presolve: bool | None = None,
    symmetry_level: int | None = None,
    max_presolve_iterations: int | None = None,
    linearization_level: int | None = None,
    log_search_progress: bool = False,
    use_lns_only: bool | None = None,
    diversify_lns_params: bool | None = None,
    lns_initial_difficulty: float | None = None,
    lns_initial_deterministic_limit: float | None = None,
) -> SolveResult:
    """Optimize one globally fixed thickness over a finite candidate universe.

    The supplied assignment is a protected incumbent, not merely a search
    hint.  If no assignment is supplied, the current internal dimensions form
    a guaranteed fallback.  A time-limited solve can therefore never make the
    returned solution worse.  When ``target_total_mills`` is supplied, CP-SAT
    solves a pure feasibility problem constrained to that total cost; if the
    target is not reached, the protected incumbent is returned.

    ``free_product_codes`` optionally defines a large-neighbourhood search:
    only those SKUs may change design and every other SKU is fixed to its
    incumbent geometry.  Supplying a neighbourhood requires
    ``initial_assignment``.  Cost tiers are still calculated over all SKUs,
    including the fixed part of the incumbent.

    ``allowed_internals_by_product`` can further restrict selected SKUs to a
    small set of geometries (for example, incumbent-or-target binary moves).
    It requires an incumbent.  The incumbent geometry is always added to each
    supplied set so the protected fallback remains feasible.  Products not in
    the mapping remain unrestricted; products fixed by ``free_product_codes``
    remain fixed even if the mapping contains additional geometries for them.

    A repeated LNS driver may provide a complete precomputed exact universe to
    avoid enumerating the same integer grid before every subproblem.  That
    universe must retain every incumbent or explicitly allowed geometry.
    """

    cp_model = _import_cp_sat()
    if candidate_strategy != "exact" and (
        preserve_individual_max_capacity
        or max_extra_pallets is not None
        or precomputed_exact_candidates is not None
        or precomputed_exact_candidate_stats is not None
    ):
        raise ValueError(
            "pallet-capacity restrictions require the complete exact candidate strategy"
        )
    if max_extra_pallets is not None and max_extra_pallets < 0:
        raise ValueError("max_extra_pallets cannot be negative")
    if target_total_mills is not None and target_total_mills < 0:
        raise ValueError("target_total_mills cannot be negative")
    if symmetry_level is not None and symmetry_level < 0:
        raise ValueError("symmetry_level cannot be negative")
    if max_presolve_iterations is not None and max_presolve_iterations < 0:
        raise ValueError("max_presolve_iterations cannot be negative")
    if linearization_level is not None and linearization_level < 0:
        raise ValueError("linearization_level cannot be negative")
    if (
        lns_initial_difficulty is not None
        and not 0.0 <= lns_initial_difficulty <= 1.0
    ):
        raise ValueError("lns_initial_difficulty must be between zero and one")
    if (
        lns_initial_deterministic_limit is not None
        and lns_initial_deterministic_limit <= 0
    ):
        raise ValueError("lns_initial_deterministic_limit must be positive")
    if (
        precomputed_exact_candidate_stats is not None
        and precomputed_exact_candidates is None
    ):
        raise ValueError(
            "precomputed exact candidate stats require a precomputed universe"
        )
    if preserve_individual_max_capacity and max_extra_pallets is not None:
        raise ValueError(
            "preserve_individual_max_capacity and max_extra_pallets are alternative restrictions"
        )
    expected_codes = {product.code for product in data.products}
    if free_product_codes is not None and initial_assignment is None:
        raise ValueError("free_product_codes requires an initial assignment")
    if allowed_internals_by_product is not None and initial_assignment is None:
        raise ValueError(
            "allowed_internals_by_product requires an initial assignment"
        )
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
    provided_initial_costs: CostBreakdown | None = None
    if initial_assignment is not None:
        if set(initial_assignment) != expected_codes:
            raise ValueError("initial assignment must contain exactly one row for every product")
        initial_thicknesses = {
            candidate.thickness_mm for candidate in initial_assignment.values()
        }
        if initial_thicknesses != {thickness_mm}:
            raise ValueError(
                f"initial assignment thicknesses {sorted(initial_thicknesses)} do not match "
                f"requested {thickness_mm:g} mm"
            )
        provided_initial_costs = evaluate_assignments(
            data.products, initial_assignment, freight_policy
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
        # An empty requested collection deliberately means "stay put".  More
        # generally, adding the incumbent here makes every restricted LNS
        # model retain the protected fallback without burdening callers with
        # duplicating it in every binary move set.
        active_allowed_internals[code] = frozenset(
            (*requested, initial_assignment[code].internal)
        )

    seed_profiles: tuple[Dimensions, ...] = ()
    retained_designs: tuple[Dimensions, ...] = tuple(
        sorted(
            (
                {
                    candidate.internal
                    for candidate in initial_assignment.values()
                }
                if initial_assignment is not None
                else (
                    set()
                    if preserve_individual_max_capacity or max_extra_pallets is not None
                    else {product.current_internal for product in data.products}
                )
            )
            | {
                internal
                for internals in active_allowed_internals.values()
                for internal in internals
            },
            key=Dimensions.as_tuple,
        )
    )
    compromise_groups: tuple[tuple[Product, ...], ...] = ()
    if initial_assignment is not None and warm_start_variant_profile_limit > 0:
        seed_volume: dict[Dimensions, int] = defaultdict(int)
        for product in data.products:
            seed_volume[initial_assignment[product.code].internal] += product.annual_volume
        seed_profiles = tuple(
            internal
            for internal, _ in sorted(
                seed_volume.items(), key=lambda item: (-item[1], item[0].as_tuple())
            )[:warm_start_variant_profile_limit]
        )
    if initial_assignment is not None and warm_start_compromise_group_limit > 0:
        products_by_design: dict[Dimensions, list[Product]] = defaultdict(list)
        for product in data.products:
            products_by_design[initial_assignment[product.code].internal].append(product)
        compromise_groups = tuple(
            tuple(group)
            for _, group in sorted(
                (
                    (internal, group)
                    for internal, group in products_by_design.items()
                    if len(group) > 1
                ),
                key=lambda item: (-sum(product.annual_volume for product in item[1]), item[0].as_tuple()),
            )[:warm_start_compromise_group_limit]
        )
    candidate_stats: ExactCandidateStats | None = None
    if candidate_strategy == "exact":
        if precomputed_exact_candidates is None:
            candidates, candidate_stats = generate_exact_candidates(
                data.products,
                thickness_mm,
                retained_designs=retained_designs,
            )
        else:
            candidates = precomputed_exact_candidates
            candidate_stats = precomputed_exact_candidate_stats
            if not candidates:
                raise ValueError("precomputed exact candidate universe cannot be empty")
            if {candidate.thickness_mm for candidate in candidates} != {thickness_mm}:
                raise ValueError(
                    "precomputed exact candidates do not match requested thickness"
                )
    elif candidate_strategy == "heuristic":
        candidates = generate_candidates(
            data.products,
            thickness_mm,
            pair_profile_limit=pair_profile_limit,
            max_extra_pair_designs=max_extra_pair_designs,
            pallet_variant_profile_limit=pallet_variant_profile_limit,
            max_pallet_variants_per_profile=max_pallet_variants_per_profile,
            seed_pallet_variant_profiles=seed_profiles,
            retained_designs=retained_designs,
            seed_compromise_groups=compromise_groups,
            max_compromise_variants_per_group=max_compromise_variants_per_group,
        )
    else:
        raise ValueError(f"unknown candidate strategy: {candidate_strategy!r}")

    # Retained designs guarantee that requested geometries survive exact
    # signature deduplication and dominance pruning.  Still validate per-SKU
    # compatibility explicitly: a geometry can be feasible for some product
    # while being invalid for the SKU to which the caller assigned it.
    compatible_internals_by_code: dict[str, set[Dimensions]] = defaultdict(set)
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

    maximum_capacity_by_code: dict[str, int] = {}
    if preserve_individual_max_capacity:
        maximum_capacity_by_code = {
            product.code: max(
                candidate.capacity_per_pallet
                for candidate in candidates
                if product.code in candidate.compatible_product_codes
            )
            for product in data.products
        }
        if initial_assignment is not None:
            for product in data.products:
                incumbent_capacity = initial_assignment[product.code].capacity_per_pallet
                if incumbent_capacity != maximum_capacity_by_code[product.code]:
                    raise ValueError(
                        "initial assignment does not preserve individual maximum capacity for "
                        f"{product.code}: {incumbent_capacity} != "
                        f"{maximum_capacity_by_code[product.code]}"
                    )
        candidates = tuple(
            candidate
            for candidate in candidates
            if any(
                candidate.capacity_per_pallet == maximum_capacity_by_code[code]
                for code in candidate.compatible_product_codes
            )
        )

    minimum_pallets_by_code: dict[str, int] = {}
    pallet_count_by_code_and_internal: dict[tuple[str, Dimensions], int] = {}
    minimum_possible_pallets: int | None = None
    if max_extra_pallets is not None:
        for product in data.products:
            compatible_pallet_counts: list[int] = []
            for candidate in candidates:
                if product.code not in candidate.compatible_product_codes:
                    continue
                pallets = sum(
                    freight_pallets(product, candidate, plant) for plant in PLANTS
                )
                pallet_count_by_code_and_internal[(product.code, candidate.internal)] = pallets
                compatible_pallet_counts.append(pallets)
            minimum_pallets_by_code[product.code] = min(compatible_pallet_counts)
        minimum_possible_pallets = sum(minimum_pallets_by_code.values())
        if (
            provided_initial_costs is not None
            and provided_initial_costs.pallets > minimum_possible_pallets + max_extra_pallets
        ):
            raise ValueError(
                "initial assignment exceeds pallet budget: "
                f"{provided_initial_costs.pallets} > "
                f"{minimum_possible_pallets + max_extra_pallets}"
            )
        candidates = tuple(
            candidate
            for candidate in candidates
            if any(
                pallet_count_by_code_and_internal[(code, candidate.internal)]
                <= minimum_pallets_by_code[code] + max_extra_pallets
                for code in candidate.compatible_product_codes
            )
        )

    # Apply the LNS restriction only after calculating the global pallet
    # minima above.  Otherwise fixing an incumbent SKU could artificially
    # raise ``minimum_possible_pallets`` and silently loosen the requested
    # extra-pallet budget.  A candidate is retained only if at least one SKU
    # can actually use it in this neighbourhood and under all active capacity
    # restrictions.
    def candidate_is_allowed_for_product(
        product: Product, candidate: CandidateBox
    ) -> bool:
        if product.code not in candidate.compatible_product_codes:
            return False
        if (
            product.code not in active_free_product_codes
            and candidate.internal != initial_assignment[product.code].internal
        ):
            return False
        if (
            product.code in active_allowed_internals
            and candidate.internal not in active_allowed_internals[product.code]
        ):
            return False
        if (
            preserve_individual_max_capacity
            and candidate.capacity_per_pallet != maximum_capacity_by_code[product.code]
        ):
            return False
        if (
            max_extra_pallets is not None
            and pallet_count_by_code_and_internal[(product.code, candidate.internal)]
            > minimum_pallets_by_code[product.code] + max_extra_pallets
        ):
            return False
        return True

    candidates = tuple(
        candidate
        for candidate in candidates
        if any(
            candidate_is_allowed_for_product(product, candidate)
            for product in data.products
        )
    )
    model = cp_model.CpModel()
    product_count, candidate_count = len(data.products), len(candidates)

    assignment_variables: dict[tuple[int, int], object] = {}
    variables_by_product: list[list[object]] = [[] for _ in range(product_count)]
    candidate_indices_by_product: list[list[int]] = [[] for _ in range(product_count)]
    variables_by_candidate: list[list[tuple[int, object]]] = [
        [] for _ in range(candidate_count)
    ]
    for product_index, product in enumerate(data.products):
        for candidate_index, candidate in enumerate(candidates):
            if not candidate_is_allowed_for_product(product, candidate):
                continue
            variable = model.NewBoolVar(f"assign_p{product_index}_c{candidate_index}")
            assignment_variables[(product_index, candidate_index)] = variable
            variables_by_product[product_index].append(variable)
            candidate_indices_by_product[product_index].append(candidate_index)
            variables_by_candidate[candidate_index].append((product_index, variable))
        if not variables_by_product[product_index]:
            raise RuntimeError(f"no feasible candidate for {product.code}")
        model.AddExactlyOne(variables_by_product[product_index])

    if max_extra_pallets is not None:
        model.Add(
            sum(
                pallet_count_by_code_and_internal[
                    (data.products[product_index].code, candidates[candidate_index].internal)
                ]
                * variable
                for (product_index, candidate_index), variable in assignment_variables.items()
            )
            <= minimum_possible_pallets + max_extra_pallets
        )

    # Rebuild the incumbent with the canonical candidates in this universe.
    # Retained designs ensure every required geometry is present even after
    # exact signature deduplication and dominance pruning.
    candidate_by_internal = {
        candidate.internal: candidate_index for candidate_index, candidate in enumerate(candidates)
    }
    incumbent_assignment: dict[str, CandidateBox] = {}
    incumbent_candidate_by_product: list[int] = []
    for product_index, product in enumerate(data.products):
        hinted_internal = (
            initial_assignment[product.code].internal
            if initial_assignment is not None
            else (
                candidates[
                    min(
                        candidate_indices_by_product[product_index],
                        key=lambda candidate_index: (
                            pallet_count_by_code_and_internal[
                                (product.code, candidates[candidate_index].internal)
                            ],
                            candidates[candidate_index].internal.as_tuple(),
                        ),
                    )
                ].internal
                if max_extra_pallets is not None
                else (
                    candidates[candidate_indices_by_product[product_index][0]].internal
                    if preserve_individual_max_capacity
                    else product.current_internal
                )
            )
        )
        if hinted_internal not in candidate_by_internal:
            raise ValueError(
                f"warm-start design {hinted_internal.as_tuple()} is not in the candidate universe"
            )
        hinted_candidate = candidate_by_internal[hinted_internal]
        if (product_index, hinted_candidate) not in assignment_variables:
            raise ValueError(
                f"incumbent design {hinted_internal.as_tuple()} is infeasible for {product.code}"
            )
        incumbent_assignment[product.code] = candidates[hinted_candidate]
        incumbent_candidate_by_product.append(hinted_candidate)

    for (product_index, candidate_index), variable in assignment_variables.items():
        model.AddHint(
            variable,
            int(candidate_index == incumbent_candidate_by_product[product_index]),
        )

    objective_terms: list[object] = []
    if preserve_individual_max_capacity:
        objective_terms.append(
            sum(
                sum(
                    (
                        product.annual_volume_by_plant[plant]
                        + maximum_capacity_by_code[product.code]
                        - 1
                    )
                    // maximum_capacity_by_code[product.code]
                    if product.annual_volume_by_plant[plant]
                    else 0
                    for plant in PLANTS
                )
                * freight_policy.expected_mills_per_pallet
                for product in data.products
            )
        )
    else:
        for (product_index, candidate_index), variable in assignment_variables.items():
            product = data.products[product_index]
            candidate = candidates[candidate_index]
            pallet_count = sum(
                freight_pallets(product, candidate, plant) for plant in PLANTS
            )
            objective_terms.append(
                pallet_count * freight_policy.expected_mills_per_pallet * variable
            )

    # At a fixed thickness every shipped unit has the same first-tier price,
    # regardless of its assigned design or plant.  This part of procurement is
    # therefore a global constant; only all-units threshold discounts depend
    # on the assignment.
    first_tier_unit_mills = unit_price_mills(
        thickness_mm, DISCOUNT_TIERS[0].lower_inclusive
    )
    objective_terms.append(
        first_tier_unit_mills
        * sum(product.annual_volume for product in data.products)
    )
    first_discount_threshold = DISCOUNT_TIERS[1].lower_inclusive

    for candidate_index, candidate in enumerate(candidates):
        candidate_assignments = variables_by_candidate[candidate_index]
        for plant in PLANTS:
            max_volume = sum(
                data.products[product_index].annual_volume_by_plant[plant]
                for product_index, _ in candidate_assignments
            )
            # Below the first threshold this candidate/plant can never earn a
            # discount, so the global base-cost constant already accounts for
            # it and no procurement variables are necessary.
            if max_volume < first_discount_threshold:
                continue
            volume = model.NewIntVar(0, max_volume, f"volume_c{candidate_index}_{plant}")
            model.Add(
                volume
                == sum(
                    data.products[product_index].annual_volume_by_plant[plant] * variable
                    for product_index, variable in candidate_assignments
                )
            )
            incumbent_volume = sum(
                data.products[product_index].annual_volume_by_plant[plant]
                for product_index, _ in candidate_assignments
                if incumbent_candidate_by_product[product_index] == candidate_index
            )
            model.AddHint(volume, incumbent_volume)
            # Procurement is an all-units discount schedule.  Express it as
            # first-tier cost minus one cumulative discount for every reached
            # threshold.  This is exactly equivalent to the former one-hot
            # tier formulation while removing the plant-active variable and
            # the mutually-exclusive tier-volume partition.
            previous_unit_mills = first_tier_unit_mills
            for tier_index, tier in enumerate(DISCOUNT_TIERS[1:], start=1):
                threshold = tier.lower_inclusive
                if threshold > max_volume:
                    break
                reached = model.NewBoolVar(
                    f"threshold_c{candidate_index}_{plant}_{tier_index}"
                )
                model.Add(volume >= threshold).OnlyEnforceIf(reached)
                model.Add(volume <= threshold - 1).OnlyEnforceIf(reached.Not())
                model.AddHint(reached, int(incumbent_volume >= threshold))

                discounted_volume = model.NewIntVar(
                    0,
                    max_volume,
                    f"discount_volume_c{candidate_index}_{plant}_{tier_index}",
                )
                model.Add(discounted_volume == volume).OnlyEnforceIf(reached)
                model.Add(discounted_volume == 0).OnlyEnforceIf(reached.Not())
                model.AddHint(
                    discounted_volume,
                    incumbent_volume if incumbent_volume >= threshold else 0,
                )

                tier_unit_mills = unit_price_mills(
                    candidate.thickness_mm, threshold
                )
                objective_terms.append(
                    -(previous_unit_mills - tier_unit_mills) * discounted_volume
                )
                previous_unit_mills = tier_unit_mills

    incumbent_costs = evaluate_assignments(
        data.products, incumbent_assignment, freight_policy
    )
    objective_expression = cp_model.LinearExpr.Sum(objective_terms)
    model.Add(objective_expression <= incumbent_costs.total_mills)
    if target_total_mills is None:
        model.Minimize(objective_expression)
    else:
        model.Add(objective_expression <= target_total_mills)
    model_validation = model.Validate()
    if model_validation:
        raise RuntimeError(f"invalid CP-SAT model for {thickness_mm} mm: {model_validation}")
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_seconds
    solver.parameters.num_search_workers = num_search_workers
    solver.parameters.random_seed = random_seed
    if cp_model_presolve is not None:
        solver.parameters.cp_model_presolve = cp_model_presolve
    if symmetry_level is not None:
        solver.parameters.symmetry_level = symmetry_level
    if max_presolve_iterations is not None:
        solver.parameters.max_presolve_iterations = max_presolve_iterations
    if linearization_level is not None:
        solver.parameters.linearization_level = linearization_level
    solver.parameters.log_search_progress = log_search_progress
    if use_lns_only is not None:
        solver.parameters.use_lns_only = use_lns_only
    if diversify_lns_params is not None:
        solver.parameters.diversify_lns_params = diversify_lns_params
    if lns_initial_difficulty is not None:
        solver.parameters.lns_initial_difficulty = lns_initial_difficulty
    if lns_initial_deterministic_limit is not None:
        solver.parameters.lns_initial_deterministic_limit = (
            lns_initial_deterministic_limit
        )
    if (
        target_total_mills is not None
        and target_total_mills < incumbent_costs.total_mills
    ):
        # The incumbent assignment is deliberately just outside the target.
        # Ask CP-SAT to use it as a repair starting point instead of discarding
        # the otherwise complete and highly informative hint.
        solver.parameters.repair_hint = True
        solver.parameters.hint_conflict_limit = 10_000
    status_code = solver.Solve(model)
    status = solver.StatusName(status_code)
    if status == "MODEL_INVALID":
        raise RuntimeError(f"CP-SAT did not find a feasible solution for {thickness_mm} mm: {status}")
    if status == "INFEASIBLE" and target_total_mills is None:
        raise RuntimeError(f"CP-SAT did not find a feasible solution for {thickness_mm} mm: {status}")

    solver_assignment: dict[str, CandidateBox] | None = None
    solver_costs: CostBreakdown | None = None
    solver_objective_mills: int | None = None
    if status in {"OPTIMAL", "FEASIBLE"}:
        solver_assignment = {}
        for product_index, product in enumerate(data.products):
            selected = next(
                candidate_index
                for candidate_index in candidate_indices_by_product[product_index]
                if solver.Value(assignment_variables[(product_index, candidate_index)])
            )
            solver_assignment[product.code] = candidates[selected]
        solver_costs = evaluate_assignments(data.products, solver_assignment, freight_policy)
        solver_objective_mills = (
            solver_costs.total_mills
            if target_total_mills is not None
            else round(solver.ObjectiveValue())
        )
        if target_total_mills is None and solver_objective_mills != solver_costs.total_mills:
            raise RuntimeError(
                "CP-SAT objective does not match independent cost evaluation: "
                f"solver={solver_objective_mills}, evaluated={solver_costs.total_mills}"
            )

    if solver_costs is not None and solver_costs.total_mills < incumbent_costs.total_mills:
        selected_assignment = solver_assignment
        selected_costs = solver_costs
        selected_source = "solver"
        improved_incumbent = True
    else:
        selected_assignment = incumbent_assignment
        selected_costs = incumbent_costs
        selected_source = "incumbent"
        improved_incumbent = False

    best_bound_mills = (
        float(solver.BestObjectiveBound())
        if target_total_mills is None and status in {"OPTIMAL", "FEASIBLE", "UNKNOWN"}
        else None
    )
    relative_gap = None
    if best_bound_mills is not None:
        relative_gap = max(
            0.0,
            (selected_costs.total_mills - best_bound_mills)
            / max(abs(selected_costs.total_mills), 1),
        )
    return SolveResult(
        thickness_mm=thickness_mm,
        assignment=selected_assignment,
        costs=selected_costs,
        status=status,
        candidate_count=candidate_count,
        candidate_strategy=candidate_strategy,
        candidate_stats=candidate_stats,
        solver_objective_mills=solver_objective_mills,
        best_bound_mills=best_bound_mills,
        candidate_universe_relative_gap=relative_gap,
        wall_time_seconds=solver.WallTime(),
        num_conflicts=solver.NumConflicts(),
        num_branches=solver.NumBranches(),
        random_seed=random_seed,
        incumbent_mills=incumbent_costs.total_mills,
        improved_incumbent=improved_incumbent,
        selected_source=selected_source,
        minimum_possible_pallets=minimum_possible_pallets,
        max_extra_pallets=max_extra_pallets,
        target_total_mills=target_total_mills,
        target_met=(
            solver_costs is not None
            and solver_costs.total_mills <= target_total_mills
            if target_total_mills is not None
            else None
        ),
        target_proven_infeasible=(
            status == "INFEASIBLE" if target_total_mills is not None else None
        ),
    )


def solve_all_thicknesses(
    data: PreparedData,
    freight_policy: FreightPolicy,
    **solver_kwargs: object,
) -> tuple[SolveResult, tuple[SolveResult, ...]]:
    """Solve all allowed global thicknesses and return the least-cost result."""

    results = tuple(
        solve_for_thickness(data, thickness, freight_policy, **solver_kwargs)
        for thickness in (3.0, 4.5, 5.0)
    )
    return min(results, key=lambda result: result.costs.total_mills), results
