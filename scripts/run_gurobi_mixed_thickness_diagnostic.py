"""Diagnostic MIP allowing one carton thickness per physical box type.

This deliberately relaxes the challenge's global-thickness rule.  It must not
be submitted as an official solution unless that rule is explicitly changed.
All remaining strict geometry, headspace, ECT, pallet, plant-tier and freight
rules are reused unchanged.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import gurobipy as gp
from gurobipy import GRB

from bonsai.config import ALLOWED_THICKNESSES, DISCOUNT_TIERS, FreightPolicy
from bonsai.costs import evaluate_assignments, freight_pallets, unit_price_mills
from bonsai.data import load_prepared_data
from bonsai.exact_candidates import generate_exact_candidates
from bonsai.geometry import product_fits_candidate
from bonsai.reporting import write_assignment_csv, write_json
from bonsai.solution_validation import validate_solution_csv


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument(
        "--warm-start",
        type=Path,
        default=Path("output_lp_pool_after_ba_15m/asignacion_optima.csv"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("output_gurobi_mixed_thickness_diagnostic")
    )
    parser.add_argument("--time-limit-seconds", type=float, default=300.0)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--mip-gap", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20260722)
    return parser


def run(args: argparse.Namespace) -> dict[str, object]:
    data = load_prepared_data(args.data_dir)
    policy = FreightPolicy()
    incumbent = validate_solution_csv(args.warm_start, data, policy)
    incumbent_thicknesses = {box.thickness_mm for box in incumbent.assignment.values()}
    if len(incumbent_thicknesses) != 1:
        raise ValueError("warm start must use the official single global thickness")

    retained_3mm = tuple(
        sorted(
            {box.internal for box in incumbent.assignment.values()},
            key=lambda dimensions: dimensions.as_tuple(),
        )
    )
    candidates = []
    stats_by_thickness: dict[str, dict[str, object]] = {}
    for thickness in ALLOWED_THICKNESSES:
        generated, stats = generate_exact_candidates(
            data.products,
            thickness,
            retained_designs=retained_3mm if thickness == 3.0 else (),
            prune_dominated=True,
        )
        candidates.extend(generated)
        stats_by_thickness[f"{thickness:g}"] = {
            **stats.__dict__,
            "candidate_count": len(generated),
        }
    candidates = tuple(candidates)

    compatible_by_product: list[list[int]] = []
    for product in data.products:
        compatible = [
            index
            for index, candidate in enumerate(candidates)
            if product.code in candidate.compatible_product_codes
        ]
        if not compatible:
            raise RuntimeError(f"mixed universe leaves {product.code} uncovered")
        compatible_by_product.append(compatible)

    model = gp.Model("bonsai_mixed_thickness_diagnostic")
    model.Params.TimeLimit = args.time_limit_seconds
    model.Params.Threads = args.threads
    model.Params.MIPGap = args.mip_gap
    model.Params.MIPGapAbs = 0.0
    model.Params.MIPFocus = 1
    model.Params.Heuristics = 0.20
    model.Params.Seed = args.seed
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.Params.LogFile = str(args.output_dir / "gurobi.log")

    assignment_vars: dict[tuple[int, int], gp.Var] = {}
    assignments_by_candidate: list[list[tuple[int, gp.Var]]] = [
        [] for _ in candidates
    ]
    for product_index, candidate_indices in enumerate(compatible_by_product):
        row_vars = []
        for candidate_index in candidate_indices:
            variable = model.addVar(
                vtype=GRB.BINARY,
                name=f"x_p{product_index}_c{candidate_index}",
            )
            assignment_vars[(product_index, candidate_index)] = variable
            assignments_by_candidate[candidate_index].append((product_index, variable))
            row_vars.append(variable)
        model.addConstr(gp.quicksum(row_vars) == 1, name=f"assign_p{product_index}")
    model.update()

    # All relevant prices and freight are exact multiples of this scale.  The
    # smaller coefficients improve numerical conditioning without rounding.
    all_unit_prices = {
        unit_price_mills(thickness, tier.lower_inclusive)
        for thickness in ALLOWED_THICKNESSES
        for tier in DISCOUNT_TIERS
    }
    objective_scale_mills = policy.expected_mills_per_pallet
    for price in all_unit_prices:
        objective_scale_mills = math.gcd(objective_scale_mills, price)
    objective_scale_mills = max(objective_scale_mills, 1)

    objective = gp.LinExpr()
    for (product_index, candidate_index), variable in assignment_vars.items():
        product = data.products[product_index]
        candidate = candidates[candidate_index]
        pallets = sum(freight_pallets(product, candidate, plant) for plant in product.annual_volume_by_plant)
        freight_mills = pallets * policy.expected_mills_per_pallet
        objective += (freight_mills // objective_scale_mills) * variable

    tier_binaries: dict[tuple[int, str, int], gp.Var] = {}
    tier_volumes: dict[tuple[int, str, int], gp.Var] = {}
    for candidate_index, candidate_assignments in enumerate(assignments_by_candidate):
        if not candidate_assignments:
            continue
        candidate = candidates[candidate_index]
        for plant in data.products[0].annual_volume_by_plant:
            positive = [
                (product_index, variable)
                for product_index, variable in candidate_assignments
                if data.products[product_index].annual_volume_by_plant[plant] > 0
            ]
            if not positive:
                continue
            maximum_volume = sum(
                data.products[product_index].annual_volume_by_plant[plant]
                for product_index, _ in positive
            )
            active_tiers = [
                (tier_index, tier)
                for tier_index, tier in enumerate(DISCOUNT_TIERS)
                if tier.lower_inclusive <= maximum_volume
            ]
            volume_parts = []
            selectors = []
            for tier_index, tier in active_tiers:
                upper = min(
                    maximum_volume,
                    tier.upper_inclusive
                    if tier.upper_inclusive is not None
                    else maximum_volume,
                )
                selector = model.addVar(
                    vtype=GRB.BINARY,
                    name=f"tier_c{candidate_index}_{plant}_t{tier_index}",
                )
                volume_part = model.addVar(
                    lb=0.0,
                    ub=upper,
                    vtype=GRB.CONTINUOUS,
                    name=f"tier_volume_c{candidate_index}_{plant}_t{tier_index}",
                )
                model.addConstr(
                    volume_part >= tier.lower_inclusive * selector,
                    name=f"tier_lower_c{candidate_index}_{plant}_t{tier_index}",
                )
                model.addConstr(
                    volume_part <= upper * selector,
                    name=f"tier_upper_c{candidate_index}_{plant}_t{tier_index}",
                )
                price = unit_price_mills(candidate.thickness_mm, tier.lower_inclusive)
                objective += (price // objective_scale_mills) * volume_part
                tier_binaries[(candidate_index, plant, tier_index)] = selector
                tier_volumes[(candidate_index, plant, tier_index)] = volume_part
                selectors.append(selector)
                volume_parts.append(volume_part)
            model.addConstr(
                gp.quicksum(volume_parts)
                == gp.quicksum(
                    data.products[product_index].annual_volume_by_plant[plant] * variable
                    for product_index, variable in positive
                ),
                name=f"volume_c{candidate_index}_{plant}",
            )
            model.addConstr(
                gp.quicksum(selectors) <= 1,
                name=f"one_tier_c{candidate_index}_{plant}",
            )

    model.setObjective(objective, GRB.MINIMIZE)
    incumbent_scaled = incumbent.costs.total_mills // objective_scale_mills
    if incumbent_scaled * objective_scale_mills != incumbent.costs.total_mills:
        raise AssertionError("objective scale does not exactly divide incumbent")
    model.addConstr(objective <= incumbent_scaled, name="protect_official_incumbent")

    candidate_index_by_key = {
        (candidate.thickness_mm, candidate.internal): index
        for index, candidate in enumerate(candidates)
    }
    warm_indices: list[int] = []
    for product_index, product in enumerate(data.products):
        warm_box = incumbent.assignment[product.code]
        candidate_index = candidate_index_by_key[(warm_box.thickness_mm, warm_box.internal)]
        warm_indices.append(candidate_index)
        for index in compatible_by_product[product_index]:
            assignment_vars[(product_index, index)].Start = float(index == candidate_index)

    warm_volume: dict[tuple[int, str], int] = defaultdict(int)
    for product_index, candidate_index in enumerate(warm_indices):
        product = data.products[product_index]
        for plant, volume in product.annual_volume_by_plant.items():
            warm_volume[(candidate_index, plant)] += volume
    for key, selector in tier_binaries.items():
        candidate_index, plant, tier_index = key
        volume = warm_volume[(candidate_index, plant)]
        selected = volume > 0 and DISCOUNT_TIERS[tier_index].contains(volume)
        selector.Start = float(selected)
        tier_volumes[key].Start = float(volume if selected else 0)

    model.optimize()
    if model.SolCount < 1:
        raise RuntimeError(f"Gurobi found no mixed-thickness solution; status={model.Status}")

    assignment = {}
    for product_index, candidate_indices in enumerate(compatible_by_product):
        selected = [
            index
            for index in candidate_indices
            if assignment_vars[(product_index, index)].X > 0.5
        ]
        if len(selected) != 1:
            raise AssertionError(
                f"product {data.products[product_index].code} has {len(selected)} selected boxes"
            )
        candidate = candidates[selected[0]]
        product = data.products[product_index]
        if not product_fits_candidate(product, candidate.internal, candidate.thickness_mm):
            raise AssertionError(f"selected box is geometrically infeasible for {product.code}")
        assignment[product.code] = candidate

    costs = evaluate_assignments(data.products, assignment, policy)
    solver_mills = round(model.ObjVal * objective_scale_mills)
    if solver_mills != costs.total_mills:
        raise AssertionError(
            f"solver objective {solver_mills} differs from independent cost {costs.total_mills}"
        )
    output_path = args.output_dir / "asignacion_mixed_thickness_DIAGNOSTIC_ONLY.csv"
    write_assignment_csv(output_path, data, assignment)

    selected_type_keys = {
        (
            candidate.thickness_mm,
            *candidate.external.as_tuple(),
        )
        for candidate in assignment.values()
    }
    type_count_by_thickness = {
        f"{thickness:g}": sum(key[0] == thickness for key in selected_type_keys)
        for thickness in ALLOWED_THICKNESSES
    }
    product_count_by_thickness = {
        f"{thickness:g}": sum(
            candidate.thickness_mm == thickness for candidate in assignment.values()
        )
        for thickness in ALLOWED_THICKNESSES
    }
    changed_products = sum(
        assignment[product.code].thickness_mm != incumbent.assignment[product.code].thickness_mm
        or assignment[product.code].external != incumbent.assignment[product.code].external
        for product in data.products
    )
    payload = {
        "diagnostic_only": True,
        "official_global_thickness_rule_relaxed": True,
        "status": int(model.Status),
        "optimal": model.Status == GRB.OPTIMAL,
        "runtime_seconds": model.Runtime,
        "nodes": model.NodeCount,
        "relative_gap": model.MIPGap,
        "best_bound_usd": model.ObjBound * objective_scale_mills / 1000,
        "objective_scale_mills": objective_scale_mills,
        "candidate_count": len(candidates),
        "assignment_variable_count": len(assignment_vars),
        "tier_binary_count": len(tier_binaries),
        "candidate_stats_by_thickness": stats_by_thickness,
        "costs": costs.as_dict(),
        "official_incumbent_costs": incumbent.costs.as_dict(),
        "savings_vs_official_incumbent_usd": (
            incumbent.costs.total_mills - costs.total_mills
        )
        / 1000,
        "type_count_by_thickness": type_count_by_thickness,
        "product_count_by_thickness": product_count_by_thickness,
        "changed_products": changed_products,
        "output_csv": str(output_path),
    }
    write_json(args.output_dir / "resumen_mixed_thickness_diagnostic.json", payload)
    return payload


if __name__ == "__main__":
    print(json.dumps(run(_parser().parse_args()), ensure_ascii=False, indent=2))
