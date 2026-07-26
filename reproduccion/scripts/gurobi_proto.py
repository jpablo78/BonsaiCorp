"""Puente mínimo entre MPModelProto de OR-Tools y el modelo de Gurobi."""

from __future__ import annotations

import math

import gurobipy as gp
from gurobipy import GRB
from ortools.linear_solver import linear_solver_pb2


def _finite(value: float) -> bool:
    return math.isfinite(value) and abs(value) < 1e100


def optional_model_attr(model: gp.Model, name: str) -> float | None:
    """Devuelve un atributo de Gurobi si existe para el estado del modelo."""

    try:
        return float(model.getAttr(name))
    except (AttributeError, gp.GurobiError):
        return None


def bound_usd(model: gp.Model, objective_scale_mills: int) -> float | None:
    """Convierte la mejor cota disponible de Gurobi a USD."""

    bound = optional_model_attr(model, "ObjBound")
    return None if bound is None else bound * objective_scale_mills / 1000


def status_name(status: int) -> str:
    """Traduce el código numérico de estado de Gurobi a su nombre técnico."""

    names = {
        GRB.OPTIMAL: "OPTIMAL",
        GRB.INFEASIBLE: "INFEASIBLE",
        GRB.INF_OR_UNBD: "INF_OR_UNBD",
        GRB.UNBOUNDED: "UNBOUNDED",
        GRB.CUTOFF: "CUTOFF",
        GRB.TIME_LIMIT: "TIME_LIMIT",
        GRB.SOLUTION_LIMIT: "SOLUTION_LIMIT",
        GRB.INTERRUPTED: "INTERRUPTED",
        GRB.NUMERIC: "NUMERIC",
        GRB.SUBOPTIMAL: "SUBOPTIMAL",
    }
    return names.get(status, f"STATUS_{status}")


def gurobi_from_proto(
    proto: linear_solver_pb2.MPModelProto,
) -> tuple[gp.Model, list[gp.Var]]:
    """Crea un modelo Gurobi equivalente, incluyendo cotas, objetivo e inicio MIP."""

    model = gp.Model("bonsai_decimal")
    model.Params.OutputFlag = 0
    variables: list[gp.Var] = []
    for variable in proto.variable:
        lower = variable.lower_bound if _finite(variable.lower_bound) else -GRB.INFINITY
        upper = variable.upper_bound if _finite(variable.upper_bound) else GRB.INFINITY
        variables.append(
            model.addVar(
                lb=lower,
                ub=upper,
                vtype=GRB.INTEGER if variable.is_integer else GRB.CONTINUOUS,
                name=variable.name,
            )
        )
    model.update()
    for row in proto.constraint:
        expression = gp.LinExpr(
            list(row.coefficient), [variables[index] for index in row.var_index]
        )
        lower, upper = row.lower_bound, row.upper_bound
        if _finite(lower) and _finite(upper) and lower == upper:
            model.addConstr(expression == lower, name=row.name)
        elif _finite(lower) and _finite(upper):
            model.addRange(expression, lower, upper, name=row.name)
        elif _finite(upper):
            model.addConstr(expression <= upper, name=row.name)
        elif _finite(lower):
            model.addConstr(expression >= lower, name=row.name)
        else:
            raise ValueError(f"fila sin cota representable: {row.name!r}")
    for index, value in zip(
        proto.solution_hint.var_index, proto.solution_hint.var_value, strict=True
    ):
        variables[index].Start = value
    objective = gp.LinExpr(
        [variable.objective_coefficient for variable in proto.variable], variables
    )
    objective += proto.objective_offset
    model.setObjective(objective, GRB.MAXIMIZE if proto.maximize else GRB.MINIMIZE)
    model.update()
    return model, variables
