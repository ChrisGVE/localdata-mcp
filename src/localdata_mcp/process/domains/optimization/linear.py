"""localdata_mcp/process/domains/optimization/linear.py — FR-301.

`solve_linear_program` and `solve_assignment_problem`'s computations,
re-authored from `main`'s optimization domain over scipy directly.
LP keeps `main`'s data layout: rows are decision variables, the
objective column is the cost vector c, each constraint column is one
constraint's coefficient vector with its right-hand side and type
(<=, >=, =). The result carries the solver's own `optimizer_status`
(HiGHS status, 0 = optimal) — the sentinel's class-2 signal, so an
infeasible or unbounded program becomes a structured error, never a
silent success. Assignment: the Hungarian algorithm
(`linear_sum_assignment`) over the named cost columns. Neighbors:
constrained.py handles the nonlinear string-objective case; tools.py
declares the ToolSpecs.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..support import invalid_source_refusal, require_columns

CONSTRAINT_TYPES = ("<=", ">=", "=")


def solve_lp(
    frame: pd.DataFrame,
    objective_column: str,
    constraint_columns: list[str] | None = None,
    constraint_values: list[float] | None = None,
    constraint_types: list[str] | None = None,
    bounds: list[list[float]] | None = None,
    integer_variables: list[int] | None = None,
) -> dict[str, Any]:
    """Minimise c·x subject to the column-declared constraints."""
    from scipy.optimize import linprog

    require_columns(frame, objective_column, *(constraint_columns or ()))
    c = pd.to_numeric(frame[objective_column], errors="coerce").dropna().to_numpy()
    if len(c) == 0:
        raise invalid_source_refusal(
            f"Column {objective_column!r} carries no numeric coefficients."
        )
    a_ub, b_ub, a_eq, b_eq = _constraint_system(
        frame, len(c), constraint_columns, constraint_values, constraint_types
    )
    box = [tuple(pair) for pair in bounds] if bounds else None
    integrality = None
    if integer_variables:
        integrality = np.zeros(len(c))
        integrality[list(integer_variables)] = 1
    outcome = linprog(
        c,
        A_ub=a_ub,
        b_ub=b_ub,
        A_eq=a_eq,
        b_eq=b_eq,
        bounds=box,
        integrality=integrality,
        method="highs",
    )
    return {
        "objective_value": float(outcome.fun) if outcome.success else None,
        "solution": [float(value) for value in outcome.x]
        if outcome.x is not None
        else None,
        "n_variables": int(len(c)),
        "optimizer_status": int(outcome.status),
        "converged": bool(outcome.success),
        "message": str(outcome.message),
    }


def _constraint_system(
    frame: pd.DataFrame,
    n_variables: int,
    columns: list[str] | None,
    values: list[float] | None,
    types: list[str] | None,
) -> tuple[Any, Any, Any, Any]:
    """(A_ub, b_ub, A_eq, b_eq) from the column-per-constraint layout."""
    if not columns:
        return None, None, None, None
    if values is None or len(values) != len(columns):
        raise invalid_source_refusal(
            "constraint_values must supply one right-hand side per constraint column."
        )
    resolved_types = list(types) if types else ["<="] * len(columns)
    if len(resolved_types) != len(columns):
        raise invalid_source_refusal(
            "constraint_types must match constraint_columns one to one."
        )
    upper_rows, upper_rhs, eq_rows, eq_rhs = [], [], [], []
    for column, rhs, kind in zip(columns, values, resolved_types):
        if kind not in CONSTRAINT_TYPES:
            raise invalid_source_refusal(
                f"Unknown constraint type {kind!r} — one of {list(CONSTRAINT_TYPES)}."
            )
        coefficients = pd.to_numeric(frame[column], errors="coerce").dropna().to_numpy()
        if len(coefficients) != n_variables:
            raise invalid_source_refusal(
                f"Constraint column {column!r} has {len(coefficients)} "
                f"coefficients for {n_variables} variables."
            )
        if kind == "<=":
            upper_rows.append(coefficients)
            upper_rhs.append(float(rhs))
        elif kind == ">=":
            upper_rows.append(-coefficients)
            upper_rhs.append(-float(rhs))
        else:
            eq_rows.append(coefficients)
            eq_rhs.append(float(rhs))
    return (
        np.array(upper_rows) if upper_rows else None,
        np.array(upper_rhs) if upper_rhs else None,
        np.array(eq_rows) if eq_rows else None,
        np.array(eq_rhs) if eq_rhs else None,
    )


def solve_assignment(
    frame: pd.DataFrame,
    cost_columns: list[str],
    agent_column: str | None = None,
) -> dict[str, Any]:
    """Hungarian assignment over the rows-as-agents cost matrix."""
    from scipy.optimize import linear_sum_assignment

    if not cost_columns:
        raise invalid_source_refusal("cost_columns must name at least one column.")
    require_columns(frame, *cost_columns, agent_column)
    matrix = frame[cost_columns].apply(pd.to_numeric, errors="coerce").dropna()
    if matrix.empty:
        raise invalid_source_refusal(
            "No complete numeric cost rows remain after dropping missing values."
        )
    rows, columns = linear_sum_assignment(matrix.to_numpy(dtype=float))
    agents = (
        [str(value) for value in frame.loc[matrix.index, agent_column]]
        if agent_column is not None
        else [str(index) for index in matrix.index]
    )
    assignments = [
        {
            "agent": agents[int(row)],
            "task": cost_columns[int(column)],
            "cost": float(matrix.iloc[int(row), int(column)]),
        }
        for row, column in zip(rows, columns)
    ]
    return {
        "assignments": assignments,
        "total_cost": float(matrix.to_numpy(dtype=float)[rows, columns].sum()),
        "n_agents": int(matrix.shape[0]),
        "n_tasks": int(matrix.shape[1]),
    }
