"""localdata_mcp/process/domains/optimization/tools.py — E10.f ToolSpecs.

The optimization family's three tools, carried by name from `main`
(DR GP2): `solve_linear_program`, `optimize_constrained` (FR-305 —
objective/constraint strings only ever meet NX-6's deny-by-default
expression service), `solve_assignment_problem`. Thin over
linear.py / constrained.py with the X-2 addressing contract.
Neighbors: spec_modules.py rosters this module.
"""

from __future__ import annotations

from typing import Any

from localdata_mcp.nexus.contract.spec import Param, TypeShape, tool_spec

from ..support import addressed_frame, source_params
from .constrained import optimize_expression
from .linear import solve_assignment, solve_lp


@tool_spec(
    name="solve_linear_program",
    summary=(
        "Minimise a linear objective on an addressed tabular source: "
        "rows are decision variables, objective_column the cost "
        "vector, each constraint column one constraint's coefficients "
        "with its constraint_values right-hand side and "
        "constraint_types (<=, >=, =). HiGHS solver; reports the "
        "solution, objective value, and solver status."
    ),
    params=(
        *source_params(),
        Param("objective_column", str, "The cost-vector column."),
        Param(
            "constraint_columns",
            list,
            "Constraint coefficient columns (one per constraint).",
            required=False,
        ),
        Param(
            "constraint_values",
            list,
            "Right-hand sides, one per constraint column.",
            required=False,
        ),
        Param(
            "constraint_types",
            list,
            "Per-constraint <= (default), >=, or =.",
            required=False,
        ),
        Param(
            "bounds",
            list,
            "Per-variable [lower, upper] pairs (default: x >= 0).",
            required=False,
        ),
        Param(
            "integer_variables",
            list,
            "Indices of variables constrained to integers.",
            required=False,
        ),
    ),
    input_shape=TypeShape.TABULAR,
    output_shape=TypeShape.SCALAR,
    domain="process",
)
def solve_linear_program(
    objective_column: str,
    endpoint: str | None = None,
    path: str | None = None,
    table: str | None = None,
    query: str | None = None,
    **knobs: Any,
) -> Any:
    frame, source = addressed_frame(endpoint, path, table, query)
    result = solve_lp(frame, objective_column, **knobs)
    result["source"] = source
    return result


@tool_spec(
    name="optimize_constrained",
    summary=(
        "Minimise a nonlinear objective expression over the decision "
        "vector x (e.g. '(x[0]-1)**2 + x[1]'), starting from "
        "initial_guess_column on an addressed tabular source. "
        "Expressions are evaluated by the deny-by-default numeric "
        "grammar — no host code can run. Optional constraint "
        "expressions (ineq: >= 0, or eq)."
    ),
    params=(
        *source_params(),
        Param(
            "objective_expression",
            str,
            "Numeric expression over x, evaluated by the safe grammar.",
        ),
        Param(
            "initial_guess_column",
            str,
            "Column holding the starting decision vector.",
        ),
        Param(
            "constraint_expressions",
            list,
            "Constraint expressions over x (safe grammar).",
            required=False,
        ),
        Param(
            "constraint_types",
            list,
            "Per-constraint ineq (default, >= 0) or eq.",
            required=False,
        ),
        Param(
            "method",
            str,
            "scipy minimize method (implementation default SLSQP).",
            required=False,
        ),
    ),
    input_shape=TypeShape.TABULAR,
    output_shape=TypeShape.SCALAR,
    domain="process",
)
def optimize_constrained(
    objective_expression: str,
    initial_guess_column: str,
    endpoint: str | None = None,
    path: str | None = None,
    table: str | None = None,
    query: str | None = None,
    **knobs: Any,
) -> Any:
    frame, source = addressed_frame(endpoint, path, table, query)
    result = optimize_expression(
        frame, objective_expression, initial_guess_column, **knobs
    )
    result["source"] = source
    return result


@tool_spec(
    name="solve_assignment_problem",
    summary=(
        "Optimal agent-task assignment (Hungarian algorithm) on an "
        "addressed tabular source: rows are agents, cost_columns the "
        "per-task cost columns; optional agent_column names the "
        "agents. Reports the assignment and total cost."
    ),
    params=(
        *source_params(),
        Param("cost_columns", list, "Per-task cost columns."),
        Param(
            "agent_column",
            str,
            "Column naming the agents (default: row index).",
            required=False,
        ),
    ),
    input_shape=TypeShape.TABULAR,
    output_shape=TypeShape.SCALAR,
    domain="process",
)
def solve_assignment_problem(
    cost_columns: list[str],
    endpoint: str | None = None,
    path: str | None = None,
    table: str | None = None,
    query: str | None = None,
    **knobs: Any,
) -> Any:
    frame, source = addressed_frame(endpoint, path, table, query)
    result = solve_assignment(frame, cost_columns, **knobs)
    result["source"] = source
    return result
