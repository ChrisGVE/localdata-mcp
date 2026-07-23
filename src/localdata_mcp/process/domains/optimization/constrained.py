"""localdata_mcp/process/domains/optimization/constrained.py — FR-301/305.

`optimize_constrained`'s computation — the FR-305 closure site. On
`main` the objective and constraint strings went through `eval()`
(#42, live RCE); here EVERY caller-supplied expression is evaluated
exclusively by NX-6's deny-by-default service through the guard seam
(`evaluate_numeric_expression`), with the decision vector bound as
the ONE symbol `x`. An unsafe or non-numeric expression is refused
before the solver starts (probed once at the initial point); scipy's
SLSQP then minimizes the guarded callable. The result carries the
solver's own `converged` verdict — the sentinel's class-2 signal.
Neighbors: linear.py holds the LP/assignment pair; tools.py declares
the ToolSpec.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd

from localdata_mcp.ingest.runtime import chokepoint
from localdata_mcp.nexus.chokepoint.guard import ExpressionRefusedError

from ..support import invalid_source_refusal, numeric_values

CONSTRAINT_KINDS = ("ineq", "eq")


def optimize_expression(
    frame: pd.DataFrame,
    objective_expression: str,
    initial_guess_column: str,
    constraint_expressions: list[str] | None = None,
    constraint_types: list[str] | None = None,
    method: str = "SLSQP",
) -> dict[str, Any]:
    """Minimise the guarded expression from the column's start point."""
    from scipy.optimize import minimize

    x0 = numeric_values(frame, initial_guess_column).to_numpy(dtype=float)
    objective = _guarded_callable(objective_expression)
    objective(x0)  # probe once: an unsafe expression refuses BEFORE solving
    constraints = _constraint_set(constraint_expressions, constraint_types, x0)
    outcome = minimize(
        objective,
        x0,
        method=method,
        constraints=constraints,
    )
    return {
        "method": method,
        "objective_expression": objective_expression,
        "n_variables": int(len(x0)),
        "solution": [float(value) for value in outcome.x],
        "objective_value": float(outcome.fun),
        "iterations": int(getattr(outcome, "nit", 0)),
        "converged": bool(outcome.success),
        "message": str(outcome.message),
    }


def _guarded_callable(
    expression: str,
) -> Callable[["np.ndarray[Any, Any]"], float]:
    """The FR-305 bridge: scipy sees a callable, the string only ever
    meets NX-6's deny-by-default interpreter."""

    def evaluate(x: "np.ndarray[Any, Any]") -> float:
        try:
            return chokepoint().evaluate_numeric_expression(
                expression, {"x": [float(value) for value in np.atleast_1d(x)]}
            )
        except ExpressionRefusedError as refused:
            raise invalid_source_refusal(
                f"expression refused by the safety gate: {refused}"
            ) from None

    return evaluate


def _constraint_set(
    expressions: list[str] | None,
    types: list[str] | None,
    x0: "np.ndarray[Any, Any]",
) -> list[dict[str, Any]]:
    if not expressions:
        return []
    resolved_types = list(types) if types else ["ineq"] * len(expressions)
    if len(resolved_types) != len(expressions):
        raise invalid_source_refusal(
            "constraint_types must match constraint_expressions one to one."
        )
    constraints: list[dict[str, Any]] = []
    for expression, kind in zip(expressions, resolved_types):
        if kind not in CONSTRAINT_KINDS:
            raise invalid_source_refusal(
                f"Unknown constraint type {kind!r} — one of "
                f"{list(CONSTRAINT_KINDS)} (scipy's convention: ineq means "
                "the expression stays >= 0)."
            )
        guarded = _guarded_callable(expression)
        guarded(x0)  # probe: refuse unsafe constraints before solving
        constraints.append({"type": kind, "fun": guarded})
    return constraints
