"""MCP tool functions for the optimization domain.

Thin adapters over :mod:`localdata_mcp.domains.optimization`. Those domain
functions read their inputs from a table themselves — they take an engine and a
table name rather than a query — so unlike the other domains there is no
DataFrame marshalling to do here. What these adapters add is validation of the
table and column names before the query is built, and a uniform structured
error rather than a raw exception.
"""

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from sqlalchemy import inspect as sa_inspect
from sqlalchemy.engine import Engine

from .logging_manager import get_logger

logger = get_logger(__name__)


def _validate_table_and_columns(
    engine: Engine, table_name: str, *columns: Optional[str]
) -> None:
    """Fail early, and by name, on a table or column that does not exist.

    The domain functions interpolate these straight into SQL, so an unknown name
    surfaces as a database error that does not say which argument was wrong.
    """
    inspector = sa_inspect(engine)
    tables = inspector.get_table_names()
    if table_name not in tables:
        raise ValueError(
            f"Table '{table_name}' not found. Available tables: {sorted(tables)}"
        )

    available = {col["name"] for col in inspector.get_columns(table_name)}
    missing = [c for c in columns if c and c not in available]
    if missing:
        raise ValueError(
            f"Column(s) {missing} not found in table '{table_name}'. "
            f"Available columns: {sorted(available)}"
        )


def _require_numeric(engine: Engine, table_name: str, *columns: Optional[str]) -> None:
    """Reject non-numeric columns where the solver needs numbers.

    The optimization solvers coerce their inputs to a float array, so a text
    column fails deep inside with "could not convert string to float" and no
    indication of which column caused it.
    """
    wanted = [c for c in columns if c]
    if not wanted:
        return

    sample = pd.read_sql(
        f"SELECT {', '.join(wanted)} FROM {table_name} LIMIT 50", engine
    )
    non_numeric = [
        col
        for col in wanted
        if pd.to_numeric(sample[col], errors="coerce").isna().all() and not sample.empty
    ]
    if non_numeric:
        raise ValueError(
            f"Column(s) {non_numeric} are not numeric. Optimization requires numeric "
            "values; map identifiers to numbers before analysis."
        )


def tool_solve_linear_program(
    engine: Engine,
    table_name: str,
    objective_column: str,
    constraint_columns: Optional[List[str]] = None,
    constraint_values: Optional[List[float]] = None,
    constraint_types: Optional[List[str]] = None,
    bounds: Optional[List[Tuple[float, float]]] = None,
    method: str = "highs",
    integer_variables: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Minimise a linear objective subject to linear constraints."""
    from .domains.optimization import solve_linear_program

    _validate_table_and_columns(
        engine, table_name, objective_column, *(constraint_columns or [])
    )
    _require_numeric(engine, table_name, objective_column, *(constraint_columns or []))

    if constraint_columns and constraint_values is None:
        raise ValueError(
            "constraint_values is required when constraint_columns is given: "
            "each constraint column needs a right-hand-side value."
        )

    return solve_linear_program(
        engine,
        table_name,
        objective_column,
        constraint_columns=constraint_columns,
        constraint_values=constraint_values,
        constraint_types=constraint_types,
        bounds=bounds,
        method=method,
        integer_variables=integer_variables,
    )


def tool_optimize_constrained(
    engine: Engine,
    table_name: str,
    objective_function: str,
    initial_guess_column: str,
    constraint_functions: Optional[List[str]] = None,
    constraint_types: Optional[List[str]] = None,
    bounds_columns: Optional[List[str]] = None,
    method: str = "SLSQP",
    multi_objective: bool = False,
) -> Dict[str, Any]:
    """Optimize a nonlinear objective from a starting point held in a column."""
    from .domains.optimization import optimize_constrained

    _validate_table_and_columns(
        engine, table_name, initial_guess_column, *(bounds_columns or [])
    )
    _require_numeric(engine, table_name, initial_guess_column)

    return optimize_constrained(
        engine,
        table_name,
        objective_function,
        initial_guess_column,
        constraint_functions=constraint_functions,
        constraint_types=constraint_types,
        bounds_columns=bounds_columns,
        method=method,
        multi_objective=multi_objective,
    )


def tool_analyze_network(
    engine: Engine,
    table_name: str,
    source_column: str,
    target_column: str,
    weight_column: Optional[str] = None,
    directed: bool = False,
    include_centrality: bool = True,
    algorithms: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Analyze a graph stored as an edge table."""
    from .domains.optimization import analyze_network

    _validate_table_and_columns(
        engine, table_name, source_column, target_column, weight_column
    )
    # The analyzer builds a numeric array from the edge list, so node identifiers
    # must be numeric. Say so here rather than failing inside the array coercion.
    _require_numeric(engine, table_name, source_column, target_column, weight_column)

    return analyze_network(
        engine,
        table_name,
        source_column,
        target_column,
        weight_column=weight_column,
        directed=directed,
        include_centrality=include_centrality,
        algorithms=algorithms,
    )


def tool_solve_assignment_problem(
    engine: Engine,
    table_name: str,
    cost_matrix_columns: List[str],
    agent_id_column: Optional[str] = None,
    task_id_column: Optional[str] = None,
    method: str = "hungarian",
    maximize: bool = False,
    allow_partial: bool = False,
) -> Dict[str, Any]:
    """Assign agents to tasks at optimal total cost."""
    from .domains.optimization import solve_assignment_problem

    if not cost_matrix_columns:
        raise ValueError("cost_matrix_columns must name at least one cost column.")

    _validate_table_and_columns(
        engine, table_name, *cost_matrix_columns, agent_id_column, task_id_column
    )
    _require_numeric(engine, table_name, *cost_matrix_columns)

    return solve_assignment_problem(
        engine,
        table_name,
        cost_matrix_columns,
        agent_id_column=agent_id_column,
        task_id_column=task_id_column,
        method=method,
        maximize=maximize,
        allow_partial=allow_partial,
    )
