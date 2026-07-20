"""MCP tool methods for the optimization domain.

Thin wrappers registered as MCP tools: resolve the named connection, delegate to
the adapter in :mod:`localdata_mcp.optimization_tools`, and serialize the result.
Kept in a mixin so ``database_manager`` does not grow with each new domain.

These tools take a ``table_name`` rather than a query, because the underlying
solvers read the columns they need directly from the table.
"""

from typing import List, Optional

from ..json_utils import safe_dumps
from ..optimization_tools import (
    tool_analyze_network,
    tool_optimize_constrained,
    tool_solve_assignment_problem,
    tool_solve_linear_program,
)


class OptimizationToolsMixin:
    """Optimization MCP tools, mixed into ``DatabaseManager``."""

    def solve_linear_program(
        self,
        connection_name: str,
        table_name: str,
        objective_column: str,
        constraint_columns: Optional[List[str]] = None,
        constraint_values: Optional[List[float]] = None,
        constraint_types: Optional[List[str]] = None,
        method: str = "highs",
        integer_variables: Optional[List[int]] = None,
    ) -> str:
        """Minimise a linear objective subject to linear constraints.

        Args:
            connection_name: Name of the connected database.
            table_name: Table holding the objective coefficients and constraints.
            objective_column: Numeric column of objective coefficients, one per variable.
            constraint_columns: Columns forming the constraint matrix.
            constraint_values: Right-hand-side value for each constraint.
            constraint_types: Per-constraint '<=', '>=' or '=' (default '<=').
            method: Solver method (default 'highs').
            integer_variables: Indices of variables constrained to integers.
        """
        engine = self._get_connection(connection_name)
        result = tool_solve_linear_program(
            engine,
            table_name,
            objective_column,
            constraint_columns=constraint_columns,
            constraint_values=constraint_values,
            constraint_types=constraint_types,
            method=method,
            integer_variables=integer_variables,
        )
        return safe_dumps(result)

    def optimize_constrained(
        self,
        connection_name: str,
        table_name: str,
        objective_function: str,
        initial_guess_column: str,
        constraint_functions: Optional[List[str]] = None,
        constraint_types: Optional[List[str]] = None,
        bounds_columns: Optional[List[str]] = None,
        method: str = "SLSQP",
    ) -> str:
        """Optimize a nonlinear objective from a starting point stored in a column.

        Args:
            connection_name: Name of the connected database.
            table_name: Table holding the starting point and any bounds.
            objective_function: Objective as a Python expression in `x`, e.g. 'np.sum(x**2)'.
            initial_guess_column: Numeric column giving the starting point.
            constraint_functions: Constraints as expressions in `x`.
            constraint_types: Per-constraint 'eq' or 'ineq'.
            bounds_columns: Two columns giving lower and upper bounds per variable.
            method: scipy.optimize method (default 'SLSQP').
        """
        engine = self._get_connection(connection_name)
        result = tool_optimize_constrained(
            engine,
            table_name,
            objective_function,
            initial_guess_column,
            constraint_functions=constraint_functions,
            constraint_types=constraint_types,
            bounds_columns=bounds_columns,
            method=method,
        )
        return safe_dumps(result)

    def analyze_network(
        self,
        connection_name: str,
        table_name: str,
        source_column: str,
        target_column: str,
        weight_column: str = "",
        directed: bool = False,
        include_centrality: bool = True,
    ) -> str:
        """Analyze a graph stored as an edge table.

        Reports graph properties, centrality measures, shortest paths and a
        minimum spanning tree. Node identifiers must be numeric.

        Args:
            connection_name: Name of the connected database.
            table_name: Table holding one edge per row.
            source_column: Numeric column identifying the source node.
            target_column: Numeric column identifying the target node.
            weight_column: Numeric edge-weight column (default: unweighted).
            directed: Treat edges as directed (default False).
            include_centrality: Compute centrality measures (default True).
        """
        engine = self._get_connection(connection_name)
        result = tool_analyze_network(
            engine,
            table_name,
            source_column,
            target_column,
            weight_column=weight_column or None,
            directed=directed,
            include_centrality=include_centrality,
        )
        return safe_dumps(result)

    def solve_assignment_problem(
        self,
        connection_name: str,
        table_name: str,
        cost_matrix_columns: List[str],
        agent_id_column: str = "",
        task_id_column: str = "",
        method: str = "hungarian",
        maximize: bool = False,
    ) -> str:
        """Assign agents to tasks at optimal total cost.

        Args:
            connection_name: Name of the connected database.
            table_name: Table holding the cost matrix, one agent per row.
            cost_matrix_columns: Numeric columns, one per task.
            agent_id_column: Column naming each agent (default: row index).
            task_id_column: Column naming each task.
            method: 'hungarian' or 'greedy' (default 'hungarian').
            maximize: Maximise total value instead of minimising cost.
        """
        engine = self._get_connection(connection_name)
        result = tool_solve_assignment_problem(
            engine,
            table_name,
            cost_matrix_columns,
            agent_id_column=agent_id_column or None,
            task_id_column=task_id_column or None,
            method=method,
            maximize=maximize,
        )
        return safe_dumps(result)
