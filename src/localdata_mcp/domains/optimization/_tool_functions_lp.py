"""
Tool functions for linear programming and constrained optimization.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ...logging_manager import get_logger
from ._constrained import ConstrainedOptimizer
from ._linear_programming import LinearProgrammingSolver

logger = get_logger(__name__)


def solve_linear_program(
    connection_name: str,
    table_name: str,
    objective_column: str,
    constraint_columns: Optional[List[str]] = None,
    constraint_values: Optional[List[float]] = None,
    constraint_types: Optional[List[str]] = None,
    bounds: Optional[List[Tuple[float, float]]] = None,
    method: str = "highs",
    integer_variables: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """
    Solve linear programming problem from database data.

    Parameters:
    -----------
    connection_name : str
        Database connection name
    table_name : str
        Table containing optimization data
    objective_column : str
        Column with objective function coefficients
    constraint_columns : list of str, optional
        Columns defining constraint matrix
    constraint_values : list of float, optional
        Right-hand side values for constraints
    constraint_types : list of str, optional
        Constraint types ('<=', '>=', '=')
    bounds : list of tuples, optional
        Variable bounds
    method : str, default='highs'
        Optimization method
    integer_variables : list of int, optional
        Indices of integer variables

    Returns:
    --------
    dict
        Linear programming results with solution and analysis
    """
    try:
        from ... import DatabaseManager

        # Get database manager and connection
        db_manager = DatabaseManager()
        engine = db_manager._get_connection(connection_name)

        # Load objective coefficients
        objective_query = f"SELECT {objective_column} FROM {table_name}"
        c = pd.read_sql(objective_query, engine)[objective_column].values

        # Build constraint matrix if provided
        A_ub = None
        b_ub = None

        if constraint_columns and constraint_values:
            constraint_query = (
                f"SELECT {', '.join(constraint_columns)} FROM {table_name}"
            )
            A_ub = pd.read_sql(
                constraint_query, engine
            ).values.T  # Transpose for proper shape

            # Convert constraint types to standard form (Ax <= b)
            b_ub = np.array(constraint_values)
            if constraint_types:
                for i, ctype in enumerate(constraint_types):
                    if ctype == ">=":
                        A_ub[i] = -A_ub[i]
                        b_ub[i] = -b_ub[i]
                    # '=' constraints would need A_eq, b_eq (not implemented here for simplicity)

        # Create and solve LP
        solver = LinearProgrammingSolver(
            method=method, integer_variables=integer_variables, bounds=bounds
        )

        result = solver.solve(c, A_ub=A_ub, b_ub=b_ub)

        # Format results
        formatted_result = {
            "success": result.success,
            "optimal_value": float(result.fun) if result.success else None,
            "optimal_solution": result.x.tolist() if result.success else None,
            "message": result.message,
            "method": result.method,
            "execution_time": result.execution_time,
            "iterations": result.nit,
            "function_evaluations": result.nfev,
            "is_integer_solution": result.is_integer_solution,
            "sensitivity_analysis": result.sensitivity,
            "dual_values": (
                result.dual_values.tolist() if result.dual_values is not None else None
            ),
        }

        # Add constraint analysis if available
        if result.slack is not None:
            formatted_result["constraint_slack"] = result.slack.tolist()

        return formatted_result

    except Exception as e:
        logger.error(f"Linear programming failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "message": f"Linear programming optimization failed: {e}",
        }


def optimize_constrained(
    connection_name: str,
    table_name: str,
    objective_function: str,
    initial_guess_column: str,
    constraint_functions: Optional[List[str]] = None,
    constraint_types: Optional[List[str]] = None,
    bounds_columns: Optional[List[str]] = None,
    method: str = "SLSQP",
    multi_objective: bool = False,
) -> Dict[str, Any]:
    """
    Solve constrained optimization problem from database data.

    Parameters:
    -----------
    connection_name : str
        Database connection name
    table_name : str
        Table containing optimization data
    objective_function : str
        Python expression for objective function using column names
    initial_guess_column : str
        Column with initial guess values
    constraint_functions : list of str, optional
        Python expressions for constraint functions
    constraint_types : list of str, optional
        Constraint types ('eq' for equality, 'ineq' for inequality)
    bounds_columns : list of str, optional
        Columns with lower and upper bounds [lb, ub]
    method : str, default='SLSQP'
        Optimization method
    multi_objective : bool, default=False
        Enable multi-objective optimization

    Returns:
    --------
    dict
        Constrained optimization results
    """
    try:
        from ... import DatabaseManager

        # Get database manager and connection
        db_manager = DatabaseManager()
        engine = db_manager._get_connection(connection_name)

        # Load data
        data = pd.read_sql(f"SELECT * FROM {table_name}", engine)

        # Get initial guess
        x0 = data[initial_guess_column].values

        # Create objective function
        def objective_func(x):
            # Create local variables for evaluation
            local_vars = {col: data[col].values for col in data.columns}
            local_vars.update({"x": x, "np": np})
            return eval(objective_function, {"__builtins__": {}}, local_vars)

        # Create constraints
        constraints = []
        if constraint_functions and constraint_types:
            for func_expr, ctype in zip(constraint_functions, constraint_types):

                def constraint_func(x, expr=func_expr):
                    local_vars = {col: data[col].values for col in data.columns}
                    local_vars.update({"x": x, "np": np})
                    return eval(expr, {"__builtins__": {}}, local_vars)

                constraints.append({"type": ctype, "fun": constraint_func})

        # Create bounds
        bounds = None
        if bounds_columns and len(bounds_columns) >= 2:
            lb = data[bounds_columns[0]].values
            ub = data[bounds_columns[1]].values
            bounds = list(zip(lb, ub))

        # Create and solve optimization
        optimizer = ConstrainedOptimizer(
            method=method,
            constraints=constraints,
            bounds=bounds,
            multi_objective=multi_objective,
        )

        result = optimizer.optimize(objective_func, x0)

        # Format results
        formatted_result = {
            "success": result.success,
            "optimal_value": float(result.fun) if result.success else None,
            "optimal_solution": result.x.tolist() if result.success else None,
            "message": result.message,
            "method": result.method,
            "execution_time": result.execution_time,
            "iterations": result.nit,
            "function_evaluations": result.nfev,
            "constraint_analysis": result.constraints,
            "constraint_violations": (
                result.constraint_violations.tolist()
                if result.constraint_violations is not None
                else None
            ),
            "lagrange_multipliers": (
                result.lagrange_multipliers.tolist()
                if result.lagrange_multipliers is not None
                else None
            ),
        }

        return formatted_result

    except Exception as e:
        logger.error(f"Constrained optimization failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "message": f"Constrained optimization failed: {e}",
        }
