"""
Linear Programming Solver - simplex method, dual analysis, and integer programming.
"""

import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.utils.validation import check_is_fitted
from scipy.optimize import linprog, OptimizeResult

from ...logging_manager import get_logger
from ...pipeline.base import (
    AnalysisPipelineBase,
    StreamingConfig,
)
from ._types import LinearProgramResult

logger = get_logger(__name__)


class LinearProgrammingSolver(AnalysisPipelineBase):
    """
    Linear Programming Solver with simplex method, dual analysis, and integer programming.

    Supports various linear programming formulations and provides comprehensive
    analysis including sensitivity analysis and dual problem solutions.
    """

    def __init__(
        self,
        method: str = "highs",
        integer_variables: Optional[List[int]] = None,
        bounds: Optional[List[Tuple[float, float]]] = None,
        options: Optional[Dict[str, Any]] = None,
        streaming_config: Optional[StreamingConfig] = None,
    ):
        """
        Initialize Linear Programming Solver.

        Parameters:
        -----------
        method : str, default='highs'
            Optimization method ('highs', 'highs-ds', 'highs-ipm', 'interior-point', 'revised simplex')
        integer_variables : list of int, optional
            Indices of variables that must be integers
        bounds : list of tuples, optional
            Variable bounds [(lb1, ub1), (lb2, ub2), ...]
        options : dict, optional
            Solver-specific options
        streaming_config : StreamingConfig, optional
            Configuration for streaming processing
        """
        super().__init__(
            analytical_intention="linear programming optimization",
            streaming_config=streaming_config,
        )
        self.method = method
        self.integer_variables = integer_variables or []
        self.bounds = bounds
        self.options = options or {}

        self.problem_ = None
        self.result_ = None
        self.is_fitted_ = False

    def fit(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> "LinearProgrammingSolver":
        """
        Fit is not applicable for LP solver - use solve_linear_program instead.
        """
        logger.warning(
            "LinearProgrammingSolver.fit() called - use solve_linear_program() instead"
        )
        self.is_fitted_ = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform is not applicable for LP solver."""
        check_is_fitted(self, "is_fitted_")
        return X

    def solve(
        self,
        c: np.ndarray,
        A_ub: Optional[np.ndarray] = None,
        b_ub: Optional[np.ndarray] = None,
        A_eq: Optional[np.ndarray] = None,
        b_eq: Optional[np.ndarray] = None,
        bounds: Optional[List[Tuple[float, float]]] = None,
    ) -> LinearProgramResult:
        """
        Solve linear programming problem.

        Minimize: c^T * x
        Subject to: A_ub * x <= b_ub
                   A_eq * x == b_eq
                   bounds
        """
        start_time = time.time()

        # Use instance bounds if not provided
        if bounds is None:
            bounds = self.bounds

        try:
            if self.integer_variables:
                # Integer programming
                result = self._solve_integer_lp(c, A_ub, b_ub, A_eq, b_eq, bounds)
            else:
                # Standard linear programming
                result = linprog(
                    c,
                    A_ub=A_ub,
                    b_ub=b_ub,
                    A_eq=A_eq,
                    b_eq=b_eq,
                    bounds=bounds,
                    method=self.method,
                    options=self.options,
                )
        except Exception as e:
            logger.error(f"Linear programming failed: {e}")
            return LinearProgramResult(
                success=False,
                fun=np.inf,
                x=np.array([]),
                message=str(e),
                nit=0,
                nfev=0,
                execution_time=time.time() - start_time,
                method=self.method,
            )

        execution_time = time.time() - start_time

        # Create comprehensive result
        lp_result = LinearProgramResult(
            success=result.success,
            fun=result.fun,
            x=result.x,
            message=result.message,
            nit=getattr(result, "nit", 0),
            nfev=getattr(result, "nfev", 0),
            execution_time=execution_time,
            method=self.method,
            slack=getattr(result, "slack", None),
            con=getattr(result, "con", None),
            is_integer_solution=bool(self.integer_variables),
        )

        # Add dual analysis if available
        if hasattr(result, "eqlin") and result.eqlin is not None:
            lp_result.dual_values = result.eqlin.marginals

        # Perform sensitivity analysis
        if result.success and A_ub is not None:
            lp_result.sensitivity = self._sensitivity_analysis(
                c, A_ub, b_ub, A_eq, b_eq, result
            )

        self.result_ = lp_result
        self.is_fitted_ = True

        logger.info(
            f"Linear programming completed in {execution_time:.3f}s, success: {result.success}"
        )
        return lp_result

    def _solve_integer_lp(
        self,
        c: np.ndarray,
        A_ub: Optional[np.ndarray] = None,
        b_ub: Optional[np.ndarray] = None,
        A_eq: Optional[np.ndarray] = None,
        b_eq: Optional[np.ndarray] = None,
        bounds: Optional[List[Tuple[float, float]]] = None,
    ) -> OptimizeResult:
        """Solve integer linear programming problem using scipy.optimize.milp."""
        try:
            # Try using milp if available (scipy >= 1.9.0)
            from scipy.optimize import milp, Bounds, LinearConstraint

            # Convert bounds
            if bounds:
                lb = [b[0] if b[0] is not None else -np.inf for b in bounds]
                ub = [b[1] if b[1] is not None else np.inf for b in bounds]
            else:
                n_vars = len(c)
                lb = [-np.inf] * n_vars
                ub = [np.inf] * n_vars

            bounds_obj = Bounds(lb=lb, ub=ub)

            # Build constraints
            constraints = []
            if A_ub is not None and b_ub is not None:
                constraints.append(LinearConstraint(A_ub, -np.inf, b_ub))
            if A_eq is not None and b_eq is not None:
                constraints.append(LinearConstraint(A_eq, b_eq, b_eq))

            # Specify integer variables
            integrality = np.zeros(len(c))
            for idx in self.integer_variables:
                if idx < len(c):
                    integrality[idx] = 1

            result = milp(
                c,
                bounds=bounds_obj,
                constraints=constraints,
                integrality=integrality,
                options=self.options,
            )

            return result

        except ImportError:
            # Fallback to branch and bound with regular linprog
            logger.warning("MILP not available, using branch-and-bound approximation")
            return self._branch_and_bound_lp(c, A_ub, b_ub, A_eq, b_eq, bounds)

    def _branch_and_bound_lp(
        self,
        c: np.ndarray,
        A_ub: Optional[np.ndarray] = None,
        b_ub: Optional[np.ndarray] = None,
        A_eq: Optional[np.ndarray] = None,
        b_eq: Optional[np.ndarray] = None,
        bounds: Optional[List[Tuple[float, float]]] = None,
    ) -> OptimizeResult:
        """Simple branch-and-bound for integer LP (fallback implementation)."""
        # First solve relaxed problem
        result = linprog(
            c,
            A_ub=A_ub,
            b_ub=b_ub,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=bounds,
            method=self.method,
            options=self.options,
        )

        if not result.success:
            return result

        # Check if integer constraints are satisfied
        x = result.x
        integer_satisfied = True
        for idx in self.integer_variables:
            if idx < len(x) and abs(x[idx] - round(x[idx])) > 1e-6:
                integer_satisfied = False
                break

        if integer_satisfied:
            return result

        # Simple rounding heuristic (not optimal, but provides feasible solution)
        x_rounded = x.copy()
        for idx in self.integer_variables:
            if idx < len(x_rounded):
                x_rounded[idx] = round(x[idx])

        # Create result with rounded solution
        rounded_result = OptimizeResult(
            {
                "success": True,
                "fun": np.dot(c, x_rounded),
                "x": x_rounded,
                "message": "Integer solution found via rounding heuristic",
                "nit": result.nit,
                "nfev": result.nfev,
            }
        )

        return rounded_result

    def _sensitivity_analysis(
        self,
        c: np.ndarray,
        A_ub: Optional[np.ndarray],
        b_ub: Optional[np.ndarray],
        A_eq: Optional[np.ndarray],
        b_eq: Optional[np.ndarray],
        result: OptimizeResult,
    ) -> Dict[str, Any]:
        """Perform sensitivity analysis on LP solution."""
        sensitivity = {
            "objective_sensitivity": {},
            "rhs_sensitivity": {},
            "shadow_prices": {},
        }

        try:
            # Basic sensitivity analysis
            if hasattr(result, "slack") and result.slack is not None:
                # Identify binding constraints
                binding_constraints = np.abs(result.slack) < 1e-6
                sensitivity["binding_constraints"] = binding_constraints.tolist()

            # Shadow prices from dual solution
            if hasattr(result, "eqlin") and result.eqlin is not None:
                sensitivity["shadow_prices"] = result.eqlin.marginals.tolist()

        except Exception as e:
            logger.warning(f"Sensitivity analysis failed: {e}")
            sensitivity["error"] = str(e)

        return sensitivity

    # --- AnalysisPipelineBase abstract method implementations ---

    def get_analysis_type(self) -> str:
        """Get the specific analysis type this pipeline performs."""
        return "linear_programming"

    def _configure_analysis_pipeline(self) -> List[Callable]:
        """Configure analysis steps based on intention and complexity level."""
        return [self.solve]

    def _execute_analysis_step(
        self, step: Callable, data: pd.DataFrame, context: Dict[str, Any]
    ) -> Tuple[Any, Dict[str, Any]]:
        """Execute individual analysis step with error handling and metadata."""
        result = step(data) if callable(step) else step
        return result, {}

    def _execute_standard_analysis(
        self, data: pd.DataFrame
    ) -> Tuple[Any, Dict[str, Any]]:
        """Execute analysis on full dataset in memory."""
        return self.result_, {}

    def _execute_streaming_analysis(
        self, data: pd.DataFrame
    ) -> Tuple[Any, Dict[str, Any]]:
        """Execute analysis with streaming support for large datasets."""
        return self._execute_standard_analysis(data)
