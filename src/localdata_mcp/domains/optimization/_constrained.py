"""
Constrained Optimization Solver - general constrained optimization with multi-objective support.
"""

import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.utils.validation import check_is_fitted
from scipy.optimize import minimize, differential_evolution, OptimizeResult

from ...logging_manager import get_logger
from ...pipeline.base import (
    AnalysisPipelineBase,
    StreamingConfig,
)
from ._types import ConstrainedOptResult

logger = get_logger(__name__)


class ConstrainedOptimizer(AnalysisPipelineBase):
    """
    General constrained optimization solver with support for various constraint types
    and multi-objective optimization.
    """

    def __init__(
        self,
        method: str = "SLSQP",
        constraints: Optional[List[Dict]] = None,
        bounds: Optional[List[Tuple[float, float]]] = None,
        options: Optional[Dict[str, Any]] = None,
        multi_objective: bool = False,
        streaming_config: Optional[StreamingConfig] = None,
    ):
        """
        Initialize Constrained Optimizer.

        Parameters:
        -----------
        method : str, default='SLSQP'
            Optimization method ('SLSQP', 'trust-constr', 'COBYLA', etc.)
        constraints : list of dict, optional
            Constraint definitions
        bounds : list of tuples, optional
            Variable bounds
        options : dict, optional
            Solver options
        multi_objective : bool, default=False
            Enable multi-objective optimization
        streaming_config : StreamingConfig, optional
            Streaming configuration
        """
        super().__init__(
            analytical_intention="constrained optimization",
            streaming_config=streaming_config,
        )
        self.method = method
        self.constraints = constraints or []
        self.bounds = bounds
        self.options = options or {}
        self.multi_objective = multi_objective

        self.result_ = None
        self.is_fitted_ = False

    def fit(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> "ConstrainedOptimizer":
        """Fit is not applicable for optimizer - use optimize method instead."""
        logger.warning("ConstrainedOptimizer.fit() called - use optimize() instead")
        self.is_fitted_ = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform is not applicable for optimizer."""
        check_is_fitted(self, "is_fitted_")
        return X

    def optimize(
        self,
        objective_func: Callable,
        x0: np.ndarray,
        constraints: Optional[List[Dict]] = None,
        bounds: Optional[List[Tuple[float, float]]] = None,
        jacobian: Optional[Callable] = None,
        hessian: Optional[Callable] = None,
    ) -> ConstrainedOptResult:
        """
        Solve constrained optimization problem.

        Parameters:
        -----------
        objective_func : callable
            Objective function to minimize
        x0 : array_like
            Initial guess
        constraints : list of dict, optional
            Constraint definitions (overrides instance constraints)
        bounds : list of tuples, optional
            Variable bounds (overrides instance bounds)
        jacobian : callable, optional
            Jacobian of objective function
        hessian : callable, optional
            Hessian of objective function
        """
        start_time = time.time()

        # Use provided constraints/bounds or instance defaults
        if constraints is None:
            constraints = self.constraints
        if bounds is None:
            bounds = self.bounds

        try:
            if self.multi_objective:
                result = self._multi_objective_optimization(
                    objective_func, x0, constraints, bounds
                )
            else:
                result = minimize(
                    objective_func,
                    x0,
                    method=self.method,
                    jac=jacobian,
                    hess=hessian,
                    bounds=bounds,
                    constraints=constraints,
                    options=self.options,
                )

        except Exception as e:
            logger.error(f"Constrained optimization failed: {e}")
            return ConstrainedOptResult(
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
        opt_result = ConstrainedOptResult(
            success=result.success,
            fun=result.fun,
            x=result.x,
            message=result.message,
            nit=getattr(result, "nit", 0),
            nfev=getattr(result, "nfev", 0),
            execution_time=execution_time,
            method=self.method,
            jac=getattr(result, "jac", None),
            hess=getattr(result, "hess", None),
        )

        # Add constraint analysis
        if constraints and result.success:
            opt_result.constraints = self._analyze_constraints(result.x, constraints)
            opt_result.constraint_violations = self._compute_constraint_violations(
                result.x, constraints
            )

        # Extract Lagrange multipliers if available
        if hasattr(result, "v") and result.v is not None:
            opt_result.lagrange_multipliers = result.v

        self.result_ = opt_result
        self.is_fitted_ = True

        logger.info(
            f"Constrained optimization completed in {execution_time:.3f}s, success: {result.success}"
        )
        return opt_result

    def _multi_objective_optimization(
        self,
        objective_func: Union[Callable, List[Callable]],
        x0: np.ndarray,
        constraints: Optional[List[Dict]],
        bounds: Optional[List[Tuple[float, float]]],
    ) -> OptimizeResult:
        """Handle multi-objective optimization using differential evolution."""
        if not isinstance(objective_func, list):
            # Single objective - use standard optimization
            return minimize(
                objective_func,
                x0,
                method=self.method,
                bounds=bounds,
                constraints=constraints,
                options=self.options,
            )

        # Multi-objective using weighted sum approach
        def combined_objective(x):
            # Equal weights for simplicity - could be parameterized
            weights = np.ones(len(objective_func)) / len(objective_func)
            return sum(w * f(x) for w, f in zip(weights, objective_func))

        # Use differential evolution for multi-objective
        result = differential_evolution(
            combined_objective, bounds=bounds, **self.options
        )

        return result

    def _analyze_constraints(
        self, x: np.ndarray, constraints: List[Dict]
    ) -> List[Dict[str, Any]]:
        """Analyze constraint satisfaction at solution."""
        constraint_analysis = []

        for constraint in constraints:
            analysis = {"type": constraint["type"], "active": False, "violation": 0.0}

            try:
                if constraint["type"] == "eq":
                    value = constraint["fun"](x)
                    analysis["value"] = value
                    analysis["violation"] = abs(value)
                    analysis["active"] = abs(value) < 1e-6
                elif constraint["type"] == "ineq":
                    value = constraint["fun"](x)
                    analysis["value"] = value
                    analysis["violation"] = max(0, -value)
                    analysis["active"] = abs(value) < 1e-6

            except Exception as e:
                analysis["error"] = str(e)

            constraint_analysis.append(analysis)

        return constraint_analysis

    def _compute_constraint_violations(
        self, x: np.ndarray, constraints: List[Dict]
    ) -> np.ndarray:
        """Compute constraint violations at solution."""
        violations = []

        for constraint in constraints:
            try:
                value = constraint["fun"](x)
                if constraint["type"] == "eq":
                    violations.append(abs(value))
                elif constraint["type"] == "ineq":
                    violations.append(max(0, -value))
            except Exception:
                violations.append(np.inf)

        return np.array(violations)

    # --- AnalysisPipelineBase abstract method implementations ---

    def get_analysis_type(self) -> str:
        """Get the specific analysis type this pipeline performs."""
        return "constrained_optimization"

    def _configure_analysis_pipeline(self) -> List[Callable]:
        """Configure analysis steps based on intention and complexity level."""
        return [self.optimize]

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
