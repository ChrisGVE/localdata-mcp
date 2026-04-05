"""
Assignment Solver - assignment and scheduling problems using optimization techniques.
"""

import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.utils.validation import check_is_fitted
from scipy.optimize import linprog

from ...logging_manager import get_logger
from ...pipeline.base import (
    AnalysisPipelineBase,
    StreamingConfig,
)
from ._types import AssignmentResult

logger = get_logger(__name__)


class AssignmentSolver(AnalysisPipelineBase):
    """
    Assignment and scheduling problem solver using various optimization techniques.

    Solves assignment problems, resource allocation, and basic scheduling using
    linear programming and specialized algorithms.
    """

    def __init__(
        self,
        method: str = "hungarian",
        maximize: bool = False,
        allow_partial: bool = False,
        streaming_config: Optional[StreamingConfig] = None,
    ):
        """
        Initialize Assignment Solver.

        Parameters:
        -----------
        method : str, default='hungarian'
            Solution method ('hungarian', 'linear_programming', 'greedy')
        maximize : bool, default=False
            Whether to maximize instead of minimize
        allow_partial : bool, default=False
            Allow partial assignments
        streaming_config : StreamingConfig, optional
            Streaming configuration
        """
        super().__init__(
            analytical_intention="assignment problem solving",
            streaming_config=streaming_config,
        )
        self.method = method
        self.maximize = maximize
        self.allow_partial = allow_partial

        self.problem_type_ = None
        self.result_ = None
        self.is_fitted_ = False

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "AssignmentSolver":
        """Fit is not applicable for assignment solver."""
        logger.warning("AssignmentSolver.fit() called - use solve_assignment() instead")
        self.is_fitted_ = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform is not applicable for assignment solver."""
        check_is_fitted(self, "is_fitted_")
        return X

    def solve_assignment(self, cost_matrix: np.ndarray) -> AssignmentResult:
        """
        Solve assignment problem with given cost matrix.

        Parameters:
        -----------
        cost_matrix : array_like
            Cost matrix where cost_matrix[i, j] is cost of assigning agent i to task j

        Returns:
        --------
        AssignmentResult
            Solution with assignment pairs and total cost
        """
        start_time = time.time()
        cost_matrix = np.array(cost_matrix)

        if cost_matrix.ndim != 2:
            raise ValueError("Cost matrix must be 2-dimensional")

        try:
            if self.method == "hungarian":
                result = self._solve_hungarian(cost_matrix)
            elif self.method == "linear_programming":
                result = self._solve_lp_assignment(cost_matrix)
            elif self.method == "greedy":
                result = self._solve_greedy_assignment(cost_matrix)
            else:
                raise ValueError(f"Unknown method: {self.method}")

        except Exception as e:
            logger.error(f"Assignment problem failed: {e}")
            return AssignmentResult(
                success=False,
                fun=np.inf,
                x=np.array([]),
                message=str(e),
                nit=0,
                nfev=0,
                execution_time=time.time() - start_time,
                method=self.method,
                assignment_matrix=np.array([]),
                assignment_pairs=[],
                total_cost=np.inf,
            )

        execution_time = time.time() - start_time
        result.execution_time = execution_time
        result.method = self.method

        self.result_ = result
        self.is_fitted_ = True

        logger.info(
            f"Assignment problem solved in {execution_time:.3f}s, cost: {result.total_cost}"
        )
        return result

    def _solve_hungarian(self, cost_matrix: np.ndarray) -> AssignmentResult:
        """Solve using Hungarian algorithm (via scipy)."""
        try:
            from scipy.optimize import linear_sum_assignment

            # Handle maximize option
            if self.maximize:
                cost_matrix = -cost_matrix

            # Solve assignment
            agent_indices, task_indices = linear_sum_assignment(cost_matrix)

            # Create assignment matrix
            assignment_matrix = np.zeros_like(cost_matrix)
            assignment_matrix[agent_indices, task_indices] = 1

            # Calculate total cost
            if self.maximize:
                total_cost = -cost_matrix[agent_indices, task_indices].sum()
            else:
                total_cost = cost_matrix[agent_indices, task_indices].sum()

            # Create assignment pairs
            assignment_pairs = list(zip(agent_indices.tolist(), task_indices.tolist()))

            # Find unassigned agents/tasks
            n_agents, n_tasks = cost_matrix.shape
            unassigned_agents = [i for i in range(n_agents) if i not in agent_indices]
            unassigned_tasks = [j for j in range(n_tasks) if j not in task_indices]

            return AssignmentResult(
                success=True,
                fun=total_cost,
                x=assignment_matrix.flatten(),
                message="Optimal assignment found",
                nit=1,
                nfev=1,
                execution_time=0.0,
                method=self.method,
                assignment_matrix=assignment_matrix,
                assignment_pairs=assignment_pairs,
                total_cost=total_cost,
                unassigned_agents=unassigned_agents,
                unassigned_tasks=unassigned_tasks,
                is_perfect_matching=len(unassigned_agents) == 0
                and len(unassigned_tasks) == 0,
            )

        except ImportError:
            # Fallback to greedy if scipy not available
            return self._solve_greedy_assignment(cost_matrix)

    def _solve_lp_assignment(self, cost_matrix: np.ndarray) -> AssignmentResult:
        """Solve as linear programming problem."""
        n_agents, n_tasks = cost_matrix.shape

        # Create decision variables: x[i,j] = 1 if agent i assigned to task j
        c = cost_matrix.flatten()
        if self.maximize:
            c = -c

        # Equality constraints: each agent assigned to exactly one task
        A_eq_agents = []
        b_eq_agents = []

        for i in range(n_agents):
            constraint = np.zeros(n_agents * n_tasks)
            for j in range(n_tasks):
                constraint[i * n_tasks + j] = 1
            A_eq_agents.append(constraint)
            b_eq_agents.append(1)

        # Equality constraints: each task assigned to exactly one agent
        A_eq_tasks = []
        b_eq_tasks = []

        for j in range(n_tasks):
            constraint = np.zeros(n_agents * n_tasks)
            for i in range(n_agents):
                constraint[i * n_tasks + j] = 1
            A_eq_tasks.append(constraint)
            b_eq_tasks.append(1)

        # Combine constraints
        A_eq = np.vstack([A_eq_agents, A_eq_tasks])
        b_eq = np.array(b_eq_agents + b_eq_tasks)

        # Variable bounds (binary variables relaxed to [0, 1])
        bounds = [(0, 1) for _ in range(n_agents * n_tasks)]

        # Solve LP
        result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")

        if not result.success:
            raise RuntimeError(f"LP solver failed: {result.message}")

        # Extract assignment from solution
        x_matrix = result.x.reshape((n_agents, n_tasks))

        # Find assignments (values close to 1)
        assignment_pairs = []
        assignment_matrix = np.zeros_like(cost_matrix)

        for i in range(n_agents):
            for j in range(n_tasks):
                if x_matrix[i, j] > 0.5:  # Threshold for binary assignment
                    assignment_pairs.append((i, j))
                    assignment_matrix[i, j] = 1

        # Calculate total cost
        total_cost = sum(cost_matrix[i, j] for i, j in assignment_pairs)
        if self.maximize:
            total_cost = -result.fun
        else:
            total_cost = result.fun

        # Find unassigned
        assigned_agents = {i for i, j in assignment_pairs}
        assigned_tasks = {j for i, j in assignment_pairs}
        unassigned_agents = [i for i in range(n_agents) if i not in assigned_agents]
        unassigned_tasks = [j for j in range(n_tasks) if j not in assigned_tasks]

        return AssignmentResult(
            success=True,
            fun=total_cost,
            x=result.x,
            message="Assignment solved via LP",
            nit=getattr(result, "nit", 0),
            nfev=getattr(result, "nfev", 0),
            execution_time=0.0,
            method=self.method,
            assignment_matrix=assignment_matrix,
            assignment_pairs=assignment_pairs,
            total_cost=total_cost,
            unassigned_agents=unassigned_agents,
            unassigned_tasks=unassigned_tasks,
            is_perfect_matching=len(unassigned_agents) == 0
            and len(unassigned_tasks) == 0,
        )

    def _solve_greedy_assignment(self, cost_matrix: np.ndarray) -> AssignmentResult:
        """Solve using greedy heuristic."""
        n_agents, n_tasks = cost_matrix.shape

        # Create list of all possible assignments with costs
        assignments = []
        for i in range(n_agents):
            for j in range(n_tasks):
                cost = cost_matrix[i, j]
                if self.maximize:
                    cost = -cost
                assignments.append((cost, i, j))

        # Sort by cost (ascending for minimization)
        assignments.sort()

        # Greedily select assignments
        selected_assignments = []
        assigned_agents = set()
        assigned_tasks = set()

        for cost, agent, task in assignments:
            if agent not in assigned_agents and task not in assigned_tasks:
                selected_assignments.append((agent, task))
                assigned_agents.add(agent)
                assigned_tasks.add(task)

        # Create assignment matrix
        assignment_matrix = np.zeros_like(cost_matrix)
        for agent, task in selected_assignments:
            assignment_matrix[agent, task] = 1

        # Calculate total cost
        total_cost = sum(
            cost_matrix[agent, task] for agent, task in selected_assignments
        )

        # Find unassigned
        unassigned_agents = [i for i in range(n_agents) if i not in assigned_agents]
        unassigned_tasks = [j for j in range(n_tasks) if j not in assigned_tasks]

        return AssignmentResult(
            success=True,
            fun=total_cost,
            x=assignment_matrix.flatten(),
            message="Greedy assignment completed",
            nit=1,
            nfev=len(assignments),
            execution_time=0.0,
            method=self.method,
            assignment_matrix=assignment_matrix,
            assignment_pairs=selected_assignments,
            total_cost=total_cost,
            unassigned_agents=unassigned_agents,
            unassigned_tasks=unassigned_tasks,
            is_perfect_matching=len(unassigned_agents) == 0
            and len(unassigned_tasks) == 0,
        )

    # --- AnalysisPipelineBase abstract method implementations ---

    def get_analysis_type(self) -> str:
        """Get the specific analysis type this pipeline performs."""
        return "assignment_solving"

    def _configure_analysis_pipeline(self) -> List[Callable]:
        """Configure analysis steps based on intention and complexity level."""
        return [self.solve_assignment]

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
