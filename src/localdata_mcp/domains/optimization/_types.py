"""
Optimization domain result dataclasses.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


@dataclass
class OptimizationResult:
    """Base class for optimization results."""

    success: bool
    fun: float  # Objective function value
    x: np.ndarray  # Solution vector
    message: str
    nit: int  # Number of iterations
    nfev: int  # Number of function evaluations
    execution_time: float
    method: str
    additional_info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LinearProgramResult(OptimizationResult):
    """Results from linear programming optimization."""

    slack: Optional[np.ndarray] = None
    con: Optional[np.ndarray] = None  # Constraint residuals
    dual_values: Optional[np.ndarray] = None  # Dual solution
    sensitivity: Optional[Dict[str, Any]] = None
    phase_one_success: bool = True
    is_integer_solution: bool = False
    integrality_gap: Optional[float] = None


@dataclass
class ConstrainedOptResult(OptimizationResult):
    """Results from constrained optimization."""

    jac: Optional[np.ndarray] = None  # Jacobian
    hess: Optional[np.ndarray] = None  # Hessian
    constraints: Optional[List[Dict[str, Any]]] = None
    lagrange_multipliers: Optional[np.ndarray] = None
    constraint_violations: Optional[np.ndarray] = None
    trust_region_radius: Optional[float] = None


@dataclass
class NetworkAnalysisResult:
    """Results from network optimization analysis."""

    graph_properties: Dict[str, Any]
    shortest_paths: Optional[Dict[str, Dict[str, Union[float, List]]]] = None
    max_flow_value: Optional[float] = None
    max_flow_dict: Optional[Dict] = None
    minimum_spanning_tree: Optional[Dict[str, Any]] = None
    tsp_solution: Optional[Dict[str, Any]] = None
    centrality_measures: Optional[Dict[str, Dict]] = None
    execution_time: float = 0.0
    method: str = ""


@dataclass
class AssignmentResult(OptimizationResult):
    """Results from assignment/scheduling problems."""

    assignment_matrix: Optional[np.ndarray] = None
    assignment_pairs: Optional[List[Tuple[int, int]]] = None
    total_cost: Optional[float] = None
    unassigned_agents: List[int] = field(default_factory=list)
    unassigned_tasks: List[int] = field(default_factory=list)
    is_perfect_matching: bool = True
