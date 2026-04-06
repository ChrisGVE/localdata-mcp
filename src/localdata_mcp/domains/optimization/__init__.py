"""
Optimization & Operations Research Domain - Comprehensive optimization capabilities.

This package implements advanced optimization and operations research tools including linear programming,
constrained optimization, network analysis, and specialized applications using scipy.optimize and networkx.

Key Features:
- Linear Programming (simplex method, dual analysis, sensitivity analysis, integer programming)
- Constrained Optimization (scipy.optimize.minimize with constraints, multi-objective optimization)
- Network Optimization (shortest paths, max flow, MST, TSP heuristics, network flow)
- Specialized Applications (portfolio optimization, resource allocation, scheduling)
- Full sklearn pipeline compatibility
- Streaming-compatible processing
- Comprehensive result formatting
"""

from ._assignment import AssignmentSolver
from ._constrained import ConstrainedOptimizer
from ._linear_programming import LinearProgrammingSolver
from ._network import NETWORKX_AVAILABLE, NetworkAnalyzer
from ._tool_functions_lp import optimize_constrained, solve_linear_program
from ._tool_functions_network import analyze_network, solve_assignment_problem
from ._types import (
    AssignmentResult,
    ConstrainedOptResult,
    LinearProgramResult,
    NetworkAnalysisResult,
    OptimizationResult,
)

__all__ = [
    # Result classes
    "OptimizationResult",
    "LinearProgramResult",
    "ConstrainedOptResult",
    "NetworkAnalysisResult",
    "AssignmentResult",
    # Core transformers
    "LinearProgrammingSolver",
    "ConstrainedOptimizer",
    "NetworkAnalyzer",
    "AssignmentSolver",
    # High-level functions
    "solve_linear_program",
    "optimize_constrained",
    "analyze_network",
    "solve_assignment_problem",
    # Constants
    "NETWORKX_AVAILABLE",
]
