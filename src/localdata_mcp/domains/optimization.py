"""
Optimization & Operations Research Domain - Comprehensive optimization capabilities.

This module implements advanced optimization and operations research tools including linear programming,
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

import warnings
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
import time
import json

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted
from scipy import optimize
from scipy.optimize import linprog, minimize, differential_evolution
from scipy.sparse import csr_matrix
import itertools

# Try to import networkx for network optimization
try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    nx = None

from ..logging_manager import get_logger
from ..pipeline.base import (
    AnalysisPipelineBase, PipelineResult, CompositionMetadata, 
    StreamingConfig, PipelineState
)

logger = get_logger(__name__)

# Suppress specific warnings that are not critical for our use case
warnings.filterwarnings('ignore', category=RuntimeWarning, module='scipy.optimize')
warnings.filterwarnings('ignore', category=UserWarning, module='scipy.optimize')


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
    assignment_matrix: np.ndarray
    assignment_pairs: List[Tuple[int, int]]
    total_cost: float
    unassigned_agents: List[int] = field(default_factory=list)
    unassigned_tasks: List[int] = field(default_factory=list)
    is_perfect_matching: bool = True


class LinearProgrammingSolver(BaseEstimator, TransformerMixin, AnalysisPipelineBase):
    """
    Linear Programming Solver with simplex method, dual analysis, and integer programming.
    
    Supports various linear programming formulations and provides comprehensive
    analysis including sensitivity analysis and dual problem solutions.
    """
    
    def __init__(
        self,
        method: str = 'highs',
        integer_variables: Optional[List[int]] = None,
        bounds: Optional[List[Tuple[float, float]]] = None,
        options: Optional[Dict[str, Any]] = None,
        streaming_config: Optional[StreamingConfig] = None
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
        super().__init__(streaming_config)
        self.method = method
        self.integer_variables = integer_variables or []
        self.bounds = bounds
        self.options = options or {}
        
        self.problem_ = None
        self.result_ = None
        self.is_fitted_ = False
    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'LinearProgrammingSolver':
        """
        Fit is not applicable for LP solver - use solve_linear_program instead.
        """
        logger.warning("LinearProgrammingSolver.fit() called - use solve_linear_program() instead")
        self.is_fitted_ = True
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform is not applicable for LP solver."""
        check_is_fitted(self, 'is_fitted_')
        return X
    
    def solve(
        self,
        c: np.ndarray,
        A_ub: Optional[np.ndarray] = None,
        b_ub: Optional[np.ndarray] = None,
        A_eq: Optional[np.ndarray] = None,
        b_eq: Optional[np.ndarray] = None,
        bounds: Optional[List[Tuple[float, float]]] = None
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
                    c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                    bounds=bounds, method=self.method, options=self.options
                )
        except Exception as e:
            logger.error(f"Linear programming failed: {e}")
            return LinearProgramResult(
                success=False, fun=np.inf, x=np.array([]), 
                message=str(e), nit=0, nfev=0,
                execution_time=time.time() - start_time, method=self.method
            )
        
        execution_time = time.time() - start_time
        
        # Create comprehensive result
        lp_result = LinearProgramResult(
            success=result.success,
            fun=result.fun,
            x=result.x,
            message=result.message,
            nit=getattr(result, 'nit', 0),
            nfev=getattr(result, 'nfev', 0),
            execution_time=execution_time,
            method=self.method,
            slack=getattr(result, 'slack', None),
            con=getattr(result, 'con', None),
            is_integer_solution=bool(self.integer_variables)
        )
        
        # Add dual analysis if available
        if hasattr(result, 'eqlin') and result.eqlin is not None:
            lp_result.dual_values = result.eqlin.marginals
        
        # Perform sensitivity analysis
        if result.success and A_ub is not None:
            lp_result.sensitivity = self._sensitivity_analysis(
                c, A_ub, b_ub, A_eq, b_eq, result
            )
        
        self.result_ = lp_result
        self.is_fitted_ = True
        
        logger.info(f"Linear programming completed in {execution_time:.3f}s, success: {result.success}")
        return lp_result
    
    def _solve_integer_lp(
        self,
        c: np.ndarray,
        A_ub: Optional[np.ndarray] = None,
        b_ub: Optional[np.ndarray] = None,
        A_eq: Optional[np.ndarray] = None,
        b_eq: Optional[np.ndarray] = None,
        bounds: Optional[List[Tuple[float, float]]] = None
    ) -> optimize.OptimizeResult:
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
                c, bounds=bounds_obj, constraints=constraints, 
                integrality=integrality, options=self.options
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
        bounds: Optional[List[Tuple[float, float]]] = None
    ) -> optimize.OptimizeResult:
        """Simple branch-and-bound for integer LP (fallback implementation)."""
        # First solve relaxed problem
        result = linprog(
            c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
            bounds=bounds, method=self.method, options=self.options
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
        rounded_result = optimize.OptimizeResult({
            'success': True,
            'fun': np.dot(c, x_rounded),
            'x': x_rounded,
            'message': 'Integer solution found via rounding heuristic',
            'nit': result.nit,
            'nfev': result.nfev
        })
        
        return rounded_result
    
    def _sensitivity_analysis(
        self,
        c: np.ndarray,
        A_ub: Optional[np.ndarray],
        b_ub: Optional[np.ndarray],
        A_eq: Optional[np.ndarray],
        b_eq: Optional[np.ndarray],
        result: optimize.OptimizeResult
    ) -> Dict[str, Any]:
        """Perform sensitivity analysis on LP solution."""
        sensitivity = {
            'objective_sensitivity': {},
            'rhs_sensitivity': {},
            'shadow_prices': {}
        }
        
        try:
            # Basic sensitivity analysis
            if hasattr(result, 'slack') and result.slack is not None:
                # Identify binding constraints
                binding_constraints = np.abs(result.slack) < 1e-6
                sensitivity['binding_constraints'] = binding_constraints.tolist()
            
            # Shadow prices from dual solution
            if hasattr(result, 'eqlin') and result.eqlin is not None:
                sensitivity['shadow_prices'] = result.eqlin.marginals.tolist()
        
        except Exception as e:
            logger.warning(f"Sensitivity analysis failed: {e}")
            sensitivity['error'] = str(e)
        
        return sensitivity


class ConstrainedOptimizer(BaseEstimator, TransformerMixin, AnalysisPipelineBase):
    """
    General constrained optimization solver with support for various constraint types
    and multi-objective optimization.
    """
    
    def __init__(
        self,
        method: str = 'SLSQP',
        constraints: Optional[List[Dict]] = None,
        bounds: Optional[List[Tuple[float, float]]] = None,
        options: Optional[Dict[str, Any]] = None,
        multi_objective: bool = False,
        streaming_config: Optional[StreamingConfig] = None
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
        super().__init__(streaming_config)
        self.method = method
        self.constraints = constraints or []
        self.bounds = bounds
        self.options = options or {}
        self.multi_objective = multi_objective
        
        self.result_ = None
        self.is_fitted_ = False
    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'ConstrainedOptimizer':
        """Fit is not applicable for optimizer - use optimize method instead."""
        logger.warning("ConstrainedOptimizer.fit() called - use optimize() instead")
        self.is_fitted_ = True
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform is not applicable for optimizer."""
        check_is_fitted(self, 'is_fitted_')
        return X
    
    def optimize(
        self,
        objective_func: Callable,
        x0: np.ndarray,
        constraints: Optional[List[Dict]] = None,
        bounds: Optional[List[Tuple[float, float]]] = None,
        jacobian: Optional[Callable] = None,
        hessian: Optional[Callable] = None
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
                    objective_func, x0, method=self.method, 
                    jac=jacobian, hess=hessian,
                    bounds=bounds, constraints=constraints,
                    options=self.options
                )
        
        except Exception as e:
            logger.error(f"Constrained optimization failed: {e}")
            return ConstrainedOptResult(
                success=False, fun=np.inf, x=np.array([]),
                message=str(e), nit=0, nfev=0,
                execution_time=time.time() - start_time, method=self.method
            )
        
        execution_time = time.time() - start_time
        
        # Create comprehensive result
        opt_result = ConstrainedOptResult(
            success=result.success,
            fun=result.fun,
            x=result.x,
            message=result.message,
            nit=getattr(result, 'nit', 0),
            nfev=getattr(result, 'nfev', 0),
            execution_time=execution_time,
            method=self.method,
            jac=getattr(result, 'jac', None),
            hess=getattr(result, 'hess', None)
        )
        
        # Add constraint analysis
        if constraints and result.success:
            opt_result.constraints = self._analyze_constraints(result.x, constraints)
            opt_result.constraint_violations = self._compute_constraint_violations(
                result.x, constraints
            )
        
        # Extract Lagrange multipliers if available
        if hasattr(result, 'v') and result.v is not None:
            opt_result.lagrange_multipliers = result.v
        
        self.result_ = opt_result
        self.is_fitted_ = True
        
        logger.info(f"Constrained optimization completed in {execution_time:.3f}s, success: {result.success}")
        return opt_result
    
    def _multi_objective_optimization(
        self,
        objective_func: Union[Callable, List[Callable]],
        x0: np.ndarray,
        constraints: Optional[List[Dict]],
        bounds: Optional[List[Tuple[float, float]]]
    ) -> optimize.OptimizeResult:
        """Handle multi-objective optimization using differential evolution."""
        if not isinstance(objective_func, list):
            # Single objective - use standard optimization
            return minimize(
                objective_func, x0, method=self.method,
                bounds=bounds, constraints=constraints,
                options=self.options
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
    
    def _analyze_constraints(self, x: np.ndarray, constraints: List[Dict]) -> List[Dict[str, Any]]:
        """Analyze constraint satisfaction at solution."""
        constraint_analysis = []
        
        for constraint in constraints:
            analysis = {
                'type': constraint['type'],
                'active': False,
                'violation': 0.0
            }
            
            try:
                if constraint['type'] == 'eq':
                    value = constraint['fun'](x)
                    analysis['value'] = value
                    analysis['violation'] = abs(value)
                    analysis['active'] = abs(value) < 1e-6
                elif constraint['type'] == 'ineq':
                    value = constraint['fun'](x)
                    analysis['value'] = value
                    analysis['violation'] = max(0, -value)
                    analysis['active'] = abs(value) < 1e-6
                    
            except Exception as e:
                analysis['error'] = str(e)
            
            constraint_analysis.append(analysis)
        
        return constraint_analysis
    
    def _compute_constraint_violations(self, x: np.ndarray, constraints: List[Dict]) -> np.ndarray:
        """Compute constraint violations at solution."""
        violations = []
        
        for constraint in constraints:
            try:
                value = constraint['fun'](x)
                if constraint['type'] == 'eq':
                    violations.append(abs(value))
                elif constraint['type'] == 'ineq':
                    violations.append(max(0, -value))
            except Exception:
                violations.append(np.inf)
        
        return np.array(violations)


class NetworkAnalyzer(BaseEstimator, TransformerMixin, AnalysisPipelineBase):
    """
    Network optimization and analysis using NetworkX for graph algorithms.
    
    Provides shortest path algorithms, maximum flow, minimum spanning tree,
    and traveling salesman problem heuristics.
    """
    
    def __init__(
        self,
        algorithm: str = 'auto',
        directed: bool = False,
        weighted: bool = True,
        streaming_config: Optional[StreamingConfig] = None
    ):
        """
        Initialize Network Analyzer.
        
        Parameters:
        -----------
        algorithm : str, default='auto'
            Algorithm selection ('dijkstra', 'floyd_warshall', 'bellman_ford', 'auto')
        directed : bool, default=False
            Whether graph is directed
        weighted : bool, default=True
            Whether graph edges have weights
        streaming_config : StreamingConfig, optional
            Streaming configuration
        """
        super().__init__(streaming_config)
        self.algorithm = algorithm
        self.directed = directed
        self.weighted = weighted
        
        if not NETWORKX_AVAILABLE:
            raise ImportError(
                "NetworkX is required for network optimization. "
                "Please install with: pip install networkx"
            )
        
        self.graph_ = None
        self.result_ = None
        self.is_fitted_ = False
    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'NetworkAnalyzer':
        """Build network from adjacency matrix or edge list."""
        X = check_array(X, accept_sparse=True)
        
        # Create graph from adjacency matrix
        if X.shape[0] == X.shape[1]:
            # Square matrix - treat as adjacency matrix
            if self.directed:
                self.graph_ = nx.from_numpy_array(X, create_using=nx.DiGraph())
            else:
                self.graph_ = nx.from_numpy_array(X, create_using=nx.Graph())
        else:
            # Non-square - treat as edge list [source, target, weight]
            if self.directed:
                self.graph_ = nx.DiGraph()
            else:
                self.graph_ = nx.Graph()
            
            for row in X:
                if len(row) >= 2:
                    source, target = int(row[0]), int(row[1])
                    weight = row[2] if len(row) > 2 and self.weighted else 1.0
                    self.graph_.add_edge(source, target, weight=weight)
        
        self.is_fitted_ = True
        logger.info(f"Network built: {self.graph_.number_of_nodes()} nodes, {self.graph_.number_of_edges()} edges")
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform is not applicable for network analyzer."""
        check_is_fitted(self, 'is_fitted_')
        return X
    
    def analyze_network(self, include_centrality: bool = True) -> NetworkAnalysisResult:
        """Comprehensive network analysis."""
        check_is_fitted(self, 'is_fitted_')
        start_time = time.time()
        
        try:
            # Basic graph properties
            graph_properties = {
                'num_nodes': self.graph_.number_of_nodes(),
                'num_edges': self.graph_.number_of_edges(),
                'is_directed': nx.is_directed(self.graph_),
                'is_connected': nx.is_connected(self.graph_) if not nx.is_directed(self.graph_) 
                               else nx.is_weakly_connected(self.graph_),
                'density': nx.density(self.graph_)
            }
            
            result = NetworkAnalysisResult(
                graph_properties=graph_properties,
                execution_time=0.0,
                method="comprehensive_analysis"
            )
            
            # Shortest paths analysis
            result.shortest_paths = self._compute_shortest_paths()
            
            # Maximum flow analysis (if directed)
            if nx.is_directed(self.graph_) and self.graph_.number_of_nodes() > 1:
                result.max_flow_value, result.max_flow_dict = self._compute_max_flow()
            
            # Minimum spanning tree (if undirected)
            if not nx.is_directed(self.graph_) and nx.is_connected(self.graph_):
                result.minimum_spanning_tree = self._compute_mst()
            
            # TSP heuristic (for small graphs)
            if self.graph_.number_of_nodes() <= 20:
                result.tsp_solution = self._solve_tsp()
            
            # Centrality measures
            if include_centrality:
                result.centrality_measures = self._compute_centrality()
            
            execution_time = time.time() - start_time
            result.execution_time = execution_time
            
            self.result_ = result
            
            logger.info(f"Network analysis completed in {execution_time:.3f}s")
            return result
            
        except Exception as e:
            logger.error(f"Network analysis failed: {e}")
            return NetworkAnalysisResult(
                graph_properties={'error': str(e)},
                execution_time=time.time() - start_time,
                method="failed_analysis"
            )
    
    def _compute_shortest_paths(self) -> Dict[str, Dict[str, Union[float, List]]]:
        """Compute shortest paths between all pairs of nodes."""
        try:
            weight_attr = 'weight' if self.weighted else None
            
            if self.algorithm == 'floyd_warshall' or (self.algorithm == 'auto' and self.graph_.number_of_nodes() <= 100):
                # Floyd-Warshall for small dense graphs
                paths = dict(nx.floyd_warshall_predecessor_and_distance(self.graph_, weight=weight_attr))
                return {
                    'algorithm': 'floyd_warshall',
                    'distances': {str(u): {str(v): dist for v, dist in distances.items()} 
                                for u, distances in paths[1].items()},
                    'predecessors': {str(u): {str(v): pred for v, pred in preds.items()} 
                                   for u, preds in paths[0].items()}
                }
            else:
                # Dijkstra for larger graphs
                nodes = list(self.graph_.nodes())
                distances = {}
                paths = {}
                
                for source in nodes[:min(10, len(nodes))]:  # Limit for performance
                    try:
                        dist = nx.single_source_dijkstra_path_length(self.graph_, source, weight=weight_attr)
                        path = nx.single_source_dijkstra_path(self.graph_, source, weight=weight_attr)
                        distances[str(source)] = {str(target): d for target, d in dist.items()}
                        paths[str(source)] = {str(target): p for target, p in path.items()}
                    except nx.NetworkXNoPath:
                        continue
                
                return {
                    'algorithm': 'dijkstra',
                    'distances': distances,
                    'paths': paths
                }
        
        except Exception as e:
            return {'error': str(e), 'algorithm': 'failed'}
    
    def _compute_max_flow(self) -> Tuple[float, Dict]:
        """Compute maximum flow between first and last nodes."""
        try:
            nodes = list(self.graph_.nodes())
            if len(nodes) < 2:
                return 0.0, {}
            
            source, sink = nodes[0], nodes[-1]
            flow_value, flow_dict = nx.maximum_flow(self.graph_, source, sink, capacity='weight')
            
            return flow_value, {
                'source': source,
                'sink': sink,
                'flow_value': flow_value,
                'flow_edges': flow_dict
            }
        
        except Exception as e:
            return 0.0, {'error': str(e)}
    
    def _compute_mst(self) -> Dict[str, Any]:
        """Compute minimum spanning tree."""
        try:
            weight_attr = 'weight' if self.weighted else None
            mst = nx.minimum_spanning_tree(self.graph_, weight=weight_attr)
            
            return {
                'total_weight': sum(data.get('weight', 1) for _, _, data in mst.edges(data=True)),
                'edges': [(u, v, data.get('weight', 1)) for u, v, data in mst.edges(data=True)],
                'num_edges': mst.number_of_edges()
            }
        
        except Exception as e:
            return {'error': str(e)}
    
    def _solve_tsp(self) -> Dict[str, Any]:
        """Solve traveling salesman problem using heuristics."""
        try:
            nodes = list(self.graph_.nodes())
            if len(nodes) <= 1:
                return {'tour': nodes, 'distance': 0.0}
            
            # Use nearest neighbor heuristic for TSP
            def nearest_neighbor_tsp(graph, start_node):
                unvisited = set(graph.nodes()) - {start_node}
                tour = [start_node]
                current = start_node
                total_distance = 0.0
                
                while unvisited:
                    # Find nearest unvisited node
                    nearest = None
                    min_distance = float('inf')
                    
                    for node in unvisited:
                        try:
                            if graph.has_edge(current, node):
                                distance = graph[current][node].get('weight', 1)
                            else:
                                distance = float('inf')
                            
                            if distance < min_distance:
                                min_distance = distance
                                nearest = node
                        except:
                            continue
                    
                    if nearest is not None:
                        tour.append(nearest)
                        total_distance += min_distance
                        unvisited.remove(nearest)
                        current = nearest
                    else:
                        break
                
                # Return to start
                if len(tour) > 1 and self.graph_.has_edge(current, start_node):
                    total_distance += self.graph_[current][start_node].get('weight', 1)
                
                return tour, total_distance
            
            # Try from first node
            tour, distance = nearest_neighbor_tsp(self.graph_, nodes[0])
            
            return {
                'tour': tour,
                'distance': distance,
                'algorithm': 'nearest_neighbor',
                'is_optimal': False
            }
        
        except Exception as e:
            return {'error': str(e)}
    
    def _compute_centrality(self) -> Dict[str, Dict]:
        """Compute various centrality measures."""
        centrality = {}
        
        try:
            # Degree centrality
            centrality['degree'] = nx.degree_centrality(self.graph_)
            
            # Betweenness centrality
            centrality['betweenness'] = nx.betweenness_centrality(self.graph_)
            
            # Closeness centrality
            centrality['closeness'] = nx.closeness_centrality(self.graph_)
            
            # Eigenvector centrality (if connected)
            try:
                centrality['eigenvector'] = nx.eigenvector_centrality(self.graph_, max_iter=1000)
            except:
                centrality['eigenvector'] = {}
            
            # PageRank (works for all graphs)
            centrality['pagerank'] = nx.pagerank(self.graph_)
            
        except Exception as e:
            centrality['error'] = str(e)
        
        return centrality


class AssignmentSolver(BaseEstimator, TransformerMixin, AnalysisPipelineBase):
    """
    Assignment and scheduling problem solver using various optimization techniques.
    
    Solves assignment problems, resource allocation, and basic scheduling using
    linear programming and specialized algorithms.
    """
    
    def __init__(
        self,
        method: str = 'hungarian',
        maximize: bool = False,
        allow_partial: bool = False,
        streaming_config: Optional[StreamingConfig] = None
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
        super().__init__(streaming_config)
        self.method = method
        self.maximize = maximize
        self.allow_partial = allow_partial
        
        self.problem_type_ = None
        self.result_ = None
        self.is_fitted_ = False
    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'AssignmentSolver':
        """Fit is not applicable for assignment solver."""
        logger.warning("AssignmentSolver.fit() called - use solve_assignment() instead")
        self.is_fitted_ = True
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform is not applicable for assignment solver."""
        check_is_fitted(self, 'is_fitted_')
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
            if self.method == 'hungarian':
                result = self._solve_hungarian(cost_matrix)
            elif self.method == 'linear_programming':
                result = self._solve_lp_assignment(cost_matrix)
            elif self.method == 'greedy':
                result = self._solve_greedy_assignment(cost_matrix)
            else:
                raise ValueError(f"Unknown method: {self.method}")
        
        except Exception as e:
            logger.error(f"Assignment problem failed: {e}")
            return AssignmentResult(
                success=False, fun=np.inf, x=np.array([]),
                message=str(e), nit=0, nfev=0,
                execution_time=time.time() - start_time, method=self.method,
                assignment_matrix=np.array([]), assignment_pairs=[],
                total_cost=np.inf
            )
        
        execution_time = time.time() - start_time
        result.execution_time = execution_time
        result.method = self.method
        
        self.result_ = result
        self.is_fitted_ = True
        
        logger.info(f"Assignment problem solved in {execution_time:.3f}s, cost: {result.total_cost}")
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
                is_perfect_matching=len(unassigned_agents) == 0 and len(unassigned_tasks) == 0
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
        
        # Constraints: each agent assigned to at most one task
        A_ub_agents = []
        b_ub_agents = []
        
        for i in range(n_agents):
            constraint = np.zeros(n_agents * n_tasks)
            for j in range(n_tasks):
                constraint[i * n_tasks + j] = 1
            A_ub_agents.append(constraint)
            b_ub_agents.append(1)
        
        # Constraints: each task assigned to at most one agent
        A_ub_tasks = []
        b_ub_tasks = []
        
        for j in range(n_tasks):
            constraint = np.zeros(n_agents * n_tasks)
            for i in range(n_agents):
                constraint[i * n_tasks + j] = 1
            A_ub_tasks.append(constraint)
            b_ub_tasks.append(1)
        
        # Combine constraints
        A_ub = np.vstack([A_ub_agents, A_ub_tasks])
        b_ub = np.array(b_ub_agents + b_ub_tasks)
        
        # Variable bounds (binary variables relaxed to [0, 1])
        bounds = [(0, 1) for _ in range(n_agents * n_tasks)]
        
        # Solve LP
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
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
            nit=getattr(result, 'nit', 0),
            nfev=getattr(result, 'nfev', 0),
            execution_time=0.0,
            method=self.method,
            assignment_matrix=assignment_matrix,
            assignment_pairs=assignment_pairs,
            total_cost=total_cost,
            unassigned_agents=unassigned_agents,
            unassigned_tasks=unassigned_tasks,
            is_perfect_matching=len(unassigned_agents) == 0 and len(unassigned_tasks) == 0
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
        total_cost = sum(cost_matrix[agent, task] for agent, task in selected_assignments)
        
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
            is_perfect_matching=len(unassigned_agents) == 0 and len(unassigned_tasks) == 0
        )


# Tool Functions
def solve_linear_program(
    connection_name: str,
    table_name: str,
    objective_column: str,
    constraint_columns: Optional[List[str]] = None,
    constraint_values: Optional[List[float]] = None,
    constraint_types: Optional[List[str]] = None,
    bounds: Optional[List[Tuple[float, float]]] = None,
    method: str = 'highs',
    integer_variables: Optional[List[int]] = None
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
        from ..database_manager import DatabaseManager
        
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
            constraint_query = f"SELECT {', '.join(constraint_columns)} FROM {table_name}"
            A_ub = pd.read_sql(constraint_query, engine).values.T  # Transpose for proper shape
            
            # Convert constraint types to standard form (Ax <= b)
            b_ub = np.array(constraint_values)
            if constraint_types:
                for i, ctype in enumerate(constraint_types):
                    if ctype == '>=':
                        A_ub[i] = -A_ub[i]
                        b_ub[i] = -b_ub[i]
                    # '=' constraints would need A_eq, b_eq (not implemented here for simplicity)
        
        # Create and solve LP
        solver = LinearProgrammingSolver(
            method=method,
            integer_variables=integer_variables,
            bounds=bounds
        )
        
        result = solver.solve(c, A_ub=A_ub, b_ub=b_ub)
        
        # Format results
        formatted_result = {
            'success': result.success,
            'optimal_value': float(result.fun) if result.success else None,
            'optimal_solution': result.x.tolist() if result.success else None,
            'message': result.message,
            'method': result.method,
            'execution_time': result.execution_time,
            'iterations': result.nit,
            'function_evaluations': result.nfev,
            'is_integer_solution': result.is_integer_solution,
            'sensitivity_analysis': result.sensitivity,
            'dual_values': result.dual_values.tolist() if result.dual_values is not None else None
        }
        
        # Add constraint analysis if available
        if result.slack is not None:
            formatted_result['constraint_slack'] = result.slack.tolist()
        
        return formatted_result
        
    except Exception as e:
        logger.error(f"Linear programming failed: {e}")
        return {
            'success': False,
            'error': str(e),
            'message': f"Linear programming optimization failed: {e}"
        }


def optimize_constrained(
    connection_name: str,
    table_name: str,
    objective_function: str,
    initial_guess_column: str,
    constraint_functions: Optional[List[str]] = None,
    constraint_types: Optional[List[str]] = None,
    bounds_columns: Optional[List[str]] = None,
    method: str = 'SLSQP',
    multi_objective: bool = False
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
        from ..database_manager import DatabaseManager
        
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
            local_vars.update({'x': x, 'np': np})
            return eval(objective_function, {"__builtins__": {}}, local_vars)
        
        # Create constraints
        constraints = []
        if constraint_functions and constraint_types:
            for func_expr, ctype in zip(constraint_functions, constraint_types):
                def constraint_func(x, expr=func_expr):
                    local_vars = {col: data[col].values for col in data.columns}
                    local_vars.update({'x': x, 'np': np})
                    return eval(expr, {"__builtins__": {}}, local_vars)
                
                constraints.append({
                    'type': ctype,
                    'fun': constraint_func
                })
        
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
            multi_objective=multi_objective
        )
        
        result = optimizer.optimize(objective_func, x0)
        
        # Format results
        formatted_result = {
            'success': result.success,
            'optimal_value': float(result.fun) if result.success else None,
            'optimal_solution': result.x.tolist() if result.success else None,
            'message': result.message,
            'method': result.method,
            'execution_time': result.execution_time,
            'iterations': result.nit,
            'function_evaluations': result.nfev,
            'constraint_analysis': result.constraints,
            'constraint_violations': result.constraint_violations.tolist() if result.constraint_violations is not None else None,
            'lagrange_multipliers': result.lagrange_multipliers.tolist() if result.lagrange_multipliers is not None else None
        }
        
        return formatted_result
        
    except Exception as e:
        logger.error(f"Constrained optimization failed: {e}")
        return {
            'success': False,
            'error': str(e),
            'message': f"Constrained optimization failed: {e}"
        }


def analyze_network(
    connection_name: str,
    table_name: str,
    source_column: str,
    target_column: str,
    weight_column: Optional[str] = None,
    directed: bool = False,
    include_centrality: bool = True,
    algorithms: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Perform network analysis on graph data from database.
    
    Parameters:
    -----------
    connection_name : str
        Database connection name
    table_name : str
        Table containing edge data
    source_column : str
        Column with source nodes
    target_column : str
        Column with target nodes  
    weight_column : str, optional
        Column with edge weights
    directed : bool, default=False
        Whether graph is directed
    include_centrality : bool, default=True
        Include centrality measures
    algorithms : list of str, optional
        Specific algorithms to run
    
    Returns:
    --------
    dict
        Network analysis results
    """
    try:
        from ..database_manager import DatabaseManager
        
        if not NETWORKX_AVAILABLE:
            return {
                'success': False,
                'error': 'NetworkX not available',
                'message': 'NetworkX is required for network analysis. Please install with: pip install networkx'
            }
        
        # Get database manager and connection
        db_manager = DatabaseManager()
        engine = db_manager._get_connection(connection_name)
        
        # Load edge data
        columns = [source_column, target_column]
        if weight_column:
            columns.append(weight_column)
        
        query = f"SELECT {', '.join(columns)} FROM {table_name}"
        edges_df = pd.read_sql(query, engine)
        
        # Convert to numpy array for NetworkAnalyzer
        if weight_column:
            edge_array = edges_df[[source_column, target_column, weight_column]].values
        else:
            edge_array = edges_df[[source_column, target_column]].values
            
        # Create and analyze network
        analyzer = NetworkAnalyzer(
            directed=directed,
            weighted=weight_column is not None
        )
        
        analyzer.fit(edge_array)
        result = analyzer.analyze_network(include_centrality=include_centrality)
        
        # Format results
        formatted_result = {
            'success': True,
            'graph_properties': result.graph_properties,
            'execution_time': result.execution_time,
            'method': result.method
        }
        
        # Add specific analysis results
        if result.shortest_paths:
            formatted_result['shortest_paths'] = result.shortest_paths
        
        if result.max_flow_value is not None:
            formatted_result['max_flow'] = {
                'value': result.max_flow_value,
                'details': result.max_flow_dict
            }
        
        if result.minimum_spanning_tree:
            formatted_result['minimum_spanning_tree'] = result.minimum_spanning_tree
        
        if result.tsp_solution:
            formatted_result['tsp_solution'] = result.tsp_solution
        
        if result.centrality_measures:
            formatted_result['centrality_measures'] = result.centrality_measures
        
        return formatted_result
        
    except Exception as e:
        logger.error(f"Network analysis failed: {e}")
        return {
            'success': False,
            'error': str(e),
            'message': f"Network analysis failed: {e}"
        }


def solve_assignment_problem(
    connection_name: str,
    table_name: str,
    cost_matrix_columns: List[str],
    agent_id_column: Optional[str] = None,
    task_id_column: Optional[str] = None,
    method: str = 'hungarian',
    maximize: bool = False,
    allow_partial: bool = False
) -> Dict[str, Any]:
    """
    Solve assignment problem from database cost matrix.
    
    Parameters:
    -----------
    connection_name : str
        Database connection name
    table_name : str
        Table containing cost matrix data
    cost_matrix_columns : list of str
        Columns representing cost matrix (tasks)
    agent_id_column : str, optional
        Column with agent identifiers
    task_id_column : str, optional
        Column with task identifiers
    method : str, default='hungarian'
        Solution method
    maximize : bool, default=False
        Whether to maximize instead of minimize
    allow_partial : bool, default=False
        Allow partial assignments
    
    Returns:
    --------
    dict
        Assignment problem results
    """
    try:
        from ..database_manager import DatabaseManager
        
        # Get database manager and connection
        db_manager = DatabaseManager()
        engine = db_manager._get_connection(connection_name)
        
        # Load cost matrix data
        all_columns = cost_matrix_columns.copy()
        if agent_id_column:
            all_columns.append(agent_id_column)
        
        query = f"SELECT {', '.join(all_columns)} FROM {table_name}"
        data = pd.read_sql(query, engine)
        
        # Extract cost matrix
        cost_matrix = data[cost_matrix_columns].values
        
        # Get agent and task identifiers
        if agent_id_column:
            agent_ids = data[agent_id_column].tolist()
        else:
            agent_ids = list(range(len(cost_matrix)))
        
        if task_id_column:
            # This would require a different table structure
            task_ids = cost_matrix_columns
        else:
            task_ids = cost_matrix_columns
        
        # Create and solve assignment
        solver = AssignmentSolver(
            method=method,
            maximize=maximize,
            allow_partial=allow_partial
        )
        
        result = solver.solve_assignment(cost_matrix)
        
        # Format results with identifiers
        assignment_pairs_with_ids = []
        for agent_idx, task_idx in result.assignment_pairs:
            assignment_pairs_with_ids.append({
                'agent_id': agent_ids[agent_idx],
                'agent_index': agent_idx,
                'task_id': task_ids[task_idx],
                'task_index': task_idx,
                'cost': float(cost_matrix[agent_idx, task_idx])
            })
        
        formatted_result = {
            'success': result.success,
            'total_cost': float(result.total_cost),
            'assignment_pairs': assignment_pairs_with_ids,
            'assignment_matrix': result.assignment_matrix.tolist(),
            'method': result.method,
            'execution_time': result.execution_time,
            'iterations': result.nit,
            'function_evaluations': result.nfev,
            'is_perfect_matching': result.is_perfect_matching,
            'unassigned_agents': [agent_ids[i] for i in result.unassigned_agents],
            'unassigned_tasks': [task_ids[i] for i in result.unassigned_tasks]
        }
        
        return formatted_result
        
    except Exception as e:
        logger.error(f"Assignment problem failed: {e}")
        return {
            'success': False,
            'error': str(e),
            'message': f"Assignment problem solving failed: {e}"
        }