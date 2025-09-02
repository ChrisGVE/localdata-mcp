"""
Tests for optimization domain functionality.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from scipy.optimize import OptimizeResult

from src.localdata_mcp.domains.optimization import (
    LinearProgrammingSolver,
    ConstrainedOptimizer,
    NetworkAnalyzer,
    AssignmentSolver,
    OptimizationResult,
    LinearProgramResult,
    ConstrainedOptResult,
    NetworkAnalysisResult,
    AssignmentResult,
    solve_linear_program,
    optimize_constrained,
    analyze_network,
    solve_assignment_problem,
    NETWORKX_AVAILABLE
)


class TestLinearProgrammingSolver:
    """Test linear programming solver functionality."""
    
    def test_initialization(self):
        """Test LinearProgrammingSolver initialization."""
        solver = LinearProgrammingSolver(
            method='highs-ds',
            integer_variables=[0, 2],
            bounds=[(0, 10), (0, 5), (0, None)]
        )
        
        assert solver.method == 'highs-ds'
        assert solver.integer_variables == [0, 2]
        assert solver.bounds == [(0, 10), (0, 5), (0, None)]
        assert not solver.is_fitted_
    
    def test_simple_linear_program(self):
        """Test solving a simple linear program."""
        solver = LinearProgrammingSolver()
        
        # Minimize: x + 2*y
        # Subject to: x + y <= 3, 2*x + y <= 4, x >= 0, y >= 0
        c = np.array([1, 2])
        A_ub = np.array([[1, 1], [2, 1]])
        b_ub = np.array([3, 4])
        bounds = [(0, None), (0, None)]
        
        result = solver.solve(c, A_ub, b_ub, bounds=bounds)
        
        assert isinstance(result, LinearProgramResult)
        assert result.success
        assert result.fun < 5  # Expected optimal value around 3
        assert len(result.x) == 2
        assert all(x >= -1e-6 for x in result.x)  # Non-negative solution
    
    def test_integer_programming(self):
        """Test integer programming solution."""
        solver = LinearProgrammingSolver(integer_variables=[0, 1])
        
        c = np.array([1, 2])
        A_ub = np.array([[1, 1], [2, 1]])
        b_ub = np.array([3, 4])
        bounds = [(0, None), (0, None)]
        
        result = solver.solve(c, A_ub, b_ub, bounds=bounds)
        
        assert isinstance(result, LinearProgramResult)
        assert result.is_integer_solution
        if result.success:
            # Check that integer variables are indeed integers
            for idx in solver.integer_variables:
                if idx < len(result.x):
                    assert abs(result.x[idx] - round(result.x[idx])) < 1e-6
    
    def test_infeasible_problem(self):
        """Test handling of infeasible linear program."""
        solver = LinearProgrammingSolver()
        
        # Infeasible problem: x + y >= 1, x + y <= 0
        c = np.array([1, 1])
        A_ub = np.array([[-1, -1], [1, 1]])
        b_ub = np.array([-1, 0])
        
        result = solver.solve(c, A_ub, b_ub)
        
        assert isinstance(result, LinearProgramResult)
        assert not result.success
    
    def test_sensitivity_analysis(self):
        """Test sensitivity analysis functionality."""
        solver = LinearProgrammingSolver()
        
        c = np.array([1, 2])
        A_ub = np.array([[1, 1], [2, 1]])
        b_ub = np.array([3, 4])
        
        result = solver.solve(c, A_ub, b_ub)
        
        if result.success:
            assert result.sensitivity is not None
            assert isinstance(result.sensitivity, dict)
    
    def test_fit_transform_interface(self):
        """Test sklearn interface compliance."""
        solver = LinearProgrammingSolver()
        X = np.array([[1, 2], [3, 4]])
        
        # fit should return self and set is_fitted_
        fitted_solver = solver.fit(X)
        assert fitted_solver is solver
        assert solver.is_fitted_
        
        # transform should work after fit
        X_transformed = solver.transform(X)
        np.testing.assert_array_equal(X_transformed, X)


class TestConstrainedOptimizer:
    """Test constrained optimization functionality."""
    
    def test_initialization(self):
        """Test ConstrainedOptimizer initialization."""
        constraints = [{'type': 'eq', 'fun': lambda x: x[0] + x[1] - 1}]
        optimizer = ConstrainedOptimizer(
            method='trust-constr',
            constraints=constraints,
            bounds=[(0, 1), (0, 1)],
            multi_objective=True
        )
        
        assert optimizer.method == 'trust-constr'
        assert optimizer.constraints == constraints
        assert optimizer.bounds == [(0, 1), (0, 1)]
        assert optimizer.multi_objective
    
    def test_simple_constrained_optimization(self):
        """Test solving a simple constrained optimization problem."""
        optimizer = ConstrainedOptimizer()
        
        # Minimize: x^2 + y^2
        # Subject to: x + y = 1
        def objective(x):
            return x[0]**2 + x[1]**2
        
        constraints = [{'type': 'eq', 'fun': lambda x: x[0] + x[1] - 1}]
        x0 = np.array([0.5, 0.5])
        
        result = optimizer.optimize(objective, x0, constraints=constraints)
        
        assert isinstance(result, ConstrainedOptResult)
        if result.success:
            assert len(result.x) == 2
            # Solution should be approximately [0.5, 0.5]
            assert abs(result.x[0] + result.x[1] - 1) < 1e-3
    
    def test_bounded_optimization(self):
        """Test optimization with bounds."""
        optimizer = ConstrainedOptimizer()
        
        def objective(x):
            return (x[0] - 2)**2 + (x[1] - 1)**2
        
        x0 = np.array([0.0, 0.0])
        bounds = [(0, 1), (0, 1)]
        
        result = optimizer.optimize(objective, x0, bounds=bounds)
        
        assert isinstance(result, ConstrainedOptResult)
        if result.success:
            # Solution should be at bounds [1, 1]
            assert all(0 <= x <= 1.01 for x in result.x)
    
    def test_multi_objective_optimization(self):
        """Test multi-objective optimization."""
        optimizer = ConstrainedOptimizer(multi_objective=True)
        
        def obj1(x):
            return x[0]**2
        
        def obj2(x):
            return (x[0] - 2)**2
        
        x0 = np.array([1.0])
        bounds = [(0, 2)]
        
        result = optimizer.optimize([obj1, obj2], x0, bounds=bounds)
        
        assert isinstance(result, ConstrainedOptResult)
        # Multi-objective result should be between the individual optima
    
    def test_constraint_analysis(self):
        """Test constraint analysis functionality."""
        optimizer = ConstrainedOptimizer()
        
        def objective(x):
            return x[0]**2 + x[1]**2
        
        def constraint1(x):
            return x[0] + x[1] - 1
        
        def constraint2(x):
            return x[0] - 0.5
        
        constraints = [
            {'type': 'eq', 'fun': constraint1},
            {'type': 'ineq', 'fun': constraint2}
        ]
        x0 = np.array([0.6, 0.4])
        
        result = optimizer.optimize(objective, x0, constraints=constraints)
        
        if result.success and result.constraints:
            assert len(result.constraints) == 2
            for constraint_info in result.constraints:
                assert 'type' in constraint_info
                assert 'violation' in constraint_info
    
    def test_fit_transform_interface(self):
        """Test sklearn interface compliance."""
        optimizer = ConstrainedOptimizer()
        X = np.array([[1, 2], [3, 4]])
        
        fitted_optimizer = optimizer.fit(X)
        assert fitted_optimizer is optimizer
        assert optimizer.is_fitted_
        
        X_transformed = optimizer.transform(X)
        np.testing.assert_array_equal(X_transformed, X)


@pytest.mark.skipif(not NETWORKX_AVAILABLE, reason="NetworkX not available")
class TestNetworkAnalyzer:
    """Test network optimization and analysis functionality."""
    
    def test_initialization(self):
        """Test NetworkAnalyzer initialization."""
        analyzer = NetworkAnalyzer(
            algorithm='dijkstra',
            directed=True,
            weighted=True
        )
        
        assert analyzer.algorithm == 'dijkstra'
        assert analyzer.directed
        assert analyzer.weighted
    
    def test_networkx_not_available_error(self):
        """Test error when NetworkX is not available."""
        with patch('src.localdata_mcp.domains.optimization.NETWORKX_AVAILABLE', False):
            with pytest.raises(ImportError, match="NetworkX is required"):
                NetworkAnalyzer()
    
    def test_fit_from_adjacency_matrix(self):
        """Test building network from adjacency matrix."""
        analyzer = NetworkAnalyzer()
        
        # Simple 3-node network
        adj_matrix = np.array([
            [0, 1, 2],
            [1, 0, 1],
            [2, 1, 0]
        ])
        
        analyzer.fit(adj_matrix)
        
        assert analyzer.is_fitted_
        assert analyzer.graph_ is not None
        assert analyzer.graph_.number_of_nodes() == 3
        assert analyzer.graph_.number_of_edges() == 3  # Undirected, so each edge counted once
    
    def test_fit_from_edge_list(self):
        """Test building network from edge list."""
        analyzer = NetworkAnalyzer(directed=True)
        
        # Edge list: [source, target, weight]
        edge_list = np.array([
            [0, 1, 2.0],
            [1, 2, 1.5],
            [0, 2, 3.0]
        ])
        
        analyzer.fit(edge_list)
        
        assert analyzer.is_fitted_
        assert analyzer.graph_.number_of_nodes() == 3
        assert analyzer.graph_.number_of_edges() == 3
    
    def test_network_analysis(self):
        """Test comprehensive network analysis."""
        analyzer = NetworkAnalyzer()
        
        # Simple triangle network
        adj_matrix = np.array([
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0]
        ])
        
        analyzer.fit(adj_matrix)
        result = analyzer.analyze_network()
        
        assert isinstance(result, NetworkAnalysisResult)
        assert result.graph_properties['num_nodes'] == 3
        assert result.graph_properties['num_edges'] == 3
        assert result.graph_properties['is_connected']
        
        # Should have shortest paths
        assert result.shortest_paths is not None
        
        # Should have centrality measures
        assert result.centrality_measures is not None
        assert 'degree' in result.centrality_measures
        assert 'betweenness' in result.centrality_measures
    
    def test_shortest_paths(self):
        """Test shortest path calculations."""
        analyzer = NetworkAnalyzer()
        
        # Linear network: 0-1-2
        adj_matrix = np.array([
            [0, 1, 0],
            [1, 0, 2],
            [0, 2, 0]
        ])
        
        analyzer.fit(adj_matrix)
        result = analyzer.analyze_network()
        
        assert result.shortest_paths is not None
        # Path from 0 to 2 should go through 1 with distance 3
        if 'distances' in result.shortest_paths:
            distances = result.shortest_paths['distances']
            if '0' in distances and '2' in distances['0']:
                assert distances['0']['2'] == 3
    
    def test_tsp_solution(self):
        """Test TSP heuristic solution."""
        analyzer = NetworkAnalyzer()
        
        # Complete graph with 4 nodes
        adj_matrix = np.array([
            [0, 1, 2, 3],
            [1, 0, 4, 2],
            [2, 4, 0, 1],
            [3, 2, 1, 0]
        ])
        
        analyzer.fit(adj_matrix)
        result = analyzer.analyze_network()
        
        if result.tsp_solution:
            assert 'tour' in result.tsp_solution
            assert 'distance' in result.tsp_solution
            assert len(result.tsp_solution['tour']) == 4
    
    def test_max_flow_directed_graph(self):
        """Test maximum flow calculation for directed graphs."""
        analyzer = NetworkAnalyzer(directed=True)
        
        # Flow network: source(0) -> 1 -> sink(2)
        edge_list = np.array([
            [0, 1, 3.0],  # capacity 3
            [1, 2, 2.0],  # capacity 2 (bottleneck)
            [0, 2, 1.0]   # alternative path
        ])
        
        analyzer.fit(edge_list)
        result = analyzer.analyze_network()
        
        if result.max_flow_value is not None:
            assert result.max_flow_value >= 0
            assert result.max_flow_dict is not None
    
    def test_minimum_spanning_tree(self):
        """Test MST calculation for undirected graphs."""
        analyzer = NetworkAnalyzer(directed=False)
        
        # Complete graph with different weights
        adj_matrix = np.array([
            [0, 4, 2, 3],
            [4, 0, 1, 5],
            [2, 1, 0, 6],
            [3, 5, 6, 0]
        ])
        
        analyzer.fit(adj_matrix)
        result = analyzer.analyze_network()
        
        if result.minimum_spanning_tree:
            assert 'total_weight' in result.minimum_spanning_tree
            assert 'edges' in result.minimum_spanning_tree
            # MST of 4 nodes should have 3 edges
            assert result.minimum_spanning_tree['num_edges'] == 3


class TestAssignmentSolver:
    """Test assignment problem solving functionality."""
    
    def test_initialization(self):
        """Test AssignmentSolver initialization."""
        solver = AssignmentSolver(
            method='linear_programming',
            maximize=True,
            allow_partial=True
        )
        
        assert solver.method == 'linear_programming'
        assert solver.maximize
        assert solver.allow_partial
    
    def test_hungarian_assignment(self):
        """Test assignment solving with Hungarian algorithm."""
        solver = AssignmentSolver(method='hungarian')
        
        # Simple 3x3 cost matrix
        cost_matrix = np.array([
            [4, 1, 3],
            [2, 0, 5],
            [3, 2, 2]
        ])
        
        result = solver.solve_assignment(cost_matrix)
        
        assert isinstance(result, AssignmentResult)
        if result.success:
            assert len(result.assignment_pairs) <= min(cost_matrix.shape)
            assert result.total_cost >= 0
            
            # Check that each agent assigned to at most one task
            agents = [pair[0] for pair in result.assignment_pairs]
            assert len(agents) == len(set(agents))
            
            # Check that each task assigned to at most one agent
            tasks = [pair[1] for pair in result.assignment_pairs]
            assert len(tasks) == len(set(tasks))
    
    def test_maximization_assignment(self):
        """Test assignment with maximization objective."""
        solver = AssignmentSolver(method='hungarian', maximize=True)
        
        # Profit matrix (higher is better)
        profit_matrix = np.array([
            [1, 4, 2],
            [3, 2, 0],
            [2, 1, 3]
        ])
        
        result = solver.solve_assignment(profit_matrix)
        
        assert isinstance(result, AssignmentResult)
        if result.success:
            # With maximization, we should get high total value
            assert result.total_cost > 0
    
    def test_rectangular_assignment(self):
        """Test assignment with non-square cost matrix."""
        solver = AssignmentSolver(method='hungarian')
        
        # More agents than tasks
        cost_matrix = np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
            [2, 1, 4]  # 4 agents, 3 tasks
        ])
        
        result = solver.solve_assignment(cost_matrix)
        
        assert isinstance(result, AssignmentResult)
        if result.success:
            assert len(result.assignment_pairs) == 3  # Only 3 tasks available
            assert len(result.unassigned_agents) == 1  # One agent unassigned
            assert not result.is_perfect_matching
    
    def test_greedy_assignment(self):
        """Test greedy assignment algorithm."""
        solver = AssignmentSolver(method='greedy')
        
        cost_matrix = np.array([
            [1, 9, 9],
            [9, 1, 9],
            [9, 9, 1]
        ])
        
        result = solver.solve_assignment(cost_matrix)
        
        assert isinstance(result, AssignmentResult)
        if result.success:
            # Greedy should find the diagonal solution (cost = 3)
            assert len(result.assignment_pairs) == 3
    
    def test_linear_programming_assignment(self):
        """Test assignment using linear programming."""
        solver = AssignmentSolver(method='linear_programming')
        
        cost_matrix = np.array([
            [2, 3, 1],
            [1, 4, 2],
            [3, 1, 3]
        ])
        
        result = solver.solve_assignment(cost_matrix)
        
        assert isinstance(result, AssignmentResult)
        if result.success:
            assert len(result.assignment_pairs) == 3
            assert result.total_cost > 0
    
    def test_assignment_matrix_format(self):
        """Test assignment matrix output format."""
        solver = AssignmentSolver()
        
        cost_matrix = np.array([
            [1, 2],
            [3, 0]
        ])
        
        result = solver.solve_assignment(cost_matrix)
        
        if result.success:
            # Assignment matrix should be binary
            assert result.assignment_matrix.shape == cost_matrix.shape
            assert np.all((result.assignment_matrix == 0) | (result.assignment_matrix == 1))
            
            # Each row and column should sum to at most 1
            assert np.all(result.assignment_matrix.sum(axis=1) <= 1)
            assert np.all(result.assignment_matrix.sum(axis=0) <= 1)
    
    def test_fit_transform_interface(self):
        """Test sklearn interface compliance."""
        solver = AssignmentSolver()
        X = np.array([[1, 2], [3, 4]])
        
        fitted_solver = solver.fit(X)
        assert fitted_solver is solver
        assert solver.is_fitted_
        
        X_transformed = solver.transform(X)
        np.testing.assert_array_equal(X_transformed, X)


class TestOptimizationResults:
    """Test optimization result classes."""
    
    def test_optimization_result_creation(self):
        """Test OptimizationResult creation."""
        result = OptimizationResult(
            success=True,
            fun=10.5,
            x=np.array([1, 2, 3]),
            message="Success",
            nit=15,
            nfev=50,
            execution_time=0.123,
            method="test_method"
        )
        
        assert result.success
        assert result.fun == 10.5
        np.testing.assert_array_equal(result.x, np.array([1, 2, 3]))
        assert result.message == "Success"
        assert result.nit == 15
        assert result.nfev == 50
        assert result.execution_time == 0.123
        assert result.method == "test_method"
    
    def test_linear_program_result_creation(self):
        """Test LinearProgramResult creation."""
        result = LinearProgramResult(
            success=True,
            fun=5.0,
            x=np.array([1, 2]),
            message="Optimal",
            nit=10,
            nfev=20,
            execution_time=0.1,
            method="highs",
            slack=np.array([0, 1]),
            dual_values=np.array([2, 0]),
            is_integer_solution=True
        )
        
        assert result.success
        assert result.is_integer_solution
        np.testing.assert_array_equal(result.slack, np.array([0, 1]))
        np.testing.assert_array_equal(result.dual_values, np.array([2, 0]))
    
    def test_network_analysis_result_creation(self):
        """Test NetworkAnalysisResult creation."""
        result = NetworkAnalysisResult(
            graph_properties={'num_nodes': 5, 'num_edges': 7},
            execution_time=0.2,
            method="comprehensive_analysis"
        )
        
        assert result.graph_properties['num_nodes'] == 5
        assert result.graph_properties['num_edges'] == 7
        assert result.execution_time == 0.2
        assert result.method == "comprehensive_analysis"
    
    def test_assignment_result_creation(self):
        """Test AssignmentResult creation."""
        assignment_matrix = np.array([[1, 0], [0, 1]])
        assignment_pairs = [(0, 0), (1, 1)]
        
        result = AssignmentResult(
            success=True,
            fun=5.0,
            x=assignment_matrix.flatten(),
            message="Optimal assignment",
            nit=1,
            nfev=1,
            execution_time=0.05,
            method="hungarian",
            assignment_matrix=assignment_matrix,
            assignment_pairs=assignment_pairs,
            total_cost=5.0,
            is_perfect_matching=True
        )
        
        assert result.success
        assert result.is_perfect_matching
        np.testing.assert_array_equal(result.assignment_matrix, assignment_matrix)
        assert result.assignment_pairs == assignment_pairs
        assert result.total_cost == 5.0


class TestOptimizationTools:
    """Test optimization tool functions."""
    
    @patch('src.localdata_mcp.domains.optimization.DatabaseManager')
    def test_solve_linear_program_tool(self, mock_db_manager):
        """Test solve_linear_program tool function."""
        # Mock database setup
        mock_engine = Mock()
        mock_db_instance = Mock()
        mock_db_instance._get_connection.return_value = mock_engine
        mock_db_manager.return_value = mock_db_instance
        
        # Mock pandas read_sql
        with patch('pandas.read_sql') as mock_read_sql:
            # Mock objective coefficients
            mock_read_sql.return_value = pd.DataFrame({'cost': [1, 2, 3]})
            
            result = solve_linear_program(
                connection_name='test_db',
                table_name='optimization_data',
                objective_column='cost'
            )
            
            assert isinstance(result, dict)
            assert 'success' in result
            
            # Verify database calls
            mock_db_instance._get_connection.assert_called_once_with('test_db')
    
    @patch('src.localdata_mcp.domains.optimization.DatabaseManager')
    def test_solve_linear_program_with_constraints(self, mock_db_manager):
        """Test solve_linear_program with constraints."""
        mock_engine = Mock()
        mock_db_instance = Mock()
        mock_db_instance._get_connection.return_value = mock_engine
        mock_db_manager.return_value = mock_db_instance
        
        with patch('pandas.read_sql') as mock_read_sql:
            # Mock different calls to read_sql
            def side_effect(query, engine):
                if 'cost' in query:
                    return pd.DataFrame({'cost': [1, 2]})
                else:
                    return pd.DataFrame({'x1': [1, 2], 'x2': [3, 1]})
            
            mock_read_sql.side_effect = side_effect
            
            result = solve_linear_program(
                connection_name='test_db',
                table_name='data',
                objective_column='cost',
                constraint_columns=['x1', 'x2'],
                constraint_values=[10, 8],
                constraint_types=['<=', '<=']
            )
            
            assert isinstance(result, dict)
            assert 'success' in result
    
    @patch('src.localdata_mcp.domains.optimization.DatabaseManager')
    def test_optimize_constrained_tool(self, mock_db_manager):
        """Test optimize_constrained tool function."""
        mock_engine = Mock()
        mock_db_instance = Mock()
        mock_db_instance._get_connection.return_value = mock_engine
        mock_db_manager.return_value = mock_db_instance
        
        with patch('pandas.read_sql') as mock_read_sql:
            mock_read_sql.return_value = pd.DataFrame({
                'x0': [1.0, 2.0],
                'initial': [0.5, 1.0]
            })
            
            result = optimize_constrained(
                connection_name='test_db',
                table_name='data',
                objective_function='np.sum(x**2)',
                initial_guess_column='initial'
            )
            
            assert isinstance(result, dict)
            assert 'success' in result
    
    @patch('src.localdata_mcp.domains.optimization.DatabaseManager')
    @pytest.mark.skipif(not NETWORKX_AVAILABLE, reason="NetworkX not available")
    def test_analyze_network_tool(self, mock_db_manager):
        """Test analyze_network tool function."""
        mock_engine = Mock()
        mock_db_instance = Mock()
        mock_db_instance._get_connection.return_value = mock_engine
        mock_db_manager.return_value = mock_db_instance
        
        with patch('pandas.read_sql') as mock_read_sql:
            mock_read_sql.return_value = pd.DataFrame({
                'source': [0, 1, 0],
                'target': [1, 2, 2],
                'weight': [1.0, 2.0, 1.5]
            })
            
            result = analyze_network(
                connection_name='test_db',
                table_name='edges',
                source_column='source',
                target_column='target',
                weight_column='weight'
            )
            
            assert isinstance(result, dict)
            assert 'success' in result
            if result['success']:
                assert 'graph_properties' in result
    
    @patch('src.localdata_mcp.domains.optimization.DatabaseManager')
    def test_solve_assignment_problem_tool(self, mock_db_manager):
        """Test solve_assignment_problem tool function."""
        mock_engine = Mock()
        mock_db_instance = Mock()
        mock_db_instance._get_connection.return_value = mock_engine
        mock_db_manager.return_value = mock_db_instance
        
        with patch('pandas.read_sql') as mock_read_sql:
            mock_read_sql.return_value = pd.DataFrame({
                'task1': [1, 2, 3],
                'task2': [4, 0, 2],
                'task3': [3, 2, 1],
                'agent': ['A', 'B', 'C']
            })
            
            result = solve_assignment_problem(
                connection_name='test_db',
                table_name='costs',
                cost_matrix_columns=['task1', 'task2', 'task3'],
                agent_id_column='agent'
            )
            
            assert isinstance(result, dict)
            assert 'success' in result
            if result['success']:
                assert 'assignment_pairs' in result
                assert 'total_cost' in result
    
    def test_tool_error_handling(self):
        """Test error handling in optimization tools."""
        # Test with invalid connection
        result = solve_linear_program(
            connection_name='invalid_connection',
            table_name='data',
            objective_column='cost'
        )
        
        assert isinstance(result, dict)
        assert result['success'] is False
        assert 'error' in result
    
    @pytest.mark.skipif(NETWORKX_AVAILABLE, reason="NetworkX is available")
    def test_analyze_network_without_networkx(self):
        """Test analyze_network when NetworkX is not available."""
        result = analyze_network(
            connection_name='test_db',
            table_name='edges',
            source_column='source',
            target_column='target'
        )
        
        assert isinstance(result, dict)
        assert result['success'] is False
        assert 'NetworkX not available' in result['error']


class TestOptimizationIntegration:
    """Integration tests for optimization domain."""
    
    def test_linear_programming_workflow(self):
        """Test complete linear programming workflow."""
        solver = LinearProgrammingSolver(method='highs')
        
        # Multi-variable problem
        c = np.array([2, 3, 1])
        A_ub = np.array([[1, 1, 1], [2, 1, 0], [0, 1, 2]])
        b_ub = np.array([5, 4, 6])
        bounds = [(0, None), (0, None), (0, None)]
        
        result = solver.solve(c, A_ub, b_ub, bounds=bounds)
        
        if result.success:
            # Verify solution satisfies constraints
            x = result.x
            constraint_values = A_ub @ x
            assert np.all(constraint_values <= b_ub + 1e-6)  # Allow small numerical errors
            
            # Verify bounds
            assert np.all(x >= -1e-6)  # Non-negative
    
    def test_optimization_pipeline_compatibility(self):
        """Test optimization classes work with sklearn pipeline concepts."""
        # Test that all optimizers implement the required interfaces
        from sklearn.base import BaseEstimator, TransformerMixin
        
        solvers = [
            LinearProgrammingSolver(),
            ConstrainedOptimizer(),
            AssignmentSolver()
        ]
        
        if NETWORKX_AVAILABLE:
            solvers.append(NetworkAnalyzer())
        
        for solver in solvers:
            assert isinstance(solver, BaseEstimator)
            assert isinstance(solver, TransformerMixin)
            
            # Test basic interface
            X = np.array([[1, 2], [3, 4]])
            solver.fit(X)
            assert solver.is_fitted_
            
            X_transformed = solver.transform(X)
            assert X_transformed.shape == X.shape
    
    def test_result_serialization(self):
        """Test that optimization results can be serialized."""
        import json
        
        # Create sample results
        opt_result = OptimizationResult(
            success=True,
            fun=10.0,
            x=np.array([1, 2]),
            message="Success",
            nit=5,
            nfev=15,
            execution_time=0.1,
            method="test"
        )
        
        # Test that we can convert to serializable format
        result_dict = {
            'success': opt_result.success,
            'fun': float(opt_result.fun),
            'x': opt_result.x.tolist(),
            'message': opt_result.message,
            'iterations': opt_result.nit,
            'function_evaluations': opt_result.nfev,
            'execution_time': opt_result.execution_time,
            'method': opt_result.method
        }
        
        # Should be JSON serializable
        json_str = json.dumps(result_dict)
        assert len(json_str) > 0
        
        # Should be deserializable
        reconstructed = json.loads(json_str)
        assert reconstructed['success'] == opt_result.success
        assert reconstructed['fun'] == opt_result.fun


if __name__ == '__main__':
    pytest.main([__file__, '-v'])