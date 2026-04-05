"""
Tool functions for network analysis and assignment problems.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ...logging_manager import get_logger
from ._network import NetworkAnalyzer, NETWORKX_AVAILABLE
from ._assignment import AssignmentSolver

logger = get_logger(__name__)


def analyze_network(
    connection_name: str,
    table_name: str,
    source_column: str,
    target_column: str,
    weight_column: Optional[str] = None,
    directed: bool = False,
    include_centrality: bool = True,
    algorithms: Optional[List[str]] = None,
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
        from ... import DatabaseManager

        if not NETWORKX_AVAILABLE:
            return {
                "success": False,
                "error": "NetworkX not available",
                "message": "NetworkX is required for network analysis. Please install with: pip install networkx",
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
            directed=directed, weighted=weight_column is not None
        )

        analyzer.fit(edge_array)
        result = analyzer.analyze_network(include_centrality=include_centrality)

        # Format results
        formatted_result = {
            "success": True,
            "graph_properties": result.graph_properties,
            "execution_time": result.execution_time,
            "method": result.method,
        }

        # Add specific analysis results
        if result.shortest_paths:
            formatted_result["shortest_paths"] = result.shortest_paths

        if result.max_flow_value is not None:
            formatted_result["max_flow"] = {
                "value": result.max_flow_value,
                "details": result.max_flow_dict,
            }

        if result.minimum_spanning_tree:
            formatted_result["minimum_spanning_tree"] = result.minimum_spanning_tree

        if result.tsp_solution:
            formatted_result["tsp_solution"] = result.tsp_solution

        if result.centrality_measures:
            formatted_result["centrality_measures"] = result.centrality_measures

        return formatted_result

    except Exception as e:
        logger.error(f"Network analysis failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "message": f"Network analysis failed: {e}",
        }


def solve_assignment_problem(
    connection_name: str,
    table_name: str,
    cost_matrix_columns: List[str],
    agent_id_column: Optional[str] = None,
    task_id_column: Optional[str] = None,
    method: str = "hungarian",
    maximize: bool = False,
    allow_partial: bool = False,
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
        from ... import DatabaseManager

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
            method=method, maximize=maximize, allow_partial=allow_partial
        )

        result = solver.solve_assignment(cost_matrix)

        # Format results with identifiers
        assignment_pairs_with_ids = []
        for agent_idx, task_idx in result.assignment_pairs:
            assignment_pairs_with_ids.append(
                {
                    "agent_id": agent_ids[agent_idx],
                    "agent_index": agent_idx,
                    "task_id": task_ids[task_idx],
                    "task_index": task_idx,
                    "cost": float(cost_matrix[agent_idx, task_idx]),
                }
            )

        formatted_result = {
            "success": result.success,
            "total_cost": float(result.total_cost),
            "assignment_pairs": assignment_pairs_with_ids,
            "assignment_matrix": result.assignment_matrix.tolist(),
            "method": result.method,
            "execution_time": result.execution_time,
            "iterations": result.nit,
            "function_evaluations": result.nfev,
            "is_perfect_matching": result.is_perfect_matching,
            "unassigned_agents": [agent_ids[i] for i in result.unassigned_agents],
            "unassigned_tasks": [task_ids[i] for i in result.unassigned_tasks],
        }

        return formatted_result

    except Exception as e:
        logger.error(f"Assignment problem failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "message": f"Assignment problem solving failed: {e}",
        }
