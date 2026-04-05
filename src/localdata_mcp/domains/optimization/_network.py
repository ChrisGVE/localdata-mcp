"""
Network Analyzer - graph algorithms using NetworkX.
"""

import sys
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.utils.validation import check_array, check_is_fitted

from ...logging_manager import get_logger
from ...pipeline.base import (
    AnalysisPipelineBase,
    StreamingConfig,
)
from ._types import NetworkAnalysisResult

logger = get_logger(__name__)

# Try to import networkx for network optimization
try:
    import networkx as nx

    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    nx = None


def _is_networkx_available() -> bool:
    """Check NETWORKX_AVAILABLE from the package namespace to support test patching."""
    pkg = sys.modules.get("localdata_mcp.domains.optimization")
    if pkg is not None:
        return getattr(pkg, "NETWORKX_AVAILABLE", NETWORKX_AVAILABLE)
    return NETWORKX_AVAILABLE


class NetworkAnalyzer(AnalysisPipelineBase):
    """
    Network optimization and analysis using NetworkX for graph algorithms.

    Provides shortest path algorithms, maximum flow, minimum spanning tree,
    and traveling salesman problem heuristics.
    """

    def __init__(
        self,
        algorithm: str = "auto",
        directed: bool = False,
        weighted: bool = True,
        input_format: str = "auto",
        streaming_config: Optional[StreamingConfig] = None,
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
        input_format : str, default='auto'
            Input data format: 'adjacency_matrix', 'edge_list', or 'auto'
            (auto uses square-matrix heuristic)
        streaming_config : StreamingConfig, optional
            Streaming configuration
        """
        super().__init__(
            analytical_intention="network analysis", streaming_config=streaming_config
        )
        self.algorithm = algorithm
        self.directed = directed
        self.weighted = weighted
        self.input_format = input_format

        if not _is_networkx_available():
            raise ImportError(
                "NetworkX is required for network optimization. "
                "Please install with: pip install networkx"
            )

        self.graph_ = None
        self.result_ = None
        self.is_fitted_ = False

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "NetworkAnalyzer":
        """Build network from adjacency matrix or edge list."""
        X = check_array(X, accept_sparse=True)

        # Determine input format
        is_adjacency = self.input_format == "adjacency_matrix" or (
            self.input_format == "auto" and X.shape[0] == X.shape[1]
        )

        # Create graph from adjacency matrix
        if is_adjacency:
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
        logger.info(
            f"Network built: {self.graph_.number_of_nodes()} nodes, {self.graph_.number_of_edges()} edges"
        )
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform is not applicable for network analyzer."""
        check_is_fitted(self, "is_fitted_")
        return X

    def analyze_network(self, include_centrality: bool = True) -> NetworkAnalysisResult:
        """Comprehensive network analysis."""
        check_is_fitted(self, "is_fitted_")
        start_time = time.time()

        try:
            # Basic graph properties
            graph_properties = {
                "num_nodes": self.graph_.number_of_nodes(),
                "num_edges": self.graph_.number_of_edges(),
                "is_directed": nx.is_directed(self.graph_),
                "is_connected": nx.is_connected(self.graph_)
                if not nx.is_directed(self.graph_)
                else nx.is_weakly_connected(self.graph_),
                "density": nx.density(self.graph_),
            }

            result = NetworkAnalysisResult(
                graph_properties=graph_properties,
                execution_time=0.0,
                method="comprehensive_analysis",
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
                graph_properties={"error": str(e)},
                execution_time=time.time() - start_time,
                method="failed_analysis",
            )

    def _compute_shortest_paths(self) -> Dict[str, Dict[str, Union[float, List]]]:
        """Compute shortest paths between all pairs of nodes."""
        try:
            weight_attr = "weight" if self.weighted else None

            if self.algorithm == "floyd_warshall" or (
                self.algorithm == "auto" and self.graph_.number_of_nodes() <= 100
            ):
                # Floyd-Warshall for small dense graphs
                paths = dict(
                    nx.floyd_warshall_predecessor_and_distance(
                        self.graph_, weight=weight_attr
                    )
                )
                return {
                    "algorithm": "floyd_warshall",
                    "distances": {
                        str(u): {str(v): dist for v, dist in distances.items()}
                        for u, distances in paths[1].items()
                    },
                    "predecessors": {
                        str(u): {str(v): pred for v, pred in preds.items()}
                        for u, preds in paths[0].items()
                    },
                }
            else:
                # Dijkstra for larger graphs
                nodes = list(self.graph_.nodes())
                distances = {}
                paths = {}

                for source in nodes[: min(10, len(nodes))]:  # Limit for performance
                    try:
                        dist = nx.single_source_dijkstra_path_length(
                            self.graph_, source, weight=weight_attr
                        )
                        path = nx.single_source_dijkstra_path(
                            self.graph_, source, weight=weight_attr
                        )
                        distances[str(source)] = {
                            str(target): d for target, d in dist.items()
                        }
                        paths[str(source)] = {
                            str(target): p for target, p in path.items()
                        }
                    except nx.NetworkXNoPath:
                        continue

                return {"algorithm": "dijkstra", "distances": distances, "paths": paths}

        except Exception as e:
            return {"error": str(e), "algorithm": "failed"}

    def _compute_max_flow(self) -> Tuple[float, Dict]:
        """Compute maximum flow between first and last nodes."""
        try:
            nodes = list(self.graph_.nodes())
            if len(nodes) < 2:
                return 0.0, {}

            source, sink = nodes[0], nodes[-1]
            flow_value, flow_dict = nx.maximum_flow(
                self.graph_, source, sink, capacity="weight"
            )

            return flow_value, {
                "source": source,
                "sink": sink,
                "flow_value": flow_value,
                "flow_edges": flow_dict,
            }

        except Exception as e:
            return 0.0, {"error": str(e)}

    def _compute_mst(self) -> Dict[str, Any]:
        """Compute minimum spanning tree."""
        try:
            weight_attr = "weight" if self.weighted else None
            mst = nx.minimum_spanning_tree(self.graph_, weight=weight_attr)

            return {
                "total_weight": sum(
                    data.get("weight", 1) for _, _, data in mst.edges(data=True)
                ),
                "edges": [
                    (u, v, data.get("weight", 1)) for u, v, data in mst.edges(data=True)
                ],
                "num_edges": mst.number_of_edges(),
            }

        except Exception as e:
            return {"error": str(e)}

    def _solve_tsp(self) -> Dict[str, Any]:
        """Solve traveling salesman problem using heuristics."""
        try:
            nodes = list(self.graph_.nodes())
            if len(nodes) <= 1:
                return {"tour": nodes, "distance": 0.0}

            # Use nearest neighbor heuristic for TSP
            def nearest_neighbor_tsp(graph, start_node):
                unvisited = set(graph.nodes()) - {start_node}
                tour = [start_node]
                current = start_node
                total_distance = 0.0

                while unvisited:
                    # Find nearest unvisited node
                    nearest = None
                    min_distance = float("inf")

                    for node in unvisited:
                        try:
                            if graph.has_edge(current, node):
                                distance = graph[current][node].get("weight", 1)
                            else:
                                distance = float("inf")

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
                    total_distance += self.graph_[current][start_node].get("weight", 1)

                return tour, total_distance

            # Try from first node
            tour, distance = nearest_neighbor_tsp(self.graph_, nodes[0])

            return {
                "tour": tour,
                "distance": distance,
                "algorithm": "nearest_neighbor",
                "is_optimal": False,
            }

        except Exception as e:
            return {"error": str(e)}

    def _compute_centrality(self) -> Dict[str, Dict]:
        """Compute various centrality measures."""
        centrality = {}

        try:
            # Degree centrality
            centrality["degree"] = nx.degree_centrality(self.graph_)

            # Betweenness centrality
            centrality["betweenness"] = nx.betweenness_centrality(self.graph_)

            # Closeness centrality
            centrality["closeness"] = nx.closeness_centrality(self.graph_)

            # Eigenvector centrality (if connected)
            try:
                centrality["eigenvector"] = nx.eigenvector_centrality(
                    self.graph_, max_iter=1000
                )
            except:
                centrality["eigenvector"] = {}

            # PageRank (works for all graphs)
            centrality["pagerank"] = nx.pagerank(self.graph_)

        except Exception as e:
            centrality["error"] = str(e)

        return centrality

    # --- AnalysisPipelineBase abstract method implementations ---

    def get_analysis_type(self) -> str:
        """Get the specific analysis type this pipeline performs."""
        return "network_analysis"

    def _configure_analysis_pipeline(self) -> List[Callable]:
        """Configure analysis steps based on intention and complexity level."""
        return [self.analyze_network]

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
