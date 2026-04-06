"""
Isochrone generation for service area analysis.
"""

import time

from ...logging_manager import get_logger
from ._base import GeospatialLibrary, _dependency_status
from ._network import IsochroneResult, SpatialNetwork

logger = get_logger(__name__)


class IsochroneGenerator:
    """Generate isochrones (equal-time/distance polygons) for service areas."""

    def __init__(self, network: SpatialNetwork):
        self.network = network

    def generate_isochrones(
        self, service_locations, time_bands, resolution=50
    ) -> IsochroneResult:
        """Generate isochrone polygons for service locations."""
        start_time = time.time()

        if not _dependency_status.is_available(GeospatialLibrary.SHAPELY):
            logger.warning("Shapely not available for isochrone polygon generation")
            return IsochroneResult(
                isochrone_polygons=[],
                time_bands=time_bands,
                coverage_areas={},
                population_coverage=None,
                service_points=service_locations,
                execution_time=time.time() - start_time,
            )

        import networkx as nx
        from shapely.geometry import Point
        from shapely.ops import unary_union

        isochrone_polygons = []
        coverage_areas = {}

        for time_threshold in sorted(time_bands):
            band_polygons = []

            for service_node in service_locations:
                reachable_nodes = []

                try:
                    path_lengths = nx.single_source_shortest_path_length(
                        self.network.network,
                        service_node,
                        cutoff=time_threshold,
                        weight="weight",
                    )
                    reachable_nodes = list(path_lengths.keys())
                except Exception:
                    continue

                if len(reachable_nodes) < 3:
                    continue

                reachable_coords = [
                    self.network.node_coordinates[node] for node in reachable_nodes
                ]

                if len(reachable_coords) >= 3:
                    try:
                        points = [Point(coord) for coord in reachable_coords]
                        buffer_distance = time_threshold / 10
                        buffered_points = [
                            point.buffer(buffer_distance) for point in points
                        ]
                        isochrone_polygon = unary_union(buffered_points)
                        band_polygons.append(isochrone_polygon)
                    except Exception as e:
                        logger.warning(f"Failed to create isochrone polygon: {e}")
                        continue

            if band_polygons:
                combined_polygon = unary_union(band_polygons)
                isochrone_polygons.append(combined_polygon)
                coverage_areas[time_threshold] = combined_polygon.area
            else:
                isochrone_polygons.append(None)
                coverage_areas[time_threshold] = 0

        execution_time = time.time() - start_time

        return IsochroneResult(
            isochrone_polygons=isochrone_polygons,
            time_bands=time_bands,
            coverage_areas=coverage_areas,
            population_coverage=None,
            service_points=service_locations,
            execution_time=execution_time,
        )
