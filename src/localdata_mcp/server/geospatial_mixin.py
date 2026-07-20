"""MCP tool methods for the geospatial domain.

Thin wrappers registered as MCP tools: resolve the named connection, delegate to
the adapter in :mod:`localdata_mcp.geospatial_tools`, and serialize the result.
Kept in a mixin so ``database_manager`` does not grow with each new domain.

Spatial data reaches these tools in one of three shapes, and the parameters
follow that split: point tables give a pair of coordinate columns, geometry
tables give a WKT text column, and networks give a node query plus an edge query.
"""

from typing import Any, List, Optional

from ..geospatial_tools import (
    tool_aggregate_points_in_polygons,
    tool_analyze_accessibility,
    tool_analyze_spatial_autocorrelation,
    tool_calculate_spatial_distances,
    tool_check_geospatial_capabilities,
    tool_find_spatial_hotspots,
    tool_generate_service_isochrones,
    tool_optimize_route,
    tool_perform_spatial_join,
    tool_perform_spatial_overlay,
)
from ..json_utils import safe_dumps


class GeospatialToolsMixin:
    """Geospatial MCP tools, mixed into ``DatabaseManager``."""

    def check_geospatial_capabilities(self) -> str:
        """Report which geospatial backends are installed and what they enable.

        Takes no connection. Call it to find out whether an analysis is
        supported before attempting it — kriging needs scikit-gstat and raster
        work needs rasterio, neither of which is a required dependency.
        """
        return safe_dumps(tool_check_geospatial_capabilities())

    def analyze_spatial_autocorrelation(
        self,
        connection_name: str,
        query: str,
        value_column: str,
        x_column: str = "x",
        y_column: str = "y",
        method: str = "moran",
        k_neighbors: int = 8,
    ) -> str:
        """Test whether nearby locations hold similar values.

        Answers "is this pattern clustered, dispersed, or random?" — the question
        to settle before treating geography as meaningful in a model.

        Args:
            connection_name: Name of the connected database.
            query: SQL returning the coordinate columns and the value column.
            value_column: Numeric column whose spatial pattern is tested.
            x_column: Longitude or easting column (default 'x').
            y_column: Latitude or northing column (default 'y').
            method: 'moran' for Moran's I or 'geary' for Geary's C.
            k_neighbors: Neighbours defining each location's neighbourhood.
        """
        engine = self._get_connection(connection_name)
        return safe_dumps(
            tool_analyze_spatial_autocorrelation(
                engine,
                query,
                value_column,
                x_column=x_column,
                y_column=y_column,
                method=method,
                k_neighbors=k_neighbors,
            )
        )

    def find_spatial_hotspots(
        self,
        connection_name: str,
        query: str,
        value_column: str,
        x_column: str = "x",
        y_column: str = "y",
        significance_level: float = 0.05,
    ) -> str:
        """Locate statistically significant hot and cold spots.

        Runs a Getis-Ord Gi* analysis and labels each location a hot spot
        (surrounded by unusually high values), a cold spot, or not significant.

        Args:
            connection_name: Name of the connected database.
            query: SQL returning the coordinate columns and the value column.
            value_column: Numeric column whose local clustering is tested.
            x_column: Longitude or easting column (default 'x').
            y_column: Latitude or northing column (default 'y').
            significance_level: p-value threshold for calling a spot (default 0.05).
        """
        engine = self._get_connection(connection_name)
        return safe_dumps(
            tool_find_spatial_hotspots(
                engine,
                query,
                value_column,
                x_column=x_column,
                y_column=y_column,
                significance_level=significance_level,
            )
        )

    def calculate_spatial_distances(
        self,
        connection_name: str,
        query: str,
        x_column: str = "x",
        y_column: str = "y",
        distance_type: str = "euclidean",
        reference_query: str = "",
        include_pairs: bool = False,
    ) -> str:
        """Measure how far apart points are, and what each one's nearest is.

        Returns a summary and per-point nearest neighbours rather than the whole
        matrix, which grows with the square of the row count.

        Args:
            connection_name: Name of the connected database.
            query: SQL returning the coordinate columns.
            x_column: Longitude or easting column (default 'x').
            y_column: Latitude or northing column (default 'y').
            distance_type: 'euclidean' in coordinate units, or 'haversine' in
                kilometres for longitude/latitude pairs.
            reference_query: Optional SQL for a second set; distances are then
                measured from the first set to this one.
            include_pairs: Also return every individual pair (capped).
        """
        engine = self._get_connection(connection_name)
        return safe_dumps(
            tool_calculate_spatial_distances(
                engine,
                query,
                x_column=x_column,
                y_column=y_column,
                distance_type=distance_type,
                reference_query=reference_query or None,
                include_pairs=include_pairs,
            )
        )

    def optimize_route(
        self,
        connection_name: str,
        nodes_query: str,
        edges_query: str,
        waypoints: List[Any],
        node_id_column: str = "id",
        x_column: str = "x",
        y_column: str = "y",
        source_column: str = "source",
        target_column: str = "target",
        weight_column: str = "",
        return_to_start: bool = False,
        optimization_method: str = "greedy",
    ) -> str:
        """Order and connect waypoints into the cheapest route across a network.

        Args:
            connection_name: Name of the connected database.
            nodes_query: SQL returning one node per row with an id and coordinates.
            edges_query: SQL returning one edge per row with source and target ids.
            waypoints: Node ids the route must visit, in any order.
            node_id_column: Node identifier column (default 'id').
            x_column: Node longitude or easting column (default 'x').
            y_column: Node latitude or northing column (default 'y').
            source_column: Edge source node column (default 'source').
            target_column: Edge target node column (default 'target').
            weight_column: Edge cost column; defaults to straight-line distance.
            return_to_start: Close the route back to its first waypoint.
            optimization_method: Waypoint ordering strategy (default 'greedy').
        """
        engine = self._get_connection(connection_name)
        return safe_dumps(
            tool_optimize_route(
                engine,
                nodes_query,
                edges_query,
                waypoints,
                node_id_column=node_id_column,
                x_column=x_column,
                y_column=y_column,
                source_column=source_column,
                target_column=target_column,
                weight_column=weight_column or None,
                return_to_start=return_to_start,
                optimization_method=optimization_method,
            )
        )

    def analyze_accessibility(
        self,
        connection_name: str,
        nodes_query: str,
        edges_query: str,
        service_locations: List[Any],
        demand_locations: List[Any],
        node_id_column: str = "id",
        x_column: str = "x",
        y_column: str = "y",
        source_column: str = "source",
        target_column: str = "target",
        weight_column: str = "",
        max_travel_time: Optional[float] = None,
        impedance_function: str = "linear",
    ) -> str:
        """Score how well a set of service points covers a set of demand points.

        Answers "who can reach a facility, how quickly, and who is left out?"

        Args:
            connection_name: Name of the connected database.
            nodes_query: SQL returning one node per row with an id and coordinates.
            edges_query: SQL returning one edge per row with source and target ids.
            service_locations: Node ids where the service sits.
            demand_locations: Node ids that need to reach it.
            node_id_column: Node identifier column (default 'id').
            x_column: Node longitude or easting column (default 'x').
            y_column: Node latitude or northing column (default 'y').
            source_column: Edge source node column (default 'source').
            target_column: Edge target node column (default 'target').
            weight_column: Edge travel-cost column; defaults to distance.
            max_travel_time: Cost beyond which a demand point counts unreachable.
            impedance_function: How cost decays into a score (default 'linear').
        """
        engine = self._get_connection(connection_name)
        return safe_dumps(
            tool_analyze_accessibility(
                engine,
                nodes_query,
                edges_query,
                service_locations,
                demand_locations,
                node_id_column=node_id_column,
                x_column=x_column,
                y_column=y_column,
                source_column=source_column,
                target_column=target_column,
                weight_column=weight_column or None,
                max_travel_time=max_travel_time,
                impedance_function=impedance_function,
            )
        )

    def generate_service_isochrones(
        self,
        connection_name: str,
        nodes_query: str,
        edges_query: str,
        service_locations: List[Any],
        time_bands: List[float],
        node_id_column: str = "id",
        x_column: str = "x",
        y_column: str = "y",
        source_column: str = "source",
        target_column: str = "target",
        weight_column: str = "",
        resolution: int = 50,
    ) -> str:
        """Draw the area reachable from service points within each travel-time band.

        Each band returns a polygon as WKT plus its area, so coverage can be
        compared between bands or between candidate sites.

        Args:
            connection_name: Name of the connected database.
            nodes_query: SQL returning one node per row with an id and coordinates.
            edges_query: SQL returning one edge per row with source and target ids.
            service_locations: Node ids the service operates from.
            time_bands: Travel-cost cutoffs, one polygon per band.
            node_id_column: Node identifier column (default 'id').
            x_column: Node longitude or easting column (default 'x').
            y_column: Node latitude or northing column (default 'y').
            source_column: Edge source node column (default 'source').
            target_column: Edge target node column (default 'target').
            weight_column: Edge travel-cost column; defaults to distance.
            resolution: Polygon smoothness (default 50).
        """
        engine = self._get_connection(connection_name)
        return safe_dumps(
            tool_generate_service_isochrones(
                engine,
                nodes_query,
                edges_query,
                service_locations,
                time_bands,
                node_id_column=node_id_column,
                x_column=x_column,
                y_column=y_column,
                source_column=source_column,
                target_column=target_column,
                weight_column=weight_column or None,
                resolution=resolution,
            )
        )

    def perform_spatial_join(
        self,
        connection_name: str,
        left_query: str,
        right_query: str,
        left_geometry_column: str = "geometry",
        right_geometry_column: str = "geometry",
        join_type: str = "intersects",
        how: str = "inner",
        x_column: str = "x",
        y_column: str = "y",
        crs: str = "",
    ) -> str:
        """Attach the attributes of one geometry set to another by spatial relation.

        Answers "which zone is each point in?" Either side may be given as WKT
        geometries or, for point data, as a pair of coordinate columns.

        Args:
            connection_name: Name of the connected database.
            left_query: SQL for the features that receive attributes.
            right_query: SQL for the features that supply them.
            left_geometry_column: WKT column on the left (default 'geometry').
            right_geometry_column: WKT column on the right (default 'geometry').
            join_type: Spatial relation: 'intersects', 'within', 'contains' or
                'nearest'.
            how: 'inner' to keep matches only, 'left' to keep every left feature.
            x_column: Coordinate column used when there is no WKT column.
            y_column: Coordinate column used when there is no WKT column.
            crs: Coordinate reference system (default 'EPSG:4326').
        """
        engine = self._get_connection(connection_name)
        return safe_dumps(
            tool_perform_spatial_join(
                engine,
                left_query,
                right_query,
                left_geometry_column=left_geometry_column,
                right_geometry_column=right_geometry_column,
                join_type=join_type,
                how=how,
                x_column=x_column,
                y_column=y_column,
                crs=crs or None,
            )
        )

    def perform_spatial_overlay(
        self,
        connection_name: str,
        left_query: str,
        right_query: str,
        operation: str = "intersection",
        left_geometry_column: str = "geometry",
        right_geometry_column: str = "geometry",
        crs: str = "",
    ) -> str:
        """Combine two geometry sets with a set operation.

        Answers "where do these two areas overlap, and where does one exclude
        the other?" Both sides must supply WKT geometries.

        Args:
            connection_name: Name of the connected database.
            left_query: SQL returning the left geometries as WKT.
            right_query: SQL returning the right geometries as WKT.
            operation: 'intersection', 'union' or 'difference'.
            left_geometry_column: WKT column on the left (default 'geometry').
            right_geometry_column: WKT column on the right (default 'geometry').
            crs: Coordinate reference system (default 'EPSG:4326').
        """
        engine = self._get_connection(connection_name)
        return safe_dumps(
            tool_perform_spatial_overlay(
                engine,
                left_query,
                right_query,
                operation=operation,
                left_geometry_column=left_geometry_column,
                right_geometry_column=right_geometry_column,
                crs=crs or None,
            )
        )

    def aggregate_points_in_polygons(
        self,
        connection_name: str,
        point_query: str,
        polygon_query: str,
        value_column: str,
        x_column: str = "x",
        y_column: str = "y",
        polygon_geometry_column: str = "geometry",
        aggregation_functions: Optional[List[str]] = None,
        crs: str = "",
    ) -> str:
        """Summarise a point measurement within each polygon that contains it.

        Answers "what is the average reading per district?" Polygons with no
        points come back with a count of zero rather than being dropped.

        Args:
            connection_name: Name of the connected database.
            point_query: SQL returning the points and the value column.
            polygon_query: SQL returning the polygons as WKT.
            value_column: Numeric point column to summarise.
            x_column: Point longitude or easting column (default 'x').
            y_column: Point latitude or northing column (default 'y').
            polygon_geometry_column: WKT column on the polygons (default 'geometry').
            aggregation_functions: Any of 'mean', 'sum', 'count', 'min', 'max',
                'median', 'std' (default mean, sum and count).
            crs: Coordinate reference system (default 'EPSG:4326').
        """
        engine = self._get_connection(connection_name)
        return safe_dumps(
            tool_aggregate_points_in_polygons(
                engine,
                point_query,
                polygon_query,
                value_column,
                x_column=x_column,
                y_column=y_column,
                polygon_geometry_column=polygon_geometry_column,
                aggregation_functions=aggregation_functions,
                crs=crs or None,
            )
        )
