# Geospatial Domain

## Overview

The geospatial domain provides spatial analysis capabilities covering distance measurement,
coordinate transformations, spatial joins and overlays, interpolation, network routing, and
spatial statistics. It is built on geopandas, shapely, and pyproj as required dependencies,
with optional enhancement from scikit-gstat (for advanced kriging) and networkx (for network
analysis).

Use this domain when you need to:

- Compute distances between geographic points (Haversine/great-circle or Euclidean)
- Reproject data between coordinate reference systems (e.g. WGS84 to UTM)
- Join point datasets to polygon datasets by spatial relationship
- Interpolate values across a spatial grid from sparse point observations
- Find shortest paths or service area isochrones on a road/graph network
- Measure spatial autocorrelation (clustering vs dispersion) in attribute values

All transformers are sklearn-compatible (`BaseEstimator`, `TransformerMixin`) and can be embedded
in scikit-learn pipelines. High-level convenience functions are also available for direct use.

---

## Available Analyses

| Analysis | Class / Function | Description |
|---|---|---|
| Haversine distance | `SpatialDistanceCalculator.haversine_distance` | Great-circle distance between lat/lon pairs |
| Batch distance matrix | `SpatialDistanceTransformer` | All-pairs or nearest-k distance matrix |
| CRS transformation | `SpatialCoordinateTransformer` | Reproject coordinates between EPSG/PROJ systems |
| Spatial join | `SpatialJoinEngine` / `perform_spatial_join` | Intersect, contains, nearest-neighbour joins |
| Spatial overlay | `SpatialOverlayEngine` / `perform_spatial_overlay` | Union, intersection, difference of polygon layers |
| Point aggregation | `aggregate_points_in_polygons` | Count/summarise points falling within polygons |
| IDW interpolation | `SpatialInterpolator` | Inverse-distance weighted interpolation |
| Kriging interpolation | `SpatialInterpolator` / `VariogramModel` | Variogram-based geostatistical interpolation |
| Moran's I | `SpatialStatistics.morans_i` | Global spatial autocorrelation index |
| Geary's C | `SpatialStatistics` | Alternative spatial autocorrelation statistic |
| Network shortest path | `NetworkRouter` / `optimize_route` | Dijkstra/A* shortest path on a graph |
| Network MST | `NetworkAnalyzer` | Minimum spanning tree |
| Accessibility analysis | `AccessibilityAnalyzer` / `analyze_accessibility` | Distance/time from each node to service locations |
| Service isochrones | `IsochroneGenerator` / `generate_service_isochrones` | Reachable area within time/distance threshold |
| Autocorrelation pipeline | `analyze_spatial_autocorrelation` | Convenience wrapper for Moran's I via pipeline |
| Spatial clustering | `perform_spatial_clustering` | K-means or DBSCAN on geographic coordinates |
| Convex hull / bounding box | `SpatialGeometryTransformer` | Geometric envelope calculations |

---

## MCP Tool Reference

The geospatial domain exposes its functionality through Python API calls. The high-level functions
below are the primary entry points.

### `calculate_spatial_distance`

Compute distances between points in a DataFrame.

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `data` | pd.DataFrame | required | DataFrame with coordinate columns |
| `coordinate_columns` | list[str] | `["x", "y"]` | Names of the x/y (or lon/lat) columns |
| `distance_metrics` | list[str] | `["euclidean"]` | Metrics: `euclidean`, `haversine`, `manhattan` |
| `output_format` | str | `"dataframe"` | Output as `"dataframe"` or `"matrix"` |

**Return format**

Returns a DataFrame with distance columns appended, or a distance matrix.

---

### `perform_spatial_join`

Join two spatial datasets by geometric relationship.

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `left_gdf` | GeoDataFrame | required | Left dataset (points or polygons) |
| `right_gdf` | GeoDataFrame | required | Right dataset to join against |
| `join_type` | str | `"intersects"` | Join predicate: `intersects`, `contains`, `within`, `nearest` |
| `how` | str | `"left"` | Join direction: `left`, `right`, `inner` |
| `distance_threshold` | float | None | Maximum distance for nearest joins |

**Return format**

```python
SpatialJoinResult(
    result_gdf=...,          # GeoDataFrame with joined attributes
    join_count=...,          # Number of matched pairs
    unmatched_left=...,      # Indices with no match
    join_type="intersects",
    metadata={...}
)
```

---

### `perform_spatial_overlay`

Combine two polygon layers with set operations.

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `gdf1` | GeoDataFrame | required | First polygon layer |
| `gdf2` | GeoDataFrame | required | Second polygon layer |
| `operation` | str | `"intersection"` | Operation: `intersection`, `union`, `difference`, `symmetric_difference` |

---

### `analyze_spatial_autocorrelation`

Measure spatial autocorrelation in a variable across a point dataset.

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `data` | pd.DataFrame | required | DataFrame with coordinate and value columns |
| `coordinate_columns` | list[str] | `["x", "y"]` | Coordinate column names |
| `value_column` | str | required | Column to test for spatial pattern |
| `method` | str | `"moran"` | Method: `moran`, `geary` |
| `k_neighbors` | int | `8` | Number of spatial neighbours for weights matrix |

**Return format**

```python
SpatialStatisticsResult(
    statistic="morans_i",
    value=0.42,              # Moran's I: -1 (dispersion) to +1 (clustering)
    p_value=0.001,
    z_score=3.8,
    interpretation="Strong positive spatial autocorrelation detected",
    weights_summary={...}
)
```

---

### `optimize_route`

Find the shortest path between two nodes in a spatial network.

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `network_data` | dict | required | Graph definition with nodes and edges |
| `origin` | str/int | required | Origin node identifier |
| `destination` | str/int | required | Destination node identifier |
| `weight` | str | `"distance"` | Edge attribute to minimise |
| `algorithm` | str | `"dijkstra"` | Algorithm: `dijkstra`, `astar`, `bellman_ford` |

**Return format**

```python
RouteResult(
    path=[node1, node2, ...],
    total_distance=12.4,
    total_time=None,
    segments=[{"from": ..., "to": ..., "distance": ...}],
    metadata={...}
)
```

---

### `generate_service_isochrones`

Compute reachable areas from service locations within a distance/time budget.

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `network_data` | dict | required | Graph definition |
| `service_locations` | list | required | Origin node IDs for service points |
| `thresholds` | list[float] | required | Distance or time budgets for each isochrone band |
| `analysis_type` | str | `"distance"` | `"distance"` or `"time"` |

**Return format**

```python
IsochroneResult(
    service_areas={threshold: [node_ids]},
    coverage_stats={"fraction_covered": 0.72, ...},
    metadata={...}
)
```

---

## Method Details

### Distance Calculations

**Haversine (great-circle) distance** accounts for Earth's curvature and is the correct method
for geographic coordinates in degrees (lat/lon). The `SpatialDistanceCalculator.haversine_distance`
method supports scalar pairs, NumPy arrays (vectorised), and tuple-based calling conventions.

```python
calc = SpatialDistanceCalculator()
dist_km = calc.haversine_distance(lat1=48.85, lon1=2.35, lat2=51.51, lon2=-0.13, unit="km")
```

Supported units: `km`, `m`, `mi`, `nmi`.

**Euclidean distance** is appropriate for projected coordinate systems (e.g., UTM metres) where
distances are small enough that Earth's curvature is negligible (typically <100 km).

**When to choose haversine:** Any time your coordinates are in degrees (EPSG:4326 / WGS84).
**When to choose Euclidean:** Projected coordinates (metres), small geographic areas.

---

### Coordinate Transformations

`SpatialCoordinateTransformer` wraps pyproj to reproject point or GeoDataFrame data between
coordinate reference systems.

```python
transformer = SpatialCoordinateTransformer(
    source_crs="EPSG:4326",   # WGS84 geographic
    target_crs="EPSG:32632",  # UTM Zone 32N
)
projected_df = transformer.fit_transform(df_with_lon_lat)
```

Common CRS codes:

| Code | Description |
|---|---|
| `EPSG:4326` | WGS84 geographic (lat/lon degrees) — GPS default |
| `EPSG:3857` | Web Mercator (metres) — web tiles |
| `EPSG:32632` | UTM Zone 32N (metres) — central Europe |
| `EPSG:27700` | British National Grid |

---

### Spatial Joins

Three join predicates are supported:

| Predicate | Meaning |
|---|---|
| `intersects` | Geometries share any point |
| `contains` | Left geometry fully contains right geometry |
| `within` | Left geometry is fully inside right geometry |
| `nearest` | Joins each left feature to its closest right feature |

For nearest joins, `distance_threshold` limits the maximum allowable distance and prevents
spurious matches when no close neighbour exists.

`aggregate_points_in_polygons` is a convenience function that counts or summarises attribute
values of points falling within each polygon — useful for spatial binning.

---

### Spatial Interpolation

#### IDW (Inverse Distance Weighting)

Estimates values at unsampled locations as a weighted average of nearby observations, with weights
proportional to 1/distance^p. No distributional assumptions; fast.

Use IDW when:
- You need a quick interpolation without stationarity assumptions
- The spatial process is expected to be locally smooth
- You have sparse but reliable point observations

#### Kriging

Kriging is a best linear unbiased estimator that models spatial dependence through a variogram.
It provides interpolated values plus prediction variance (uncertainty estimates).

The `VariogramModel` class fits empirical variograms using one of four theoretical models:

| Model | Shape | Best for |
|---|---|---|
| `spherical` | Gradual flattening with finite range | Most natural phenomena |
| `exponential` | Asymptotic approach to sill | Soil properties, precipitation |
| `gaussian` | Smooth curve, no nugget | Very smooth processes |
| `linear` | No sill | Unbounded processes |

When scikit-gstat is installed, advanced variogram fitting and model selection are available.
Without it, a fallback implementation is used.

**Interpretation:** Low prediction variance indicates the interpolated value is well-constrained
by nearby observations. High variance flags locations far from data points.

---

### Spatial Statistics / Autocorrelation

#### Moran's I

Global statistic measuring whether similar values cluster together spatially.

- I near +1: values cluster (hot-spots and cold-spots)
- I near 0: random spatial arrangement (no spatial pattern)
- I near -1: values are dispersed (chess-board pattern)

The test uses a normal approximation. `p_value` from the z-score tests the null hypothesis
of complete spatial randomness.

#### Geary's C

Complementary to Moran's I, Geary's C uses squared differences between neighbouring values.
C = 1 means no spatial autocorrelation; C < 1 indicates positive autocorrelation.

**Spatial weights matrix methods:** `knn` (k-nearest neighbours, default `k=8`) or `distance_band`
(all neighbours within a threshold distance).

---

### Network Analysis

NetworkX is required for network operations. Check availability with `NETWORKX_AVAILABLE`.

**Shortest paths:** Dijkstra's algorithm (non-negative weights), Bellman-Ford (handles negative
weights), or A* (with heuristic, faster on geographic networks).

**Minimum spanning tree:** Kruskal's or Prim's algorithm. Returns the subgraph connecting all
nodes with minimum total edge weight.

**Accessibility analysis:** For each node, computes distance/cost to the nearest service location.
Results show coverage statistics (fraction of nodes within each threshold band).

**Isochrones:** Service areas computed by expanding from service nodes until the budget is
exhausted. Returns the set of reachable nodes per threshold.

**Centrality measures** (when `include_centrality=True`):

| Measure | Interpretation |
|---|---|
| Degree centrality | Fraction of nodes connected to this node |
| Betweenness centrality | How often a node lies on shortest paths between other pairs |
| Closeness centrality | Mean inverse distance to all other nodes |
| Eigenvector centrality | Influence score weighted by neighbour influence |

---

## Composition

| After geospatial analysis | Chain to | Purpose |
|---|---|---|
| Distance matrix | Regression/Modeling | Spatial lag features for prediction |
| Interpolated grid | Statistical Analysis | Test distribution of interpolated surface |
| Moran's I result | Pattern Recognition | Cluster detection based on confirmed autocorrelation |
| Isochrone coverage | Business Intelligence | Segment customers by accessibility |
| Spatial join result | Statistical Analysis | Compare attribute distributions across spatial zones |

---

## Examples

### Compute haversine distances between store locations and customers

```python
from localdata_mcp.domains.geospatial_analysis import SpatialDistanceCalculator

calc = SpatialDistanceCalculator()
distances = calc.haversine_distance(
    lat1=customer_df["lat"].values,
    lon1=customer_df["lon"].values,
    lat2=store_lat,
    lon2=store_lon,
    unit="km",
)
```

### Run autocorrelation analysis via pipeline

```python
result = analyze_spatial_autocorrelation(
    data=df,
    coordinate_columns=["longitude", "latitude"],
    value_column="house_price",
    method="moran",
    k_neighbors=6,
)
print(f"Moran's I = {result.value:.3f}, p = {result.p_value:.4f}")
print(result.interpretation)
```

### Spatial join: assign census polygon attributes to point data

```python
from localdata_mcp.domains.geospatial_analysis import perform_spatial_join

joined = perform_spatial_join(
    left_gdf=customer_points,
    right_gdf=census_polygons,
    join_type="intersects",
    how="left",
)
# Each customer point now carries attributes from the census polygon it falls within
```

### Interpolate sensor readings onto a regular grid

```python
from localdata_mcp.domains.geospatial_analysis import SpatialInterpolator

interpolator = SpatialInterpolator(method="kriging", variogram_model="spherical")
result = interpolator.interpolate(
    points=sensor_locations,     # list of (x, y) tuples
    values=sensor_readings,      # np.ndarray of observed values
    grid_resolution=100,
)
# result.interpolated_values: grid of estimated values
# result.prediction_variance: uncertainty per grid cell
```

### Route optimisation on a delivery network

```python
route = optimize_route(
    network_data={"nodes": node_list, "edges": edge_list},
    origin="depot_A",
    destination="customer_42",
    weight="travel_time_minutes",
)
print(f"Route: {route.path}, total time: {route.total_distance} min")
```
