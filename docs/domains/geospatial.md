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

The domain is reached through the ten MCP tools listed below. The classes named
in the table that follows are the internal implementation those tools call — they
are sklearn-compatible (`BaseEstimator`, `TransformerMixin`) and usable from
Python, but an MCP client never sees them.

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

The geospatial domain exposes ten MCP tools. Like every other analytical tool,
each takes the name of a live connection and one or more SQL queries — there is
no data-frame parameter and no separate load step. The classes listed under
*Available Analyses* above are the internal implementation those tools call;
they are not reachable from an MCP client.

Spatial data arrives from SQL in one of three shapes, and the parameters follow
that split:

| Shape | How it is given | Tools |
|---|---|---|
| Points | `x_column` and `y_column` (default `x`, `y`) | autocorrelation, hotspots, distances |
| Geometries | a WKT text column (default `geometry`) | join, overlay, aggregation |
| Networks | a nodes query and an edges query | route, accessibility, isochrones |

Full parameter tables for all ten live in the
[tools reference](../tools-reference.md#geospatial-10-tools). This page covers
what each tool is for and when to reach for it.

### `check_geospatial_capabilities`

Reports which backends are installed. geopandas, shapely, pyproj and fiona are
required dependencies; scikit-gstat (advanced kriging) and rasterio (raster
data) are optional, so call this before an analysis that needs one. It is the
only tool in the server that takes no connection.

### `analyze_spatial_autocorrelation`

Answers "is this pattern clustered, dispersed, or random?" — the question to
settle before treating geography as meaningful in a model. Returns Moran's I
(`method="moran"`) or Geary's C (`method="geary"`) with its expected value,
variance, z-score, p-value and an interpretation. Clustering pushes Moran's I
above its expected value and Geary's C below 1.

### `find_spatial_hotspots`

Answers "where are the clusters?" — a Getis-Ord Gi* analysis labelling each
location a hot spot, a cold spot, or not significant, with its statistic and
p-value. A hot spot is a location *surrounded by* unusually high values, which
is not the same as a high value: a lone peak among low neighbours is not one.
Run `analyze_spatial_autocorrelation` first; without significant global
autocorrelation, local clusters are noise.

### `calculate_spatial_distances`

Answers "how far apart is everything, and what is each point's nearest
neighbour?" Returns a summary and per-point nearest neighbours rather than the
matrix itself, which grows with the square of the row count; pass
`include_pairs=True` for individual pairs. Use `distance_type="haversine"` for
longitude/latitude, which reports kilometres — treating degrees as a plane
understates distance badly away from the equator.

### `optimize_route`

Answers "what is the cheapest way to visit all of these?" Orders the waypoints
and connects them across the network, returning the node path, its coordinates,
the total cost and the route as a WKT `LINESTRING`.

### `analyze_accessibility`

Answers "who can reach a facility, how quickly, and who is left out?" Returns a
score and travel time per demand point plus an explicit
`unreachable_locations` list — the answer a coverage percentage hides.

### `generate_service_isochrones`

Answers "how much ground does this site cover?" Returns one polygon per
travel-time band as WKT, with its area. Compare areas between bands to find
where coverage stops growing, or between candidate sites to choose one.

### `perform_spatial_join`

Answers "which zone is each point in?" Either side may be given as WKT
geometries or, for point data, as a coordinate pair — point tables rarely carry
a WKT column, and joining points to zones is the commonest spatial join there
is. Left and right columns come back suffixed `_left` and `_right`.

### `perform_spatial_overlay`

Answers "where do these two areas overlap, and where does one exclude the
other?" Both sides must supply WKT geometries, since a set operation on points
is not meaningful. Supports `intersection`, `union` and `difference`.

### `aggregate_points_in_polygons`

Answers "what is the average reading per district?" — the spatial equivalent of
a GROUP BY, and the usual bridge from point measurements into the Data Science
tools, which then work on the resulting per-polygon table. Polygons containing
no points are kept with a count of zero rather than dropped.

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

Every example below is an MCP tool call, the way an agent would issue it.

### Are house prices spatially clustered?

```python
analyze_spatial_autocorrelation(
    "listings", "SELECT lon, lat, price FROM properties",
    value_column="price", x_column="lon", y_column="lat",
)
```

A Moran's I well above its expected value with a p-value below 0.05 means price
depends on location, and a model that ignores geography will be biased.

### Where are the expensive neighbourhoods?

```python
find_spatial_hotspots(
    "listings", "SELECT lon, lat, price FROM properties",
    value_column="price", x_column="lon", y_column="lat",
)
```

Each returned point carries a `cluster_id` of 1, -1 or 0. Run this only after
the autocorrelation test above comes back significant.

### How far is each customer from the nearest store?

```python
calculate_spatial_distances(
    "retail", "SELECT lon, lat FROM customers",
    x_column="lon", y_column="lat", distance_type="haversine",
    reference_query="SELECT lon, lat FROM stores",
)
```

`haversine` reports kilometres. The `summary` gives the distribution; the
per-point nearest neighbour identifies the closest store to each customer.

### Which sales region does each customer fall in?

```python
perform_spatial_join(
    "retail",
    "SELECT customer_id, lon, lat FROM customers",
    "SELECT region_name, geometry FROM sales_regions",
    x_column="lon", y_column="lat",
)
```

The left side is point data given as coordinates and the right side is WKT
polygons; each side is read in whichever form its query provides.

### What is average revenue per region?

```python
aggregate_points_in_polygons(
    "retail",
    "SELECT lon, lat, revenue FROM customers",
    "SELECT region_name, geometry FROM sales_regions",
    value_column="revenue", x_column="lon", y_column="lat",
)
```

The result is one row per region, ready to feed into `analyze_hypothesis_test`
or `analyze_regression` as an ordinary table.

### What is the cheapest delivery round?

```python
optimize_route(
    "logistics",
    "SELECT id, x, y FROM depots",
    "SELECT source, target, travel_minutes FROM roads",
    waypoints=[3, 17, 42],
    weight_column="travel_minutes",
    return_to_start=True,
)
```

### Who is more than 20 minutes from a clinic?

```python
analyze_accessibility(
    "health",
    "SELECT id, x, y FROM junctions",
    "SELECT source, target, travel_minutes FROM roads",
    service_locations=[11, 58],
    demand_locations=[2, 7, 19, 33],
    weight_column="travel_minutes",
    max_travel_time=20,
)
```

`unreachable_locations` names them directly, rather than leaving them implied by
a coverage percentage.

### How much ground would a new site cover?

```python
generate_service_isochrones(
    "health",
    "SELECT id, x, y FROM junctions",
    "SELECT source, target, travel_minutes FROM roads",
    service_locations=[58],
    time_bands=[10, 20, 30],
    weight_column="travel_minutes",
)
```

Each band comes back as a WKT polygon with its area, so two candidate sites can
be compared band for band.
