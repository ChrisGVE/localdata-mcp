---
name: geospatial
description: Analyze spatial patterns, compute distances, detect geographic clusters, and assess accessibility. Use when data has coordinates or geographic dimensions.
allowed-tools: mcp__localdata__connect_database mcp__localdata__describe_database mcp__localdata__execute_query mcp__localdata__get_data_quality_report mcp__localdata__describe_table mcp__localdata__analyze_clusters
argument-hint: "<database-name>"
---

# Geospatial Analysis

Analyze spatial patterns, distances, and geographic relationships in location-based data.

## Steps

1. **Identify spatial columns.** Call `describe_database` with the database name from `$ARGUMENTS`. Look for columns containing coordinates (latitude/longitude, x/y), addresses, ZIP codes, or geometry fields. Call `describe_table` for detailed column inspection.

2. **Validate coordinates.** Call `execute_query` to check coordinate ranges: latitude should be -90 to 90, longitude -180 to 180. Flag nulls, zeros (often default values, not actual locations), and points in unexpected regions. Report the geographic extent of the data.

3. **Assess spatial distribution.** Call `execute_query` to compute basic spatial statistics: centroid (mean lat/lon), spread (standard deviation of coordinates), and bounding box. Determine whether points are concentrated in a small area or spread across a wide region.

4. **Compute distances.** Calculate distances between points of interest using the haversine formula (suitable for lat/lon data). Call `execute_query` with distance computations to find nearest neighbors, average spacing, and distance distributions.

5. **Detect spatial clusters.** Use `analyze_clusters` with DBSCAN and geographic distance to identify spatial hotspots. Alternatively, compute density by gridding the area and counting points per cell. Report cluster locations, sizes, and any relationship to attributes.

6. **Cross-reference with attributes.** Call `execute_query` to analyze how non-spatial attributes vary across geographic clusters or regions. Spatial patterns are most valuable when they correlate with other variables.

7. **Present results.** Provide:
   - Spatial data summary: coordinate system, extent, point count, coverage
   - Distribution pattern: clustered, dispersed, or random
   - Key spatial clusters with locations and characterization
   - Distance statistics relevant to the analysis question
   - Geographic insights tied to the domain context

8. **Recommend next steps.** Suggest:
   - `/localdata-mcp:find-reference-data` to add geographic context (boundaries, demographics)
   - `/localdata-mcp:analyze-correlations` for spatial vs. attribute relationships
   - `/localdata-mcp:forecast` for spatiotemporal patterns
