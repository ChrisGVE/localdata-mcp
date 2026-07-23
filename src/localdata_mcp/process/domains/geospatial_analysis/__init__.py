"""localdata_mcp/process/domains/geospatial_analysis — FR-301's geo family.

The extras-tier geospatial domain: ten tools carried by name from
`main` (tools.py declares them). The whole family depends on the
opt-in `geospatial` extra (geopandas/shapely/pyproj); the single
capability probe answers unconditionally, and every other tool refuses
with a structured "install the geospatial extra" message when the
native stack is absent (support.py's dependency guard). Spatial
statistics use a k-NN weights matrix over the addressed coordinates;
geometry and network tools take their second input as inline WKT /
edge lists so the X-2 single-source addressing contract holds.
"""
