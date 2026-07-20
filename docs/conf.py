"""Sphinx configuration for LocalData MCP documentation."""

project = "LocalData MCP"
author = "Christian C. Berclaz"
copyright = "2025, Christian C. Berclaz"

extensions = [
    "myst_parser",
    "sphinx_copybutton",
]

myst_enable_extensions = [
    "colon_fence",
    "fieldlist",
    "deflist",
]

# Generate anchors for h1-h3 so cross-page links can target a section by its
# heading, which docs/domains/geospatial.md already assumed was possible.
myst_heading_anchors = 3

templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    # The architecture section publishes its index and the two normative design
    # documents. The four below propose machinery that was never built, so they
    # stay in the repository for their reasoning but out of the shipped docs,
    # where every page is expected to describe the current system.
    "architecture/CORE_PATTERNS.md",
    "architecture/LIBRARY_STRATEGY.md",
    "architecture/domain-integration-layer.md",
    "architecture/integration-shims-architecture.md",
    "integration",
    "BACKWARD_COMPATIBILITY.md",
    "PERFORMANCE_BENCHMARKS.md",
    "TIMEOUT_SYSTEM.md",
    "core-pipeline-framework-design.md",
    "streaming-integration-architecture.md",
    "time_series_analysis_guide.md",
]

html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "navigation_depth": 3,
    "collapse_navigation": False,
}
html_static_path = ["_static"]
html_logo = "../assets/logo.png"

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
