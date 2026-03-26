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

templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "architecture",
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
