"""tests/v3/test_layout.py — asserts the v3 package layout exists per ARCHITECTURE.md section 9.

Every new v3 subpackage must be importable and carry a context docstring
saying what lives there. Legacy packages (server/, domains/, pipeline/, ...)
are deliberately not covered: they are excluded from new-code gates until E15.
"""

import importlib

import pytest

# The new v3 packages scaffolded beside the legacy tree (ARCHITECTURE.md section 9).
V3_PACKAGES = [
    "localdata_mcp.nexus",
    "localdata_mcp.nexus.contract",
    "localdata_mcp.nexus.config",
    "localdata_mcp.nexus.error",
    "localdata_mcp.nexus.observability",
    "localdata_mcp.nexus.persistence",
    "localdata_mcp.nexus.chokepoint",
    "localdata_mcp.nexus.response",
    "localdata_mcp.nexus.export",
    "localdata_mcp.ingest",
    "localdata_mcp.ingest.connectors",
    "localdata_mcp.ingest.connectors.sql",
    "localdata_mcp.ingest.connectors.file",
    "localdata_mcp.ingest.connectors.kv",
    "localdata_mcp.ingest.connectors.graph_tree",
    "localdata_mcp.explore",
    "localdata_mcp.process",
    "localdata_mcp.process.domains",
    "localdata_mcp.process.composition",
    "localdata_mcp.process.preprocessing",
    "localdata_mcp.visualize",
    "localdata_mcp.visualize.charts",
    "localdata_mcp.visualize.render",
    "localdata_mcp.testbench",
    "localdata_mcp.testbench.fixtures",
    "localdata_mcp.testbench.batteries",
    "localdata_mcp.testbench.batteries.base",
    "localdata_mcp.testbench.batteries.security",
    "localdata_mcp.testbench.batteries.domain",
    "localdata_mcp.testbench.batteries.pipeline",
    "localdata_mcp.testbench.batteries.perf_memory",
    "localdata_mcp.testbench.results_store",
]


@pytest.mark.parametrize("package_name", V3_PACKAGES)
def test_package_importable_with_context_docstring(package_name: str) -> None:
    """Each v3 package imports cleanly and documents its area of concern."""
    module = importlib.import_module(package_name)
    assert module.__doc__, f"{package_name} must carry a context docstring"
    assert module.__doc__.strip(), f"{package_name} docstring must not be blank"
