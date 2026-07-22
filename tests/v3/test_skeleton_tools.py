"""tests/v3/test_skeleton_tools.py — E3.4/E3.6: walking-skeleton ToolSpecs.

Pins server/skeleton_tools.py (trivial pure tools proving the NX-1
pipeline end to end) and nexus/contract/spec_modules.py (the one
declared list of spec-registering modules generate.py and the
generated wrapper both import). Every non-DYNAMIC TypeShape must be
covered by the skeleton set (E3.6 / FR-701/704 acceptance floor).
"""

from __future__ import annotations

from localdata_mcp.nexus.contract.registry import default_registry
from localdata_mcp.nexus.contract.spec import TypeShape
from localdata_mcp.nexus.contract.spec_modules import (
    SPEC_MODULES,
    load_spec_modules,
)


class TestSpecModules:
    def test_skeleton_module_is_declared(self) -> None:
        assert "localdata_mcp.server.skeleton_tools" in SPEC_MODULES

    def test_load_is_idempotent(self) -> None:
        load_spec_modules()
        before = len(default_registry())
        load_spec_modules()
        assert len(default_registry()) == before


class TestPingSpec:
    def test_ping_registered_as_source_scalar(self) -> None:
        load_spec_modules()
        spec = default_registry().lookup("ping")
        assert spec.input_shape is TypeShape.NONE
        assert spec.output_shape is TypeShape.SCALAR
        assert spec.func() == "pong"
