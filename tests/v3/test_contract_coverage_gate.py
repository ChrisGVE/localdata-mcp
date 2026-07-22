"""tests/v3/test_contract_coverage_gate.py — E3.6's failing-stub demonstration.

FR-702's always-on L3-presence gate, demonstrated end to end: a
ToolSpec registered AFTER the stub module was generated makes the
generated coverage check FAIL. The stale stub module is rendered from
a smaller registry, loaded from a scratch file with its registry
lookups pointed at the larger one, and its own coverage test is
invoked — the assertion it raises names the uncovered tool.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

# Imported BEFORE any registry monkeypatching: the rendered stub module
# imports mcp_app, whose first import registers the real tool population
# against the real registry — priming the module cache here keeps that
# registration out of the patched-registry window below.
import localdata_mcp.server.mcp_app  # noqa: F401
from localdata_mcp.nexus.contract.generators.test_stub import render_test_module
from localdata_mcp.nexus.contract.registry import ToolRegistry
from localdata_mcp.nexus.contract.spec import Param, ToolSpec, TypeShape


def _spec(name: str) -> ToolSpec:
    return ToolSpec(
        name=name,
        summary=f"Walking-skeleton probe {name}.",
        params=(Param("text", str, "probe payload"),),
        input_shape=TypeShape.NONE,
        output_shape=TypeShape.SCALAR,
        func=lambda text: text,
    )


def _load_stale_stub_module(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, registry: ToolRegistry
) -> ModuleType:
    """Import a rendered stub module with its lookups pointed at `registry`."""
    import localdata_mcp.nexus.contract.registry as registry_module
    import localdata_mcp.nexus.contract.spec_modules as roster_module

    monkeypatch.setattr(registry_module, "default_registry", lambda: registry)
    monkeypatch.setattr(roster_module, "load_spec_modules", lambda: None)
    module_path = tmp_path / "stale_contract_stub.py"
    spec = importlib.util.spec_from_file_location("stale_contract_stub", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["stale_contract_stub"] = module
    try:
        spec.loader.exec_module(module)
    finally:
        del sys.modules["stale_contract_stub"]
    return module


def test_late_registered_spec_fails_the_generated_coverage_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    generated_from = ToolRegistry()
    generated_from.register(_spec("covered_tool"))
    (tmp_path / "stale_contract_stub.py").write_text(
        render_test_module(generated_from), encoding="utf-8"
    )

    live = ToolRegistry()
    live.register(_spec("covered_tool"))
    live.register(_spec("late_registered_tool"))  # never regenerated

    module = _load_stale_stub_module(tmp_path, monkeypatch, live)
    with pytest.raises(AssertionError, match="late_registered_tool"):
        module.test_every_registered_spec_has_a_generated_entry()


def test_regenerated_stub_covers_the_late_spec(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    live = ToolRegistry()
    live.register(_spec("covered_tool"))
    live.register(_spec("late_registered_tool"))
    (tmp_path / "stale_contract_stub.py").write_text(
        render_test_module(live), encoding="utf-8"
    )

    module = _load_stale_stub_module(tmp_path, monkeypatch, live)
    module.test_every_registered_spec_has_a_generated_entry()
