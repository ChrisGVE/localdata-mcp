"""tests/v3/test_response_shaping.py — E7.2 wrapper seam + S3.3 sentinel.

The exit-gate assertions for shaping.py/sentinel.py: every shaped call
returns the four-region wire envelope (never a bare payload or
traceback — the T12 defect class), each of the four sentinel signal
classes converts to a structured NX-3 error envelope (never a silent
success), guard failures pass their already-shaped NX-3 form through,
and configure_shaping swaps the live NX-2 model in.
"""

from __future__ import annotations

from typing import Any

import pytest

from localdata_mcp.nexus.chokepoint.guard import GuardedExecutionError
from localdata_mcp.nexus.config.models import (
    ConfigModel,
    ProcessConfig,
    ResponseConfig,
)
from localdata_mcp.nexus.contract.registry import ToolRegistry
from localdata_mcp.nexus.contract.spec import Param, ToolSpec, TypeShape
from localdata_mcp.nexus.error.model import ErrorType, StructuredError
from localdata_mcp.nexus.response import shaping
from localdata_mcp.nexus.response.sentinel import inspect
from localdata_mcp.nexus.response.shaping import configure_shaping, shaped_call

_KAPPA_MAX = 1e6  # deliberately non-S8 test threshold


def _spec(name: str) -> ToolSpec:
    return ToolSpec(
        name=name,
        summary="a test tool",
        params=(Param("x", int, "arg"),),
        input_shape=TypeShape.NONE,
        output_shape=TypeShape.SCALAR,
        domain="core",
    )


@pytest.fixture()
def registry() -> ToolRegistry:
    r = ToolRegistry()
    for name in ("clean", "broken", "guarded", "degenerate"):
        r.register(_spec(name))
    return r


@pytest.fixture(autouse=True)
def configured(registry: ToolRegistry) -> Any:
    """Install a test shaper; restore the default state afterwards so
    the process-wide seam cannot leak between test modules."""
    configure_shaping(ConfigModel(), registry)
    yield
    configure_shaping(ConfigModel(), shaping.default_registry())


class TestShapedCall:
    def test_success_returns_the_wire_envelope(self) -> None:
        wire = shaped_call("clean", lambda x: x + 1, {"x": 1})
        assert set(wire) == {"inline", "data", "composition_metadata", "error"}
        assert wire["inline"] == "2"
        assert wire["data"] == 2
        assert wire["error"] is None

    def test_exception_becomes_the_error_envelope(self) -> None:
        def boom(x: int) -> int:
            raise ValueError("implementation failure")

        wire = shaped_call("broken", boom, {"x": 1})
        assert wire["error"] is not None
        assert wire["inline"] is None and wire["data"] is None
        assert wire["error"]["error_type"]

    def test_guard_failure_passes_its_shaped_form_through(self) -> None:
        structured = StructuredError(
            error_type=ErrorType.CONNECTION_ERROR,
            message="endpoint down",
            suggestion="re-check the endpoint",
            retryable=True,
        )

        def guarded(x: int) -> int:
            raise GuardedExecutionError(structured)

        wire = shaped_call("guarded", guarded, {"x": 1})
        assert wire["error"] == structured.to_wire()

    def test_sentinel_trip_is_never_a_silent_success(self) -> None:
        wire = shaped_call("degenerate", lambda x: {"mean": float("nan")}, {"x": 1})
        assert wire["error"] is not None
        assert "degenerate" in wire["error"]["message"]

    def test_configure_swaps_the_live_budgets(self, registry: ToolRegistry) -> None:
        configure_shaping(
            ConfigModel(response=ResponseConfig(inline_max_bytes=4)), registry
        )
        wire = shaped_call("clean", lambda x: {"k": "v" * 50}, {"x": 1})
        # Mapping payloads render inline regardless of the table budget —
        # but the swap is observable through the config identity below.
        assert wire["error"] is None
        shaper, config, _ = shaping._STATE.current()
        assert config.response.inline_max_bytes == 4


class TestSentinelSignalClasses:
    def test_class1_non_finite_anywhere(self) -> None:
        assert inspect({"coef": [1.0, float("inf")]}, _KAPPA_MAX) is not None
        assert inspect([1.0, 2.0], _KAPPA_MAX) is None

    def test_class2_convergence_flags(self) -> None:
        assert inspect({"converged": False, "fit": 1.0}, _KAPPA_MAX) is not None
        assert inspect({"optimizer_status": 2}, _KAPPA_MAX) is not None
        assert inspect({"converged": True, "optimizer_status": 0}, _KAPPA_MAX) is None

    def test_class3_degenerate_shapes_inside_a_result(self) -> None:
        assert inspect({"groups": []}, _KAPPA_MAX) is not None
        assert inspect({"clusters": {"a": [], "b": [1]}}, _KAPPA_MAX) is not None
        assert inspect({"groups": {"a": [1], "b": [2]}}, _KAPPA_MAX) is None

    def test_class4_rank_deficiency_and_conditioning(self) -> None:
        """The necessary fourth class: full-shaped, NaN-free, no
        convergence concept — only the rank/kappa signal catches it."""
        assert (
            inspect(
                {"rank": 2, "design_columns": 3, "coef": [1.0, 2.0, 0.5]}, _KAPPA_MAX
            )
            is not None
        )
        assert inspect({"condition_number": _KAPPA_MAX * 10}, _KAPPA_MAX) is not None
        assert (
            inspect(
                {"rank": 3, "design_columns": 3, "condition_number": 10.0}, _KAPPA_MAX
            )
            is None
        )

    def test_verdict_is_a_structured_nx3_error(self) -> None:
        verdict = inspect({"groups": []}, _KAPPA_MAX)
        assert verdict is not None
        assert verdict.error_type is ErrorType.DATA_VALIDATION
        assert verdict.retryable is False

    def test_threshold_comes_from_the_caller_not_a_literal(self) -> None:
        """S8 row 32 has one home: the same result flips verdict when
        the caller-passed threshold moves."""
        result = {"condition_number": 1e8}
        assert inspect(result, 1e6) is not None
        assert inspect(result, 1e9) is None

    def test_process_config_carries_the_declared_default(self) -> None:
        assert ProcessConfig().sentinel_max_condition_number == 1e10
