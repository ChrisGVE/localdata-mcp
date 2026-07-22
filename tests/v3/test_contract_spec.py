"""tests/v3/test_contract_spec.py — E3.1: ToolSpec/Param/TypeShape + registry.

Exercises nexus/contract/spec.py (the one declaration vocabulary,
ARCHITECTURE.md section 6.1) and nexus/contract/registry.py (register,
lookup, iterate, duplicate-name refusal). TypeShape is a CLOSED enum:
the test pins the exact member set so a new shape is a deliberate,
reviewed change, never an accident.
"""

from __future__ import annotations

import dataclasses

import pytest

from localdata_mcp.nexus.contract.errors import (
    DuplicateToolNameError,
    SpecValidationError,
    ToolContractError,
    UnknownToolError,
)
from localdata_mcp.nexus.contract.registry import ToolRegistry
from localdata_mcp.nexus.contract.spec import Param, ToolSpec, TypeShape, tool_spec

EXPECTED_SHAPES = {
    "TABULAR",
    "SCALAR",
    "VECTOR",
    "MATRIX",
    "FITTED_MODEL",
    "GRAPH",
    "GEO",
    "CHART_SPEC",
    "NONE",
    "DYNAMIC",
}


def _make_spec(name: str = "demo_tool", **overrides: object) -> ToolSpec:
    fields: dict = dict(
        name=name,
        summary="A demo tool.",
        params=(Param("x", int, "An input."),),
        input_shape=TypeShape.TABULAR,
        output_shape=TypeShape.SCALAR,
        streaming_capable=False,
        domain=None,
        func=lambda x: x,
    )
    fields.update(overrides)
    return ToolSpec(**fields)


class TestTypeShape:
    def test_closed_member_set(self) -> None:
        assert {m.name for m in TypeShape} == EXPECTED_SHAPES

    def test_member_count_is_ten(self) -> None:
        assert len(TypeShape) == 10


class TestParam:
    def test_frozen(self) -> None:
        p = Param("query", str, "SQL SELECT statement.")
        with pytest.raises(dataclasses.FrozenInstanceError):
            p.name = "other"  # type: ignore[misc]

    def test_fields(self) -> None:
        p = Param("query", str, "SQL SELECT statement.")
        assert (p.name, p.annotation, p.description) == (
            "query",
            str,
            "SQL SELECT statement.",
        )


class TestToolSpec:
    def test_frozen(self) -> None:
        spec = _make_spec()
        with pytest.raises(dataclasses.FrozenInstanceError):
            spec.name = "other"  # type: ignore[misc]

    def test_params_stored_as_tuple(self) -> None:
        spec = _make_spec(params=[Param("x", int, "An input.")])
        assert isinstance(spec.params, tuple)

    def test_empty_name_refused(self) -> None:
        with pytest.raises(SpecValidationError):
            _make_spec(name="")

    def test_empty_summary_refused(self) -> None:
        with pytest.raises(SpecValidationError):
            _make_spec(summary="")

    def test_non_param_entry_refused(self) -> None:
        with pytest.raises(SpecValidationError):
            _make_spec(params=("not-a-param",))

    def test_validation_error_is_contract_error(self) -> None:
        assert issubclass(SpecValidationError, ToolContractError)


class TestToolRegistry:
    def test_register_and_lookup(self) -> None:
        registry = ToolRegistry()
        spec = _make_spec()
        registry.register(spec)
        assert registry.lookup("demo_tool") is spec

    def test_iterate_yields_specs_in_registration_order(self) -> None:
        registry = ToolRegistry()
        first = _make_spec("alpha")
        second = _make_spec("beta")
        registry.register(first)
        registry.register(second)
        assert list(registry) == [first, second]

    def test_duplicate_name_refused(self) -> None:
        registry = ToolRegistry()
        registry.register(_make_spec())
        with pytest.raises(DuplicateToolNameError):
            registry.register(_make_spec())

    def test_unknown_lookup_refused(self) -> None:
        registry = ToolRegistry()
        with pytest.raises(UnknownToolError):
            registry.lookup("missing")

    def test_dynamic_shape_registrable(self) -> None:
        # compose_pipeline (E11) is the ONE legitimate DYNAMIC tool; the
        # registry accepts DYNAMIC — the stage-bar rule is checked at
        # dag_spec validation, not at registration (section 6.1).
        registry = ToolRegistry()
        spec = _make_spec(
            name="compose_pipeline",
            input_shape=TypeShape.DYNAMIC,
            output_shape=TypeShape.DYNAMIC,
        )
        registry.register(spec)
        assert registry.lookup("compose_pipeline").output_shape is TypeShape.DYNAMIC


class TestToolSpecDecorator:
    def test_decorator_attaches_spec_and_registers(self) -> None:
        registry = ToolRegistry()

        @tool_spec(
            name="double",
            summary="Double an integer.",
            params=[Param("x", int, "The integer to double.")],
            input_shape=TypeShape.SCALAR,
            output_shape=TypeShape.SCALAR,
            streaming_capable=False,
            domain=None,
            registry=registry,
        )
        def double(x: int) -> int:
            return x * 2

        assert double(21) == 42
        spec = registry.lookup("double")
        assert spec.func is double
        assert double.__tool_spec__ is spec  # type: ignore[attr-defined]

    def test_decorator_defaults(self) -> None:
        registry = ToolRegistry()

        @tool_spec(
            name="ping",
            summary="Liveness probe.",
            params=[],
            input_shape=TypeShape.NONE,
            output_shape=TypeShape.SCALAR,
            registry=registry,
        )
        def ping() -> str:
            return "pong"

        spec = registry.lookup("ping")
        assert spec.streaming_capable is False
        assert spec.domain is None
        assert spec.params == ()
