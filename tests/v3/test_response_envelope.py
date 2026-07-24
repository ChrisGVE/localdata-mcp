"""tests/v3/test_response_envelope.py — E7.1 FR-401/403 envelope + metadata.

The exit-gate assertions for envelope.py/metadata.py: the four named
regions always present on the wire (FR-401), error exclusive with the
others at construction (FR-403), the S8 23a/23b inline/stream cutover
by rows AND bytes, O-1's explicit zero statements (never an empty
table or bare list), and applicable-next-steps derived live from the
FR-606 registry + adjacency table.
"""

from __future__ import annotations

import pytest

from localdata_mcp.nexus.chokepoint.guard import Result
from localdata_mcp.nexus.config.models import ConfigModel, ResponseConfig
from localdata_mcp.nexus.contract.registry import ToolRegistry
from localdata_mcp.nexus.contract.spec import Param, ToolSpec, TypeShape
from localdata_mcp.nexus.error.model import ErrorType, StructuredError
from localdata_mcp.nexus.response.envelope import (
    EnvelopeContractError,
    ResponseEnvelope,
    ResponseShaper,
    error_envelope,
)
from localdata_mcp.nexus.response.metadata import (
    CompositionMetadata,
    applicable_next_steps,
)


def spec(
    name: str,
    output_shape: TypeShape = TypeShape.TABULAR,
    input_shape: TypeShape = TypeShape.NONE,
    domain: str | None = "ingest",
) -> ToolSpec:
    return ToolSpec(
        name=name,
        summary=f"{name} summary",
        params=(Param("x", int, "an argument"),),
        input_shape=input_shape,
        output_shape=output_shape,
        domain=domain,
    )


@pytest.fixture()
def registry() -> ToolRegistry:
    r = ToolRegistry()
    r.register(spec("source_tool", TypeShape.TABULAR, TypeShape.NONE))
    r.register(spec("tabular_consumer", TypeShape.SCALAR, TypeShape.TABULAR))
    r.register(spec("vector_consumer", TypeShape.SCALAR, TypeShape.VECTOR))
    r.register(spec("chart_sink", TypeShape.NONE, TypeShape.TABULAR))
    r.register(spec("dynamic_stage", TypeShape.DYNAMIC, TypeShape.DYNAMIC))
    return r


@pytest.fixture()
def shaper(registry: ToolRegistry) -> ResponseShaper:
    return ResponseShaper(ConfigModel(), registry)


class TestEnvelopeContract:
    def test_wire_form_always_names_all_four_regions(
        self, shaper: ResponseShaper
    ) -> None:
        """FR-401's named-field assertion."""
        wire = shaper.shape_envelope(
            Result(columns=("a",), rows=((1,),), category="query"),
            spec("source_tool"),
        ).to_wire()
        assert set(wire) == {"inline", "data", "composition_metadata", "error"}
        assert wire["error"] is None

    def test_error_is_exclusive_with_the_other_regions(self) -> None:
        structured = StructuredError(
            error_type=ErrorType.QUERY_EXECUTION,
            message="boom",
            suggestion="retry",
            retryable=True,
        )
        with pytest.raises(EnvelopeContractError):
            ResponseEnvelope(inline="text", error=structured.to_wire())

    def test_success_requires_inline(self) -> None:
        """FR-401: the LLM-readable region is not optional."""
        with pytest.raises(EnvelopeContractError):
            ResponseEnvelope(data={"k": 1})

    def test_error_envelope_carries_the_wire_shape_alone(self) -> None:
        structured = StructuredError(
            error_type=ErrorType.SECURITY_VIOLATION,
            message="refused",
            suggestion="fix the statement",
            retryable=False,
        )
        wire = error_envelope(structured).to_wire()
        assert wire["error"] == structured.to_wire()
        assert wire["inline"] is None
        assert wire["data"] is None
        assert wire["composition_metadata"] is None


class TestInlineTable:
    def test_within_budget_renders_a_markdown_table(
        self, shaper: ResponseShaper
    ) -> None:
        result = Result(
            columns=("id", "label"),
            rows=((1, "a"), (2, "b")),
            category="query",
        )
        envelope = shaper.shape_envelope(result, spec("source_tool"))
        assert envelope.inline is not None
        assert "| id | label |" in envelope.inline
        assert "| 2 | b |" in envelope.inline
        assert envelope.data == {
            "columns": ["id", "label"],
            "rows": [[1, "a"], [2, "b"]],
        }

    def test_pipe_characters_cannot_break_the_table(
        self, shaper: ResponseShaper
    ) -> None:
        result = Result(columns=("v",), rows=(("a|b",),), category="query")
        envelope = shaper.shape_envelope(result, spec("source_tool"))
        assert envelope.inline is not None
        assert "a\\|b" in envelope.inline


class TestInlineStreamCutover:
    def test_row_bound_cuts_over_to_the_stream_reference(
        self, registry: ToolRegistry
    ) -> None:
        config = ConfigModel(response=ResponseConfig(inline_max_rows=2))
        shaper = ResponseShaper(config, registry)
        result = Result(columns=("n",), rows=((1,), (2,), (3,)), category="query")
        envelope = shaper.shape_envelope(result, spec("source_tool"), stream_id="s-1")
        assert envelope.data == {"stream_id": "s-1"}
        assert envelope.inline is not None
        assert "3 rows" in envelope.inline and "s-1" in envelope.inline

    def test_byte_bound_trips_independently_of_rows(
        self, registry: ToolRegistry
    ) -> None:
        """23b: wide rows cut over even under the row bound."""
        config = ConfigModel(response=ResponseConfig(inline_max_bytes=64))
        shaper = ResponseShaper(config, registry)
        result = Result(columns=("wide",), rows=(("x" * 500,),), category="query")
        envelope = shaper.shape_envelope(result, spec("source_tool"), stream_id="s-2")
        assert envelope.data == {"stream_id": "s-2"}

    def test_cutover_without_a_stream_states_the_streaming_path(
        self, registry: ToolRegistry
    ) -> None:
        config = ConfigModel(response=ResponseConfig(inline_max_rows=1))
        shaper = ResponseShaper(config, registry)
        result = Result(columns=("n",), rows=((1,), (2,)), category="query")
        envelope = shaper.shape_envelope(result, spec("source_tool"))
        assert envelope.inline is not None
        assert "streaming" in envelope.inline
        assert envelope.data == {"row_count": 2}


class TestEmptyResultSemantics:
    def test_zero_rows_render_the_explicit_statement(
        self, shaper: ResponseShaper
    ) -> None:
        """O-1: never an empty table — the agent must be able to tell
        no-data from truncation or defect."""
        result = Result(columns=("id",), rows=(), category="query")
        envelope = shaper.shape_envelope(result, spec("source_tool"))
        assert envelope.inline is not None
        assert "Zero rows" in envelope.inline
        assert "not a truncation or failure" in envelope.inline
        assert "|" not in envelope.inline  # no empty table skeleton

    def test_zero_items_render_the_explicit_statement(
        self, shaper: ResponseShaper
    ) -> None:
        envelope = shaper.shape_envelope([], spec("source_tool"))
        assert envelope.inline is not None
        assert "Zero items" in envelope.inline

    def test_scalar_and_mapping_payloads_render_inline(
        self, shaper: ResponseShaper
    ) -> None:
        scalar = shaper.shape_envelope(42, spec("source_tool", TypeShape.SCALAR))
        assert scalar.inline == "42"
        mapping = shaper.shape_envelope(
            {"mean": 1.5}, spec("source_tool", TypeShape.SCALAR)
        )
        assert mapping.inline is not None
        assert "mean: 1.5" in mapping.inline


class TestMappingInlineBudget:
    """CR-001: a mapping carrying a large embedded list (clustering
    labels, CLV/RFM customer lists, LP assignments) must not render its
    whole payload into an unbounded inline string. The inline region is
    admission-subject like tabular results; the full typed mapping stays
    in `data` for composition (GP4)."""

    def _shaper(self, registry: ToolRegistry) -> ResponseShaper:
        config = ConfigModel(
            response=ResponseConfig(inline_max_rows=5, inline_max_bytes=256)
        )
        return ResponseShaper(config, registry)

    def test_large_embedded_list_value_is_bounded_inline(
        self, registry: ToolRegistry
    ) -> None:
        shaper = self._shaper(registry)
        result = {"labels": list(range(10_000)), "n_clusters": 3}
        envelope = shaper.shape_envelope(result, spec("source_tool", TypeShape.SCALAR))
        assert envelope.inline is not None
        # The inline region is bounded (well under the ~50 KB the raw
        # list would render to), and the full mapping survives in data.
        assert len(envelope.inline.encode("utf-8")) <= 2 * 256 + 512
        assert envelope.data == result
        assert "truncat" in envelope.inline.lower()

    def test_many_entry_mapping_truncates_with_a_note(
        self, registry: ToolRegistry
    ) -> None:
        shaper = self._shaper(registry)
        result = {f"k{i}": i for i in range(50)}
        envelope = shaper.shape_envelope(result, spec("source_tool", TypeShape.SCALAR))
        assert envelope.inline is not None
        assert "omitted" in envelope.inline
        assert envelope.data == result

    def test_small_mapping_renders_in_full_without_a_note(
        self, registry: ToolRegistry
    ) -> None:
        shaper = self._shaper(registry)
        result = {"mean": 1.5, "n": 3}
        envelope = shaper.shape_envelope(result, spec("source_tool", TypeShape.SCALAR))
        assert envelope.inline is not None
        assert "mean: 1.5" in envelope.inline
        assert "n: 3" in envelope.inline
        assert "omitted" not in envelope.inline
        assert "truncat" not in envelope.inline.lower()

    def test_true_scalar_still_renders_directly(self, registry: ToolRegistry) -> None:
        shaper = self._shaper(registry)
        envelope = shaper.shape_envelope(42, spec("source_tool", TypeShape.SCALAR))
        assert envelope.inline == "42"


class TestCompositionMetadataDerivation:
    def test_next_steps_derive_from_the_registry_adjacency(
        self, registry: ToolRegistry
    ) -> None:
        """FR-606 one home: TABULAR feeds TABULAR-input stages only."""
        assert applicable_next_steps(TypeShape.TABULAR, registry) == (
            "tabular_consumer",
            "chart_sink",
        )

    def test_vector_additionally_feeds_tabular_consumers(
        self, registry: ToolRegistry
    ) -> None:
        steps = applicable_next_steps(TypeShape.VECTOR, registry)
        assert set(steps) == {"vector_consumer", "tabular_consumer", "chart_sink"}

    def test_terminal_and_dynamic_shapes_recommend_nothing(
        self, registry: ToolRegistry
    ) -> None:
        assert applicable_next_steps(TypeShape.NONE, registry) == ()
        assert applicable_next_steps(TypeShape.DYNAMIC, registry) == ()

    def test_envelope_embeds_the_derived_metadata(self, shaper: ResponseShaper) -> None:
        envelope = shaper.shape_envelope(
            Result(columns=("a",), rows=((1,),), category="query"),
            spec("source_tool"),
        )
        metadata = envelope.composition_metadata
        assert metadata is not None
        assert metadata.domain == "ingest"
        assert metadata.analysis_type == "source_tool"
        assert metadata.result_type == "tabular"
        assert {"tool": "tabular_consumer"} in metadata.recommended_next_steps

    def test_metadata_wire_form_is_plain_json_shapes(self) -> None:
        metadata = CompositionMetadata(
            domain="d",
            analysis_type="a",
            result_type="r",
            recommended_next_steps=({"tool": "t"},),
        )
        wire = metadata.to_wire()
        assert wire["recommended_next_steps"] == [{"tool": "t"}]
        assert wire["compatible_tools"] == []
