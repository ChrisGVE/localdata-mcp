"""tests/v3/test_config_provenance.py — E1.3 provenance types.

Per-field provenance carries the winning (value, source) plus every
losing contribution as (layer, attempted_value) entries, queryable —
what makes the shadowed half of the startup pinned/shadowed report
(E2.6) derivable (ARCHITECTURE.md section 5).
"""

from __future__ import annotations

from localdata_mcp.nexus.config.provenance import (
    Contribution,
    FieldProvenance,
    Layer,
    LayerSource,
    Provenance,
)


def _entry() -> FieldProvenance:
    return FieldProvenance(
        field_path="query.default_chunk_size",
        value=250,
        winning_source="project-file",
        contributions=(
            Contribution(
                source="user-file",
                layer=Layer.USER,
                value=200,
                disposition="overridden",
            ),
            Contribution(
                source="project-file",
                layer=Layer.PROJECT,
                value=250,
                disposition="won",
            ),
        ),
    )


class TestLayer:
    def test_trust_order_is_system_over_user_over_project(self) -> None:
        assert Layer.SYSTEM > Layer.USER > Layer.PROJECT

    def test_layer_source_carries_raw_values(self) -> None:
        source = LayerSource(
            name="env", layer=Layer.USER, order=1, values={"query": {}}
        )
        assert source.layer is Layer.USER
        assert source.values == {"query": {}}


class TestProvenanceQueries:
    def test_winner_is_queryable_by_path(self) -> None:
        provenance = Provenance({_entry().field_path: _entry()})
        assert provenance.winner("query.default_chunk_size") == (
            250,
            "project-file",
        )

    def test_default_valued_field_reports_default_source(self) -> None:
        entry = FieldProvenance(
            field_path="query.stream_idle_ttl_seconds",
            value=600,
            winning_source=None,
            contributions=(),
        )
        provenance = Provenance({entry.field_path: entry})
        assert provenance.winner("query.stream_idle_ttl_seconds") == (
            600,
            "default",
        )

    def test_shadowed_lists_only_losing_contributions(self) -> None:
        provenance = Provenance({_entry().field_path: _entry()})
        shadowed = provenance.shadowed("query.default_chunk_size")
        assert [(c.source, c.value) for c in shadowed] == [("user-file", 200)]

    def test_iteration_and_membership(self) -> None:
        provenance = Provenance({_entry().field_path: _entry()})
        assert "query.default_chunk_size" in provenance
        assert list(provenance) == ["query.default_chunk_size"]
