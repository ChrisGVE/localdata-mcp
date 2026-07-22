"""tests/v3/test_config_merge.py — E1.2 two-tier layer merge, scalar side.

Ordinary fields merge last-wins across system -> user -> project (env at
user rank, after the user file); pin-eligible fields merge first-wins by
trust, refusing lower-trust shadowing without failing startup
(ARCHITECTURE.md section 5). Unknown fields and type mismatches are
fatal typed errors — a malformed config is a broken config.
"""

from __future__ import annotations

import pytest

from localdata_mcp.nexus.config.errors import (
    InvalidValueError,
    PinShadowingError,
    TypeMismatchError,
    UnknownFieldError,
)
from localdata_mcp.nexus.config.merge import merge_sources
from localdata_mcp.nexus.config.provenance import Layer, LayerSource

GIB = 2**30


def system(values: dict) -> LayerSource:
    return LayerSource(name="system-file", layer=Layer.SYSTEM, values=values)


def user(values: dict) -> LayerSource:
    return LayerSource(name="user-file", layer=Layer.USER, values=values)


def env(values: dict) -> LayerSource:
    return LayerSource(name="env", layer=Layer.USER, order=1, values=values)


def project(values: dict) -> LayerSource:
    return LayerSource(name="project-file", layer=Layer.PROJECT, values=values)


class TestOrdinaryLastWins:
    def test_no_sources_yields_pure_defaults(self) -> None:
        result = merge_sources([])
        assert result.model.query.default_chunk_size == 100
        assert result.refusals == ()
        assert result.provenance.winner("query.default_chunk_size") == (
            100,
            "default",
        )

    def test_lower_layer_wins_an_ordinary_field(self) -> None:
        result = merge_sources(
            [
                user({"query": {"default_chunk_size": 200}}),
                project({"query": {"default_chunk_size": 250}}),
            ]
        )
        assert result.model.query.default_chunk_size == 250
        assert result.provenance.winner("query.default_chunk_size") == (
            250,
            "project-file",
        )
        shadowed = result.provenance.shadowed("query.default_chunk_size")
        assert [(c.source, c.value) for c in shadowed] == [("user-file", 200)]

    def test_env_beats_the_user_file_on_ordinary_fields(self) -> None:
        result = merge_sources(
            [
                user({"query": {"default_chunk_size": 200}}),
                env({"query": {"default_chunk_size": 300}}),
            ]
        )
        assert result.model.query.default_chunk_size == 300

    def test_source_input_order_is_irrelevant(self) -> None:
        sources = [
            project({"query": {"default_chunk_size": 250}}),
            user({"query": {"default_chunk_size": 200}}),
        ]
        assert (
            merge_sources(sources).model.query.default_chunk_size
            == merge_sources(list(reversed(sources))).model.query.default_chunk_size
            == 250
        )


class TestPinFirstWins:
    def test_operator_pin_refuses_project_shadowing(self) -> None:
        result = merge_sources(
            [
                system({"resources": {"memory_ceiling_bytes": 4 * GIB}}),
                project({"resources": {"memory_ceiling_bytes": 64 * GIB}}),
            ]
        )
        assert result.model.resources.memory_ceiling_bytes == 4 * GIB
        (refusal,) = result.refusals
        assert isinstance(refusal, PinShadowingError)
        assert refusal.field_path == "resources.memory_ceiling_bytes"
        assert refusal.source == "project-file"

    def test_user_pin_is_subordinate_to_system(self) -> None:
        result = merge_sources(
            [
                system({"resources": {"query_timeout_seconds": 60}}),
                user({"resources": {"query_timeout_seconds": 900}}),
            ]
        )
        assert result.model.resources.query_timeout_seconds == 60
        (refusal,) = result.refusals
        assert isinstance(refusal, PinShadowingError)

    def test_same_trust_pin_is_first_wins_without_refusal(self) -> None:
        result = merge_sources(
            [
                user({"resources": {"query_timeout_seconds": 120}}),
                env({"resources": {"query_timeout_seconds": 240}}),
            ]
        )
        assert result.model.resources.query_timeout_seconds == 120
        assert result.refusals == ()

    def test_pin_defends_shadowing_only_not_setting(self) -> None:
        # No operator layer set it, so the project value stands (the
        # introduction rule gates declarations, not scalar ceilings).
        result = merge_sources([project({"resources": {"query_timeout_seconds": 30}})])
        assert result.model.resources.query_timeout_seconds == 30
        assert result.refusals == ()


class TestFatalValidation:
    def test_unknown_section_is_fatal(self) -> None:
        with pytest.raises(UnknownFieldError):
            merge_sources([user({"quEry": {"default_chunk_size": 1}})])

    def test_unknown_field_is_fatal(self) -> None:
        with pytest.raises(UnknownFieldError) as excinfo:
            merge_sources([user({"query": {"chunk_size": 1}})])
        assert excinfo.value.field_path == "query.chunk_size"

    def test_type_mismatch_is_fatal(self) -> None:
        with pytest.raises(TypeMismatchError):
            merge_sources([user({"query": {"default_chunk_size": "big"}})])

    def test_bool_is_not_an_int(self) -> None:
        with pytest.raises(TypeMismatchError):
            merge_sources([user({"query": {"default_chunk_size": True}})])

    def test_int_is_accepted_for_a_float_field(self) -> None:
        result = merge_sources([user({"testbench": {"png_ssim_threshold": 1}})])
        assert result.model.testbench.png_ssim_threshold == 1.0

    def test_derive_sentinel_is_not_a_legal_value(self) -> None:
        with pytest.raises(InvalidValueError):
            merge_sources([user({"query": {"max_analysis_rows": -1}})])


class TestDerivedField:
    def test_unset_derives_from_the_merged_ceiling(self) -> None:
        result = merge_sources([user({"resources": {"memory_ceiling_bytes": 8 * GIB}})])
        assert result.model.query.max_analysis_rows == 8 * GIB // 8192

    def test_explicit_layer_value_beats_derivation(self) -> None:
        result = merge_sources([user({"query": {"max_analysis_rows": 4096}})])
        assert result.model.query.max_analysis_rows == 4096


class TestProvenanceCompleteness:
    def test_every_scalar_field_has_an_entry(self) -> None:
        result = merge_sources([])
        assert "resources.memory_ceiling_bytes" in result.provenance
        assert "testbench.results_retention_runs" in result.provenance
        assert len(result.provenance) >= 36
