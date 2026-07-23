"""tests/v3/test_config_model.py — E1.1 ConfigModel contract tests.

Asserts the NX-2 dataclass-per-truth model carries exactly the S8 rows
whose Home is NX-2 (testbench section included, CI-homed rows excluded),
each with exactly its S8 default; that `query.max_analysis_rows` is
derived from the memory ceiling (S8 row 13), never a literal; and that
pin-eligibility metadata follows ARCHITECTURE.md section 5 (declared on
the field, fail-closed on security-classed sections).
"""

from __future__ import annotations

import dataclasses

import pytest

from localdata_mcp.nexus.config.fields import DERIVED
from localdata_mcp.nexus.config.models import (
    ConfigModel,
    QueryConfig,
    ResourcesConfig,
    is_pin_eligible,
    iter_config_fields,
)

GIB = 2**30
MIB = 2**20
KIB = 2**10

# The S8 contract, one entry per NX-2-homed row (PRD tmp/v3/PRD.md S8).
# Restating the defaults HERE is the frozen test expectation guarding the
# one default site — the NFR-403 check scans src/, never tests.
EXPECTED_DEFAULTS = {
    # resources — S8 rows 1, 2, 3, 5, 6
    "resources.memory_ceiling_bytes": 4 * GIB,
    "resources.query_timeout_seconds": 300,
    "resources.max_connections_per_endpoint": 8,
    "resources.max_spill_bytes": 8 * GIB,
    "resources.min_free_disk_bytes": 2 * GIB,
    # query — S8 rows 8, 9, 10, 12, 13 (derived), 24
    "query.chunk_buffer_max_chunks": 4,
    "query.chunk_buffer_max_bytes": 256 * MIB,
    "query.stream_idle_ttl_seconds": 600,
    "query.default_chunk_size": 100,
    "query.max_analysis_rows": 4 * GIB // 8192,  # derived, S8 row 13
    "query.max_concurrent_streams_per_endpoint": 4,
    # security — S8 rows 11, 19
    "security.validation_cache_entries": 1024,
    "security.allowed_paths": (),
    # security — the E5.3 ephemeral rw grant (ARCHITECTURE section 5)
    "security.ephemeral_write_paths": (),
    # composition — S8 row 14
    "composition.max_pipeline_length": 4,
    # response — S8 rows 23a, 23b
    "response.inline_max_rows": 100,
    "response.inline_max_bytes": 256 * KIB,
    # process — S8 rows 30, 31, 32
    "process.bootstrap_default_resamples": 1000,
    "process.monte_carlo_default_iterations": 10000,
    "process.sentinel_max_condition_number": 1e10,
    # visualize — chart styling defaults (FR-503 styling layer, E12.6)
    "visualize.default_palette": "colorblind",
    "visualize.default_sequential_cmap": "viridis",
    "visualize.figure_width_inches": 6.4,
    "visualize.figure_height_inches": 4.8,
    "visualize.figure_dpi": 100,
    "visualize.grid": True,
    "visualize.despine": True,
    "visualize.fit_line_color": "#d55e00",
    "visualize.edge_color": "#8c8c8c",
    # testbench — S8 rows 15a-f, 20, 21, 22, 25, 27, 28, 29
    "testbench.tol_closed_form_rtol": 1e-6,
    "testbench.tol_iterative_rtol": 1e-2,
    "testbench.tol_quality_rtol": 1e-9,
    "testbench.tol_stream_parity_rtol": 1e-9,
    "testbench.png_ssim_threshold": 0.95,
    "testbench.embedding_trustworthiness_min": 0.95,
    "testbench.choke_latency_ms_miss": 25,
    "testbench.choke_latency_ms_hit": 1,
    "testbench.hypothesis_max_examples": 200,
    "testbench.leak_loop_iterations": 500,
    "testbench.leak_loop_max_drift_mib": 50,
    "testbench.pipeline_chain_p95_seconds": 10,
    "testbench.pipeline_chain_p95_seconds_long": 15,
    "testbench.stretch_max_length": 6,
    "testbench.stretch_sample_chains": 200,
    "testbench.discovery_p99_ms": 100,
    "testbench.results_retention_runs": 200,
}

# Sections whose fields must be pin-eligible (ARCHITECTURE.md section 5:
# resource ceilings and security fields are first-wins-by-layer).
SECURITY_CLASSED_SECTIONS = {"resources", "security"}


def _resolve(model: ConfigModel, path: str) -> object:
    section_name, field_name = path.split(".")
    return getattr(getattr(model, section_name), field_name)


class TestS8Defaults:
    def test_every_s8_field_present_with_exact_default(self) -> None:
        model = ConfigModel()
        for path, expected in EXPECTED_DEFAULTS.items():
            assert _resolve(model, path) == expected, path

    def test_field_set_is_exactly_the_s8_nx2_rows(self) -> None:
        declared = {f"{section}.{fld.name}" for section, fld in iter_config_fields()}
        assert declared == set(EXPECTED_DEFAULTS)

    def test_default_types_match_declarations(self) -> None:
        model = ConfigModel()
        for path, expected in EXPECTED_DEFAULTS.items():
            value = _resolve(model, path)
            assert type(value) is type(expected), path


class TestDerivedMaxAnalysisRows:
    def test_default_reproduces_the_524288_order(self) -> None:
        assert ConfigModel().query.max_analysis_rows == 524288

    def test_scales_with_an_operator_raised_ceiling(self) -> None:
        model = ConfigModel(resources=ResourcesConfig(memory_ceiling_bytes=8 * GIB))
        assert model.query.max_analysis_rows == 8 * GIB // 8192

    def test_explicit_value_is_never_overwritten(self) -> None:
        model = ConfigModel(query=QueryConfig(max_analysis_rows=1000))
        assert model.query.max_analysis_rows == 1000

    def test_derivation_is_declared_on_the_field(self) -> None:
        (fld,) = [
            f for f in dataclasses.fields(QueryConfig) if f.name == "max_analysis_rows"
        ]
        assert fld.default is DERIVED
        assert callable(fld.metadata["derive"])


class TestPinEligibility:
    def test_security_classed_sections_pin_every_field(self) -> None:
        for section, fld in iter_config_fields():
            if section in SECURITY_CLASSED_SECTIONS:
                assert is_pin_eligible(section, fld), f"{section}.{fld.name}"

    def test_ordinary_sections_are_not_pinned(self) -> None:
        for section, fld in iter_config_fields():
            if section not in SECURITY_CLASSED_SECTIONS:
                assert not is_pin_eligible(section, fld), f"{section}.{fld.name}"


class TestModelShape:
    def test_sections_are_frozen(self) -> None:
        model = ConfigModel()
        with pytest.raises(dataclasses.FrozenInstanceError):
            model.query.default_chunk_size = 1  # type: ignore[misc]
        with pytest.raises(dataclasses.FrozenInstanceError):
            model.resources = ResourcesConfig()  # type: ignore[misc]

    def test_every_field_declares_a_rationale(self) -> None:
        for section, fld in iter_config_fields():
            assert fld.metadata.get("doc"), f"{section}.{fld.name}"
