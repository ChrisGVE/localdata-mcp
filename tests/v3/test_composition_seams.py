"""tests/v3/test_composition_seams.py — E11.1's two engine seams.

The guard's composition seams (`composition_limits`,
`charge_composition`/`release_composition` — NFR-105's one aggregate
ledger) and the addressing home's stage-injection channel
(`pipeline_input` — the upstream handoff that keeps X-2's
exactly-one-source contract intact for standalone calls)."""

from __future__ import annotations

import os

import pandas as pd
import pytest

from localdata_mcp.explore.addressing import pipeline_input, resolve_frame
from localdata_mcp.nexus.chokepoint.guard import (
    Chokepoint,
    GuardedExecutionError,
    ResourceRefusedError,
)
from localdata_mcp.nexus.config.models import ConfigModel


class TestGuardCompositionSeams:
    def _guard(self) -> Chokepoint:
        return Chokepoint.boot(ConfigModel(), environ=dict(os.environ))

    def test_composition_limits_reads_the_config_value(self) -> None:
        guard = self._guard()
        try:
            limits = guard.composition_limits()
            assert (
                limits.max_pipeline_length
                == ConfigModel().composition.max_pipeline_length
            )
        finally:
            guard.shutdown()

    def test_charge_and_release_round_trip(self) -> None:
        guard = self._guard()
        try:
            guard.charge_composition("p1", 1024)
            guard.charge_composition("p1", 4096)  # re-charge replaces
            guard.release_composition("p1")
            guard.release_composition("p1")  # idempotent
        finally:
            guard.shutdown()

    def test_charge_over_ceiling_is_refused(self) -> None:
        guard = self._guard()
        try:
            ceiling = ConfigModel().resources.memory_ceiling_bytes
            with pytest.raises(ResourceRefusedError):
                guard.charge_composition("hog", ceiling + 1)
        finally:
            guard.shutdown()


class TestPipelineInputSeam:
    def test_injected_frame_resolves_with_no_source(self) -> None:
        upstream = pd.DataFrame({"value": [1, 2, 3]})
        with pipeline_input(upstream, "pipeline:clean"):
            frame, label = resolve_frame(None, None, None, None)
        assert label == "pipeline:clean"
        pd.testing.assert_frame_equal(frame, upstream)

    def test_injection_hands_out_a_copy(self) -> None:
        upstream = pd.DataFrame({"value": [1, 2, 3]})
        with pipeline_input(upstream, "pipeline:clean"):
            frame, _ = resolve_frame(None, None, None, None)
            frame.loc[0, "value"] = 99
        assert upstream.loc[0, "value"] == 1

    def test_standalone_zero_source_still_refuses(self) -> None:
        with pytest.raises(GuardedExecutionError):
            resolve_frame(None, None, None, None)

    def test_injection_ends_with_the_context(self) -> None:
        with pipeline_input(pd.DataFrame({"a": [1]}), "pipeline:x"):
            pass
        with pytest.raises(GuardedExecutionError):
            resolve_frame(None, None, None, None)

    def test_nested_injection_restores_the_outer_frame(self) -> None:
        outer = pd.DataFrame({"a": [1]})
        inner = pd.DataFrame({"b": [2]})
        with pipeline_input(outer, "pipeline:outer"):
            with pipeline_input(inner, "pipeline:inner"):
                frame, label = resolve_frame(None, None, None, None)
                assert label == "pipeline:inner"
                assert list(frame.columns) == ["b"]
            frame, label = resolve_frame(None, None, None, None)
            assert label == "pipeline:outer"
            assert list(frame.columns) == ["a"]
