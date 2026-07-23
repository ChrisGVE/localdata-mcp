"""tests/v3/test_composition_combiners.py — E11.1 stable combiners (C-1).

FR-603's soundness constraint: per-chunk variance/covariance partials
merge via the Welford/Chan parallel update, matching the materialized
computation within `testbench.tol_stream_parity_rtol`. The adversarial
cases put a large common offset on the data — exactly where the
forbidden naive E[X²]−E[X]² one-pass formula loses the answer to
catastrophic cancellation — so the tolerance is proven sound, not
hoped."""

from __future__ import annotations

import numpy as np
import pytest

from localdata_mcp.nexus.config.models import ConfigModel
from localdata_mcp.process.composition.streaming_exec.combiners import (
    RunningComoments,
    RunningMoments,
)

_RTOL = ConfigModel().testbench.tol_stream_parity_rtol


def _chunks(values: np.ndarray, size: int) -> "list[np.ndarray]":
    return [values[start : start + size] for start in range(0, len(values), size)]


class TestRunningMoments:
    def test_single_pass_matches_numpy(self) -> None:
        rng = np.random.default_rng(42)
        values = rng.normal(loc=5.0, scale=3.0, size=997)
        moments = RunningMoments.of(values)
        assert moments.count == len(values)
        assert moments.mean == pytest.approx(float(np.mean(values)), rel=_RTOL)
        assert moments.variance() == pytest.approx(
            float(np.var(values, ddof=1)), rel=_RTOL
        )
        assert moments.std() == pytest.approx(float(np.std(values, ddof=1)), rel=_RTOL)

    def test_chunked_merge_matches_materialized(self) -> None:
        rng = np.random.default_rng(7)
        values = rng.normal(size=1013)
        merged = RunningMoments.empty()
        for chunk in _chunks(values, 89):
            merged = merged.merged(RunningMoments.of(chunk))
        assert merged.count == len(values)
        assert merged.mean == pytest.approx(float(np.mean(values)), rel=_RTOL)
        assert merged.variance() == pytest.approx(
            float(np.var(values, ddof=1)), rel=_RTOL
        )

    def test_large_offset_survives_merge_where_naive_dies(self) -> None:
        """The catastrophic-cancellation trap: mean ~1e6, variance ~1
        (condition number ~1e12). The forbidden naive E[X²]−E[X]²
        formula loses ~4 digits here and misses the parity tolerance
        by orders of magnitude; the Welford/Chan merge holds it."""
        rng = np.random.default_rng(3)
        values = rng.normal(loc=1e6, scale=1.0, size=503)
        expected = float(np.var(values, ddof=1))
        merged = RunningMoments.empty()
        for chunk in _chunks(values, 41):
            merged = merged.merged(RunningMoments.of(chunk))
        stable = merged.variance()
        assert stable is not None
        assert stable == pytest.approx(expected, rel=_RTOL)
        # Documentation of the trap: the naive one-pass formula fails
        # the same tolerance on the same data.
        naive = (
            float(np.sum(values**2)) - len(values) * float(np.mean(values)) ** 2
        ) / (len(values) - 1)
        assert abs(naive - expected) / expected > _RTOL

    def test_merge_is_order_insensitive_within_tolerance(self) -> None:
        rng = np.random.default_rng(11)
        values = rng.normal(loc=1e6, size=200)
        chunks = _chunks(values, 23)
        forward = RunningMoments.empty()
        for chunk in chunks:
            forward = forward.merged(RunningMoments.of(chunk))
        backward = RunningMoments.empty()
        for chunk in reversed(chunks):
            backward = backward.merged(RunningMoments.of(chunk))
        assert forward.variance() == pytest.approx(backward.variance(), rel=_RTOL)

    def test_empty_and_degenerate_counts(self) -> None:
        empty = RunningMoments.empty()
        assert empty.count == 0
        assert empty.variance() is None
        one = RunningMoments.of(np.asarray([2.5]))
        assert one.variance() is None  # ddof=1 undefined at n=1
        assert one.mean == pytest.approx(2.5)
        # merging with empty is the identity
        merged = empty.merged(one)
        assert merged.count == 1
        assert merged.mean == pytest.approx(2.5)


class TestRunningComoments:
    def test_chunked_covariance_and_correlation_match(self) -> None:
        rng = np.random.default_rng(19)
        xs = rng.normal(loc=1e7, scale=2.0, size=811)
        ys = 0.5 * xs + rng.normal(scale=3.0, size=811)
        merged = RunningComoments.empty()
        for start in range(0, len(xs), 67):
            merged = merged.merged(
                RunningComoments.of(xs[start : start + 67], ys[start : start + 67])
            )
        assert merged.count == len(xs)
        expected_cov = float(np.cov(xs, ys, ddof=1)[0, 1])
        expected_corr = float(np.corrcoef(xs, ys)[0, 1])
        assert merged.covariance() == pytest.approx(expected_cov, rel=_RTOL)
        assert merged.correlation() == pytest.approx(expected_corr, rel=_RTOL)

    def test_degenerate_pairs(self) -> None:
        assert RunningComoments.empty().covariance() is None
        one = RunningComoments.of(np.asarray([1.0]), np.asarray([2.0]))
        assert one.covariance() is None
        # zero-variance side: correlation undefined, never a crash
        xs = np.asarray([3.0, 3.0, 3.0])
        ys = np.asarray([1.0, 2.0, 3.0])
        flat = RunningComoments.of(xs, ys)
        assert flat.correlation() is None

    def test_mismatched_lengths_are_refused(self) -> None:
        with pytest.raises(ValueError):
            RunningComoments.of(np.asarray([1.0, 2.0]), np.asarray([1.0]))
