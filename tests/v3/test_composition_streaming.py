"""tests/v3/test_composition_streaming.py — E11.1 streaming execution (FR-603).

FR-603's acceptance shape: the same chain over a chunked source and
over the fully-materialized source agree within
`testbench.tol_stream_parity_rtol` — exact for row-local transforms,
tolerance-bounded for the variance-class folds. Plus the declared
cutover: a non-capable stage materializes at its boundary, and the
partial_fit adapter refuses estimators that cannot honestly stream."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from localdata_mcp.nexus.config.models import ConfigModel
from localdata_mcp.process.composition.streaming_exec.executor import (
    StreamingStage,
    execute_streaming,
    fold_column_comoments,
    fold_column_moments,
    iter_chunks,
)
from localdata_mcp.process.composition.streaming_exec.sklearn_adapter import (
    SklearnStreamingAdapter,
)

_RTOL = ConfigModel().testbench.tol_stream_parity_rtol


def _frame(rows: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(23)
    return pd.DataFrame(
        {
            "value": rng.normal(loc=1e6, scale=2.0, size=rows),
            "other": rng.normal(size=rows),
        }
    )


def _double(chunk: pd.DataFrame) -> pd.DataFrame:
    result = chunk.copy()
    result["value"] = result["value"] * 2.0
    return result


def _shift(chunk: pd.DataFrame) -> pd.DataFrame:
    result = chunk.copy()
    result["other"] = result["other"] + 1.0
    return result


def _materializing_zscore(frame: pd.DataFrame) -> pd.DataFrame:
    # Whole-frame statistics: honest only on materialized input.
    result = frame.copy()
    result["value"] = (result["value"] - result["value"].mean()) / result["value"].std(
        ddof=1
    )
    return result


class TestIterChunks:
    def test_chunks_partition_the_frame(self) -> None:
        frame = _frame(103)
        chunks = list(iter_chunks(frame, 25))
        assert sum(len(c) for c in chunks) == 103
        rebuilt = pd.concat(chunks, ignore_index=True)
        pd.testing.assert_frame_equal(rebuilt, frame.reset_index(drop=True))

    def test_non_positive_chunk_rows_refused(self) -> None:
        with pytest.raises(ValueError):
            list(iter_chunks(_frame(5), 0))


class TestExecuteStreaming:
    def test_capable_chain_chunked_equals_materialized(self) -> None:
        frame = _frame()
        stages = (
            StreamingStage("double", _double, streaming_capable=True),
            StreamingStage("shift", _shift, streaming_capable=True),
        )
        streamed = execute_streaming(iter_chunks(frame, 17), stages)
        materialized = execute_streaming(iter([frame]), stages)
        pd.testing.assert_frame_equal(streamed, materialized)

    def test_non_capable_stage_materializes_at_its_boundary(self) -> None:
        """The whole-frame z-score is only correct on the materialized
        frame — the executor's declared cutover must hand it ALL rows,
        chunked upstream or not (FR-603 parity through the boundary)."""
        frame = _frame()
        stages = (
            StreamingStage("double", _double, streaming_capable=True),
            StreamingStage("zscore", _materializing_zscore, streaming_capable=False),
        )
        streamed = execute_streaming(iter_chunks(frame, 29), stages)
        materialized = execute_streaming(iter([frame]), stages)
        assert streamed["value"].to_numpy() == pytest.approx(
            materialized["value"].to_numpy(), rel=_RTOL
        )

    def test_empty_stream_is_an_empty_frame(self) -> None:
        stages = (StreamingStage("double", _double, streaming_capable=True),)
        result = execute_streaming(iter([]), stages)
        assert result.empty


class TestVarianceClassFolds:
    def test_moments_fold_matches_materialized(self) -> None:
        frame = _frame(541)
        merged = fold_column_moments(iter_chunks(frame, 37), "value")
        values = frame["value"].to_numpy()
        assert merged.count == len(frame)
        assert merged.mean == pytest.approx(float(np.mean(values)), rel=_RTOL)
        variance = merged.variance()
        assert variance is not None
        assert variance == pytest.approx(float(np.var(values, ddof=1)), rel=_RTOL)

    def test_comoments_fold_matches_materialized(self) -> None:
        frame = _frame(419)
        merged = fold_column_comoments(iter_chunks(frame, 43), "value", "other")
        xs = frame["value"].to_numpy()
        ys = frame["other"].to_numpy()
        covariance = merged.covariance()
        correlation = merged.correlation()
        assert covariance is not None and correlation is not None
        assert covariance == pytest.approx(
            float(np.cov(xs, ys, ddof=1)[0, 1]), rel=_RTOL
        )
        assert correlation == pytest.approx(float(np.corrcoef(xs, ys)[0, 1]), rel=_RTOL)


class TestSklearnAdapter:
    def test_partial_fit_scaler_streams(self) -> None:
        from sklearn.preprocessing import StandardScaler

        frame = _frame(300).rename(columns={"value": "a", "other": "b"})
        adapter = SklearnStreamingAdapter(StandardScaler())
        adapter.fit_chunks(iter_chunks(frame, 31))
        streamed = adapter.transform_chunks(iter_chunks(frame, 31))
        reference = StandardScaler().fit(frame).transform(frame)
        assert streamed.to_numpy() == pytest.approx(reference, rel=1e-5)

    def test_estimator_without_partial_fit_is_refused(self) -> None:
        from sklearn.decomposition import PCA

        with pytest.raises(ValueError, match="partial_fit"):
            SklearnStreamingAdapter(PCA())

    def test_transform_before_fit_is_refused(self) -> None:
        from sklearn.preprocessing import StandardScaler

        adapter = SklearnStreamingAdapter(StandardScaler())
        with pytest.raises(ValueError, match="fit_chunks"):
            adapter.transform_chunks(iter([_frame(5)]))
