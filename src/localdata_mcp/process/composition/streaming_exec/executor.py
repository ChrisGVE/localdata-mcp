"""localdata_mcp/process/composition/streaming_exec/executor.py — the chunked loop (E11.1).

The harvest-rewrite of `StreamingDataPipeline`'s chunked fit/transform
loop (`pipeline/core/streaming.py`) onto FR-603's honest semantics:
streaming through a chain is per-stage-conditional, never a chain-wide
guarantee (§6.3). A `streaming_capable` stage maps chunk-by-chunk with
bounded residency; the FIRST non-capable stage materializes at that
boundary (declared, not implied away) and the chain continues from the
materialized frame as a single chunk. The variance-class folds hand
per-chunk partials to combiners.py's Welford/Chan merge — the pairing
that makes the S8 row-15d parity tolerance sound. v2's adaptive
chunk-sizing/memory-monitor scaffolding is dead mass, not harvested
(NX-6 owns admission, §4c). Neighbors: sklearn_adapter.py carries
`partial_fit` estimators through the same chunk iterator.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass

import pandas as pd

from .combiners import RunningComoments, RunningMoments


@dataclass(frozen=True)
class StreamingStage:
    """One chain link for the chunked loop: its name, the frame
    transform, and its honest streaming_capable declaration (§6.1)."""

    name: str
    apply: Callable[[pd.DataFrame], pd.DataFrame]
    streaming_capable: bool


def iter_chunks(frame: pd.DataFrame, chunk_rows: int) -> Iterator[pd.DataFrame]:
    """`frame` as row-bounded chunks (the DataFrameStreamingSource
    harvest, minus its estimator scaffolding)."""
    if chunk_rows <= 0:
        raise ValueError(f"chunk_rows must be positive, got {chunk_rows}")
    for start in range(0, len(frame), chunk_rows):
        yield frame.iloc[start : start + chunk_rows].reset_index(drop=True)


def execute_streaming(
    chunks: Iterable[pd.DataFrame], stages: Sequence[StreamingStage]
) -> pd.DataFrame:
    """Fold the chunk stream through the chain and materialize the
    final result. Capable stages compose lazily (one chunk resident at
    a time); a non-capable stage concatenates the stream at its
    boundary — exactly the declared cutover §6.3 promises."""
    current: Iterator[pd.DataFrame] = iter(chunks)
    for stage in stages:
        if stage.streaming_capable:
            current = _mapped(current, stage.apply)
        else:
            current = iter([stage.apply(_materialized(current))])
    return _materialized(current)


def fold_column_moments(chunks: Iterable[pd.DataFrame], column: str) -> RunningMoments:
    """The streaming variance-class path for one column: per-chunk
    partials merged via Welford/Chan (FR-603's soundness constraint)."""
    merged = RunningMoments.empty()
    for chunk in chunks:
        merged = merged.merged(RunningMoments.of(chunk[column].to_numpy(dtype=float)))
    return merged


def fold_column_comoments(
    chunks: Iterable[pd.DataFrame], column_x: str, column_y: str
) -> RunningComoments:
    """The streaming covariance/correlation path for a column pair."""
    merged = RunningComoments.empty()
    for chunk in chunks:
        merged = merged.merged(
            RunningComoments.of(
                chunk[column_x].to_numpy(dtype=float),
                chunk[column_y].to_numpy(dtype=float),
            )
        )
    return merged


def _mapped(
    stream: Iterator[pd.DataFrame], apply: Callable[[pd.DataFrame], pd.DataFrame]
) -> Iterator[pd.DataFrame]:
    for chunk in stream:
        yield apply(chunk)


def _materialized(stream: Iterator[pd.DataFrame]) -> pd.DataFrame:
    collected = list(stream)
    if not collected:
        return pd.DataFrame()
    if len(collected) == 1:
        return collected[0]
    return pd.concat(collected, ignore_index=True)
