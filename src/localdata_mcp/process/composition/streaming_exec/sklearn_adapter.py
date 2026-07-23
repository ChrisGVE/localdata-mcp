"""localdata_mcp/process/composition/streaming_exec/sklearn_adapter.py — partial_fit adapter (E11.1).

The harvest-rewrite of v2's `SklearnStreamingAdapter`
(`pipeline/core/streaming.py`) reduced to its sound core: carry an
estimator that HONESTLY supports incremental learning (`partial_fit`)
through the chunk stream, and refuse one that does not — the
materializing path for full-fit estimators is the chain's declared
cutover (executor.py), never a silent adapter fallback (v2's
collect-then-fit disguise is dead mass, not harvested). Duck-typed on
partial_fit/transform, so this module needs no sklearn import of its
own. Neighbors: executor.py supplies the chunk stream.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import pandas as pd


class SklearnStreamingAdapter:
    """Chunk-wise incremental fit/transform for one estimator."""

    def __init__(self, estimator: Any, *, classes: Any = None) -> None:
        if not hasattr(estimator, "partial_fit"):
            raise ValueError(
                f"{type(estimator).__name__} declares no partial_fit — it "
                "cannot honestly stream; run it on the materialized path "
                "instead (the §6.3 per-stage cutover)"
            )
        self._estimator = estimator
        self._classes = classes
        self._fitted = False

    @property
    def estimator(self) -> Any:
        return self._estimator

    def fit_chunks(
        self,
        chunks: Iterable[pd.DataFrame],
        target_column: "str | None" = None,
    ) -> "SklearnStreamingAdapter":
        """partial_fit chunk by chunk; with a target column the chunk
        splits into features/target (the supervised shape), classes
        forwarded on the first classifier call as sklearn requires."""
        first = True
        for chunk in chunks:
            features, target = _split(chunk, target_column)
            if first and self._classes is not None:
                self._estimator.partial_fit(features, target, classes=self._classes)
            else:
                self._estimator.partial_fit(features, target)
            first = False
            self._fitted = True
        return self

    def transform_chunks(self, chunks: Iterable[pd.DataFrame]) -> pd.DataFrame:
        """transform chunk by chunk, concatenated — bounded residency
        per chunk, one result out."""
        if not self._fitted:
            raise ValueError("fit_chunks must run before transform_chunks")
        pieces = [pd.DataFrame(self._estimator.transform(chunk)) for chunk in chunks]
        if not pieces:
            return pd.DataFrame()
        return pd.concat(pieces, ignore_index=True)


def _split(
    chunk: pd.DataFrame, target_column: "str | None"
) -> "tuple[pd.DataFrame, Any]":
    if target_column is None:
        return chunk, None
    return chunk.drop(columns=[target_column]), chunk[target_column]
