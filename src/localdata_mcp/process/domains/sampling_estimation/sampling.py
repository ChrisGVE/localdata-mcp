"""localdata_mcp/process/domains/sampling_estimation/sampling.py — FR-301.

`generate_sample`'s computation, re-authored from `main`'s
`SamplingTransformer`: the five designs kept by name —
simple_random, stratified (proportional within stratify_column),
systematic (every k-th row from a seeded random start), cluster
(whole clusters drawn by cluster_column), weighted (probability
proportional to weight_column). `sample_size` keeps `main`'s
convention: an integer is a row count, a fraction is a share. Output
is TABULAR (the drawn relation, missing values as None) plus the
design summary. Neighbors: tools.py declares the ToolSpec.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from ..support import invalid_source_refusal, require_columns

METHODS = ("simple_random", "stratified", "systematic", "cluster", "weighted")

# main's default: a ten-percent share when the caller names no size.
_DEFAULT_SHARE = 0.1


def draw_sample(
    frame: pd.DataFrame,
    sampling_method: str = "simple_random",
    sample_size: float = _DEFAULT_SHARE,
    stratify_column: str | None = None,
    cluster_column: str | None = None,
    weight_column: str | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """The drawn relation plus the design that produced it."""
    if sampling_method not in METHODS:
        raise invalid_source_refusal(
            f"Unknown sampling_method {sampling_method!r} — one of {list(METHODS)}."
        )
    count = _resolve_count(frame, sample_size)
    rng = np.random.default_rng(seed)
    drawn = _DESIGNS[sampling_method](
        frame, count, stratify_column, cluster_column, weight_column, rng
    )
    cleaned = drawn.astype(object).where(pd.notna(drawn), None)
    return {
        "sampling_method": sampling_method,
        "requested_size": count,
        "sample_rows": int(len(cleaned)),
        "population_rows": int(len(frame)),
        "columns": [str(name) for name in cleaned.columns],
        "rows": cleaned.to_numpy().tolist(),
    }


def _resolve_count(frame: pd.DataFrame, sample_size: float) -> int:
    """main's convention: integers count rows, fractions share them."""
    if sample_size <= 0:
        raise invalid_source_refusal("sample_size must be positive.")
    if float(sample_size).is_integer() and sample_size >= 1:
        count = int(sample_size)
    else:
        if sample_size >= 1:
            raise invalid_source_refusal("A fractional sample_size must be below 1.")
        count = max(1, math.floor(len(frame) * sample_size))
    if count > len(frame):
        raise invalid_source_refusal(
            f"sample_size {count} exceeds the {len(frame)} available rows."
        )
    return count


def _simple(
    frame: pd.DataFrame,
    count: int,
    stratify: str | None,
    cluster: str | None,
    weight: str | None,
    rng: np.random.Generator,
) -> pd.DataFrame:
    return frame.sample(n=count, random_state=rng)


def _stratified(
    frame: pd.DataFrame,
    count: int,
    stratify: str | None,
    cluster: str | None,
    weight: str | None,
    rng: np.random.Generator,
) -> pd.DataFrame:
    if stratify is None:
        raise invalid_source_refusal("stratified sampling needs stratify_column=.")
    require_columns(frame, stratify)
    share = count / len(frame)
    parts = [
        group.sample(n=max(1, round(len(group) * share)), random_state=rng)
        for _label, group in frame.groupby(stratify, sort=True)
    ]
    return pd.concat(parts)


def _systematic(
    frame: pd.DataFrame,
    count: int,
    stratify: str | None,
    cluster: str | None,
    weight: str | None,
    rng: np.random.Generator,
) -> pd.DataFrame:
    step = len(frame) // count
    start = int(rng.integers(0, step)) if step > 1 else 0
    return frame.iloc[start::step].head(count)


def _cluster(
    frame: pd.DataFrame,
    count: int,
    stratify: str | None,
    cluster: str | None,
    weight: str | None,
    rng: np.random.Generator,
) -> pd.DataFrame:
    if cluster is None:
        raise invalid_source_refusal("cluster sampling needs cluster_column=.")
    require_columns(frame, cluster)
    labels = sorted(frame[cluster].dropna().unique().tolist(), key=str)
    drawn_rows: list[pd.DataFrame] = []
    order = rng.permutation(len(labels))
    for position in order:
        drawn_rows.append(frame[frame[cluster] == labels[position]])
        if sum(len(part) for part in drawn_rows) >= count:
            break
    return pd.concat(drawn_rows)


def _weighted(
    frame: pd.DataFrame,
    count: int,
    stratify: str | None,
    cluster: str | None,
    weight: str | None,
    rng: np.random.Generator,
) -> pd.DataFrame:
    if weight is None:
        raise invalid_source_refusal("weighted sampling needs weight_column=.")
    require_columns(frame, weight)
    weights = pd.to_numeric(frame[weight], errors="coerce").fillna(0.0)
    if (weights <= 0).all():
        raise invalid_source_refusal(f"Column {weight!r} carries no positive weights.")
    return frame.sample(n=count, weights=weights, random_state=rng)


_DESIGNS = {
    "simple_random": _simple,
    "stratified": _stratified,
    "systematic": _systematic,
    "cluster": _cluster,
    "weighted": _weighted,
}
