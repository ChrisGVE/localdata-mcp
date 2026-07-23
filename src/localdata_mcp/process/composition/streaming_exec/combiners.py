"""localdata_mcp/process/composition/streaming_exec/combiners.py — C-1 combiners.

FR-603's soundness constraint, new code by design (no harvest source):
any streaming stage computing variance/covariance-class statistics
combines per-chunk partials via the numerically stable Welford/Chan
parallel update (Chan–Golub–LeVeque 1979) — the same algorithm family
the materialized path uses, which is what makes the S8 row-15d
stream-parity tolerance sound rather than a flakiness source. The
naive E[X²]−E[X]² one-pass formula is forbidden (catastrophic
cancellation on offset-heavy data). Per-chunk partials are computed
two-pass (mean, then centered squares) — stable at chunk scope — and
chunks merge pairwise via the parallel update. Neighbors: executor.py
folds chunks through these; the pipeline battery asserts parity.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RunningMoments:
    """Mean/variance sufficient statistics: count, mean, and M2 (the
    centered sum of squares) — mergeable without revisiting data."""

    count: int
    mean: float
    m2: float

    @classmethod
    def empty(cls) -> "RunningMoments":
        return cls(count=0, mean=0.0, m2=0.0)

    @classmethod
    def of(cls, values: "np.ndarray") -> "RunningMoments":
        """One chunk's partial, two-pass (mean first, then centered
        squares — stable at chunk scope)."""
        data = np.asarray(values, dtype=float)
        if data.size == 0:
            return cls.empty()
        mean = float(np.mean(data))
        m2 = float(np.sum((data - mean) ** 2))
        return cls(count=int(data.size), mean=mean, m2=m2)

    def merged(self, other: "RunningMoments") -> "RunningMoments":
        """The Chan parallel update: exact count/mean, M2 corrected by
        the between-chunk mean shift."""
        if self.count == 0:
            return other
        if other.count == 0:
            return self
        total = self.count + other.count
        delta = other.mean - self.mean
        mean = self.mean + delta * other.count / total
        m2 = self.m2 + other.m2 + delta * delta * self.count * other.count / total
        return RunningMoments(count=total, mean=mean, m2=m2)

    def variance(self, ddof: int = 1) -> "float | None":
        """Sample variance, None when undefined (count <= ddof)."""
        if self.count <= ddof:
            return None
        return self.m2 / (self.count - ddof)

    def std(self, ddof: int = 1) -> "float | None":
        variance = self.variance(ddof)
        return None if variance is None else float(np.sqrt(variance))


@dataclass(frozen=True)
class RunningComoments:
    """Covariance sufficient statistics for a paired series: the two
    marginal moment sets plus C2, the centered cross-product sum."""

    count: int
    mean_x: float
    mean_y: float
    m2_x: float
    m2_y: float
    c2: float

    @classmethod
    def empty(cls) -> "RunningComoments":
        return cls(count=0, mean_x=0.0, mean_y=0.0, m2_x=0.0, m2_y=0.0, c2=0.0)

    @classmethod
    def of(cls, xs: "np.ndarray", ys: "np.ndarray") -> "RunningComoments":
        """One chunk's paired partial (two-pass, aligned lengths)."""
        data_x = np.asarray(xs, dtype=float)
        data_y = np.asarray(ys, dtype=float)
        if data_x.size != data_y.size:
            raise ValueError(
                f"paired chunk lengths differ: {data_x.size} vs {data_y.size}"
            )
        if data_x.size == 0:
            return cls.empty()
        mean_x = float(np.mean(data_x))
        mean_y = float(np.mean(data_y))
        return cls(
            count=int(data_x.size),
            mean_x=mean_x,
            mean_y=mean_y,
            m2_x=float(np.sum((data_x - mean_x) ** 2)),
            m2_y=float(np.sum((data_y - mean_y) ** 2)),
            c2=float(np.sum((data_x - mean_x) * (data_y - mean_y))),
        )

    def merged(self, other: "RunningComoments") -> "RunningComoments":
        """The Chan parallel update extended to the cross term: C2
        gains delta_x * delta_y scaled by the same count factor M2
        uses."""
        if self.count == 0:
            return other
        if other.count == 0:
            return self
        total = self.count + other.count
        delta_x = other.mean_x - self.mean_x
        delta_y = other.mean_y - self.mean_y
        scale = self.count * other.count / total
        return RunningComoments(
            count=total,
            mean_x=self.mean_x + delta_x * other.count / total,
            mean_y=self.mean_y + delta_y * other.count / total,
            m2_x=self.m2_x + other.m2_x + delta_x * delta_x * scale,
            m2_y=self.m2_y + other.m2_y + delta_y * delta_y * scale,
            c2=self.c2 + other.c2 + delta_x * delta_y * scale,
        )

    def covariance(self, ddof: int = 1) -> "float | None":
        """Sample covariance, None when undefined (count <= ddof)."""
        if self.count <= ddof:
            return None
        return self.c2 / (self.count - ddof)

    def correlation(self) -> "float | None":
        """Pearson correlation, None when either side is degenerate
        (zero variance) or the count cannot support it."""
        if self.count <= 1 or self.m2_x <= 0.0 or self.m2_y <= 0.0:
            return None
        return self.c2 / float(np.sqrt(self.m2_x * self.m2_y))
