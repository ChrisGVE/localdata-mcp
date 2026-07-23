"""testbench/batteries/domain/pattern_battery_test.py — E10.c slice.

The NFR-502c domain battery's pattern-recognition rows: FR-301 L3
coverage and the FR-304/NFR-505 dual-assertion oracle. Published
fixture: Fisher's iris (via sklearn's bundled copy — the canonical
published dataset): PCA's first component explains 92.46% of the raw
data's variance (sklearn's own documented decomposition,
explained_variance_ratio_[0] = 0.92462). Reference legs recompute
with sklearn under the SAME pinned seed on both sides (NFR-505);
label comparisons use adjusted Rand (permutation-invariant — the
Hungarian-alignment concern for label-indeterminate methods). The
t-SNE row gates on S8 row 15f's trustworthiness floor via config.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterator

import anyio
import numpy as np
import pandas as pd
import pytest
from fastmcp import Client

import localdata_mcp.ingest.runtime as runtime
from localdata_mcp.nexus.chokepoint.guard import Chokepoint
from localdata_mcp.nexus.config.models import ConfigModel, SecurityConfig
from localdata_mcp.nexus.contract.registry import default_registry
from localdata_mcp.nexus.response.shaping import configure_shaping
from localdata_mcp.server.mcp_app import app

# Published: PCA on raw iris — PC1 explains 92.46% (sklearn's own
# documented decomposition of the canonical Fisher dataset).
_IRIS_PC1_RATIO = 0.9246
_PUBLISHED_RTOL = 1e-3

_SEED = 42


@pytest.fixture()
def bench(tmp_path: Path) -> Iterator[Path]:
    config = ConfigModel(
        security=SecurityConfig(allowed_paths=(str(tmp_path),)),
    )
    guard = Chokepoint.boot(config, environ=dict(os.environ))
    configure_shaping(config, default_registry())
    runtime.configure_ingest(guard)
    yield tmp_path
    runtime._CHOKEPOINT = None
    configure_shaping(ConfigModel(), default_registry())
    guard.shutdown()


def _call(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    async def session() -> dict[str, Any]:
        async with Client(app) as client:
            result = await client.call_tool(name, arguments)
            assert not result.is_error
            if isinstance(result.structured_content, dict) and (
                "inline" in result.structured_content
            ):
                return result.structured_content
            payload = json.loads(result.content[0].text)
            assert isinstance(payload, dict)
            return payload

    return anyio.run(session)


def _data(envelope: dict[str, Any]) -> Any:
    assert envelope["error"] is None, envelope["error"]
    return envelope["data"]


def _iris_csv(tmp_path: Path) -> tuple[str, "np.ndarray[Any, Any]"]:
    from sklearn.datasets import load_iris

    iris = load_iris()
    frame = pd.DataFrame(iris.data, columns=[str(n) for n in iris.feature_names])
    target = tmp_path / "iris.csv"
    frame.to_csv(target, index=False)
    return str(target), iris.data


def _blobs_csv(
    tmp_path: Path,
) -> tuple[str, "np.ndarray[Any, Any]", "np.ndarray[Any, Any]"]:
    from sklearn.datasets import make_blobs

    points, labels = make_blobs(
        n_samples=90, centers=3, cluster_std=0.6, random_state=_SEED
    )
    target = tmp_path / "blobs.csv"
    pd.DataFrame(points, columns=["x", "y"]).to_csv(target, index=False)
    return str(target), points, labels


def test_pca_matches_reference_and_published_iris(bench: Path) -> None:
    """reduce_dimensions pca — both oracle legs + FR-308's ratio key."""
    from sklearn.decomposition import PCA

    path, raw = _iris_csv(bench)
    data = _data(_call("reduce_dimensions", {"path": path, "n_components": 2}))
    config = ConfigModel()
    rtol = config.testbench.tol_closed_form_rtol
    reference = PCA(n_components=2).fit(raw)
    assert data["explained_variance_ratio"] == pytest.approx(
        list(reference.explained_variance_ratio_), rel=rtol
    )
    assert data["explained_variance_ratio"][0] == pytest.approx(
        _IRIS_PC1_RATIO, rel=_PUBLISHED_RTOL
    )
    assert len(data["components"]) == 150


def test_tsne_trustworthiness_meets_the_s8_floor(bench: Path) -> None:
    """reduce_dimensions tsne — S8 row 15f via config (NFR-505 seed pin)."""
    path, _points, _labels = _blobs_csv(bench)
    data = _data(
        _call(
            "reduce_dimensions",
            {"path": path, "method": "tsne", "seed": _SEED},
        )
    )
    config = ConfigModel()
    assert data["trustworthiness"] >= config.testbench.embedding_trustworthiness_min
    assert len(data["components"]) == 90


def test_kmeans_recovers_blobs_against_seeded_reference(bench: Path) -> None:
    """analyze_clusters kmeans — reference leg, permutation-invariant."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score

    path, points, true_labels = _blobs_csv(bench)
    data = _data(
        _call(
            "analyze_clusters",
            {"path": path, "n_clusters": 3, "seed": _SEED},
        )
    )
    reference = KMeans(n_clusters=3, n_init="auto", random_state=_SEED).fit_predict(
        points
    )
    assert adjusted_rand_score(data["labels"], reference) == pytest.approx(1.0)
    assert adjusted_rand_score(data["labels"], true_labels) == pytest.approx(1.0)
    assert data["n_clusters"] == 3
    assert sum(data["clusters"].values()) == 90
    assert data["silhouette_score"] > 0.5


def test_cluster_auto_k_picks_three_on_blobs(bench: Path) -> None:
    """The silhouette sweep lands on the true k for separated blobs."""
    path, _points, _labels = _blobs_csv(bench)
    data = _data(_call("analyze_clusters", {"path": path, "seed": _SEED}))
    assert data["n_clusters"] == 3


def test_zscore_anomalies_find_the_planted_outlier(bench: Path) -> None:
    """detect_anomalies zscore — deterministic three-sigma rule."""
    target = bench / "outlier.csv"
    # A single outlier among n inliers caps at z = (n-1)/sqrt(n) (it
    # inflates its own baseline), so n must be large enough for the
    # planted point to clear the three-sigma cut: 30 rows give z ~ 5.3.
    rng = np.random.default_rng(_SEED)
    values = [float(v) for v in rng.normal(0.0, 0.1, size=29)] + [50.0]
    pd.DataFrame({"metric": values}).to_csv(target, index=False)
    data = _data(_call("detect_anomalies", {"path": str(target), "method": "zscore"}))
    assert data["anomaly_indices"] == [29]
    assert data["anomaly_count"] == 1


def test_isolation_forest_matches_seeded_reference(bench: Path) -> None:
    """detect_anomalies isolation_forest — reference leg, same seed."""
    from sklearn.ensemble import IsolationForest

    path, points, _labels = _blobs_csv(bench)
    data = _data(
        _call(
            "detect_anomalies",
            {"path": path, "seed": _SEED, "contamination": 0.05},
        )
    )
    reference = IsolationForest(contamination=0.05, random_state=_SEED).fit_predict(
        points
    )
    assert data["anomaly_indices"] == [int(i) for i in np.nonzero(reference == -1)[0]]


def test_transform_data_rewrites_the_column(bench: Path) -> None:
    """transform_data — deterministic rewrite with the safety screen."""
    target = bench / "codes.csv"
    pd.DataFrame({"code": ["AB-1", "CD-2", "EF-3"], "keep": [1, 2, 3]}).to_csv(
        target, index=False
    )
    data = _data(
        _call(
            "transform_data",
            {
                "path": str(target),
                "column": "code",
                "find": "-",
                "replace": "_",
            },
        )
    )
    assert data["transformed_count"] == 3
    assert [row[0] for row in data["rows"]] == ["AB_1", "CD_2", "EF_3"]
    assert [row[1] for row in data["rows"]] == [1, 2, 3]
