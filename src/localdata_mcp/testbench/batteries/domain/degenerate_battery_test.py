"""testbench/batteries/domain/degenerate_battery_test.py — E10.x10 + x3 slice.

The degenerate-input rows: each of the shared sentinel's four signal
classes (S3.3), driven end to end through a real domain tool at the L3
seam, must land as a STRUCTURED error envelope — never a silent
success, never a bare traceback:

- **empty group** (class 3) — a grouping column with one group refused
  before it reaches the statistic;
- **singular / rank-deficient matrix** (class 4) — a collinear design
  trips the rank/condition sentinel;
- **zero-variance column** (class 1) — a constant column drives the
  statistic to NaN, caught as non-finite;
- **non-converged optimizer** (class 2) — an unbounded LP reports a
  non-zero optimizer status.

The x3 fault-injection leg confirms the same wrapper turns an
arbitrary domain-library exception (a malformed argument that makes
scipy raise deep inside) into a structured error the client can read,
not a transport-level crash.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterator

import anyio
import pandas as pd
import pytest
from fastmcp import Client

import localdata_mcp.ingest.runtime as runtime
from localdata_mcp.nexus.chokepoint.guard import Chokepoint
from localdata_mcp.nexus.config.models import ConfigModel, SecurityConfig
from localdata_mcp.nexus.contract.registry import default_registry
from localdata_mcp.nexus.response.shaping import configure_shaping
from localdata_mcp.server.mcp_app import app


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


def _envelope(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    async def session() -> dict[str, Any]:
        async with Client(app) as client:
            result = await client.call_tool(name, arguments)
            assert not result.is_error  # transport ok; error lives in the envelope
            if isinstance(result.structured_content, dict) and (
                "inline" in result.structured_content
            ):
                return result.structured_content
            payload = json.loads(result.content[0].text)
            assert isinstance(payload, dict)
            return payload

    return anyio.run(session)


def _assert_structured_error(envelope: dict[str, Any]) -> None:
    assert envelope["error"] is not None, envelope
    assert envelope["inline"] is None
    assert envelope["data"] is None
    assert isinstance(envelope["error"].get("message"), str)


def _write(tmp_path: Path, name: str, frame: pd.DataFrame) -> str:
    target = tmp_path / name
    frame.to_csv(target, index=False)
    return str(target)


def test_empty_group_is_refused(bench: Path) -> None:
    """Class 3 (degenerate shape): a one-group comparison is refused."""
    path = _write(
        bench,
        "one_group.csv",
        pd.DataFrame({"v": [1.0, 2.0, 3.0], "g": ["a", "a", "a"]}),
    )
    _assert_structured_error(
        _envelope(
            "analyze_hypothesis_test",
            {
                "path": path,
                "test_type": "ttest_ind",
                "column": "v",
                "group_column": "g",
            },
        )
    )


def test_rank_deficient_design_is_refused(bench: Path) -> None:
    """Class 4 (rank deficiency): a perfectly collinear feature pair
    trips the condition-number/rank sentinel."""
    path = _write(
        bench,
        "collinear.csv",
        pd.DataFrame(
            {
                "y": [1.0, 2.0, 3.0, 4.0, 5.0],
                "x1": [1.0, 2.0, 3.0, 4.0, 5.0],
                "x2": [2.0, 4.0, 6.0, 8.0, 10.0],
            }
        ),
    )
    _assert_structured_error(
        _envelope(
            "analyze_regression",
            {"path": path, "target_column": "y", "feature_columns": ["x1", "x2"]},
        )
    )


def test_zero_variance_column_is_refused(bench: Path) -> None:
    """Class 1 (non-finite): a constant column makes the correlation
    statistic NaN, caught by the sentinel."""
    path = _write(
        bench,
        "constant.csv",
        pd.DataFrame({"a": [5.0, 5.0, 5.0, 5.0], "b": [1.0, 2.0, 3.0, 4.0]}),
    )
    _assert_structured_error(
        _envelope(
            "analyze_hypothesis_test",
            {
                "path": path,
                "test_type": "correlation",
                "column": "a",
                "second_column": "b",
            },
        )
    )


def test_non_converged_optimizer_is_refused(bench: Path) -> None:
    """Class 2 (non-convergence): an unbounded LP reports a non-zero
    optimizer status, converted to a structured error."""
    # Minimise -x with no upper bound: unbounded below.
    path = _write(bench, "unbounded.csv", pd.DataFrame({"cost": [-1.0]}))
    _assert_structured_error(
        _envelope(
            "solve_linear_program",
            {
                "path": path,
                "objective_column": "cost",
                "bounds": [[0.0, None]],
            },
        )
    )


def test_fault_injection_yields_a_structured_error(bench: Path) -> None:
    """E10.x3: a domain-library exception (an impossible cluster count)
    reaches the client as a structured error, not a bare traceback."""
    path = _write(bench, "tiny.csv", pd.DataFrame({"x": [1.0, 2.0], "y": [1.0, 2.0]}))
    _assert_structured_error(
        _envelope(
            "analyze_clusters",
            {"path": path, "method": "kmeans", "n_clusters": 50},
        )
    )
