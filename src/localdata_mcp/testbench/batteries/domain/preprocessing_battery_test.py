"""testbench/batteries/domain/preprocessing_battery_test.py — E10.x8 slice (FR-303).

FR-303's acceptance: the data-prep stages produce correct output on a
fixture with known missing/mistyped values, and — the explicit L3
test the PRD names — omitting `missing_strategy` applies `drop` (the
value-fabricating-nothing default). The prep-then-process pipeline
fixture (a prep stage feeding a downstream Process tool) is exercised
here by cleaning a source, writing it back, and profiling the result.
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


def _messy_csv(tmp_path: Path) -> str:
    target = tmp_path / "messy.csv"
    pd.DataFrame({"a": [1.0, None, 3.0, None], "b": [10.0, 20.0, None, 40.0]}).to_csv(
        target, index=False
    )
    return str(target)


def test_omitting_missing_strategy_applies_drop(bench: Path) -> None:
    """FR-303's named L3 test: no missing_strategy → drop."""
    data = _data(_call("prepare_missing_values", {"path": _messy_csv(bench)}))
    assert data["missing_strategy"] == "drop"
    # Only the one fully-present row survives (row 0: a=1, b=10).
    assert data["total_rows"] == 1
    assert data["missing_after"] == 0
    assert data["rows"] == [[1.0, 10.0]]


def test_mean_imputation_fills_the_column_mean(bench: Path) -> None:
    """mean strategy fills a's gaps with a's mean (2.0)."""
    data = _data(
        _call(
            "prepare_missing_values",
            {
                "path": _messy_csv(bench),
                "columns": ["a"],
                "missing_strategy": "mean",
            },
        )
    )
    assert data["total_rows"] == 4
    a_values = [row[0] for row in data["rows"]]
    assert a_values == pytest.approx([1.0, 2.0, 3.0, 2.0])


def test_convert_types_reports_failed_casts(bench: Path) -> None:
    """convert_types coerces and counts the cells that failed."""
    target = bench / "typed.csv"
    pd.DataFrame({"n": ["1", "2", "oops", "4"]}).to_csv(target, index=False)
    data = _data(
        _call(
            "convert_types",
            {"path": str(target), "conversions": {"n": "numeric"}},
        )
    )
    assert data["conversions"]["n"]["target"] == "numeric"
    assert data["conversions"]["n"]["failed_cells"] == 1
    assert data["rows"][2][0] is None


def test_prep_then_profile_pipeline(bench: Path) -> None:
    """The prep-then-process fixture: clean a source, write it back,
    profile the cleaned result (FR-303 composability)."""
    cleaned = _data(
        _call(
            "prepare_missing_values",
            {
                "path": _messy_csv(bench),
                "missing_strategy": "constant",
                "fill_value": "0",
            },
        )
    )
    written = bench / "cleaned.csv"
    pd.DataFrame(cleaned["rows"], columns=cleaned["columns"]).to_csv(
        written, index=False
    )
    profile = _data(_call("profile_data", {"path": str(written)}))
    assert profile["row_count"] == 4
    assert profile["columns"]["a"]["null_count"] == 0
