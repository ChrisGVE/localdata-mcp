"""testbench/batteries/domain/bi_battery_test.py — E10.h slice.

The NFR-502c domain battery's business-intelligence rows: FR-301 L3
coverage, the FR-307 (E10.x2) segment-reachability enumeration, and
the FR-309 (E10.x4) configurable-customer-column closure. The BI
formulas are arithmetic (no external statistical library owns them),
so the FR-304 oracle is the independent closed-form recomputation in
this file — every CLV number is re-derived from the fixture by hand
alongside the tool call.
"""

from __future__ import annotations

import json
import os
from itertools import product
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
from localdata_mcp.process.domains.business_intelligence.rfm import (
    SEGMENT_RULES,
    _assign_segment,
)
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


def test_every_documented_segment_is_reachable() -> None:
    """FR-307 (E10.x2): the enumeration over the whole 1..5 score
    space assigns EVERY documented segment — main's dead branches
    ('Need Attention', 'Lost') included."""
    documented = {name for name, _rule in SEGMENT_RULES}
    assigned = {
        _assign_segment(r, f, m)
        for r, f, m in product(range(1, 6), range(1, 6), range(1, 6))
    }
    assert assigned == documented


def test_rfm_segments_a_transaction_fixture(bench: Path) -> None:
    """analyze_rfm at L3: scores, segments, and summary coherence."""
    target = bench / "transactions.csv"
    rows = []
    # Ten customers with spread-out behavior; c0 buys often, recently,
    # and big — c9 bought once, long ago, small.
    for customer in range(10):
        orders = 10 - customer
        for order in range(orders):
            rows.append(
                {
                    "buyer": f"c{customer}",
                    "when": f"2026-{customer + 1:02d}-{order + 1:02d}",
                    "total": float((10 - customer) * 10),
                }
            )
    pd.DataFrame(rows).to_csv(target, index=False)
    data = _data(
        _call(
            "analyze_rfm",
            {
                "path": str(target),
                "customer_column": "buyer",
                "date_column": "when",
                "value_column": "total",
            },
        )
    )
    assert data["n_customers"] == 10
    assert sum(entry["customer_count"] for entry in data["segments"].values()) == 10
    scores = {entry["customer"]: entry for entry in data["customers"]}
    assert scores["c0"]["frequency"] == 10
    assert scores["c9"]["frequency"] == 1
    assert scores["c0"]["M"] > scores["c9"]["M"]
    # c9's single order is the LATEST month in the fixture, so its
    # recency score tops c0's (whose orders all sit in month one).
    assert scores["c0"]["R"] < scores["c9"]["R"]


def test_clv_uses_the_named_customer_column_and_the_documented_formula(
    bench: Path,
) -> None:
    """FR-309 (E10.x4): a non-customer_id identifier column produces
    correct CLV — every number re-derived by hand here (FR-304)."""
    target = bench / "purchases.csv"
    pd.DataFrame(
        {
            "member": ["a", "a", "a", "b"],
            "day": ["2026-01-01", "2026-01-11", "2026-01-21", "2026-03-01"],
            "spend": [100.0, 200.0, 300.0, 50.0],
        }
    ).to_csv(target, index=False)
    data = _data(
        _call(
            "calculate_clv",
            {
                "path": str(target),
                "customer_column": "member",
                "date_column": "day",
                "value_column": "spend",
            },
        )
    )
    assert data["customer_column"] == "member"
    assert data["n_customers"] == 2
    by_customer = {entry["customer"]: entry for entry in data["customers"]}
    # Hand derivation for 'a': AOV 200, 3 orders over 20 days -> freq
    # 3/21, CLV = 200 * (3/21) * 0.2 * 365.
    config = ConfigModel()
    rtol = config.testbench.tol_closed_form_rtol
    a = by_customer["a"]
    assert a["total_spent"] == pytest.approx(600.0, rel=rtol)
    assert a["avg_order_value"] == pytest.approx(200.0, rel=rtol)
    assert a["lifespan_days"] == 20
    assert a["clv_estimate"] == pytest.approx(200.0 * (3 / 21) * 0.2 * 365, rel=rtol)
    # 'b': one order, lifespan 0 -> freq 1, CLV = 50 * 1 * 0.2 * 365.
    b = by_customer["b"]
    assert b["clv_estimate"] == pytest.approx(50.0 * 0.2 * 365, rel=rtol)
    assert data["clv_distribution"]["max"] >= data["clv_distribution"]["min"]
