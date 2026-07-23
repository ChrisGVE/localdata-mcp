"""testbench/batteries/perf_memory/discovery_latency_test.py — X-5 (E9.5).

NFR-201/FR-205's warm discovery floor as a perf-battery assertion:
`list_tools()` p99 under `testbench.discovery_p99_ms` (S8 row 28 —
the LOCKED restatement lives in the ConfigModel, never here), WARM
(an untimed first call absorbs one-time import/registration cost),
dataset-independent by construction — no endpoint is declared, no
data exists, and section 6.1's build-time generation keeps codegen
off this path — at the FULL registered v3 tool count. The sample
count is a named non-config constant: enough draws that p99 is the
tail, cheap enough for the per-PR smoke tier (S7.4).
"""

from __future__ import annotations

import math
import time

import anyio
from fastmcp import Client

from localdata_mcp.nexus.config.models import ConfigModel
from localdata_mcp.nexus.contract.registry import default_registry
from localdata_mcp.nexus.contract.spec_modules import load_spec_modules
from localdata_mcp.server.mcp_app import app

_SAMPLES = 40  # tail-resolving draw count (non-config bench shape)


def test_warm_list_tools_p99_under_the_row_28_floor() -> None:
    bound_ms = ConfigModel().testbench.discovery_p99_ms
    load_spec_modules()
    registered = {spec.name for spec in default_registry()}

    async def session() -> tuple[list[float], int]:
        async with Client(app) as client:
            served = await client.list_tools()  # warm-up, untimed
            timings: list[float] = []
            for _ in range(_SAMPLES):
                started = time.perf_counter()
                await client.list_tools()
                timings.append((time.perf_counter() - started) * 1000.0)
            return timings, len(served)

    timings, served_count = anyio.run(session)
    # The full v3 tool count is on the path — the floor is asserted
    # against the real surface, not a subset.
    assert served_count >= len(registered)
    ordered = sorted(timings)
    p99 = ordered[min(math.ceil(len(ordered) * 0.99) - 1, len(ordered) - 1)]
    assert p99 < bound_ms, (
        f"warm list_tools p99 {p99:.2f} ms breaches the "
        f"testbench.discovery_p99_ms floor ({bound_ms} ms) at "
        f"{served_count} tools"
    )
