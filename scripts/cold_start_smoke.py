#!/usr/bin/env python3
"""scripts/cold_start_smoke.py — E8.6's cold-start drift signal.

Measures process-launch → serve-ready in a FRESH interpreter: import
the entrypoint module (which registers every generated tool wrapper),
load-and-validate a config, and boot the chokepoint — everything
`mcp_app.main()` does short of binding the stdio transport. The S8
row-17 bound (≤ 5 s p99 on the CI reference runner, with a pre-seeded
matplotlib font cache — a cold `fontManager` rebuild reflects no
operator steady state) is passed in via `--bound`; the check is
NON-GATING until E16.3's re-baseline: an exceeded bound prints a
loud warning and still exits 0 (E16.3 flips `--gate` on with a
recorded finding if the bound must move). Home of the numeric bound:
the CI workflow invocation, per S8 row 17 (Home: CI assertion).

Usage: python scripts/cold_start_smoke.py [--bound SECONDS] [--gate]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time

# What the fresh interpreter runs: serve-ready is "tools registered +
# config validated + guard booted" — main() minus the transport bind.
_CHILD = """
import time

started = time.perf_counter()

from localdata_mcp.server.mcp_app import app  # registers every tool
from localdata_mcp.nexus.config.models import ConfigModel
from localdata_mcp.nexus.chokepoint.guard import Chokepoint
from localdata_mcp.nexus.contract.registry import default_registry
from localdata_mcp.nexus.response.shaping import configure_shaping
import localdata_mcp.ingest.runtime as runtime

model = ConfigModel()
configure_shaping(model, default_registry())
guard = Chokepoint.boot(model, environ={})
runtime.configure_ingest(guard)
guard.shutdown()

print(time.perf_counter() - started)
"""


def measure_once() -> float:
    """One cold start in a fresh interpreter; child prints its own
    import-to-ready seconds (interpreter startup excluded — row 17
    measures the SERVER's readiness cost, not Python's)."""
    completed = subprocess.run(
        [sys.executable, "-c", _CHILD],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(completed.stdout.strip().splitlines()[-1])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bound",
        type=float,
        default=None,
        help="seconds; the S8 row-17 bound as configured by CI",
    )
    parser.add_argument("--runs", type=int, default=3, help="cold starts to sample")
    parser.add_argument(
        "--gate",
        action="store_true",
        help="exit non-zero past the bound (E16.3 turns this on)",
    )
    options = parser.parse_args()

    # Pre-seed the matplotlib font cache OUTSIDE the measurement — the
    # row-17 measurement condition (an ephemeral runner's cold
    # fontManager rebuild costs seconds and reflects no steady state).
    subprocess.run(
        [sys.executable, "-c", "import matplotlib.pyplot"],
        capture_output=True,
        check=True,
    )

    samples = sorted(measure_once() for _ in range(options.runs))
    worst = samples[-1]
    print(
        json.dumps(
            {
                "samples_seconds": [round(s, 3) for s in samples],
                "worst_seconds": round(worst, 3),
                "bound_seconds": options.bound,
            }
        )
    )
    if options.bound is not None and worst > options.bound:
        print(
            f"WARNING: cold start {worst:.3f}s exceeds the S8 row-17 "
            f"bound {options.bound}s — early drift signal (non-gating "
            "until E16.3)",
            file=sys.stderr,
        )
        return 1 if options.gate else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
