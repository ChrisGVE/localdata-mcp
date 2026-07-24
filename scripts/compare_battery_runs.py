#!/usr/bin/env python3
"""scripts/compare_battery_runs.py — E14.4's nightly longitudinal diff.

Reports the difference between the two most-recent runs of a battery in
the canonical results store (NFR-508's longitudinal comparison), or
between two explicitly named run ids. Prints the classified diff —
regressions, fixes, added/removed tests, numeric-output changes — as
JSON. Report-only by default; `--fail-on-regression` makes a regression
exit non-zero (the nightly job runs it report-only, surfacing drift
without gating).

Usage:
  python scripts/compare_battery_runs.py --store S.db --battery pipeline
  python scripts/compare_battery_runs.py --store S.db --base R1 --head R2
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from typing import Optional, Sequence

from localdata_mcp.testbench.results_store import compare, schema

_BUSY_TIMEOUT_MS = 5000


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", required=True, help="canonical results store path")
    parser.add_argument("--battery", help="battery name (diff its two latest runs)")
    parser.add_argument("--mode", default="deterministic", help="run mode filter")
    parser.add_argument("--base", help="explicit older run id")
    parser.add_argument("--head", help="explicit newer run id")
    parser.add_argument(
        "--fail-on-regression",
        action="store_true",
        help="exit non-zero when a regression is found",
    )
    options = parser.parse_args(argv)

    connection = schema.connect(options.store, busy_timeout_ms=_BUSY_TIMEOUT_MS)
    schema.ensure_schema(connection)
    try:
        if options.base and options.head:
            diff: Optional[compare.RunDiff] = compare.diff_runs(
                connection, base_run_id=options.base, head_run_id=options.head
            )
        elif options.battery:
            diff = compare.diff_latest(
                connection, battery_name=options.battery, run_mode=options.mode
            )
        else:
            parser.error("provide either --battery or both --base and --head")
    finally:
        connection.close()

    if diff is None:
        print(json.dumps({"comparison": None, "reason": "fewer than two runs"}))
        return 0

    print(json.dumps({"comparison": asdict(diff)}, indent=2))
    if options.fail_on_regression and diff.has_regressions():
        print(
            f"regression: {len(diff.regressions)} test(s) newly failing",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
