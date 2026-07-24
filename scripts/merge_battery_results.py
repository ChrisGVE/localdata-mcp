#!/usr/bin/env python3
"""scripts/merge_battery_results.py — E14.4's post-job results merge.

Folds every per-worker results file a battery job produced into the
canonical results store (NFR-508; ARCHITECTURE.md section 5). Each CI
battery worker writes its own SQLite file (batteries/conftest.py's
plugin); this step, run once after the matrix finishes, merges them and
applies the retention prune in the same transaction (merge.py). The
retention bound is read from the config nexus — `testbench.
results_retention_runs` — never restated here (its one home is the
ConfigModel). The merge is idempotent under the (run_id, test_id) key,
so re-running the step is safe.

Usage:
  python scripts/merge_battery_results.py --canonical STORE.db WORKER.db ...
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path
from typing import List, Optional, Sequence

from localdata_mcp.nexus.config import load_config
from localdata_mcp.testbench.results_store import merge, schema

# Write tunables for this trusted, non-LLM-facing merge path — not
# operator ConfigModel fields (the store's DDL owns them).
_BUSY_TIMEOUT_MS = 5000
_MAX_WRITE_ATTEMPTS = 5


def _expand(patterns: Sequence[str]) -> List[Path]:
    """Resolve worker arguments, accepting both literal paths and globs."""
    paths: List[Path] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        paths.extend(Path(match) for match in (matches or [pattern]))
    return paths


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--canonical", required=True, help="canonical store path (created if absent)"
    )
    parser.add_argument("workers", nargs="+", help="per-worker files or globs")
    options = parser.parse_args(argv)

    retention = load_config().model.testbench.results_retention_runs
    worker_paths = _expand(options.workers)

    connection = schema.connect(options.canonical, busy_timeout_ms=_BUSY_TIMEOUT_MS)
    schema.ensure_schema(connection)
    merged: List[str] = []
    refused: List[dict] = []
    try:
        for worker in worker_paths:
            if not worker.exists():
                refused.append({"file": str(worker), "reason": "missing"})
                continue
            try:
                merge.merge_worker_file(
                    connection,
                    worker,
                    retention_runs=retention,
                    max_write_attempts=_MAX_WRITE_ATTEMPTS,
                )
                merged.append(str(worker))
            except merge.MergeRefusedError as error:
                refused.append({"file": str(worker), "reason": str(error)})
        total_runs = int(
            connection.execute("SELECT COUNT(*) FROM battery_runs").fetchone()[0]
        )
    finally:
        connection.close()

    print(
        json.dumps(
            {
                "canonical": options.canonical,
                "retention_runs": retention,
                "merged": merged,
                "refused": refused,
                "canonical_run_count": total_runs,
            }
        )
    )
    return 1 if refused else 0


if __name__ == "__main__":
    sys.exit(main())
