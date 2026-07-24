#!/usr/bin/env python3
"""scripts/check_battery_run_trailer.py — E14.4's NFR-506 merge gate.

The merge-blocking half of batch-then-diagnose. For every commit in the
checked range that is a fix (`fix:`/`fix(...)` subject) touching
bench-covered source, this gate REQUIRES a `Battery-Run:` trailer citing
the antecedent full-battery run and verifies the two mechanical
conditions against the canonical results store (trailer.verify): the
cited run predates the commit and recorded a failure, and a newer run of
the same battery exists after it. Non-fix commits and commits that touch
no bench-covered path are exempt. The store carries no source-to-test
map, so the "diff addresses THIS failure" clause is the reviewer's — this
gate enforces exactly what a machine can decide, and says so.

Usage:
  python scripts/check_battery_run_trailer.py --store canonical.db \
      --base ORIGIN_SHA --head HEAD_SHA
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from datetime import datetime, timezone
import subprocess
import sys
from typing import List, Optional, Sequence

from localdata_mcp.testbench.results_store import schema, trailer

# Bench-covered source: a fix here must follow batch-then-diagnose. The
# same v3 packages the gates cover (gated_tree.V3_PACKAGES), as paths.
_COVERED_PREFIXES = tuple(
    f"src/localdata_mcp/{package}/"
    for package in ("nexus", "ingest", "explore", "process", "visualize", "output")
)
_FIX_SUBJECT = re.compile(r"^fix(\([^)]*\))?!?:", re.IGNORECASE)


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], capture_output=True, text=True, check=True
    ).stdout


def _commits(base: str, head: str) -> List[str]:
    out = _git("rev-list", "--no-merges", f"{base}..{head}")
    return [line for line in out.splitlines() if line]


def _subject(sha: str) -> str:
    return _git("show", "-s", "--format=%s", sha).strip()


def _message(sha: str) -> str:
    return _git("show", "-s", "--format=%B", sha)


def _commit_iso(sha: str) -> str:
    """Committer time in the store's fixed-width ISO-8601 UTC shape.

    Read as a Unix timestamp (`%ct`) and formatted in UTC here, so the
    comparison is timezone-independent — a `format-local` git date would
    render in the runner's local zone while the store is always UTC.
    """
    unix = int(_git("show", "-s", "--format=%ct", sha).strip())
    return datetime.fromtimestamp(unix, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _touches_covered(sha: str) -> bool:
    files = _git("show", "--name-only", "--format=", sha).splitlines()
    return any(f.startswith(_COVERED_PREFIXES) for f in files if f)


def _check_commit(connection: sqlite3.Connection, sha: str) -> List[str]:
    """Violations for one fix commit; empty when it is exempt or clean."""
    if not _FIX_SUBJECT.match(_subject(sha)) or not _touches_covered(sha):
        return []
    cited = trailer.parse_trailer(_message(sha))
    if cited is None:
        return [
            f"{sha[:8]} is a fix touching bench-covered code but carries no"
            " Battery-Run: trailer (NFR-506 batch-then-diagnose)"
        ]
    return [
        f"{sha[:8]}: {violation}"
        for violation in trailer.verify(
            connection, cited_run_id=cited, commit_iso=_commit_iso(sha)
        )
    ]


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", required=True, help="canonical results store path")
    parser.add_argument("--base", required=True, help="base ref of the checked range")
    parser.add_argument("--head", default="HEAD", help="head ref (default HEAD)")
    parser.add_argument(
        "--busy-timeout-ms", type=int, default=5000, help="store busy timeout"
    )
    options = parser.parse_args(argv)

    connection = schema.connect(options.store, busy_timeout_ms=options.busy_timeout_ms)
    schema.ensure_schema(connection)
    try:
        violations: List[str] = []
        for sha in _commits(options.base, options.head):
            violations.extend(_check_commit(connection, sha))
    finally:
        connection.close()

    print(json.dumps({"violations": violations}, indent=2))
    if violations:
        print(
            f"NFR-506 batch-then-diagnose gate: {len(violations)} violation(s)",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
