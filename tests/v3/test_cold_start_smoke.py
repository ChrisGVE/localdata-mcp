"""tests/v3/test_cold_start_smoke.py — E8.6: the drift-signal harness.

The cold-start smoke script itself works: a fresh-interpreter
measurement comes back as parseable JSON with positive samples, and
the non-gating contract holds — an exceeded bound WARNS but exits 0
(E16.3 flips `--gate` on). The row-17 numeric bound lives in the CI
workflow invocation, never here (S8 Home discipline).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "cold_start_smoke.py"


def _run(*extra: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(_SCRIPT), "--runs", "1", *extra],
        capture_output=True,
        text=True,
    )


class TestColdStartSmoke:
    def test_measurement_is_parseable_and_positive(self) -> None:
        completed = _run()
        assert completed.returncode == 0, completed.stderr
        report = json.loads(completed.stdout.strip().splitlines()[-1])
        assert report["worst_seconds"] > 0
        assert len(report["samples_seconds"]) == 1

    def test_exceeded_bound_warns_but_does_not_gate(self) -> None:
        """Non-gating until E16.3: an absurdly tight bound trips the
        warning yet exits 0."""
        completed = _run("--bound", "0.001")
        assert completed.returncode == 0
        assert "row-17" in completed.stderr

    def test_gate_flag_makes_the_bound_blocking(self) -> None:
        """The E16.3 flip already works: --gate turns the same breach
        into a non-zero exit."""
        completed = _run("--bound", "0.001", "--gate")
        assert completed.returncode == 1
