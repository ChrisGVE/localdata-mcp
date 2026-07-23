"""testbench/batteries/base/purity_wrap_test.py — NFR-303 full-scope wrap (E14.6).

The stdout-purity primitive (testbench/purity_runner.py) applied at
full scope: a real child-process session that drives base ingest AND
domain analysis tools — the code paths that import and run the heavy
C-extension stack (numpy/scipy/sklearn/statsmodels/matplotlib), the one
place a stray `print()` or a library writing to fd 1 would corrupt the
JSON-RPC transport. The child loads its endpoints and `allowed_paths`
from a `./.localdata.toml` in its working directory (the NX-2
project-file layer), so the domain tools resolve their addressed CSV
fixture for real. `run_session` captures TRUE fd 1 and raises
PurityViolation on any non-frame byte; the assertions below prove the
positive round-trip too (every request answered, clean exit).

E2.5 owns the primitive (exercised on `ping` in tests/v3/
test_purity_runner.py); this module is E14.6's extension of it across
the base + domain surface (S7.1: extended cost-free to the others,
since the capture primitive is shared). Neighbors: purity_runner.py is
the mechanism; ingest_battery_test.py / the domain battery are the
in-process L3 slices this re-drives through the guarded descriptor.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator

import pandas as pd
import pytest

from localdata_mcp.testbench.purity_runner import (
    SessionResult,
    initialize_request,
    initialized_notification,
    run_session,
    tool_call_request,
)

_KV_ENDPOINT = "store_kv"
_FIXTURE_CSV = "numeric.csv"


def _write_workspace(workspace: Path) -> None:
    """A child working directory carrying its config and one numeric
    CSV fixture — three float columns, rows enough for clustering and
    regression to run their real fits."""
    frame = pd.DataFrame(
        {
            "a": [float(index) for index in range(30)],
            "b": [float(index) * 2.0 + 1.0 for index in range(30)],
            "c": [float((index * 7) % 5) for index in range(30)],
        }
    )
    frame.to_csv(workspace / _FIXTURE_CSV, index=False)
    kv_path = workspace / "kv.db"
    (workspace / ".localdata.toml").write_text(
        "[security]\n"
        f'allowed_paths = ["{workspace}"]\n'
        "\n"
        f"[endpoints.{_KV_ENDPOINT}]\n"
        f'dsn = "kv+sqlite:///{kv_path}"\n'
        'posture = "read_write"\n'
    )


def _child_env(child_home: Path) -> dict[str, str]:
    """Hermetic child environment: no LOCALDATA_* override leaks in, and
    HOME points at an empty dir so only the project-file config (written
    into the child's cwd) and the model defaults apply."""
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("LOCALDATA_")
    }
    env["HOME"] = str(child_home)
    return env


# The full base + domain call sequence, request ids ascending from 2
# (initialize is id 1). Each domain tool addresses the CSV fixture by
# path; a domain-level refusal is still a JSON-RPC frame, so purity
# holds regardless of analytical outcome — what matters is that the
# heavy library code ran under the captured descriptor.
def _session_messages() -> list[dict]:
    csv = _FIXTURE_CSV
    return [
        initialize_request(request_id=1),
        initialized_notification(),
        # base ingest + kv round-trip through the guarded descriptor
        tool_call_request(request_id=2, name="read_file", arguments={"path": csv}),
        tool_call_request(
            request_id=3,
            name="set_value",
            arguments={
                "endpoint": _KV_ENDPOINT,
                "path": "root",
                "key": "k",
                "value": "1",
                "value_type": "integer",
            },
        ),
        tool_call_request(
            request_id=4,
            name="get_value",
            arguments={"endpoint": _KV_ENDPOINT, "path": "root", "key": "k"},
        ),
        # domain analysis — scipy, sklearn, statsmodels, matplotlib
        tool_call_request(
            request_id=5,
            name="analyze_hypothesis_test",
            arguments={"path": csv, "column": "a"},
        ),
        tool_call_request(
            request_id=6,
            name="analyze_regression",
            arguments={"path": csv, "target_column": "b"},
        ),
        tool_call_request(
            request_id=7,
            name="analyze_clusters",
            arguments={"path": csv, "seed": 42},
        ),
        tool_call_request(
            request_id=8,
            name="render_chart",
            arguments={"path": csv, "kind": "histogram", "encoding": {"x": "a"}},
        ),
    ]


@pytest.fixture(scope="module")
def wrapped_session(tmp_path_factory: pytest.TempPathFactory) -> SessionResult:
    """One real child session across the base + domain surface; the
    capture verifies stdout purity before returning."""
    workspace = tmp_path_factory.mktemp("purity-workspace")
    child_home = tmp_path_factory.mktemp("purity-home")
    _write_workspace(workspace)
    return run_session(
        _session_messages(),
        cwd=str(workspace),
        env=_child_env(child_home),
    )


class TestFullScopeStdoutPurity:
    def test_child_exits_cleanly(self, wrapped_session: SessionResult) -> None:
        assert wrapped_session.returncode == 0

    def test_stdout_is_only_jsonrpc_frames(
        self, wrapped_session: SessionResult
    ) -> None:
        # run_session already raised on any impurity; assert the stream
        # was non-empty (the session really ran) and every frame is a
        # well-formed JSON-RPC object.
        assert wrapped_session.frames
        assert all("jsonrpc" in frame for frame in wrapped_session.frames)

    def test_every_request_was_answered(self, wrapped_session: SessionResult) -> None:
        answered = wrapped_session.responses_by_id()
        # initialize (id 1) plus the seven base + domain calls (ids 2-8).
        assert set(answered) == set(range(1, 9))

    def test_nothing_leaked_to_stderr_as_a_frame(
        self, wrapped_session: SessionResult
    ) -> None:
        # Diagnostics belong on stderr (bootstrap logging); fd 1 carries
        # frames only. This asserts the split held: stdout parsed as
        # frames (above) while stderr is where any log text went.
        assert wrapped_session.raw_stdout.endswith(b"\n")
