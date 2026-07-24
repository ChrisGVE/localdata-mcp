"""testbench/batteries/security/path_containment_test.py — NFR-108 at the L3 seam.

The `allowed_paths` filesystem containment (E6.4, unit-tested at the
`contain()` seam in tests/v3/test_path_contain.py, and at the export
function in tests/v3/test_output_export.py) proven through the wire the
agent uses. Every file-accepting tool parameter — read side (`read_file`,
`query_file`) and write side (`export_result`) — is driven with the full
traversal payload set NFR-108's acceptance names (`../` sequences,
absolute paths outside the root, symlink escapes, `~` expansion) and each
is refused as a structured FR-403 error; the canonical real path is
resolved before the comparison, so no payload reaches a byte of the
protected file, and a positive control per side proves containment admits
the legitimate in-root path.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Callable

import pandas as pd
import pytest

from . import _seam

_SECRET_MARKER = "TOPSECRET-do-not-leak"


@pytest.fixture()
def contained(tmp_path: Path):
    """A root contained to `<tmp>/root`, with a secret file planted just
    outside it and a symlink inside pointing at that secret — the material
    every traversal payload below tries, and fails, to reach."""
    root = tmp_path / "root"
    root.mkdir()
    secret = tmp_path / "secret.txt"
    secret.write_text(_SECRET_MARKER)
    escape_link = root / "innocent.txt"
    escape_link.symlink_to(secret)
    with _seam.booted(allowed_paths=(str(root),)):
        yield root, secret, escape_link


# Each payload builds an out-of-root path from (root, secret, link). The
# four kinds are exactly NFR-108's named traversal classes.
_TRAVERSAL: tuple[tuple[str, Callable[[Path, Path, Path], str]], ...] = (
    ("absolute_outside", lambda root, secret, link: str(secret)),
    ("dotdot_escape", lambda root, secret, link: str(root / ".." / "secret.txt")),
    ("symlink_escape", lambda root, secret, link: str(link)),
    ("tilde_expansion", lambda root, secret, link: "~/secret.txt"),
)


@pytest.mark.parametrize("tool", ["read_file", "query_file"])
@pytest.mark.parametrize(("kind", "build"), _TRAVERSAL, ids=[k for k, _ in _TRAVERSAL])
def test_read_side_traversal_is_refused(
    contained, tool: str, kind: str, build: Callable[[Path, Path, Path], str]
) -> None:
    root, secret, link = contained
    path = build(root, secret, link)
    arguments = {"path": path}
    if tool == "query_file":
        arguments["sql"] = "SELECT 1"
    envelope = _seam.call_envelope(tool, arguments)
    _seam.expect_refused(envelope)
    assert _SECRET_MARKER not in str(envelope.get("data"))


@pytest.mark.parametrize(("kind", "build"), _TRAVERSAL, ids=[k for k, _ in _TRAVERSAL])
def test_write_side_traversal_is_refused(
    contained, kind: str, build: Callable[[Path, Path, Path], str]
) -> None:
    root, secret, link = contained
    path = build(root, secret, link)
    before = secret.read_text()
    envelope = _seam.call_envelope(
        "export_result", {"format": "csv", "path": path, "source": [{"x": 1}]}
    )
    _seam.expect_refused(envelope)
    # The protected file is untouched and no out-of-root artifact appeared.
    assert secret.read_text() == before


# -- positive controls: containment admits the legitimate in-root path --


def test_read_file_inside_root_succeeds(contained) -> None:
    root, _secret, _link = contained
    target = root / "ok.csv"
    pd.DataFrame({"id": [1, 2]}).to_csv(target, index=False)
    data = _seam.expect_ok(_seam.call_envelope("read_file", {"path": str(target)}))
    assert data["rows"][0][0] == 1


def test_query_file_inside_root_succeeds(contained) -> None:
    root, _secret, _link = contained
    db = root / "ok.db"
    with sqlite3.connect(db) as connection:
        connection.execute("CREATE TABLE t (id INTEGER)")
        connection.executemany("INSERT INTO t VALUES (?)", [(1,), (2,)])
    data = _seam.expect_ok(
        _seam.call_envelope(
            "query_file", {"path": str(db), "sql": "SELECT id FROM t ORDER BY id"}
        )
    )
    assert data["rows"] == [[1], [2]]


def test_export_inside_root_succeeds(contained) -> None:
    root, _secret, _link = contained
    target = root / "out.csv"
    _seam.expect_ok(
        _seam.call_envelope(
            "export_result",
            {"format": "csv", "path": str(target), "source": [{"x": 1}, {"x": 2}]},
        )
    )
    assert target.exists()
