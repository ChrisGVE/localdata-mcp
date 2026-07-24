"""testbench/batteries/security/destructive_ops_test.py — NFR-115 at the L3 seam.

A destructive local operation must carry an explicit, unambiguous request
or be refused — never defaulted to proceed. Proven through the wire the
agent uses, across the two named destructive surfaces:

- **export over an existing file** — `export_result` at an existing target
  is refused unless the caller passes the explicit `overwrite=true`
  disambiguator (the guard at nexus/export/interface.py, E11); the refused
  call leaves the prior file byte-for-byte intact, and the same call *with*
  the disambiguator is the positive control.
- **wildcard/bulk delete** — the key/tree/graph delete tools address a
  single named target; a wildcard or pattern key resolves as a literal
  property miss, never a bulk match, so a wildcard delete is refused while
  the genuinely-named target survives. A named single-key delete is the
  positive control that explicit deletion still works.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from . import _seam


@pytest.fixture()
def bench(tmp_path: Path):
    """A contained root plus a writable key-value endpoint over a seeded
    file — the substrate both destructive surfaces act on."""
    kv_dsn = f"kv+sqlite:///{tmp_path / 'kv.db'}"
    declarations = _seam.declare(kv=(kv_dsn, "read_write"))
    with _seam.booted(allowed_paths=(str(tmp_path),), declarations=declarations):
        _seam.expect_ok(
            _seam.call_envelope(
                "set_value",
                {"endpoint": "kv", "path": "root.n", "key": "keep", "value": "1"},
            )
        )
        yield tmp_path


# -- export-over-existing-file requires the explicit disambiguator -------


def test_export_over_existing_file_without_overwrite_is_refused(bench) -> None:
    target = bench / "out.csv"
    original = "prior,content\n1,2\n"
    target.write_text(original, encoding="utf-8")
    envelope = _seam.call_envelope(
        "export_result",
        {"format": "csv", "path": str(target), "source": [{"x": 9}]},
    )
    error = _seam.expect_refused(envelope)
    assert "exists" in error["message"], error["message"]
    # The prior file is byte-for-byte intact — the refused write touched
    # nothing.
    assert target.read_text(encoding="utf-8") == original


def test_export_over_existing_file_with_overwrite_succeeds(bench) -> None:
    target = bench / "out.csv"
    target.write_text("prior,content\n1,2\n", encoding="utf-8")
    _seam.expect_ok(
        _seam.call_envelope(
            "export_result",
            {
                "format": "csv",
                "path": str(target),
                "source": [{"x": 9}],
                "overwrite": True,
            },
        )
    )
    replaced = pd.read_csv(target)
    assert list(replaced.columns) == ["x"]
    assert replaced["x"].tolist() == [9]


# -- wildcard/bulk delete is refused; only a named target deletes --------


@pytest.mark.parametrize("wildcard", ["*", "%", "keep*"])
def test_wildcard_delete_is_refused_and_named_target_survives(
    bench, wildcard: str
) -> None:
    envelope = _seam.call_envelope(
        "delete_key", {"endpoint": "kv", "path": "root.n", "key": wildcard}
    )
    _seam.expect_refused(envelope)
    # The genuinely-named key was never a wildcard match — it survives.
    data = _seam.expect_ok(
        _seam.call_envelope(
            "get_value", {"endpoint": "kv", "path": "root.n", "key": "keep"}
        )
    )
    assert data["value"] == 1


def test_named_key_delete_succeeds(bench) -> None:
    _seam.expect_ok(
        _seam.call_envelope(
            "delete_key", {"endpoint": "kv", "path": "root.n", "key": "keep"}
        )
    )
    gone = _seam.call_envelope(
        "get_value", {"endpoint": "kv", "path": "root.n", "key": "keep"}
    )
    _seam.expect_refused(gone)
