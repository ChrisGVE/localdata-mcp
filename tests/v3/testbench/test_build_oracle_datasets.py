"""tests/v3/testbench/test_build_oracle_datasets.py — the E14.1 script.

Exercises the network-free surface of scripts/build_oracle_datasets.py: the
--print-hash CLI (what CI assigns to LOCALDATA_DATASET_HASH) and the committed
authored .xls fixture the base battery reads. The provisioning/fetch path is
network-bound and not run here — its logic is the pure manifest module, tested
in test_manifest.py.
"""

from __future__ import annotations

import subprocess
import sys
from importlib.resources import files
from pathlib import Path

from localdata_mcp.testbench.fixtures import manifest as m

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT = _REPO_ROOT / "scripts" / "build_oracle_datasets.py"


def _fixtures_dir() -> Path:
    return Path(str(files("localdata_mcp.testbench.fixtures")))


def test_print_hash_matches_committed_manifest() -> None:
    result = subprocess.run(
        [sys.executable, str(_SCRIPT), "--print-hash"],
        capture_output=True,
        text=True,
        check=True,
    )
    committed = m.load_manifest(_fixtures_dir() / "dataset_manifest.json")
    assert result.stdout.strip() == committed.combined_sha256


def test_committed_xls_matches_its_manifest_hash() -> None:
    xls = _fixtures_dir() / "datasets" / "base_excel.xls"
    assert xls.is_file()
    committed = m.load_manifest(_fixtures_dir() / "dataset_manifest.json")
    entry = next(e for e in committed.entries if e.name == "excel_xls_fixture")
    assert m.sha256_bytes(xls.read_bytes()) == entry.sha256


def test_committed_xls_reads_as_the_base_frame() -> None:
    from localdata_mcp.ingest.connectors.file.readers import _read_xls

    xls = _fixtures_dir() / "datasets" / "base_excel.xls"
    frame = _read_xls(xls)
    assert list(frame.columns) == ["id", "label"]
    assert frame.iloc[0]["id"] == 1
    assert frame.iloc[0]["label"] == "a"
