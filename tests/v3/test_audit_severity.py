"""tests/v3/test_audit_severity.py — tests for scripts/check_audit_severity.py.

The CVE gate blocks a PR on any vulnerability scoring CVSS >= the given
threshold and reports every severity. These tests drive the pure logic —
report parsing, severity resolution, threshold gating — with injected OSV
records and a stubbed vector scorer, so no network and no cvss library are
needed to run them.
"""

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Dict

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "check_audit_severity.py"


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("check_audit_severity", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Registration must precede exec: the dataclass machinery resolves the
    # module through sys.modules while the class is being created.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def script() -> ModuleType:
    return _load_script()


PIP_AUDIT_REPORT: Dict[str, Any] = {
    "dependencies": [
        {"name": "clean-package", "version": "1.0.0", "vulns": []},
        {
            "name": "hit-package",
            "version": "2.0.0",
            "vulns": [{"id": "GHSA-high", "fix_versions": ["2.0.1"], "aliases": []}],
        },
        {
            "name": "noise-package",
            "version": "3.0.0",
            "vulns": [{"id": "GHSA-low", "fix_versions": [], "aliases": []}],
        },
    ]
}

OSV_RECORDS: Dict[str, Dict[str, Any]] = {
    "GHSA-high": {"severity": [{"type": "CVSS_V3", "score": "VECTOR-9.8"}]},
    "GHSA-low": {"database_specific": {"severity": "LOW"}},
    "GHSA-unknown": {},
}

FAKE_VECTOR_SCORES = {"VECTOR-9.8": 9.8, "VECTOR-5.0": 5.0}


@pytest.fixture()
def patched_scorer(script: ModuleType, monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    monkeypatch.setattr(
        script, "score_cvss_vector", lambda vector: FAKE_VECTOR_SCORES[vector]
    )
    return script


class TestSeverityResolution:
    def test_cvss_vector_wins(self, patched_scorer: ModuleType) -> None:
        score, source = patched_scorer.score_from_osv(OSV_RECORDS["GHSA-high"])
        assert score == 9.8
        assert source == "cvss-vector"

    def test_qualitative_fallback(self, script: ModuleType) -> None:
        score, source = script.score_from_osv(OSV_RECORDS["GHSA-low"])
        assert score is not None and score < 7.0
        assert source == "qualitative"

    def test_unknown_severity_resolves_to_none(self, script: ModuleType) -> None:
        score, source = script.score_from_osv(OSV_RECORDS["GHSA-unknown"])
        assert score is None
        assert source == "unknown"


class TestGate:
    def _findings(self, script: ModuleType) -> list:
        return script.collect_findings(
            PIP_AUDIT_REPORT, fetch=lambda vuln_id: OSV_RECORDS[vuln_id]
        )

    def test_collects_only_vulnerable_packages(
        self, patched_scorer: ModuleType
    ) -> None:
        findings = self._findings(patched_scorer)
        assert {f.package for f in findings} == {"hit-package", "noise-package"}

    def test_blocks_at_and_above_threshold(self, patched_scorer: ModuleType) -> None:
        findings = self._findings(patched_scorer)
        assert patched_scorer.gate(findings, threshold=7.0) is True

    def test_passes_below_threshold(self, patched_scorer: ModuleType) -> None:
        findings = self._findings(patched_scorer)
        below_only = [f for f in findings if f.package == "noise-package"]
        assert patched_scorer.gate(below_only, threshold=7.0) is False

    def test_unknown_severity_blocks(self, script: ModuleType) -> None:
        """Fail-closed: a vulnerability we cannot score must block, not slip."""
        finding = script.Finding(
            package="mystery",
            version="1.0",
            vuln_id="GHSA-unknown",
            fix_versions=[],
            score=None,
            source="unknown",
        )
        assert script.gate([finding], threshold=7.0) is True


class TestEndToEnd:
    def test_main_reports_all_and_gates(
        self,
        patched_scorer: ModuleType,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        report_path = tmp_path / "audit.json"
        report_path.write_text(json.dumps(PIP_AUDIT_REPORT), encoding="utf-8")
        monkeypatch.setattr(
            patched_scorer,
            "fetch_osv_record",
            lambda vuln_id: OSV_RECORDS[vuln_id],
        )
        exit_code = patched_scorer.main(
            ["--report", str(report_path), "--threshold", "7.0"]
        )
        output = capsys.readouterr().out
        assert exit_code == 1
        assert "GHSA-high" in output and "GHSA-low" in output  # all severities shown

    def test_main_clean_report_passes(self, script: ModuleType, tmp_path: Path) -> None:
        report_path = tmp_path / "audit.json"
        report_path.write_text(json.dumps({"dependencies": []}), encoding="utf-8")
        exit_code = script.main(["--report", str(report_path), "--threshold", "7.0"])
        assert exit_code == 0
