#!/usr/bin/env python3
"""scripts/check_audit_severity.py — CVSS threshold gate over pip-audit output.

Reads a pip-audit JSON report, resolves each vulnerability's CVSS score from
OSV (https://osv.dev), prints every finding at every severity, and exits
non-zero when any finding scores at or above the given threshold. The
threshold is a required argument — its one configuration home is the CI
workflow that invokes this gate, never a default in this file.

Fail-closed: a vulnerability whose severity cannot be resolved blocks the
build rather than slipping through unscored.

Scoring order per vulnerability: a CVSS vector in the OSV record's
`severity` list (needs the `cvss` package, installed by the CI job), else
the qualitative `database_specific.severity` label, else unknown.

Run: python3 scripts/check_audit_severity.py --report audit.json --threshold 7.0
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

OSV_VULN_URL = "https://api.osv.dev/v1/vulns/{vuln_id}"

# Representative scores for OSV qualitative labels, used only when no CVSS
# vector is published: the label's CVSS-range floor, so HIGH/CRITICAL land
# at/above a 7.0 threshold and MODERATE/LOW below it.
QUALITATIVE_FLOOR = {
    "CRITICAL": 9.0,
    "HIGH": 7.0,
    "MODERATE": 4.0,
    "MEDIUM": 4.0,
    "LOW": 0.1,
}


@dataclass
class Finding:
    """One vulnerability affecting one installed package."""

    package: str
    version: str
    vuln_id: str
    fix_versions: list[str] = field(default_factory=list)
    score: float | None = None
    source: str = "unknown"  # "cvss-vector" | "qualitative" | "unknown"


def score_cvss_vector(vector: str) -> float:
    """Compute the base score of a CVSS v3/v4 vector string."""
    import cvss  # CI-job dependency; lazy so tests need no cvss install

    if vector.startswith("CVSS:4"):
        return float(cvss.CVSS4(vector).base_score)
    return float(cvss.CVSS3(vector).scores()[0])


def score_from_osv(record: dict[str, Any]) -> tuple[float | None, str]:
    """Resolve (score, source) from an OSV record; (None, "unknown") if bare."""
    vector_scores = []
    for entry in record.get("severity", []):
        if entry.get("type", "").startswith("CVSS") and entry.get("score"):
            vector_scores.append(score_cvss_vector(entry["score"]))
    if vector_scores:
        return max(vector_scores), "cvss-vector"

    label = str(record.get("database_specific", {}).get("severity", "")).upper()
    if label in QUALITATIVE_FLOOR:
        return QUALITATIVE_FLOOR[label], "qualitative"
    return None, "unknown"


def fetch_osv_record(vuln_id: str) -> dict[str, Any]:
    """Fetch one vulnerability record from the OSV API."""
    with urllib.request.urlopen(OSV_VULN_URL.format(vuln_id=vuln_id)) as response:
        return json.loads(response.read().decode("utf-8"))


def collect_findings(
    report: dict[str, Any],
    fetch: Callable[[str], dict[str, Any]] | None = None,
) -> list[Finding]:
    """Turn a pip-audit JSON report into scored findings."""
    if fetch is None:  # late-bound so a patched fetch_osv_record is honored
        fetch = fetch_osv_record
    findings = []
    for dependency in report.get("dependencies", []):
        for vuln in dependency.get("vulns", []):
            score, source = score_from_osv(fetch(vuln["id"]))
            findings.append(
                Finding(
                    package=dependency["name"],
                    version=dependency["version"],
                    vuln_id=vuln["id"],
                    fix_versions=list(vuln.get("fix_versions", [])),
                    score=score,
                    source=source,
                )
            )
    return findings


def gate(findings: list[Finding], threshold: float) -> bool:
    """True when the build must fail: any score >= threshold, or unscorable."""
    return any(f.score is None or f.score >= threshold for f in findings)


def render_report(findings: list[Finding], threshold: float) -> str:
    """Human-readable table of every finding, blockers marked."""
    if not findings:
        return "pip-audit: no known vulnerabilities in the audited set"
    lines = [f"{'':2} {'package':30} {'vulnerability':22} {'score':>7}  fix versions"]
    for f in sorted(findings, key=lambda f: -(f.score or 10.0)):
        blocking = f.score is None or f.score >= threshold
        marker = "!!" if blocking else "  "
        score_text = "unknown" if f.score is None else f"{f.score:.1f}"
        fixes = ", ".join(f.fix_versions) or "-"
        lines.append(
            f"{marker} {f.package + ' ' + f.version:30} {f.vuln_id:22}"
            f" {score_text:>7}  {fixes}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--threshold", type=float, required=True)
    args = parser.parse_args(argv)

    report = json.loads(args.report.read_text(encoding="utf-8"))
    findings = collect_findings(report)
    print(render_report(findings, args.threshold))

    if gate(findings, args.threshold):
        print(f"FAIL: finding at or above CVSS {args.threshold}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
