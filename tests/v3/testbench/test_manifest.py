"""tests/v3/testbench/test_manifest.py — the pure NFR-503 manifest logic.

Covers the hashing, build, JSON round-trip, and verify decisions of
``testbench.fixtures.manifest``, plus the internal consistency of the
committed ``dataset_manifest.json`` (E14.1). No network: the fetch side lives
in scripts/build_oracle_datasets.py and is exercised separately.
"""

from __future__ import annotations

import json
from importlib.resources import files
from pathlib import Path

from localdata_mcp.testbench.fixtures import manifest as m


def _entry(name: str, sha: str | None) -> m.DatasetEntry:
    return m.DatasetEntry(
        name=name,
        domain="d",
        source=m.SOURCE_BUNDLED if sha else m.SOURCE_AUTHORED,
        retrieval="r",
        license="MIT",
        sha256=sha,
    )


def test_combined_hash_is_order_independent() -> None:
    a = _entry("alpha", "aa")
    b = _entry("beta", "bb")
    assert m.combined_hash([a, b]) == m.combined_hash([b, a])


def test_combined_hash_ignores_unhashed_entries() -> None:
    hashed = [_entry("alpha", "aa")]
    with_authored = hashed + [_entry("inline", None)]
    assert m.combined_hash(hashed) == m.combined_hash(with_authored)


def test_combined_hash_moves_when_any_dataset_rehashes() -> None:
    before = m.combined_hash([_entry("alpha", "aa"), _entry("beta", "bb")])
    after = m.combined_hash([_entry("alpha", "aa"), _entry("beta", "cc")])
    assert before != after


def test_build_sorts_entries_and_sets_hash() -> None:
    built = m.Manifest.build([_entry("beta", "bb"), _entry("alpha", "aa")])
    assert [e.name for e in built.entries] == ["alpha", "beta"]
    assert built.combined_sha256 == m.combined_hash(built.entries)
    assert built.version == m.MANIFEST_VERSION


def test_json_round_trip(tmp_path: Path) -> None:
    built = m.Manifest.build([_entry("alpha", "aa"), _entry("inline", None)])
    path = tmp_path / "manifest.json"
    path.write_text(built.to_json(), encoding="utf-8")
    loaded = m.load_manifest(path)
    assert loaded == built


def test_verify_accepts_matching_provisioned() -> None:
    built = m.Manifest.build([_entry("alpha", "aa"), _entry("beta", "bb")])
    assert m.verify(built, {"alpha": "aa", "beta": "bb"}) == []


def test_verify_flags_missing_dataset() -> None:
    built = m.Manifest.build([_entry("alpha", "aa")])
    violations = m.verify(built, {})
    assert any("alpha" in v and "not provisioned" in v for v in violations)


def test_verify_flags_hash_mismatch() -> None:
    built = m.Manifest.build([_entry("alpha", "aa")])
    violations = m.verify(built, {"alpha": "zz"})
    assert any("alpha" in v and "mismatch" in v for v in violations)


def test_verify_flags_tampered_combined_hash(tmp_path: Path) -> None:
    built = m.Manifest.build([_entry("alpha", "aa")])
    raw = json.loads(built.to_json())
    raw["combined_sha256"] = "deadbeef"
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    tampered = m.load_manifest(path)
    violations = m.verify(tampered, {"alpha": "aa"})
    assert any("combined hash mismatch" in v for v in violations)


# --- the committed manifest ------------------------------------------------


def _committed_manifest() -> m.Manifest:
    path = Path(
        str(files("localdata_mcp.testbench.fixtures") / "dataset_manifest.json")
    )
    return m.load_manifest(path)


def test_committed_manifest_is_internally_consistent() -> None:
    committed = _committed_manifest()
    assert committed.combined_sha256 == m.combined_hash(committed.entries)


def test_committed_manifest_covers_the_s7_2_domains() -> None:
    committed = _committed_manifest()
    domains = {e.domain for e in committed.entries}
    for expected in (
        "regression",
        "time_series",
        "pattern_recognition",
        "statistical",
        "optimization",
        "network_graph",
        "geospatial",
        "business_intelligence",
        "sampling",
        "base_ingest",
    ):
        assert expected in domains, expected


def test_committed_manifest_hashes_network_and_bundled_but_not_authored_inline() -> (
    None
):
    committed = _committed_manifest()
    by_name = {e.name: e for e in committed.entries}
    # Network + bundled + the authored .xls binary carry a hash.
    assert by_name["california_housing"].sha256 is not None
    assert by_name["iris"].sha256 is not None
    assert by_name["excel_xls_fixture"].sha256 is not None
    # Authored-inline fixtures record a license but no artifact hash.
    assert by_name["network_graph"].sha256 is None
    assert by_name["network_graph"].license
