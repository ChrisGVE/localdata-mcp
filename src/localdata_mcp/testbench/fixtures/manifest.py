"""testbench/fixtures/manifest.py — NFR-503 oracle-dataset manifest (pure).

The committed provenance record for the S7.2 oracle datasets. One entry per
dataset carries its FR-301 domain, source class, retrieval recipe, license,
and — for every provisionable artifact — the SHA-256 of its canonical
serialization. The combined hash pins the external-oracle data identity end
to end: it is what CI exports as ``LOCALDATA_DATASET_HASH`` (the results
store's provenance column, NFR-508), so a reference-library version bump
that shifts a bundled dataset surfaces as a ``--verify`` hash mismatch, not
silent oracle drift.

This module is pure by design: it (de)serializes a manifest already in
memory and decides hash/verify questions, but performs no network I/O and
imports no pandas. The collect-and-build side — fetching the datasets,
serializing frames, authoring the ``.xls`` binary, printing progress — lives
in ``scripts/build_oracle_datasets.py`` (outside the gated tree) and
delegates every hashing decision here.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

MANIFEST_VERSION = 1

# The three source classes an S7.2 row can have. ``network`` is fetched once
# from the reference library's mirror; ``bundled`` ships inside a pinned
# package (no network); ``authored`` is an in-repo fixture. Only the first
# two, plus the one authored binary, carry a hash — the authored-inline
# fixtures are Python literals in the battery modules, pinned by their own
# source, so they record a license but no artifact hash.
SOURCE_NETWORK = "network"
SOURCE_BUNDLED = "bundled"
SOURCE_AUTHORED = "authored"


@dataclass(frozen=True)
class DatasetEntry:
    """One S7.2 oracle-dataset row.

    ``sha256`` is ``None`` exactly for authored-inline fixtures that have no
    standalone provisioned artifact; every network/bundled dataset and the
    authored ``.xls`` binary carry a hash and take part in the combined hash.
    """

    name: str
    domain: str
    source: str
    retrieval: str
    license: str
    sha256: str | None

    @property
    def hashed(self) -> bool:
        return self.sha256 is not None


def sha256_bytes(data: bytes) -> str:
    """Hex SHA-256 of a byte string — the one hashing primitive used here."""
    return hashlib.sha256(data).hexdigest()


def combined_hash(entries: Sequence[DatasetEntry]) -> str:
    """Deterministic hash over every hashed entry, keyed by name.

    Sorting by name makes the result independent of entry order, and keying
    each dataset's hash by its name means adding, removing, or re-hashing any
    single dataset moves the combined value — the property that lets it stand
    in as a provenance fingerprint for the whole oracle set.
    """
    keyed = {entry.name: entry.sha256 for entry in entries if entry.hashed}
    payload = json.dumps(keyed, sort_keys=True, separators=(",", ":"))
    return sha256_bytes(payload.encode())


@dataclass(frozen=True)
class Manifest:
    version: int
    combined_sha256: str
    entries: tuple[DatasetEntry, ...]

    @classmethod
    def build(cls, entries: Sequence[DatasetEntry]) -> Manifest:
        """Assemble a manifest, ordering entries and computing the hash."""
        ordered = tuple(sorted(entries, key=lambda entry: entry.name))
        return cls(MANIFEST_VERSION, combined_hash(ordered), ordered)

    def to_json(self) -> str:
        body = {
            "version": self.version,
            "combined_sha256": self.combined_sha256,
            "entries": [asdict(entry) for entry in self.entries],
        }
        return json.dumps(body, indent=2) + "\n"


def load_manifest(path: Path) -> Manifest:
    """Read a committed manifest file back into a ``Manifest``."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    entries = tuple(DatasetEntry(**entry) for entry in raw["entries"])
    return Manifest(
        version=int(raw["version"]),
        combined_sha256=str(raw["combined_sha256"]),
        entries=entries,
    )


def verify(manifest: Manifest, provisioned: Mapping[str, str]) -> list[str]:
    """Compare freshly-provisioned hashes to the committed manifest.

    ``provisioned`` maps a dataset name to the SHA-256 computed this run.
    Returns a list of human-readable violations (empty ⇒ the provisioned
    data matches the pinned oracle set). Checks each hashed entry is present
    and matching, and re-derives the combined hash to catch a hand-edited
    manifest whose recorded fingerprint no longer matches its own entries.
    """
    violations: list[str] = []
    for entry in manifest.entries:
        if not entry.hashed:
            continue
        got = provisioned.get(entry.name)
        if got is None:
            violations.append(f"{entry.name}: not provisioned")
        elif got != entry.sha256:
            assert entry.sha256 is not None  # hashed ⇒ non-None, for mypy
            violations.append(
                f"{entry.name}: hash mismatch "
                f"(manifest {entry.sha256[:12]}, provisioned {got[:12]})"
            )
    recomputed = combined_hash(manifest.entries)
    if recomputed != manifest.combined_sha256:
        violations.append(
            f"combined hash mismatch (recorded {manifest.combined_sha256[:12]}, "
            f"recomputed {recomputed[:12]})"
        )
    return violations
