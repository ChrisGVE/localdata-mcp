#!/usr/bin/env python3
"""Collect-and-build the S7.2 oracle datasets + manifest (NFR-503, E14.1).

Fresh v3 script — the legacy ``download_test_datasets.py`` (NYC taxi / Ames /
World Bank) is a separate v1 helper and stays as-is. This one provisions the
oracle datasets the domain and base batteries pin against, and records their
provenance:

  * network-fetched once (California housing, AirPassengers, sleep,
    PlantGrowth, mtcars) — retrieved through the same reference libraries the
    dual-assertion oracle (FR-304 / NFR-505) recomputes with;
  * package-bundled (diabetes, iris, macrodata) — shipped inside the pinned
    scikit-learn / statsmodels wheels, hashed here so a silent bundle change
    is caught;
  * one authored binary — the legacy ``.xls`` the base battery reads (pandas
    2.x dropped its writer engine and no maintained pure-writer exists), so
    it is authored here with ``xlwt`` and committed alongside the manifest.

The per-dataset SHA-256 + license land in the committed manifest
(``src/localdata_mcp/testbench/fixtures/dataset_manifest.json``); its combined
hash is what CI exports as ``LOCALDATA_DATASET_HASH``.

Modes:
  (default) / --build   provision every dataset, (re)write the manifest and
                        the committed ``.xls`` fixture. ``--out DIR`` also
                        drops each dataset's canonical CSV there for
                        inspection (NFR-503 "provisions on a clean env").
  --verify              re-provision and compare hashes to the committed
                        manifest; exit 1 with a report on any mismatch (the
                        CI drift guard — a reference-library bump fails here).
  --print-hash          print the committed manifest's combined hash and exit
                        (reads the manifest only, no network) — the value CI
                        assigns to LOCALDATA_DATASET_HASH.

Authoring the ``.xls`` needs xlwt, which is not a runtime dependency; run the
build once under ``uv run --with xlwt python scripts/build_oracle_datasets.py``.
The runtime dep ``xlrd`` reads the committed result.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable

# The script sits in scripts/; the package tree is one level up under src/.
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from localdata_mcp.testbench.fixtures.manifest import (  # noqa: E402
    SOURCE_AUTHORED,
    SOURCE_BUNDLED,
    SOURCE_NETWORK,
    DatasetEntry,
    Manifest,
    load_manifest,
    sha256_bytes,
    verify,
)

_FIXTURES = _REPO_ROOT / "src" / "localdata_mcp" / "testbench" / "fixtures"
MANIFEST_PATH = _FIXTURES / "dataset_manifest.json"
XLS_PATH = _FIXTURES / "datasets" / "base_excel.xls"


# --- dataset provisioners ---------------------------------------------------
#
# Each returns the canonical CSV bytes whose SHA-256 is the recorded hash.
# Frames serialize with index=False so the hash tracks the data, not pandas'
# row labels. Retrieval strings double as the manifest's reproducibility note.


def _frame_csv(frame: object) -> bytes:
    # `frame` is a pandas DataFrame; typed loosely so this module stays import-
    # light and the pandas dependency lives only where the fetch happens.
    return frame.to_csv(index=False).encode()  # type: ignore[attr-defined]


def _california() -> bytes:
    from sklearn.datasets import fetch_california_housing

    return _frame_csv(fetch_california_housing(as_frame=True).frame)


def _diabetes() -> bytes:
    from sklearn.datasets import load_diabetes

    return _frame_csv(load_diabetes(as_frame=True).frame)


def _iris() -> bytes:
    from sklearn.datasets import load_iris

    return _frame_csv(load_iris(as_frame=True).frame)


def _macrodata() -> bytes:
    from statsmodels.datasets import macrodata

    return _frame_csv(macrodata.load_pandas().data)


def _rdataset(name: str) -> Callable[[], bytes]:
    def fetch() -> bytes:
        from statsmodels.datasets import get_rdataset

        return _frame_csv(get_rdataset(name, "datasets", cache=True).data)

    return fetch


# --- the S7.2 oracle table --------------------------------------------------
#
# (name, domain, source, retrieval, license, provisioner|None). A None
# provisioner is an authored-inline fixture: a Python literal living in the
# battery module (the published-oracle value embedded for determinism), so it
# records a license but no standalone artifact hash.

_Spec = tuple[str, str, str, str, str, "Callable[[], bytes] | None"]

_DATASETS: tuple[_Spec, ...] = (
    (
        "california_housing",
        "regression",
        SOURCE_NETWORK,
        "sklearn.datasets.fetch_california_housing(as_frame=True)",
        "Public domain (StatLib census-derived)",
        _california,
    ),
    (
        "diabetes",
        "regression",
        SOURCE_BUNDLED,
        "sklearn.datasets.load_diabetes(as_frame=True)",
        "BSD-3-Clause (scikit-learn bundle)",
        _diabetes,
    ),
    (
        "mtcars",
        "regression",
        SOURCE_NETWORK,
        "statsmodels get_rdataset('mtcars', 'datasets')",
        "Public-domain data; Rdatasets packaging GPL-3 (data cached, packaging not redistributed)",
        _rdataset("mtcars"),
    ),
    (
        "air_passengers",
        "time_series",
        SOURCE_NETWORK,
        "statsmodels get_rdataset('AirPassengers', 'datasets')",
        "Public-domain series; Rdatasets packaging GPL-3 (data cached, packaging not redistributed)",
        _rdataset("AirPassengers"),
    ),
    (
        "macrodata",
        "time_series",
        SOURCE_BUNDLED,
        "statsmodels.datasets.macrodata.load_pandas()",
        "US public domain (statsmodels bundle)",
        _macrodata,
    ),
    (
        "iris",
        "pattern_recognition",
        SOURCE_BUNDLED,
        "sklearn.datasets.load_iris(as_frame=True)",
        "CC BY 4.0 (UCI) via scikit-learn bundle",
        _iris,
    ),
    (
        "sleep",
        "statistical",
        SOURCE_NETWORK,
        "statsmodels get_rdataset('sleep', 'datasets')",
        "Public domain (Student 1908 / Cushny & Peebles)",
        _rdataset("sleep"),
    ),
    (
        "plant_growth",
        "statistical",
        SOURCE_NETWORK,
        "statsmodels get_rdataset('PlantGrowth', 'datasets')",
        "Public domain",
        _rdataset("PlantGrowth"),
    ),
    # Authored-inline fixtures (S7.2 rows with no published-analysis
    # convention): hand-verified literals in the battery modules, hashed by
    # their own source. Recorded here for the license trail; sha256 is None.
    (
        "optimization_lp_assignment",
        "optimization",
        SOURCE_AUTHORED,
        "in-repo literal (optimization_battery_test.py)",
        "MIT (repo)",
        None,
    ),
    (
        "network_graph",
        "network_graph",
        SOURCE_AUTHORED,
        "in-repo literal (network_battery_test.py)",
        "MIT (repo)",
        None,
    ),
    (
        "geospatial",
        "geospatial",
        SOURCE_AUTHORED,
        "in-repo literal (geospatial_battery_test.py)",
        "MIT (repo)",
        None,
    ),
    (
        "rfm_clv",
        "business_intelligence",
        SOURCE_AUTHORED,
        "in-repo literal (bi_battery_test.py)",
        "MIT (repo)",
        None,
    ),
    (
        "finite_population",
        "sampling",
        SOURCE_AUTHORED,
        "in-repo literal (sampling_battery_test.py)",
        "MIT (repo)",
        None,
    ),
)


def _author_xls(path: Path) -> bytes:
    """Author the legacy ``.xls`` binary and return its bytes.

    Content mirrors the base battery's ``_FRAME`` (id/label → 1/a, 2/b) so
    the ``.xls`` row of the format matrix asserts the same tabular result as
    the other 14 formats. xlwt is imported lazily — it is a build-only tool,
    never a runtime dependency.
    """
    import xlwt

    workbook = xlwt.Workbook()
    sheet = workbook.add_sheet("Sheet1")
    for col, header in enumerate(("id", "label")):
        sheet.write(0, col, header)
    for row, (identifier, label) in enumerate(((1, "a"), (2, "b")), start=1):
        sheet.write(row, 0, identifier)
        sheet.write(row, 1, label)
    path.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(str(path))
    return path.read_bytes()


def _provision(out: Path | None) -> dict[str, str]:
    """Fetch/serialize every hashed dataset; return name → SHA-256.

    Writes a canonical CSV per dataset into ``out`` when given, and always
    (re)authors the committed ``.xls`` fixture.
    """
    hashes: dict[str, str] = {}
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
    for name, _domain, _source, _retrieval, _license, provisioner in _DATASETS:
        if provisioner is None:
            continue
        data = provisioner()
        hashes[name] = sha256_bytes(data)
        if out is not None:
            (out / f"{name}.csv").write_bytes(data)
        print(f"  provisioned {name} ({len(data)} bytes)")
    xls_bytes = _author_xls(XLS_PATH)
    hashes["excel_xls_fixture"] = sha256_bytes(xls_bytes)
    print(f"  authored {XLS_PATH.name} ({len(xls_bytes)} bytes)")
    return hashes


def _entries(hashes: dict[str, str]) -> list[DatasetEntry]:
    entries = [
        DatasetEntry(
            name=name,
            domain=domain,
            source=source,
            retrieval=retrieval,
            license=license_,
            sha256=hashes.get(name) if provisioner is not None else None,
        )
        for name, domain, source, retrieval, license_, provisioner in _DATASETS
    ]
    entries.append(
        DatasetEntry(
            name="excel_xls_fixture",
            domain="base_ingest",
            source=SOURCE_AUTHORED,
            retrieval="authored by scripts/build_oracle_datasets.py (xlwt)",
            license="MIT (repo)",
            sha256=hashes["excel_xls_fixture"],
        )
    )
    return entries


def _build(out: Path | None) -> int:
    print("Building oracle datasets + manifest...")
    hashes = _provision(out)
    manifest = Manifest.build(_entries(hashes))
    MANIFEST_PATH.write_text(manifest.to_json(), encoding="utf-8")
    print(f"Wrote {MANIFEST_PATH.relative_to(_REPO_ROOT)}")
    print(f"Combined hash: {manifest.combined_sha256}")
    return 0


def _verify() -> int:
    print("Verifying provisioned datasets against the committed manifest...")
    manifest = load_manifest(MANIFEST_PATH)
    hashes = _provision(out=None)
    violations = verify(manifest, hashes)
    if violations:
        print("DRIFT — provisioned data does not match the manifest:")
        for violation in violations:
            print(f"  - {violation}")
        return 1
    print("OK — provisioned data matches the pinned oracle set.")
    return 0


def _print_hash() -> int:
    print(load_manifest(MANIFEST_PATH).combined_sha256)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--build",
        action="store_true",
        help="Provision datasets and (re)write the manifest + .xls (default).",
    )
    mode.add_argument(
        "--verify",
        action="store_true",
        help="Re-provision and compare hashes to the manifest; exit 1 on drift.",
    )
    mode.add_argument(
        "--print-hash",
        action="store_true",
        help="Print the committed manifest's combined hash and exit (no network).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Also write each dataset's canonical CSV into this directory.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.verify:
        sys.exit(_verify())
    if args.print_hash:
        sys.exit(_print_hash())
    sys.exit(_build(args.out))


if __name__ == "__main__":
    main()
