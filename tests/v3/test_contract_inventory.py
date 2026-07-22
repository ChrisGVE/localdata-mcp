"""tests/v3/test_contract_inventory.py — E3.3: the section 7.2 inventory SSOT.

Pins nexus/contract/inventory.py: the tier-annotated connector/format
registry, and the drift check that keeps pyproject.toml's extras groups
consistent with the registry's tier annotations (the axis T13 proved
drifts when hand-synced). PRD S3.1/FR-101 is authoritative for the
format list: 14 core formats, with Excel's .xlsx/.xls as two registry
entries sharing one format family.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

from localdata_mcp.nexus.contract.inventory import (
    INVENTORY,
    InventoryEntry,
    Kind,
    Tier,
    entries,
)

PYPROJECT = Path(__file__).resolve().parents[2] / "pyproject.toml"


def _extras_groups() -> dict[str, list[str]]:
    with PYPROJECT.open("rb") as fh:
        data = tomllib.load(fh)
    return data["project"]["optional-dependencies"]


def _core_dependencies() -> list[str]:
    with PYPROJECT.open("rb") as fh:
        data = tomllib.load(fh)
    return data["project"]["dependencies"]


class TestSqlEngines:
    def test_core_floor(self) -> None:
        core_sql = {e.name for e in entries(kind=Kind.SQL_ENGINE, tier=Tier.CORE)}
        assert core_sql == {"sqlite", "postgresql", "mysql", "duckdb"}

    def test_extras_engines(self) -> None:
        extra_sql = {
            e.name: e.extra_group
            for e in entries(kind=Kind.SQL_ENGINE, tier=Tier.EXTRA)
        }
        assert extra_sql == {"mssql": "mssql", "oracle": "enterprise"}


class TestFileFormats:
    def test_fifteen_entries_fourteen_families(self) -> None:
        formats = entries(kind=Kind.FILE_FORMAT)
        assert len(formats) == 15
        assert len({e.family for e in formats}) == 14

    def test_all_core(self) -> None:
        assert all(e.tier is Tier.CORE for e in entries(kind=Kind.FILE_FORMAT))

    def test_amended_list_names(self) -> None:
        names = {e.name for e in entries(kind=Kind.FILE_FORMAT)}
        assert names == {
            "csv",
            "tsv",
            "json",
            "yaml",
            "toml",
            "ini",
            "xml",
            "excel_xlsx",
            "excel_xls",
            "ods",
            "numbers",
            "parquet",
            "feather",
            "arrow",
            "hdf5",
        }

    def test_excel_variants_share_family(self) -> None:
        by_name = {e.name: e for e in entries(kind=Kind.FILE_FORMAT)}
        assert by_name["excel_xlsx"].family == by_name["excel_xls"].family


class TestNonRelationalStores:
    def test_core_floor_present(self) -> None:
        core_stores = {
            e.name
            for e in INVENTORY
            if e.tier is Tier.CORE
            and e.kind in {Kind.KV_STORE, Kind.GRAPH_TREE_STORE, Kind.RDF_STORE}
        }
        assert core_stores == {"kv", "graph_tree", "rdf_sparql"}

    def test_modern_databases_entries(self) -> None:
        modern = {e.name for e in INVENTORY if e.extra_group == "modern-databases"}
        assert modern == {
            "redis",
            "elasticsearch",
            "mongodb",
            "influxdb",
            "neo4j",
            "couchdb",
        }


class TestTierDiscipline:
    def test_core_entries_have_no_extras_group(self) -> None:
        for entry in entries(tier=Tier.CORE):
            assert entry.extra_group is None, entry.name

    def test_extra_entries_name_a_group(self) -> None:
        for entry in entries(tier=Tier.EXTRA):
            assert entry.extra_group, entry.name

    def test_entry_names_unique(self) -> None:
        names = [e.name for e in INVENTORY]
        assert len(names) == len(set(names))


class TestPyprojectDrift:
    """The registry's tier annotations must match pyproject reality."""

    def test_referenced_extras_groups_exist(self) -> None:
        groups = _extras_groups()
        for entry in entries(tier=Tier.EXTRA):
            assert entry.extra_group in groups, (
                f"{entry.name} names extras group {entry.extra_group!r} "
                "absent from pyproject.toml"
            )

    def test_extra_driver_packages_live_in_their_group(self) -> None:
        groups = _extras_groups()
        for entry in entries(tier=Tier.EXTRA):
            declared = " ".join(groups[entry.extra_group or ""])
            for package in entry.driver_packages:
                assert package in declared, (
                    f"{entry.name}: driver {package!r} not in extras "
                    f"group {entry.extra_group!r}"
                )

    def test_core_driver_packages_live_in_core_dependencies(self) -> None:
        core = " ".join(_core_dependencies())
        for entry in entries(tier=Tier.CORE):
            for package in entry.driver_packages:
                assert package in core, (
                    f"{entry.name}: driver {package!r} not in core dependencies"
                )

    def test_entries_are_frozen(self) -> None:
        assert isinstance(INVENTORY[0], InventoryEntry)
        assert INVENTORY[0].__dataclass_fields__  # dataclass, and...
        assert getattr(type(INVENTORY[0]).__dataclass_params__, "frozen")  # ...frozen
