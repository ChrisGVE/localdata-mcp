"""localdata_mcp/nexus/contract/inventory.py — the section 7.2 inventory SSOT.

The single tier-annotated declaration of every connector and file
format v3 ships or gates behind an extras group. ARCHITECTURE.md 7.2:
the SSOT is THIS code — the doc tables are rendered views, the
collect-and-build fixture script (NFR-503/504) consumes this registry,
and tests/v3/test_contract_inventory.py CI-asserts pyproject.toml's
extras groups stay consistent with the tier annotations here (the
one-declaration discipline NX-1 applies to the tool contract, applied
to its sibling cross-cutting truth). PRD S3.1/FR-101's amended list is
authoritative for formats: 14 core formats, Excel's .xlsx/.xls being
two entries in one format family.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Final


class Kind(enum.Enum):
    """What sort of data source an inventory entry names."""

    SQL_ENGINE = "sql_engine"
    FILE_FORMAT = "file_format"
    KV_STORE = "kv_store"
    GRAPH_TREE_STORE = "graph_tree_store"
    RDF_STORE = "rdf_store"
    DOCUMENT_STORE = "document_store"
    TIMESERIES_STORE = "timeseries_store"
    SEARCH_STORE = "search_store"


class Tier(enum.Enum):
    """CORE ships with the base install; EXTRA gates behind a
    pyproject extras group named by the entry."""

    CORE = "core"
    EXTRA = "extra"


@dataclass(frozen=True)
class InventoryEntry:
    """One connector or format declaration.

    `family` groups variant entries counted as one format (the Excel
    pair); it defaults to the entry name. `driver_packages` names the
    pyproject dependencies the drift test asserts are present in the
    entry's tier home (core list or extras group); empty means the
    entry needs no dedicated driver (stdlib, or shared machinery).
    """

    name: str
    kind: Kind
    tier: Tier
    family: str = ""
    extra_group: str | None = None
    driver_packages: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.family:
            object.__setattr__(self, "family", self.name)


def _core(name: str, kind: Kind, *drivers: str, family: str = "") -> InventoryEntry:
    return InventoryEntry(
        name=name, kind=kind, tier=Tier.CORE, family=family, driver_packages=drivers
    )


def _extra(name: str, kind: Kind, group: str, *drivers: str) -> InventoryEntry:
    return InventoryEntry(
        name=name,
        kind=kind,
        tier=Tier.EXTRA,
        extra_group=group,
        driver_packages=drivers,
    )


INVENTORY: Final[tuple[InventoryEntry, ...]] = (
    # SQL engines — core (NFR-504 floor). SQLite is stdlib-backed.
    _core("sqlite", Kind.SQL_ENGINE, "sqlalchemy"),
    _core("postgresql", Kind.SQL_ENGINE, "psycopg2-binary"),
    _core("mysql", Kind.SQL_ENGINE, "mysql-connector-python"),
    _core("duckdb", Kind.SQL_ENGINE, "duckdb"),
    # SQL engines — extras tiers.
    _extra("mssql", Kind.SQL_ENGINE, "mssql", "pyodbc", "pymssql"),
    _extra("oracle", Kind.SQL_ENGINE, "enterprise", "oracledb"),
    # File formats — the FR-101 amended list: 14 core formats, the
    # Excel pair sharing one family (15 entries).
    _core("csv", Kind.FILE_FORMAT),
    _core("tsv", Kind.FILE_FORMAT),
    _core("json", Kind.FILE_FORMAT),
    _core("yaml", Kind.FILE_FORMAT, "pyyaml"),
    _core("toml", Kind.FILE_FORMAT, "toml"),
    _core("ini", Kind.FILE_FORMAT),
    _core("xml", Kind.FILE_FORMAT, "defusedxml", "lxml"),
    _core("excel_xlsx", Kind.FILE_FORMAT, "openpyxl", family="excel"),
    _core("excel_xls", Kind.FILE_FORMAT, "xlrd", family="excel"),
    _core("ods", Kind.FILE_FORMAT, "odfpy"),
    _core("numbers", Kind.FILE_FORMAT, "numbers-parser"),
    _core("parquet", Kind.FILE_FORMAT, "pyarrow"),
    _core("feather", Kind.FILE_FORMAT, "pyarrow"),
    _core("arrow", Kind.FILE_FORMAT, "pyarrow"),
    _core("hdf5", Kind.FILE_FORMAT, "h5py"),
    # Non-relational stores — core floor (FR-103 MUST, section 7.2).
    _core("kv", Kind.KV_STORE),
    _core("graph_tree", Kind.GRAPH_TREE_STORE, "networkx", "pydot"),
    _core("rdf_sparql", Kind.RDF_STORE, "rdflib", "SPARQLWrapper"),
    # Non-relational stores — [modern-databases] extras.
    _extra("redis", Kind.KV_STORE, "modern-databases", "redis"),
    _extra("elasticsearch", Kind.SEARCH_STORE, "modern-databases", "elasticsearch"),
    _extra("mongodb", Kind.DOCUMENT_STORE, "modern-databases", "pymongo"),
    _extra("influxdb", Kind.TIMESERIES_STORE, "modern-databases", "influxdb-client"),
    _extra("neo4j", Kind.GRAPH_TREE_STORE, "modern-databases", "neo4j"),
    _extra("couchdb", Kind.DOCUMENT_STORE, "modern-databases", "couchdb"),
)


def entries(
    *, kind: Kind | None = None, tier: Tier | None = None
) -> tuple[InventoryEntry, ...]:
    """The inventory filtered by kind and/or tier, declaration order."""
    return tuple(
        e
        for e in INVENTORY
        if (kind is None or e.kind is kind) and (tier is None or e.tier is tier)
    )
