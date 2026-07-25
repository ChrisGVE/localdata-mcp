"""What the datasource slot registry must do.

The model has one idea in it: **a slot is always a database, addressed by a
nickname.** A flat file becomes a new in-memory database holding one table named
from the file; a SQLite file is attached read-only with whatever tables it
already has; a URL naming a service is a separate engine. Nothing is special-cased
by source, so addressing is uniformly ``nickname.table`` and a later
``create_table`` can add a second table to a slot that began life as a CSV.

Two properties are worth stating because they are what the tests defend:

* **Ten slots, and the number is measured.** Every slot is an attached database,
  and SQLite refuses the eleventh ``ATTACH``.
* **Failure is explained, not merely reported.** An evicted slot and a join
  across two engines both fail; both must say why, because an LLM that gets
  ``no such table`` re-plans blind.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from localdata_mcp import config as config_module
from localdata_mcp.config import Config
from localdata_mcp.slots import (
    AttachRefused,
    Registry,
    SlotNotAvailable,
    url_scheme,
)

ASSETS = Path(__file__).parent / "assets"


@pytest.fixture()
def root(monkeypatch, tmp_path):
    """A scratch directory holding the hostile corpus, and in scope."""
    shared = tmp_path / "root"
    shared.mkdir()
    for asset in ASSETS.iterdir():
        (shared / asset.name).write_bytes(asset.read_bytes())
    config_module.use(Config(roots=(shared,)))
    return shared


@pytest.fixture()
def registry(root):
    reg = Registry()
    yield reg
    reg.close()


def build_database(path: Path, table: str = "products") -> Path:
    connection = sqlite3.connect(path)
    connection.execute(f"CREATE TABLE {table} (sku TEXT, name TEXT)")
    connection.executemany(
        f"INSERT INTO {table} VALUES (?, ?)",
        [("a", "Widget"), ("b", "Gadget")],
    )
    connection.commit()
    connection.close()
    return path


def csv_at(path: Path, text: str = "sku,qty\na,3\nb,4\n") -> Path:
    path.write_text(text)
    return path


# ---------------------------------------------------------------------------
# Recognising what a datasource is
# ---------------------------------------------------------------------------


def test_a_scheme_is_recognised_as_a_url():
    assert url_scheme("postgresql://user@host/db") == "postgresql"
    assert url_scheme("mysql+pymysql://host/db") == "mysql+pymysql"


def test_a_windows_drive_letter_is_not_a_scheme():
    """A single-letter 'scheme' is a drive, and there is no '://' either way."""
    assert url_scheme(r"C:\data\sales.csv") is None
    assert url_scheme("C:/data/sales.csv") is None


def test_an_ordinary_path_is_not_a_url():
    assert url_scheme("sales.csv") is None
    assert url_scheme("/var/data/sales.csv") is None
    assert url_scheme("./nested/sales.csv") is None


# ---------------------------------------------------------------------------
# A flat file becomes a database with one table
# ---------------------------------------------------------------------------


def test_a_csv_becomes_a_database_holding_one_table(registry, root):
    csv_at(root / "sales.csv")
    attachment = registry.attach(str(root / "sales.csv"), "shop")

    assert attachment.slot.nickname == "shop"
    assert attachment.slot.tables == ("sales",)
    assert attachment.evicted is None


def test_the_table_is_named_from_the_file_not_from_the_nickname(registry, root):
    """The nickname names the database; the file names the table inside it."""
    csv_at(root / "quarterly_report.csv")
    attachment = registry.attach(str(root / "quarterly_report.csv"), "q3")
    assert attachment.slot.tables == ("quarterly_report",)


def test_a_file_slot_is_addressed_as_nickname_dot_table(registry, root):
    csv_at(root / "sales.csv")
    registry.attach(str(root / "sales.csv"), "shop")
    columns, rows = registry.query(
        "shop", "SELECT sku, qty FROM shop.sales ORDER BY sku"
    )
    assert columns == ["sku", "qty"]
    assert rows == [("a", 3), ("b", 4)]


def test_a_tsv_is_read_by_its_own_reader(registry, root):
    (root / "tabbed.tsv").write_text("sku\tqty\na\t3\n")
    attachment = registry.attach(str(root / "tabbed.tsv"), "tabbed")
    assert attachment.slot.tables == ("tabbed",)


def test_the_hostile_corpus_still_reports_its_mixed_columns(registry, root):
    registry.attach(str(root / "messy_mixed_types.csv"), "messy")
    described = registry.describe("messy", "messy_mixed_types")
    assert described.mixed_columns


def test_an_unsupported_extension_is_refused_naming_what_works(registry, root):
    (root / "notes.docx").write_bytes(b"not tabular")
    with pytest.raises(AttachRefused, match=r"\.csv"):
        registry.attach(str(root / "notes.docx"), "notes")


# ---------------------------------------------------------------------------
# A database file is attached, not copied
# ---------------------------------------------------------------------------


def test_a_database_file_arrives_with_its_existing_tables(registry, root):
    build_database(root / "warehouse.db")
    attachment = registry.attach(str(root / "warehouse.db"), "wh")
    assert attachment.slot.tables == ("products",)


def test_a_database_is_recognised_by_its_header_not_its_extension(registry, root):
    """`.db`, `.sqlite`, `.db3` and no extension at all are all real in the wild."""
    build_database(root / "warehouse.sqlite3")
    build_database(root / "extensionless")
    assert registry.attach(str(root / "warehouse.sqlite3"), "a").slot.kind == "database"
    assert registry.attach(str(root / "extensionless"), "b").slot.kind == "database"


def test_a_file_claiming_to_be_a_database_but_is_not_is_refused(registry, root):
    (root / "impostor.db").write_bytes(b"this is not a SQLite file at all")
    with pytest.raises(AttachRefused):
        registry.attach(str(root / "impostor.db"), "fake")


def test_an_attached_database_cannot_be_written_through(registry, root):
    build_database(root / "warehouse.db")
    registry.attach(str(root / "warehouse.db"), "wh")
    with pytest.raises(Exception):
        registry.query("wh", "DELETE FROM wh.products")


def test_a_path_outside_the_allowed_area_is_refused(registry, tmp_path):
    outside = csv_at(tmp_path / "secret.csv")
    with pytest.raises(AttachRefused, match="outside the allowed paths"):
        registry.attach(str(outside), "secret")


# ---------------------------------------------------------------------------
# The flagship: one statement across two slots
# ---------------------------------------------------------------------------


def test_a_file_slot_joins_a_database_slot_in_one_statement(registry, root):
    csv_at(root / "sales.csv")
    build_database(root / "warehouse.db")
    registry.attach(str(root / "sales.csv"), "shop")
    registry.attach(str(root / "warehouse.db"), "wh")

    columns, rows = registry.query(
        "shop",
        "SELECT p.name, s.qty FROM shop.sales s "
        "JOIN wh.products p ON s.sku = p.sku ORDER BY p.name",
    )
    assert columns == ["name", "qty"]
    assert rows == [("Gadget", 4), ("Widget", 3)]


def test_two_file_slots_join_each_other(registry, root):
    csv_at(root / "left.csv", "k,v\n1,a\n")
    csv_at(root / "right.csv", "k,w\n1,b\n")
    registry.attach(str(root / "left.csv"), "l")
    registry.attach(str(root / "right.csv"), "r")

    _, rows = registry.query("l", "SELECT v, w FROM l.left JOIN r.right USING (k)")
    assert rows == [("a", "b")]


# ---------------------------------------------------------------------------
# Nicknames
# ---------------------------------------------------------------------------


def test_a_nickname_is_rejected_rather_than_silently_mangled(registry, root):
    """Silent mangling hands back a handle that is not the one requested."""
    csv_at(root / "sales.csv")
    with pytest.raises(AttachRefused, match="my-data"):
        registry.attach(str(root / "sales.csv"), "my-data")


@pytest.mark.parametrize("nickname", ["main", "temp", "sqlite_master", "MAIN"])
def test_names_sqlite_reserves_are_rejected(registry, root, nickname):
    csv_at(root / "sales.csv")
    with pytest.raises(AttachRefused):
        registry.attach(str(root / "sales.csv"), nickname)


@pytest.mark.parametrize("nickname", ["", "1st", "a b", "a;b", "a'b"])
def test_unusable_nicknames_are_rejected(registry, root, nickname):
    csv_at(root / "sales.csv")
    with pytest.raises(AttachRefused):
        registry.attach(str(root / "sales.csv"), nickname)


def test_reattaching_a_nickname_replaces_that_slot_in_place(registry, root):
    csv_at(root / "first.csv", "a\n1\n")
    csv_at(root / "second.csv", "b\n2\n")
    registry.attach(str(root / "first.csv"), "slot")
    attachment = registry.attach(str(root / "second.csv"), "slot")

    assert attachment.slot.tables == ("second",)
    assert len(registry.slots()) == 1
    _, rows = registry.query("slot", "SELECT b FROM slot.second")
    assert rows == [(2,)]


# ---------------------------------------------------------------------------
# Ten slots, evicted oldest first
# ---------------------------------------------------------------------------


def fill(registry: Registry, root: Path, count: int, start: int = 0) -> None:
    for index in range(start, start + count):
        csv_at(root / f"f{index}.csv", "a\n1\n")
        registry.attach(str(root / f"f{index}.csv"), f"s{index}")


def test_the_cap_comes_from_the_configuration(root):
    config_module.use(Config(roots=(root,), slots=2))
    registry = Registry()
    try:
        fill(registry, root, 3)
        assert [slot.nickname for slot in registry.slots()] == ["s1", "s2"]
    finally:
        registry.close()


def test_the_eleventh_attachment_evicts_the_first(registry, root):
    fill(registry, root, 10)
    attachment = registry.attach(str(root / "f0.csv"), "eleventh")

    assert attachment.evicted is not None
    assert attachment.evicted.nickname == "s0"
    assert len(registry.slots()) == 10


def test_eviction_takes_the_oldest_not_the_newest(root):
    config_module.use(Config(roots=(root,), slots=3))
    registry = Registry()
    try:
        fill(registry, root, 3)
        registry.attach(str(root / "f0.csv"), "fresh")
        assert [slot.nickname for slot in registry.slots()] == ["s1", "s2", "fresh"]
    finally:
        registry.close()


def test_the_eviction_record_carries_what_is_needed_to_rebuild_the_slot(root):
    """Once a slot can hold several tables, its source alone is not enough."""
    config_module.use(Config(roots=(root,), slots=1))
    registry = Registry()
    try:
        source = csv_at(root / "sales.csv")
        registry.attach(str(source), "shop")
        csv_at(root / "other.csv", "a\n1\n")
        evicted = registry.attach(str(root / "other.csv"), "next").evicted

        assert evicted.nickname == "shop"
        assert evicted.source == str(source.resolve())
        assert evicted.tables == ("sales",)
        assert "slot" in evicted.reason.lower() or "limit" in evicted.reason.lower()
    finally:
        registry.close()


def test_using_an_evicted_nickname_explains_the_eviction(root):
    config_module.use(Config(roots=(root,), slots=1))
    registry = Registry()
    try:
        fill(registry, root, 2)
        with pytest.raises(SlotNotAvailable, match="evicted"):
            registry.query("s0", "SELECT * FROM s0.f0")
    finally:
        registry.close()


def test_an_unknown_nickname_is_distinguished_from_an_evicted_one(registry, root):
    with pytest.raises(SlotNotAvailable) as raised:
        registry.query("neverexisted", "SELECT 1")
    assert "evicted" not in str(raised.value).lower()


def test_a_qualified_reference_to_an_evicted_slot_is_also_explained(root):
    """The nickname routed on is live; the one inside the SQL is gone."""
    config_module.use(Config(roots=(root,), slots=2))
    registry = Registry()
    try:
        fill(registry, root, 3)
        with pytest.raises(SlotNotAvailable, match="evicted"):
            registry.query("s2", "SELECT * FROM s2.f2 JOIN s0.f0 USING (a)")
    finally:
        registry.close()


def test_a_refused_attachment_costs_no_live_slot_its_place(root):
    """Every refusal happens while the shelf is still untouched."""
    config_module.use(Config(roots=(root,), slots=1))
    registry = Registry()
    try:
        csv_at(root / "keeper.csv")
        registry.attach(str(root / "keeper.csv"), "keeper")

        (root / "empty.csv").write_text("")
        with pytest.raises(AttachRefused):
            registry.attach(str(root / "empty.csv"), "doomed")
        with pytest.raises(AttachRefused):
            registry.attach(str(root / "absent.csv"), "doomed")
        with pytest.raises(AttachRefused):
            registry.attach(str(root / "keeper.csv"), "not-a-name")

        assert [slot.nickname for slot in registry.slots()] == ["keeper"]
        _, rows = registry.query("keeper", "SELECT count(*) FROM keeper.keeper")
        assert rows == [(2,)]
    finally:
        registry.close()


def test_a_slot_can_be_attached_again_after_eviction(root):
    config_module.use(Config(roots=(root,), slots=1))
    registry = Registry()
    try:
        fill(registry, root, 2)
        registry.attach(str(root / "f0.csv"), "s0")
        _, rows = registry.query("s0", "SELECT a FROM s0.f0")
        assert rows == [(1,)]
    finally:
        registry.close()


def test_a_genuine_missing_table_still_reports_itself_plainly(registry, root):
    csv_at(root / "sales.csv")
    registry.attach(str(root / "sales.csv"), "shop")
    with pytest.raises(Exception) as raised:
        registry.query("shop", "SELECT * FROM shop.absent")
    assert "evicted" not in str(raised.value).lower()


# ---------------------------------------------------------------------------
# A slot on its own engine
# ---------------------------------------------------------------------------


def test_a_network_url_is_refused_while_the_network_is_closed(registry, root):
    with pytest.raises(AttachRefused, match="network"):
        registry.attach("postgresql://user:hunter2@db.example.com/sales", "pg")


def test_the_refusal_does_not_echo_the_password(registry, root):
    with pytest.raises(AttachRefused) as raised:
        registry.attach("postgresql://user:hunter2@db.example.com/sales", "pg")
    assert "hunter2" not in str(raised.value)


def test_opening_the_network_gets_past_the_gate(root):
    """No driver is installed, so the failure must be the driver, not the gate."""
    config_module.use(Config(roots=(root,), network_enabled=True))
    registry = Registry()
    try:
        with pytest.raises(AttachRefused) as raised:
            registry.attach("postgresql://user:hunter2@db.example.com/sales", "pg")
        message = str(raised.value)
        assert "network" not in message.lower()
        assert "hunter2" not in message
    finally:
        registry.close()


def test_a_slot_on_its_own_engine_answers_its_own_sql(registry, root):
    """Exercised through the internal constructor: the only driver available
    without a server is SQLite, and a sqlite: URL routes to ATTACH at the
    surface, so the separate-engine path is reached directly here."""
    build_database(root / "remote.db")
    slot = registry._attach_engine(f"sqlite:///{root / 'remote.db'}", "remote")

    assert slot.kind == "engine"
    assert slot.tables == ("products",)
    columns, rows = registry.query("remote", "SELECT sku FROM products ORDER BY sku")
    assert columns == ["sku"]
    assert rows == [("a",), ("b",)]


def test_a_join_across_two_engines_explains_the_one_engine_rule(registry, root):
    build_database(root / "remote.db")
    csv_at(root / "sales.csv")
    registry.attach(str(root / "sales.csv"), "shop")
    registry._attach_engine(f"sqlite:///{root / 'remote.db'}", "remote")

    with pytest.raises(Exception) as raised:
        registry.query("remote", "SELECT * FROM products JOIN shop.sales USING (sku)")
    assert "engine" in str(raised.value).lower()


def test_an_engine_slot_reports_a_source_without_its_password(registry, root):
    build_database(root / "remote.db")
    slot = registry._attach_engine(f"sqlite:///{root / 'remote.db'}", "remote")
    assert "remote.db" in slot.source


# ---------------------------------------------------------------------------
# Listing
# ---------------------------------------------------------------------------


def test_slots_are_listed_in_the_order_they_were_attached(registry, root):
    fill(registry, root, 3)
    assert [slot.nickname for slot in registry.slots()] == ["s0", "s1", "s2"]


def test_a_fresh_registry_holds_nothing(registry):
    assert registry.slots() == []
