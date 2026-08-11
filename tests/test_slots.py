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

import foreign
import pytest
from sqlalchemy import Index, MetaData, Table, Text, create_engine, text

from localdata_mcp import config as config_module
from localdata_mcp.config import Config
from localdata_mcp.slots import (
    AttachRefused,
    NotWritable,
    Registry,
    SlotError,
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
    return foreign.build_database(
        path,
        table,
        [("sku", Text), ("name", Text)],
        [("a", "Widget"), ("b", "Gadget")],
    )


def csv_at(path: Path, text: str = "sku,qty\na,3\nb,4\n") -> Path:
    path.write_text(text)
    return path


def view_from_outside(registry: Registry, tag: str, sql: str) -> None:
    """Create a view the only way one can still come into existence.

    ``query`` refuses every write, so the surface cannot make a view at all any
    more. One can still *arrive* inside a SQLite file somebody else built — this
    reaches past the surface to produce that state, writing through the tag's own
    engine because that is the only connection its database has.
    """
    with registry.workspace.engine(tag).begin() as connection:
        connection.execute(text(sql))


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
    columns, rows = registry.query("shop", "SELECT sku, qty FROM sales ORDER BY sku")
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
        registry.query("wh", "DELETE FROM products")


def test_a_path_outside_the_allowed_area_is_refused(registry, tmp_path):
    outside = csv_at(tmp_path / "secret.csv")
    with pytest.raises(AttachRefused, match="outside the allowed paths"):
        registry.attach(str(outside), "secret")


# ---------------------------------------------------------------------------
# The flagship: one statement across two slots
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Nicknames
# ---------------------------------------------------------------------------


def test_a_nickname_is_rejected_rather_than_silently_mangled(registry, root):
    """Silent mangling hands back a handle that is not the one requested."""
    csv_at(root / "sales.csv")
    with pytest.raises(AttachRefused, match="my-data"):
        registry.attach(str(root / "sales.csv"), "my-data")


@pytest.mark.parametrize("nickname", ["main", "temp", "sqlite_master", "MAIN"])
def test_names_sqlite_once_reserved_are_now_ordinary(registry, root, nickname):
    """These were refused because a nickname became a SQL schema name.

    A slot is its own database now and its nickname never appears in a
    statement, so there is nothing left for them to collide with. Kept as a test
    rather than deleted: the refusal was visible in the surface, and this is what
    says it went away on purpose.
    """
    csv_at(root / "sales.csv")
    attachment = registry.attach(str(root / "sales.csv"), nickname)
    assert attachment.slot.nickname == nickname
    assert registry.query(nickname, "SELECT count(*) FROM sales")[1] == [(2,)]


@pytest.mark.parametrize("nickname", ["", "1st", "a b", "a;b", "a'b"])
def test_unusable_nicknames_are_rejected(registry, root, nickname):
    csv_at(root / "sales.csv")
    with pytest.raises(AttachRefused):
        registry.attach(str(root / "sales.csv"), nickname)


def test_a_colliding_nickname_is_disambiguated_rather_than_replacing_a_slot(
    registry, root
):
    """Two real datasources both deserve a slot; only detach drops one."""
    csv_at(root / "first.csv", "a\n1\n")
    csv_at(root / "second.csv", "b\n2\n")
    registry.attach(str(root / "first.csv"), "slot")
    attachment = registry.attach(str(root / "second.csv"), "slot")

    assert attachment.slot.nickname == "slot_2"
    assert attachment.slot.tables == ("second",)
    assert [s.nickname for s in registry.slots()] == ["slot", "slot_2"]

    _, rows = registry.query("slot_2", "SELECT b FROM second")
    assert rows == [(2,)]
    # The first slot is untouched, which is the whole point of not replacing it.
    _, first = registry.query("slot", "SELECT a FROM first")
    assert first == [(1,)]


# ---------------------------------------------------------------------------
# Nicknames: derived, disambiguated, and never silently rewritten
# ---------------------------------------------------------------------------


def test_a_nickname_is_derived_from_the_filename_when_none_is_given(registry, root):
    csv_at(root / "sales.csv")
    attachment = registry.attach(str(root / "sales.csv"))
    assert attachment.slot.nickname == "sales"
    assert attachment.collided_with is None


def test_a_filename_that_is_not_a_legal_identifier_still_yields_one(registry, root):
    """`2024 Sales Report.csv` has no legal spelling the caller chose for us."""
    csv_at(root / "2024 Sales Report.csv")
    attachment = registry.attach(str(root / "2024 Sales Report.csv"))

    nickname = attachment.slot.nickname
    (table,) = attachment.slot.tables
    # Both are prefixed rather than left starting with a digit, and each says
    # what it is: the database is a db_, the table inside it a table_.
    assert nickname == "db_2024_sales_report"
    assert table == "table_2024_sales_report"

    # The proof they are usable is that they address the data.
    _, rows = registry.query(nickname, f"SELECT qty FROM {table}")
    assert sorted(r[0] for r in rows) == [3, 4]


def test_a_file_called_main_keeps_its_own_name(registry, root):
    """It used to become main_2, to stay off SQLite's own schema name."""
    csv_at(root / "main.csv")
    attachment = registry.attach(str(root / "main.csv"))

    assert attachment.slot.nickname == "main"
    _, rows = registry.query("main", "SELECT count(*) FROM main")
    assert rows == [(2,)]


def test_two_same_named_files_in_different_directories_both_get_a_slot(registry, root):
    """The nickname collides; the datasources do not. Both deserve a slot."""
    (root / "q1").mkdir()
    (root / "q2").mkdir()
    csv_at(root / "q1" / "sales.csv", "sku,qty\na,1\n")
    csv_at(root / "q2" / "sales.csv", "sku,qty\nb,2\n")

    first = registry.attach(str(root / "q1" / "sales.csv"))
    second = registry.attach(str(root / "q2" / "sales.csv"))

    assert first.slot.nickname == "sales"
    assert second.slot.nickname == "sales_2"
    assert second.collided_with is not None
    assert second.collided_with.nickname == "sales"
    # The source is what actually distinguishes them, so it has to come back.
    assert second.collided_with.source == str(root / "q1" / "sales.csv")

    _, rows = registry.query("sales_2", "SELECT qty FROM sales")
    assert rows == [(2,)]


def test_disambiguation_keeps_counting_past_the_second_collision(registry, root):
    for index, directory in enumerate(("a", "b", "c")):
        (root / directory).mkdir()
        csv_at(root / directory / "sales.csv", f"sku,qty\nx,{index}\n")
    names = [
        registry.attach(str(root / directory / "sales.csv")).slot.nickname
        for directory in ("a", "b", "c")
    ]
    assert names == ["sales", "sales_2", "sales_3"]


def test_the_same_source_twice_is_refused_and_says_where_it_lives(registry, root):
    """A second copy of identical data burns a slot for nothing."""
    csv_at(root / "sales.csv")
    registry.attach(str(root / "sales.csv"), "shop")

    with pytest.raises(AttachRefused, match="already attached as 'shop'"):
        registry.attach(str(root / "sales.csv"), "shop")


def test_asking_for_a_different_nickname_does_not_get_around_the_duplicate_check(
    registry, root
):
    csv_at(root / "sales.csv")
    registry.attach(str(root / "sales.csv"), "shop")

    with pytest.raises(AttachRefused, match="'shop'"):
        registry.attach(str(root / "sales.csv"), "elsewhere")
    assert [slot.nickname for slot in registry.slots()] == ["shop"]


def test_a_refused_duplicate_costs_no_live_slot_its_place(root):
    """Every refusal must happen while the shelf is still untouched."""
    config_module.use(Config(roots=(root,), slots=2))
    registry = Registry()
    try:
        csv_at(root / "first.csv", "a\n1\n")
        csv_at(root / "second.csv", "b\n2\n")
        registry.attach(str(root / "first.csv"), "one")
        registry.attach(str(root / "second.csv"), "two")

        with pytest.raises(AttachRefused):
            registry.attach(str(root / "first.csv"), "three")

        assert [slot.nickname for slot in registry.slots()] == ["one", "two"]
        _, rows = registry.query("one", "SELECT a FROM first")
        assert rows == [(1,)]
    finally:
        registry.close()


def test_an_explicitly_requested_nickname_is_refused_rather_than_corrected(
    registry, root
):
    """Handing back a silently corrected handle is the defect this avoids."""
    csv_at(root / "sales.csv")
    with pytest.raises(AttachRefused, match="cannot be a nickname"):
        registry.attach(str(root / "sales.csv"), "2 bad")


# ---------------------------------------------------------------------------
# Write is not the default
# ---------------------------------------------------------------------------


def test_a_database_we_built_from_a_file_is_writable(registry, root):
    """Nothing outside it is at risk, so composition needs no grant."""
    csv_at(root / "sales.csv")
    slot = registry.attach(str(root / "sales.csv"), "shop").slot

    assert slot.writable is True
    # The grant is what add_table and drop_table consult. query never writes,
    # whatever the grant says, so it is no longer the way to observe this.
    csv_at(root / "prices.csv", "sku,price\na,10\n")
    registry.create_table("shop", source=str(root / "prices.csv"))
    assert "prices" in registry.tables("shop")
    registry.drop_table("shop", "prices")
    assert "prices" not in registry.tables("shop")


def test_an_outside_database_arrives_read_only(registry, root):
    build_database(root / "warehouse.db")
    slot = registry.attach(str(root / "warehouse.db"), "wh").slot

    assert slot.writable is False
    csv_at(root / "prices.csv", "sku,price\na,10\n")
    with pytest.raises(NotWritable, match="writable=true"):
        registry.create_table("wh", source=str(root / "prices.csv"))


def test_the_write_grant_is_honoured_when_it_is_asked_for(registry, root):
    build_database(root / "warehouse.db")
    slot = registry.attach(str(root / "warehouse.db"), "wh", writable=True).slot

    assert slot.writable is True
    csv_at(root / "prices.csv", "sku,price\na,10\n")
    added = registry.create_table("wh", source=str(root / "prices.csv"))
    assert added.row_count == 1
    assert "prices" in registry.tables("wh")


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
    csv_at(root / "eleventh.csv", "a\n1\n")
    attachment = registry.attach(str(root / "eleventh.csv"), "eleventh")

    assert attachment.evicted is not None
    assert attachment.evicted.nickname == "s0"
    assert len(registry.slots()) == 10


def test_eviction_takes_the_oldest_not_the_newest(root):
    config_module.use(Config(roots=(root,), slots=3))
    registry = Registry()
    try:
        fill(registry, root, 3)
        csv_at(root / "fresh.csv", "a\n1\n")
        registry.attach(str(root / "fresh.csv"), "fresh")
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


def test_the_eviction_record_names_a_table_that_create_added(root):
    """The composed half is the half the source cannot rebuild.

    A slot's ``tables`` is the snapshot taken at attach time, and the whole
    point of ``create`` is that the slot stops matching it. Reporting the
    snapshot names the one table the caller could have got back anyway and
    stays silent about the one that is actually gone.
    """
    config_module.use(Config(roots=(root,), slots=1))
    registry = Registry()
    try:
        registry.attach(str(csv_at(root / "sales.csv")), "shop")
        csv_at(root / "prices.csv", "sku,price\na,10\n")
        registry.create_table("shop", source=str(root / "prices.csv"))

        csv_at(root / "other.csv", "a\n1\n")
        evicted = registry.attach(str(root / "other.csv"), "next").evicted

        assert set(evicted.tables) == {"sales", "prices"}
    finally:
        registry.close()


def test_reaching_for_an_evicted_slot_names_every_table_it_held(root):
    """The advice attached to this message has to be true.

    It ends "attach it again to use it", and re-attaching the source restores
    only what the source holds. A caller reaching for the composed table needs
    to be told that table was there, or the instruction sends it after
    something that will not come back.
    """
    config_module.use(Config(roots=(root,), slots=1))
    registry = Registry()
    try:
        registry.attach(str(csv_at(root / "sales.csv")), "shop")
        csv_at(root / "prices.csv", "sku,price\na,10\n")
        registry.create_table("shop", source=str(root / "prices.csv"))
        csv_at(root / "other.csv", "a\n1\n")
        registry.attach(str(root / "other.csv"), "next")

        with pytest.raises(SlotNotAvailable, match="prices"):
            registry.slot("shop")
    finally:
        registry.close()


def test_refusing_a_duplicate_source_names_every_table_the_slot_holds(registry, root):
    """Same snapshot, third reader: "query it there" has to say where there is."""
    source = csv_at(root / "sales.csv")
    registry.attach(str(source), "shop")
    csv_at(root / "prices.csv", "sku,price\na,10\n")
    registry.create_table("shop", source=str(root / "prices.csv"))

    with pytest.raises(AttachRefused, match="prices"):
        registry.attach(str(source), "shop_again")


def test_using_an_evicted_nickname_explains_the_eviction(root):
    config_module.use(Config(roots=(root,), slots=1))
    registry = Registry()
    try:
        fill(registry, root, 2)
        with pytest.raises(SlotNotAvailable, match="evicted"):
            registry.query("s0", "SELECT * FROM f0")
    finally:
        registry.close()


def test_an_unknown_nickname_is_distinguished_from_an_evicted_one(registry, root):
    with pytest.raises(SlotNotAvailable) as raised:
        registry.query("neverexisted", "SELECT 1")
    assert "evicted" not in str(raised.value).lower()


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
        _, rows = registry.query("keeper", "SELECT count(*) FROM keeper")
        assert rows == [(2,)]
    finally:
        registry.close()


def test_a_slot_can_be_attached_again_after_eviction(root):
    config_module.use(Config(roots=(root,), slots=1))
    registry = Registry()
    try:
        fill(registry, root, 2)
        registry.attach(str(root / "f0.csv"), "s0")
        _, rows = registry.query("s0", "SELECT a FROM f0")
        assert rows == [(1,)]
    finally:
        registry.close()


def test_a_genuine_missing_table_still_reports_itself_plainly(registry, root):
    csv_at(root / "sales.csv")
    registry.attach(str(root / "sales.csv"), "shop")
    with pytest.raises(Exception) as raised:
        registry.query("shop", "SELECT * FROM absent")
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
    slot = registry._attach_engine(
        f"sqlite:///{root / 'remote.db'}", "remote", writable=False
    )

    assert slot.kind == "engine"
    assert slot.tables == ("products",)
    columns, rows = registry.query("remote", "SELECT sku FROM products ORDER BY sku")
    assert columns == ["sku"]
    assert rows == [("a",), ("b",)]


def test_an_engine_slot_reports_a_source_without_its_password(registry, root):
    build_database(root / "remote.db")
    slot = registry._attach_engine(
        f"sqlite:///{root / 'remote.db'}", "remote", writable=False
    )
    assert "remote.db" in slot.source


# ---------------------------------------------------------------------------
# A URL-addressed datasource is a slot like any other
#
# These are the tests the two-code-path design could not pass. A slot reached
# over a URL used to live outside the workspace, with its own implementations of
# query, describe and tables — and *no* implementation of the rest, so add_table
# and save were refused by kind rather than by capability. One path means the
# verbs are the same verbs, and these say so.
# ---------------------------------------------------------------------------


def test_a_url_slot_composes_like_any_other(registry, root):
    """add_table lands a file inside a URL-addressed database."""
    build_database(root / "remote.db")
    csv_at(root / "stock.csv")
    registry._attach_engine(f"sqlite:///{root / 'remote.db'}", "remote", writable=True)

    added = registry.create_table("remote", source=str(root / "stock.csv"))

    assert added.name == "stock"
    assert added.row_count == 2


def test_a_url_slot_lists_what_was_added_to_it(registry, root):
    """``tables`` asks the database, so composition is visible afterwards."""
    build_database(root / "remote.db")
    csv_at(root / "stock.csv")
    registry._attach_engine(f"sqlite:///{root / 'remote.db'}", "remote", writable=True)
    registry.create_table("remote", source=str(root / "stock.csv"))

    assert registry.tables("remote") == ("products", "stock")


def test_a_url_slot_joins_the_table_that_was_added_to_it(registry, root):
    build_database(root / "remote.db")
    csv_at(root / "stock.csv")
    registry._attach_engine(f"sqlite:///{root / 'remote.db'}", "remote", writable=True)
    registry.create_table("remote", source=str(root / "stock.csv"))

    _, rows = registry.query(
        "remote",
        "SELECT p.name, s.qty FROM products p JOIN stock s ON s.sku = p.sku "
        "ORDER BY p.sku",
    )
    assert rows == [("Widget", 3), ("Gadget", 4)]


def test_a_url_slot_describes_a_table_it_was_given(registry, root):
    build_database(root / "remote.db")
    csv_at(root / "stock.csv")
    registry._attach_engine(f"sqlite:///{root / 'remote.db'}", "remote", writable=True)
    registry.create_table("remote", source=str(root / "stock.csv"))

    info = registry.describe("remote", "stock")
    assert [column.name for column in info.columns] == ["sku", "qty"]
    assert info.row_count == 2


def test_a_read_only_url_slot_still_refuses_composition(registry, root):
    """Unified does not mean permissive: the grant still governs."""
    build_database(root / "remote.db")
    csv_at(root / "stock.csv")
    registry._attach_engine(f"sqlite:///{root / 'remote.db'}", "remote", writable=False)

    with pytest.raises(NotWritable):
        registry.create_table("remote", source=str(root / "stock.csv"))


def test_a_url_slot_backed_by_a_real_file_can_be_saved(registry, root):
    """``save`` is refused by capability now, not by how the slot was opened."""
    build_database(root / "remote.db")
    registry._attach_engine(f"sqlite:///{root / 'remote.db'}", "remote", writable=False)

    saved = registry.save("remote", str(root / "kept.db"))

    assert saved.exists()
    connection = sqlite3.connect(saved)
    try:
        assert connection.execute("SELECT count(*) FROM products").fetchone()[0] == 2
    finally:
        connection.close()


# ---------------------------------------------------------------------------
# Listing
# ---------------------------------------------------------------------------


def test_slots_are_listed_in_the_order_they_were_attached(registry, root):
    fill(registry, root, 3)
    assert [slot.nickname for slot in registry.slots()] == ["s0", "s1", "s2"]


def test_a_fresh_registry_holds_nothing(registry):
    assert registry.slots() == []


# ---------------------------------------------------------------------------
# Detaching on purpose
# ---------------------------------------------------------------------------


def test_detaching_frees_the_slot_for_another_datasource(root):
    """The deliberate counterpart to eviction: it recovers real capacity."""
    config_module.use(Config(roots=(root,), slots=1))
    registry = Registry()
    try:
        csv_at(root / "first.csv", "a\n1\n")
        csv_at(root / "second.csv", "b\n2\n")
        registry.attach(str(root / "first.csv"), "one")

        registry.detach("one")
        attachment = registry.attach(str(root / "second.csv"), "two")

        assert attachment.evicted is None
        assert [slot.nickname for slot in registry.slots()] == ["two"]
        _, rows = registry.query("two", "SELECT b FROM second")
        assert rows == [(2,)]
    finally:
        registry.close()


def test_a_detached_source_can_be_attached_again(registry, root):
    """Detach is also how you get around the same-source refusal on purpose."""
    csv_at(root / "sales.csv")
    registry.attach(str(root / "sales.csv"), "shop")
    registry.detach("shop")

    again = registry.attach(str(root / "sales.csv"), "shop")
    assert again.slot.nickname == "shop"
    _, rows = registry.query("shop", "SELECT count(*) FROM sales")
    assert rows == [(2,)]


def test_detaching_a_nickname_nobody_holds_explains_itself(registry, root):
    csv_at(root / "sales.csv")
    registry.attach(str(root / "sales.csv"), "shop")
    with pytest.raises(SlotNotAvailable, match="shop"):
        registry.detach("elsewhere")


# ---------------------------------------------------------------------------
# Composing a slot: the lookup arc
# ---------------------------------------------------------------------------


def test_a_second_file_joins_the_first_inside_one_database(registry, root):
    """The whole point of adding rather than attaching: one database, so a

    view over the join is stable instead of going invalid on the next eviction.
    """
    csv_at(root / "sales.csv", "sku,qty\na,3\nb,4\n")
    csv_at(root / "prices.csv", "sku,price\na,10\nb,20\n")
    registry.attach(str(root / "sales.csv"), "shop")

    added = registry.create_table("shop", source=str(root / "prices.csv"))

    # `qualified` identifies the table; it is not how a statement addresses it.
    assert added.qualified == "shop.prices"
    assert added.row_count == 2
    _, rows = registry.query(
        "shop",
        "SELECT s.sku, s.qty * p.price FROM sales s "
        "JOIN prices p ON s.sku = p.sku ORDER BY s.sku",
    )
    assert rows == [("a", 30), ("b", 80)]


def test_the_added_table_is_named_from_its_file_unless_told_otherwise(registry, root):
    csv_at(root / "sales.csv")
    csv_at(root / "2025 prices.csv", "sku,price\na,10\n")
    registry.attach(str(root / "sales.csv"), "shop")

    derived = registry.create_table("shop", source=str(root / "2025 prices.csv"))
    assert derived.name == "table_2025_prices"

    csv_at(root / "more.csv", "sku,x\na,1\n")
    named = registry.create_table("shop", table="extra", source=str(root / "more.csv"))
    assert named.name == "extra"


def test_adding_over_an_existing_table_is_refused(registry, root):
    """Never silently replace: the rows already there are unrecoverable."""
    csv_at(root / "sales.csv")
    registry.attach(str(root / "sales.csv"), "shop")

    with pytest.raises(SlotError, match="already exists"):
        registry.create_table("shop", table="sales", source=str(root / "sales.csv"))

    _, rows = registry.query("shop", "SELECT count(*) FROM sales")
    assert rows == [(2,)]


def test_a_read_only_slot_refuses_composition_and_says_how_to_allow_it(registry, root):
    build_database(root / "warehouse.db")
    registry.attach(str(root / "warehouse.db"), "wh")
    csv_at(root / "extra.csv", "sku,x\na,1\n")

    with pytest.raises(NotWritable, match="writable=true"):
        registry.create_table("wh", source=str(root / "extra.csv"))
    with pytest.raises(NotWritable):
        registry.drop_table("wh", "products")


def test_composition_is_allowed_once_write_is_granted(registry, root):
    build_database(root / "warehouse.db")
    registry.attach(str(root / "warehouse.db"), "wh", writable=True)
    csv_at(root / "extra.csv", "sku,x\na,1\n")

    registry.create_table("wh", source=str(root / "extra.csv"))
    _, rows = registry.query(
        "wh", "SELECT p.name FROM products p JOIN extra e ON p.sku = e.sku"
    )
    assert rows == [("Widget",)]


def test_dropping_a_table_leaves_the_rest_of_the_slot_answering(registry, root):
    csv_at(root / "sales.csv")
    csv_at(root / "prices.csv", "sku,price\na,10\n")
    registry.attach(str(root / "sales.csv"), "shop")
    registry.create_table("shop", source=str(root / "prices.csv"))

    registry.drop_table("shop", "prices")

    assert registry.tables("shop") == ("sales",)
    _, rows = registry.query("shop", "SELECT count(*) FROM sales")
    assert rows == [(2,)]


def test_dropping_a_table_that_is_not_there_lists_the_ones_that_are(registry, root):
    csv_at(root / "sales.csv")
    registry.attach(str(root / "sales.csv"), "shop")
    with pytest.raises(SlotNotAvailable, match="sales"):
        registry.drop_table("shop", "prices")


# ---------------------------------------------------------------------------
# Indexes: asked for, never inferred
# ---------------------------------------------------------------------------


def test_an_index_is_created_on_the_columns_asked_for_and_names_itself(registry, root):
    csv_at(root / "sales.csv", "sku,qty\na,3\nb,4\n")
    registry.attach(str(root / "sales.csv"), "shop")

    made = registry.create_index("shop", table="sales", columns=["sku"])

    assert made.name == "ix_sales_sku"
    assert made.table == "sales"
    assert made.columns == ("sku",)
    assert [index.name for index in registry.indexes("shop")] == ["ix_sales_sku"]


def test_a_composite_index_keeps_the_column_order_it_was_given(registry, root):
    """Order is not cosmetic — an index on (a, b) does not serve a lookup on b."""
    csv_at(root / "sales.csv", "sku,qty\na,3\n")
    registry.attach(str(root / "sales.csv"), "shop")

    made = registry.create_index("shop", table="sales", columns=["qty", "sku"])

    assert made.columns == ("qty", "sku")
    assert made.name == "ix_sales_qty_sku"


def test_asking_twice_for_the_same_index_is_refused_by_name(registry, root):
    csv_at(root / "sales.csv", "sku,qty\na,3\n")
    registry.attach(str(root / "sales.csv"), "shop")
    registry.create_index("shop", table="sales", columns=["sku"])

    with pytest.raises(SlotError, match="already indexed on sku, by ix_sales_sku"):
        registry.create_index("shop", table="sales", columns=["sku"])


def test_an_index_on_a_column_that_is_not_there_lists_the_ones_that_are(registry, root):
    csv_at(root / "sales.csv", "sku,qty\na,3\n")
    registry.attach(str(root / "sales.csv"), "shop")

    with pytest.raises(SlotError, match="sku, qty"):
        registry.create_index("shop", table="sales", columns=["price"])


def test_an_index_needs_at_least_one_column(registry, root):
    csv_at(root / "sales.csv")
    registry.attach(str(root / "sales.csv"), "shop")

    with pytest.raises(SlotError, match="at least one column"):
        registry.create_index("shop", table="sales", columns=[])


def test_an_index_is_dropped_by_the_name_creation_returned(registry, root):
    csv_at(root / "sales.csv", "sku,qty\na,3\n")
    registry.attach(str(root / "sales.csv"), "shop")
    made = registry.create_index("shop", table="sales", columns=["sku"])

    gone = registry.drop_index("shop", made.name)

    assert gone.name == made.name
    assert gone.table == "sales"
    assert registry.indexes("shop") == ()


def test_dropping_an_index_that_is_not_there_lists_the_ones_that_are(registry, root):
    csv_at(root / "sales.csv", "sku,qty\na,3\n")
    registry.attach(str(root / "sales.csv"), "shop")
    registry.create_index("shop", table="sales", columns=["sku"])

    with pytest.raises(SlotError, match="No such index.*ix_sales_sku"):
        registry.drop_index("shop", "ix_sales_price")


def test_a_read_only_slot_refuses_indexes_in_both_directions(registry, root):
    """The write grant governs indexes exactly as it governs tables."""
    build_database(root / "warehouse.db")
    registry.attach(str(root / "warehouse.db"), "wh")

    with pytest.raises(NotWritable, match="writable=true"):
        registry.create_index("wh", table="products", columns=["sku"])
    with pytest.raises(NotWritable):
        registry.drop_index("wh", "anything")


def test_indexes_a_datasource_arrived_with_are_reported(registry, root):
    """Reported whoever made them — an attached database is not a blank slate.

    The pre-existing index is built through SQLAlchemy over a URL, not through
    the ``sqlite3`` driver: a fixture that reaches past the abstraction proves
    the abstraction only for the one backend it reached past.
    """
    build_database(root / "warehouse.db")
    outside = create_engine(f"sqlite:///{root / 'warehouse.db'}")
    products = Table("products", MetaData(), autoload_with=outside)
    Index("ix_products_sku", products.c.sku).create(outside)
    outside.dispose()
    registry.attach(str(root / "warehouse.db"), "wh")

    assert [index.name for index in registry.indexes("wh")] == ["ix_products_sku"]


def test_an_index_survives_the_save_that_carries_its_table(registry, root):
    csv_at(root / "sales.csv", "sku,qty\na,3\n")
    registry.attach(str(root / "sales.csv"), "shop")
    registry.create_index("shop", table="sales", columns=["sku"])

    saved = registry.save("shop", str(root / "keep.db"))
    registry.detach("shop")
    registry.attach(str(saved), "kept")

    assert [index.name for index in registry.indexes("kept")] == ["ix_sales_sku"]


def test_whether_a_join_lines_up_is_answerable_in_plain_sql(registry, root):
    """The completeness check moved to the caller; it did not disappear.

    This is the anti-join the old ``join_on`` parameter ran internally, written
    the way a caller writes it. Both directions, because which one matters is
    the caller's question rather than ours: ``b`` and ``c`` sold with no price,
    and ``z`` priced with nothing sold.
    """
    csv_at(root / "sales.csv", "sku,qty\na,3\nb,4\nc,5\n")
    csv_at(root / "prices.csv", "sku,price\na,10\nz,99\n")
    registry.attach(str(root / "sales.csv"), "shop")
    registry.create_table("shop", source=str(root / "prices.csv"))

    _, unpriced = registry.query(
        "shop",
        "SELECT sku FROM sales WHERE sku NOT IN (SELECT sku FROM prices) ORDER BY sku",
    )
    _, unsold = registry.query(
        "shop",
        "SELECT sku FROM prices WHERE sku NOT IN (SELECT sku FROM sales) ORDER BY sku",
    )

    assert unpriced == [("b",), ("c",)]
    assert unsold == [("z",)]


# ---------------------------------------------------------------------------
# Saving: the escape from ephemerality
# ---------------------------------------------------------------------------


def test_a_saved_database_can_be_attached_again_with_its_rows(registry, root):
    csv_at(root / "sales.csv")
    csv_at(root / "prices.csv", "sku,price\na,10\nb,20\n")
    registry.attach(str(root / "sales.csv"), "shop")
    registry.create_table("shop", source=str(root / "prices.csv"))

    saved = registry.save("shop", str(root / "keep.db"))
    registry.detach("shop")
    again = registry.attach(str(saved), "kept")

    assert sorted(again.slot.tables) == ["prices", "sales"]
    _, rows = registry.query(
        "kept",
        "SELECT s.sku, s.qty * p.price FROM sales s "
        "JOIN prices p ON s.sku = p.sku ORDER BY s.sku",
    )
    assert rows == [("a", 30), ("b", 80)]


def test_a_saved_database_comes_back_read_only(registry, root):
    """Like any other outside database — being ours once does not persist."""
    csv_at(root / "sales.csv")
    registry.attach(str(root / "sales.csv"), "shop")
    saved = registry.save("shop", str(root / "keep.db"))
    registry.detach("shop")

    slot = registry.attach(str(saved), "kept").slot
    assert slot.writable is False
    with pytest.raises(NotWritable, match="writable=true"):
        registry.drop_table("kept", "sales")


def test_saving_keeps_the_slot_answering_and_writable(registry, root):
    """Copied out, not moved away: the session is not disturbed by keeping it."""
    csv_at(root / "sales.csv")
    registry.attach(str(root / "sales.csv"), "shop")

    registry.save("shop", str(root / "keep.db"))

    csv_at(root / "prices.csv", "sku,price\na,10\n")
    registry.create_table("shop", source=str(root / "prices.csv"))
    assert "prices" in registry.tables("shop")
    _, rows = registry.query("shop", "SELECT count(*) FROM sales")
    assert rows == [(2,)]


def test_saving_refuses_an_existing_file_then_replaces_it_when_forced(registry, root):
    """A second save must not cost the user the first one by accident.

    The path came from the user via an agent, so replacing what is there is the
    user's decision — `force` is that decision arriving, not the agent's own.
    """
    csv_at(root / "sales.csv")
    registry.attach(str(root / "sales.csv"), "shop")
    saved = registry.save("shop", str(root / "keep.db"))
    kept = saved.read_bytes()

    csv_at(root / "prices.csv", "sku,price\na,10\n")
    registry.create_table("shop", source=str(root / "prices.csv"))

    with pytest.raises(SlotError, match="Ask the user"):
        registry.save("shop", str(root / "keep.db"))
    assert saved.read_bytes() == kept

    replaced = registry.save("shop", str(root / "keep.db"), force=True)
    connection = sqlite3.connect(replaced)
    try:
        # The forced write carries the slot as it stands now, not as it was:
        # the table added after the first save is in the replacement.
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        assert tables == {"sales", "prices"}
    finally:
        connection.close()


def test_forcing_a_save_over_a_live_slots_own_file_is_refused(registry, root):
    """Not the same slot only — any live slot's file, including a spilled one.

    The unlink would succeed and the slot would keep answering from an unnamed
    inode, so the divergence would never surface as an error.
    """
    csv_at(root / "sales.csv")
    csv_at(root / "prices.csv")
    registry.attach(str(root / "sales.csv"), "shop")
    registry.attach(str(root / "prices.csv"), "wh")

    for target, holder in ((root / "sales.csv", "shop"), (root / "prices.csv", "wh")):
        with pytest.raises(SlotError, match=f"'{holder}'"):
            registry.save("shop", str(target), force=True)
        assert target.exists()

    assert registry.claimed_paths() == {
        (root / "sales.csv").resolve(): "shop",
        (root / "prices.csv").resolve(): "wh",
    }


def test_a_saved_file_is_not_world_readable(registry, root):
    """It holds the user's actual data; SQLite would have created it 0o644."""
    csv_at(root / "sales.csv")
    registry.attach(str(root / "sales.csv"), "shop")

    saved = registry.save("shop", str(root / "keep.db"))

    assert saved.stat().st_mode & 0o077 == 0


def test_saving_outside_the_allowed_area_is_refused(registry, root, tmp_path):
    csv_at(root / "sales.csv")
    registry.attach(str(root / "sales.csv"), "shop")
    with pytest.raises(SlotError, match="outside the allowed paths"):
        registry.save("shop", str(tmp_path / "elsewhere.db"))


# ---------------------------------------------------------------------------
# Outgrowing memory: spill to disk, in place, without saying so
# ---------------------------------------------------------------------------


def bulky_csv(path: Path, rows: int = 60_000, marker: str = "x") -> Path:
    """A file large enough to cross a one-megabyte budget on its own."""
    lines = "\n".join(f"{index},{marker}{index},{index * 3}" for index in range(rows))
    path.write_text(f"id,label,amount\n{lines}\n")
    return path


def budgeted(root: Path, megabytes: int = 1, slots: int = 10) -> Registry:
    config_module.use(Config(roots=(root,), slots=slots, memory_budget_mb=megabytes))
    return Registry()


def test_a_database_that_outgrows_the_budget_still_answers_from_disk(root):
    """Assert on what the queries return — a file appearing proves nothing."""
    registry = budgeted(root)
    try:
        bulky_csv(root / "big.csv")
        registry.attach(str(root / "big.csv"), "big")
        before = registry.query(
            "big", "SELECT count(*), sum(amount), min(label), max(label) FROM big"
        )[1]

        moved = registry.relieve_memory()

        assert [slot.nickname for slot in moved] == ["big"]
        after = registry.query(
            "big", "SELECT count(*), sum(amount), min(label), max(label) FROM big"
        )[1]
        assert after == before
        assert after[0][0] == 60_000
    finally:
        registry.close()


def test_the_overshoot_is_tolerated_once_and_paid_for_next_time(root):
    """The load that crosses the line finishes; the next operation clears it."""
    registry = budgeted(root)
    try:
        bulky_csv(root / "big.csv")
        registry.attach(str(root / "big.csv"), "big")

        # Straight after the attach, nothing has moved: the overshoot stands.
        assert registry.slot("big").spill_path is None

        assert registry.relieve_memory() != ()
        assert registry.slot("big").spill_path is not None
    finally:
        registry.close()


def test_a_session_inside_its_budget_moves_nothing(root):
    registry = budgeted(root, megabytes=100)
    try:
        csv_at(root / "sales.csv")
        registry.attach(str(root / "sales.csv"), "shop")

        assert registry.relieve_memory() == ()
        assert registry.slot("shop").spill_path is None
    finally:
        registry.close()


def test_the_spilled_slot_keeps_its_name_and_its_place_in_eviction_order(root):
    """A database that moved to disk must not become the youngest slot."""
    registry = budgeted(root, slots=3)
    try:
        csv_at(root / "first.csv", "a\n1\n")
        bulky_csv(root / "big.csv")
        csv_at(root / "third.csv", "c\n3\n")
        registry.attach(str(root / "first.csv"), "one")
        registry.attach(str(root / "big.csv"), "big")
        registry.attach(str(root / "third.csv"), "three")

        registry.relieve_memory()

        assert [slot.nickname for slot in registry.slots()] == ["one", "big", "three"]

        # The shelf is full, so the next attach evicts the oldest — which must
        # still be 'one', not the slot that happens to have moved most recently.
        csv_at(root / "fourth.csv", "d\n4\n")
        evicted = registry.attach(str(root / "fourth.csv"), "four").evicted
        assert evicted is not None
        assert evicted.nickname == "one"
    finally:
        registry.close()


def test_the_largest_database_is_the_one_that_moves(root):
    registry = budgeted(root)
    try:
        csv_at(root / "small.csv", "a\n1\n")
        bulky_csv(root / "big.csv")
        registry.attach(str(root / "small.csv"), "small")
        registry.attach(str(root / "big.csv"), "big")

        moved = registry.relieve_memory()

        assert [slot.nickname for slot in moved] == ["big"]
        assert registry.slot("small").spill_path is None
    finally:
        registry.close()


def test_a_spilled_database_is_still_writable_and_composable(root):
    """Moving it must not quietly take away rights the caller already had."""
    registry = budgeted(root)
    try:
        bulky_csv(root / "big.csv")
        registry.attach(str(root / "big.csv"), "big")
        registry.relieve_memory()

        csv_at(root / "labels.csv", "id,note\n1,first\n")
        registry.create_table("big", source=str(root / "labels.csv"))
        csv_at(root / "more.csv", "id,note\n2,second\n")
        registry.create_table("big", source=str(root / "more.csv"), table="more")

        _, rows = registry.query(
            "big",
            "SELECT l.note FROM big b JOIN labels l ON b.id = l.id ORDER BY l.note",
        )
        assert rows == [("first",)]
        # Composable after the move: both added tables landed in the same slot.
        assert {"labels", "more"} <= set(registry.tables("big"))
    finally:
        registry.close()


def test_relieving_twice_does_not_move_an_already_spilled_database_again(root):
    registry = budgeted(root)
    try:
        bulky_csv(root / "big.csv")
        registry.attach(str(root / "big.csv"), "big")
        first = registry.relieve_memory()

        assert registry.relieve_memory() == ()
        assert registry.slot("big").spill_path == first[0].spill_path
    finally:
        registry.close()


def test_the_temp_file_goes_when_the_slot_is_detached(root):
    registry = budgeted(root)
    try:
        bulky_csv(root / "big.csv")
        registry.attach(str(root / "big.csv"), "big")
        registry.relieve_memory()
        spilled = registry.slot("big").spill_path
        assert spilled is not None and spilled.exists()

        registry.detach("big")

        assert not spilled.exists()
    finally:
        registry.close()


def test_the_temp_file_goes_when_the_slot_is_evicted(root):
    registry = budgeted(root, slots=1)
    try:
        bulky_csv(root / "big.csv")
        registry.attach(str(root / "big.csv"), "big")
        registry.relieve_memory()
        spilled = registry.slot("big").spill_path
        assert spilled is not None and spilled.exists()

        csv_at(root / "next.csv", "a\n1\n")
        registry.attach(str(root / "next.csv"), "next")

        assert not spilled.exists()
    finally:
        registry.close()


def test_the_temp_directory_goes_when_the_session_ends_with_slots_live(root):
    """The third exit, beside eviction and detach: closing with data attached."""
    registry = budgeted(root)
    bulky_csv(root / "big.csv")
    registry.attach(str(root / "big.csv"), "big")
    registry.relieve_memory()
    spilled = registry.slot("big").spill_path
    assert spilled is not None and spilled.exists()

    registry.close()

    assert not spilled.exists()
    assert not spilled.parent.exists()


def test_a_spilled_database_can_still_be_saved(root):
    registry = budgeted(root)
    try:
        bulky_csv(root / "big.csv")
        registry.attach(str(root / "big.csv"), "big")
        registry.relieve_memory()

        saved = registry.save("big", str(root / "keep.db"))

        connection = sqlite3.connect(saved)
        try:
            assert (
                connection.execute("SELECT count(*) FROM big").fetchone()[0] == 60_000
            )
        finally:
            connection.close()
    finally:
        registry.close()


def test_nothing_is_created_on_disk_until_something_has_to_move(root):
    """A session that stays inside its budget writes nothing anywhere."""
    registry = budgeted(root, megabytes=100)
    try:
        csv_at(root / "sales.csv")
        registry.attach(str(root / "sales.csv"), "shop")
        registry.relieve_memory()
        assert registry._temp_dir is None
    finally:
        registry.close()


def test_the_spill_really_moves_the_database_off_the_heap(root):
    """A near-zero effect has the same shape as a no-op, so check the mechanism.

    SQLite names the file behind each attached schema, and an in-memory database
    has no file. Watching that entry go from empty to the temp path is direct
    evidence the pages left the heap, which is not something the queries above
    could distinguish from a rename.
    """
    registry = budgeted(root)
    try:
        bulky_csv(root / "big.csv")
        registry.attach(str(root / "big.csv"), "big")
        assert _backing_file(registry, "big") == ""

        registry.relieve_memory()

        spilled = registry.slot("big").spill_path
        assert spilled is not None
        assert _backing_file(registry, "big") == str(spilled)
        assert spilled.stat().st_size > 0
    finally:
        registry.close()


def _backing_file(registry: Registry, tag: str) -> str:
    """The file behind a tag's database; empty while it is still in memory.

    The tag dict carries this directly: ``uri`` is where the data came from and
    never moves, ``location`` is where it sits now. A spill is exactly the moment
    those two stop being equal.
    """
    location = registry.workspace.location(tag)
    return "" if location == ":memory:" else location


# ---------------------------------------------------------------------------
# A view carries the nickname it was built under, and that travels badly
# ---------------------------------------------------------------------------


def test_a_view_naming_tables_unqualified_survives_being_saved_and_renamed(
    registry, root
):
    """Inside a view, an unqualified name already means *this* database.

    So the portable spelling is the short one, and it keeps working under a
    nickname nobody had thought of when the view was written.
    """
    csv_at(root / "sales.csv", "sku,qty\na,3\n")
    csv_at(root / "prices.csv", "sku,price\na,10\n")
    registry.attach(str(root / "sales.csv"), "shop")
    registry.create_table("shop", source=str(root / "prices.csv"))
    view_from_outside(
        registry,
        "shop",
        "CREATE VIEW revenue AS SELECT s.sku, s.qty * p.price AS total "
        "FROM sales s JOIN prices p ON s.sku = p.sku",
    )

    saved = registry.save("shop", str(root / "keep.db"))
    registry.detach("shop")
    registry.attach(str(saved), "renamed")

    _, rows = registry.query("renamed", "SELECT total FROM revenue")
    assert rows == [(30,)]


def test_a_table_aliased_to_the_nickname_does_not_trip_the_check(registry, root):
    """The check attaches the file rather than scanning the view's text.

    Text-scanning for the nickname would flag this view, whose 'shop' is an
    alias and not a schema at all.
    """
    csv_at(root / "sales.csv", "sku,qty\na,3\n")
    registry.attach(str(root / "sales.csv"), "shop")
    view_from_outside(
        registry, "shop", "CREATE VIEW totals AS SELECT shop.qty FROM sales shop"
    )

    saved = registry.save("shop", str(root / "keep.db"))
    registry.detach("shop")
    registry.attach(str(saved), "renamed")

    assert registry.query("renamed", "SELECT qty FROM totals")[1] == [(3,)]


def test_attaching_a_database_poisoned_by_such_a_view_explains_itself(registry, root):
    """SQLite says 'malformed database schema', which reads as corruption.

    The file is built the way one arrives in the wild: a view referencing its
    *own* schema by name, which SQLite accepts at create time and only rejects
    once the database is opened under some other name.
    """
    poisoned = root / "poisoned.db"
    connection = sqlite3.connect(":memory:")
    connection.execute("ATTACH DATABASE ':memory:' AS shop")
    connection.execute("CREATE TABLE shop.sales (sku TEXT, qty INT)")
    connection.execute("INSERT INTO shop.sales VALUES ('a', 3)")
    connection.execute("CREATE VIEW shop.v AS SELECT qty FROM shop.sales")
    connection.commit()
    connection.execute("VACUUM shop INTO ?", (str(poisoned),))
    connection.close()

    with pytest.raises(AttachRefused, match="unqualified"):
        registry.attach(str(poisoned), "kept")

    # And under the name it was built with too. A tag is not a schema alias any
    # more, so there is no name that makes the view resolve — which is why the
    # refusal is the whole answer rather than a hint to try another nickname.
    with pytest.raises(AttachRefused, match="unqualified"):
        registry.attach(str(poisoned), "shop")

    # A refusal costs nothing: no slot was taken by either attempt.
    assert registry.slots() == []


# ---------------------------------------------------------------------------
# A table name the database folds
# ---------------------------------------------------------------------------


def mixed_case_database(path: Path, table: str = "MyTable") -> Path:
    """A SQLite file whose table is stored under a name that is not lowercase.

    Built with ``sqlite3`` rather than from a flat file on purpose: the loader
    lowercases a name it derives from a filename, which would hide the very
    thing these tests are about.
    """
    connection = sqlite3.connect(path)
    connection.execute(f'CREATE TABLE "{table}" (id INTEGER, val TEXT)')
    connection.execute(f'INSERT INTO "{table}" VALUES (1, \'x\')')
    connection.commit()
    connection.close()
    return path


def test_a_table_query_can_read_is_not_reported_as_missing_by_the_other_verbs(
    registry, root
):
    """The five table-taking verbs answer one question the same way.

    SQLite resolves an unquoted identifier case-insensitively, so ``query``
    reads ``mytable`` out of a table stored as ``MyTable``. A verb that takes
    the same name as an *argument* and refuses it — while listing that table as
    what the slot holds — is telling the caller two things at once.
    """
    mixed_case_database(root / "mixed.sqlite")
    registry.attach(str(root / "mixed.sqlite"), "mixed", writable=True)

    assert registry.query("mixed", "SELECT val FROM mytable")[1] == [("x",)]
    assert registry.describe("mixed", "mytable").row_count == 1
    registry.create_index("mixed", table="mytable", columns=["id"])
    registry.rename_table("mixed", "mytable", "renamed")
    registry.drop_table("mixed", "RENAMED")
    assert registry.tables("mixed") == ()


def test_a_verb_answers_under_the_name_the_database_stores(registry, root):
    """The payload names the table, not the spelling the caller happened to use.

    A caller who is handed back their own input has learned nothing, and the
    name they were given will not match the slot's own listing.
    """
    mixed_case_database(root / "mixed.sqlite")
    registry.attach(str(root / "mixed.sqlite"), "mixed", writable=True)

    assert registry.describe("mixed", "mytable").name == "MyTable"
    assert registry.rename_table("mixed", "MYTABLE", "renamed").name == "renamed"


def test_a_genuinely_absent_table_is_still_refused_with_what_is_there(registry, root):
    """Folding a name must not soften the refusal for a name that is not there."""
    mixed_case_database(root / "mixed.sqlite")
    registry.attach(str(root / "mixed.sqlite"), "mixed", writable=True)

    for call in (
        lambda: registry.describe("mixed", "nope"),
        lambda: registry.rename_table("mixed", "nope", "other"),
        lambda: registry.drop_table("mixed", "nope"),
        lambda: registry.create_index("mixed", table="nope", columns=["id"]),
    ):
        with pytest.raises(SlotNotAvailable) as raised:
            call()
        assert str(raised.value) == "No such table: mixed.nope. In mixed: MyTable."


def test_landing_a_table_beside_one_that_differs_only_in_case_is_a_collision(
    registry, root
):
    """SQLite would not hold both, so the collision is answered here."""
    mixed_case_database(root / "mixed.sqlite")
    registry.attach(str(root / "mixed.sqlite"), "mixed", writable=True)
    csv_at(root / "mytable.csv", "id,val\n2,y\n")

    with pytest.raises(SlotError, match="already exists"):
        registry.create_table("mixed", source=str(root / "mytable.csv"))


def test_renaming_a_table_to_its_own_name_in_another_case_says_why_it_cannot(
    registry, root
):
    """SQLite holds one name for both spellings, so this is not a free name."""
    mixed_case_database(root / "mixed.sqlite")
    registry.attach(str(root / "mixed.sqlite"), "mixed", writable=True)

    with pytest.raises(SlotError, match="differ only in case"):
        registry.rename_table("mixed", "MyTable", "mytable")
    # And the table is untouched by the refusal.
    assert registry.tables("mixed") == ("MyTable",)


def test_two_stored_names_differing_only_in_case_resolve_to_neither(registry, root):
    """A case-sensitive engine can hold both, and then a fold is a guess.

    Modelled by handing the resolver the name list such an engine reports —
    which is the whole of what it reads — since SQLite cannot store the pair.
    """
    mixed_case_database(root / "mixed.sqlite")
    registry.attach(str(root / "mixed.sqlite"), "mixed", writable=True)
    workspace = registry._workspace
    workspace.table_names = lambda tag: ("MyTable", "mytable")

    assert workspace.resolve_table("mixed", "MyTable") == "MyTable"
    assert workspace.resolve_table("mixed", "mytable") == "mytable"
    assert workspace.resolve_table("mixed", "MYTABLE") is None
