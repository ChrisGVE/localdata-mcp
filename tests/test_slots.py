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

    _, rows = registry.query("slot_2", "SELECT b FROM slot_2.second")
    assert rows == [(2,)]
    # The first slot is untouched, which is the whole point of not replacing it.
    _, first = registry.query("slot", "SELECT a FROM slot.first")
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
    _, rows = registry.query(nickname, f"SELECT qty FROM {nickname}.{table}")
    assert sorted(r[0] for r in rows) == [3, 4]


def test_a_derived_nickname_that_sqlite_reserves_is_stepped_over(registry, root):
    """A file called main.csv must not try to claim SQLite's own schema name."""
    csv_at(root / "main.csv")
    attachment = registry.attach(str(root / "main.csv"))

    assert attachment.slot.nickname == "main_2"
    _, rows = registry.query("main_2", "SELECT count(*) FROM main_2.main")
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

    _, rows = registry.query("sales_2", "SELECT qty FROM sales_2.sales")
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
        _, rows = registry.query("one", "SELECT a FROM one.first")
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
    registry.query("shop", "INSERT INTO shop.sales VALUES ('c', 5)")
    _, rows = registry.query("shop", "SELECT qty FROM shop.sales WHERE sku='c'")
    assert rows == [(5,)]


def test_an_outside_database_arrives_read_only(registry, root):
    build_database(root / "warehouse.db")
    slot = registry.attach(str(root / "warehouse.db"), "wh").slot

    assert slot.writable is False
    with pytest.raises(Exception, match="readonly"):
        registry.query("wh", "INSERT INTO wh.products VALUES ('c', 'Thing')")


def test_the_write_grant_is_honoured_when_it_is_asked_for(registry, root):
    build_database(root / "warehouse.db")
    slot = registry.attach(str(root / "warehouse.db"), "wh", writable=True).slot

    assert slot.writable is True
    registry.query("wh", "INSERT INTO wh.products VALUES ('c', 'Thing')")
    _, rows = registry.query("wh", "SELECT name FROM wh.products WHERE sku='c'")
    assert rows == [("Thing",)]


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
    slot = registry._attach_engine(
        f"sqlite:///{root / 'remote.db'}", "remote", writable=False
    )

    assert slot.kind == "engine"
    assert slot.tables == ("products",)
    columns, rows = registry.query("remote", "SELECT sku FROM products ORDER BY sku")
    assert columns == ["sku"]
    assert rows == [("a",), ("b",)]


def test_a_join_across_two_engines_explains_the_one_engine_rule(registry, root):
    build_database(root / "remote.db")
    csv_at(root / "sales.csv")
    registry.attach(str(root / "sales.csv"), "shop")
    registry._attach_engine(f"sqlite:///{root / 'remote.db'}", "remote", writable=False)

    with pytest.raises(Exception) as raised:
        registry.query("remote", "SELECT * FROM products JOIN shop.sales USING (sku)")
    assert "engine" in str(raised.value).lower()


def test_an_engine_slot_reports_a_source_without_its_password(registry, root):
    build_database(root / "remote.db")
    slot = registry._attach_engine(
        f"sqlite:///{root / 'remote.db'}", "remote", writable=False
    )
    assert "remote.db" in slot.source


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
        _, rows = registry.query("two", "SELECT b FROM two.second")
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
    _, rows = registry.query("shop", "SELECT count(*) FROM shop.sales")
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

    added = registry.add_table("shop", source=str(root / "prices.csv"))

    assert added.info.qualified == "shop.prices"
    assert added.info.row_count == 2
    _, rows = registry.query(
        "shop",
        "SELECT s.sku, s.qty * p.price FROM shop.sales s "
        "JOIN shop.prices p ON s.sku = p.sku ORDER BY s.sku",
    )
    assert rows == [("a", 30), ("b", 80)]


def test_a_view_can_only_be_built_over_tables_in_one_database(registry, root):
    """This is why the lookup arc adds a table instead of attaching a slot.

    The cross-database view is not merely fragile — SQLite refuses to create it
    at all, while the same join written over two tables in one database is an
    ordinary view that keeps answering. So "add the second file here" is not a
    preference about mental models; it is the only route to a stored join.
    """
    csv_at(root / "sales.csv", "sku,qty\na,3\n")
    csv_at(root / "prices.csv", "sku,price\na,10\n")
    build_database(root / "warehouse.db")
    registry.attach(str(root / "sales.csv"), "shop")
    registry.add_table("shop", source=str(root / "prices.csv"))
    registry.attach(str(root / "warehouse.db"), "wh")

    registry.query(
        "shop",
        "CREATE VIEW shop.revenue AS SELECT s.sku, s.qty * p.price AS total "
        "FROM shop.sales s JOIN shop.prices p ON s.sku = p.sku",
    )
    assert registry.query("shop", "SELECT total FROM shop.revenue")[1] == [(30,)]

    with pytest.raises(Exception, match="cannot reference objects in database"):
        registry.query(
            "shop",
            "CREATE VIEW shop.named AS SELECT w.name FROM shop.sales s "
            "JOIN wh.products w ON s.sku = w.sku",
        )

    # The join itself is fine across databases — it is only *storing* it that
    # cannot cross the line. That distinction is what the skill has to teach.
    assert registry.query(
        "shop",
        "SELECT w.name FROM shop.sales s JOIN wh.products w ON s.sku = w.sku",
    )[1] == [("Widget",)]


def test_a_declared_table_can_be_added_and_filled(registry, root):
    csv_at(root / "sales.csv")
    registry.attach(str(root / "sales.csv"), "shop")

    added = registry.add_table(
        "shop", table="notes", columns={"sku": "TEXT", "note": "TEXT"}
    )

    assert added.info.qualified == "shop.notes"
    assert added.info.row_count == 0
    registry.query("shop", "INSERT INTO shop.notes VALUES ('a', 'backordered')")
    _, rows = registry.query(
        "shop", "SELECT n.note FROM shop.sales s JOIN shop.notes n ON s.sku = n.sku"
    )
    assert rows == [("backordered",)]


def test_the_added_table_is_named_from_its_file_unless_told_otherwise(registry, root):
    csv_at(root / "sales.csv")
    csv_at(root / "2025 prices.csv", "sku,price\na,10\n")
    registry.attach(str(root / "sales.csv"), "shop")

    derived = registry.add_table("shop", source=str(root / "2025 prices.csv"))
    assert derived.info.name == "table_2025_prices"

    csv_at(root / "more.csv", "sku,x\na,1\n")
    named = registry.add_table("shop", table="extra", source=str(root / "more.csv"))
    assert named.info.name == "extra"


def test_adding_over_an_existing_table_is_refused(registry, root):
    """Never silently replace: the rows already there are unrecoverable."""
    csv_at(root / "sales.csv")
    registry.attach(str(root / "sales.csv"), "shop")

    with pytest.raises(SlotError, match="already exists"):
        registry.add_table("shop", table="sales", source=str(root / "sales.csv"))

    _, rows = registry.query("shop", "SELECT count(*) FROM shop.sales")
    assert rows == [(2,)]


def test_adding_needs_exactly_one_of_a_source_or_a_schema(registry, root):
    csv_at(root / "sales.csv")
    registry.attach(str(root / "sales.csv"), "shop")

    with pytest.raises(SlotError, match="exactly one"):
        registry.add_table("shop", table="x")
    with pytest.raises(SlotError, match="exactly one"):
        registry.add_table(
            "shop", table="x", source=str(root / "sales.csv"), columns={"a": "TEXT"}
        )


def test_a_read_only_slot_refuses_composition_and_says_how_to_allow_it(registry, root):
    build_database(root / "warehouse.db")
    registry.attach(str(root / "warehouse.db"), "wh")
    csv_at(root / "extra.csv", "sku,x\na,1\n")

    with pytest.raises(NotWritable, match="writable=true"):
        registry.add_table("wh", source=str(root / "extra.csv"))
    with pytest.raises(NotWritable):
        registry.drop_table("wh", "products")


def test_composition_is_allowed_once_write_is_granted(registry, root):
    build_database(root / "warehouse.db")
    registry.attach(str(root / "warehouse.db"), "wh", writable=True)
    csv_at(root / "extra.csv", "sku,x\na,1\n")

    registry.add_table("wh", source=str(root / "extra.csv"))
    _, rows = registry.query(
        "wh", "SELECT p.name FROM wh.products p JOIN wh.extra e ON p.sku = e.sku"
    )
    assert rows == [("Widget",)]


def test_dropping_a_table_leaves_the_rest_of_the_slot_answering(registry, root):
    csv_at(root / "sales.csv")
    csv_at(root / "prices.csv", "sku,price\na,10\n")
    registry.attach(str(root / "sales.csv"), "shop")
    registry.add_table("shop", source=str(root / "prices.csv"))

    registry.drop_table("shop", "prices")

    assert registry.tables("shop") == ("sales",)
    _, rows = registry.query("shop", "SELECT count(*) FROM shop.sales")
    assert rows == [(2,)]


def test_dropping_a_table_that_is_not_there_lists_the_ones_that_are(registry, root):
    csv_at(root / "sales.csv")
    registry.attach(str(root / "sales.csv"), "shop")
    with pytest.raises(SlotNotAvailable, match="sales"):
        registry.drop_table("shop", "prices")


# ---------------------------------------------------------------------------
# Does the join actually line up?
# ---------------------------------------------------------------------------


def test_a_complete_join_reports_nothing_missing(registry, root):
    csv_at(root / "sales.csv", "sku,qty\na,3\nb,4\n")
    csv_at(root / "prices.csv", "sku,price\na,10\nb,20\n")
    registry.attach(str(root / "sales.csv"), "shop")

    report = registry.add_table(
        "shop", source=str(root / "prices.csv"), join_on="sku"
    ).join

    assert report is not None
    assert report.complete is True
    assert report.matched_keys == 2
    assert report.missing_from_added == ()
    assert report.missing_from_existing == ()


def test_an_incomplete_join_names_the_values_on_the_correct_side(registry, root):
    """Both directions, because which one matters is not ours to guess."""
    csv_at(root / "sales.csv", "sku,qty\na,3\nb,4\nc,5\n")
    csv_at(root / "prices.csv", "sku,price\na,10\nz,99\n")
    registry.attach(str(root / "sales.csv"), "shop")

    report = registry.add_table(
        "shop", source=str(root / "prices.csv"), join_on="sku"
    ).join

    assert report is not None
    assert report.complete is False
    assert report.key == "sku"
    assert report.existing_table == "shop.sales"
    assert report.added_table == "shop.prices"
    assert report.matched_keys == 1
    # b and c are in sales with no price.
    assert report.missing_from_added == ("b", "c")
    assert report.missing_from_added_total == 2
    # z has a price nothing was sold under.
    assert report.missing_from_existing == ("z",)
    assert report.missing_from_existing_total == 1


def test_the_unmatched_sample_is_bounded_but_the_count_is_not(registry, root):
    rows = "\n".join(f"k{index},1" for index in range(50))
    csv_at(root / "sales.csv", f"sku,qty\n{rows}\n")
    csv_at(root / "prices.csv", "sku,price\nk0,10\n")
    registry.attach(str(root / "sales.csv"), "shop")

    report = registry.add_table(
        "shop", source=str(root / "prices.csv"), join_on="sku"
    ).join

    assert report is not None
    assert report.missing_from_added_total == 49
    assert len(report.missing_from_added) == 10


def test_a_null_key_is_not_reported_as_an_unmatched_value(registry, root):
    """A null key has no partner anywhere; saying so is noise, not a finding."""
    csv_at(root / "sales.csv", "sku,qty\na,3\n,4\n")
    csv_at(root / "prices.csv", "sku,price\na,10\n")
    registry.attach(str(root / "sales.csv"), "shop")

    report = registry.add_table(
        "shop", source=str(root / "prices.csv"), join_on="sku"
    ).join

    assert report is not None
    assert report.complete is True


def test_the_partner_table_must_be_said_when_the_slot_holds_several(registry, root):
    csv_at(root / "sales.csv", "sku,qty\na,3\n")
    csv_at(root / "prices.csv", "sku,price\na,10\n")
    csv_at(root / "stock.csv", "sku,on_hand\na,7\nz,1\n")
    registry.attach(str(root / "sales.csv"), "shop")
    registry.add_table("shop", source=str(root / "prices.csv"))

    with pytest.raises(SlotError, match="explicitly"):
        registry.add_table("shop", source=str(root / "stock.csv"), join_on="sku")


def test_the_partner_table_is_honoured_when_it_is_named(registry, root):
    csv_at(root / "sales.csv", "sku,qty\na,3\n")
    csv_at(root / "prices.csv", "sku,price\na,10\n")
    csv_at(root / "stock.csv", "sku,on_hand\na,7\nz,1\n")
    registry.attach(str(root / "sales.csv"), "shop")
    registry.add_table("shop", source=str(root / "prices.csv"))

    report = registry.add_table(
        "shop", source=str(root / "stock.csv"), join_on="sku", join_table="prices"
    ).join

    assert report is not None
    assert report.existing_table == "shop.prices"
    assert report.missing_from_existing == ("z",)


def test_a_key_missing_from_either_side_lists_the_columns_that_exist(registry, root):
    csv_at(root / "sales.csv", "sku,qty\na,3\n")
    csv_at(root / "prices.csv", "code,price\na,10\n")
    registry.attach(str(root / "sales.csv"), "shop")

    with pytest.raises(SlotError, match="code"):
        registry.add_table("shop", source=str(root / "prices.csv"), join_on="sku")


# ---------------------------------------------------------------------------
# Saving: the escape from ephemerality
# ---------------------------------------------------------------------------


def test_a_saved_database_can_be_attached_again_with_its_rows(registry, root):
    csv_at(root / "sales.csv")
    csv_at(root / "prices.csv", "sku,price\na,10\nb,20\n")
    registry.attach(str(root / "sales.csv"), "shop")
    registry.add_table("shop", source=str(root / "prices.csv"))

    saved = registry.save("shop", str(root / "keep.db"))
    registry.detach("shop")
    again = registry.attach(str(saved), "kept")

    assert sorted(again.slot.tables) == ["prices", "sales"]
    _, rows = registry.query(
        "kept",
        "SELECT s.sku, s.qty * p.price FROM kept.sales s "
        "JOIN kept.prices p ON s.sku = p.sku ORDER BY s.sku",
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
    with pytest.raises(Exception, match="readonly"):
        registry.query("kept", "DELETE FROM kept.sales")


def test_saving_keeps_the_slot_answering_and_writable(registry, root):
    """Copied out, not moved away: the session is not disturbed by keeping it."""
    csv_at(root / "sales.csv")
    registry.attach(str(root / "sales.csv"), "shop")

    registry.save("shop", str(root / "keep.db"))

    registry.query("shop", "INSERT INTO shop.sales VALUES ('c', 9)")
    _, rows = registry.query("shop", "SELECT count(*) FROM shop.sales")
    assert rows == [(3,)]


def test_saving_refuses_an_existing_file_and_leaves_it_untouched(registry, root):
    """A second save to the same name must not cost the user the first one.

    There is deliberately no overwrite parameter to reach for: the path came
    from the user via an agent, so replacing what is there is not the agent's
    call to make.
    """
    csv_at(root / "sales.csv")
    registry.attach(str(root / "sales.csv"), "shop")
    saved = registry.save("shop", str(root / "keep.db"))
    kept = saved.read_bytes()

    registry.query("shop", "INSERT INTO shop.sales VALUES ('c', 9)")

    with pytest.raises(SlotError, match="will not replace it"):
        registry.save("shop", str(root / "keep.db"))

    with pytest.raises(TypeError):
        registry.save("shop", str(root / "keep.db"), overwrite=True)

    assert saved.read_bytes() == kept


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
            "big", "SELECT count(*), sum(amount), min(label), max(label) FROM big.big"
        )[1]

        moved = registry.relieve_memory()

        assert [slot.nickname for slot in moved] == ["big"]
        after = registry.query(
            "big", "SELECT count(*), sum(amount), min(label), max(label) FROM big.big"
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
        registry.add_table("big", source=str(root / "labels.csv"))
        registry.query("big", "INSERT INTO big.labels VALUES (2, 'second')")

        _, rows = registry.query(
            "big",
            "SELECT l.note FROM big.big b JOIN big.labels l ON b.id = l.id "
            "ORDER BY l.note",
        )
        assert rows == [("first",), ("second",)]
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


def _backing_file(registry: Registry, schema: str) -> str:
    """The file SQLite has behind an attached schema; empty for in-memory."""
    rows = registry.workspace._conn.execute("PRAGMA database_list").fetchall()
    return next(row[2] for row in rows if row[1] == schema)


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
    registry.add_table("shop", source=str(root / "prices.csv"))
    registry.query(
        "shop",
        "CREATE VIEW shop.revenue AS SELECT s.sku, s.qty * p.price AS total "
        "FROM sales s JOIN prices p ON s.sku = p.sku",
    )

    saved = registry.save("shop", str(root / "keep.db"))
    registry.detach("shop")
    registry.attach(str(saved), "renamed")

    _, rows = registry.query("renamed", "SELECT total FROM renamed.revenue")
    assert rows == [(30,)]


def test_saving_a_database_that_could_not_be_opened_again_is_refused(registry, root):
    """A file that looks saved and cannot be attached is the worst outcome.

    The whole database is rejected, not just the offending view, so this is not
    a small blemish on an otherwise fine artifact.
    """
    csv_at(root / "sales.csv", "sku,qty\na,3\n")
    csv_at(root / "prices.csv", "sku,price\na,10\n")
    registry.attach(str(root / "sales.csv"), "shop")
    registry.add_table("shop", source=str(root / "prices.csv"))
    registry.query(
        "shop",
        "CREATE VIEW shop.revenue AS SELECT s.sku, s.qty * p.price AS total "
        "FROM shop.sales s JOIN shop.prices p ON s.sku = p.sku",
    )

    with pytest.raises(SlotError, match="unqualified"):
        registry.save("shop", str(root / "keep.db"))

    # Refused means nothing left behind that could be mistaken for a save.
    assert not (root / "keep.db").exists()
    # And the session is undisturbed.
    assert registry.query("shop", "SELECT total FROM shop.revenue")[1] == [(30,)]


def test_a_table_aliased_to_the_nickname_does_not_trip_the_check(registry, root):
    """The check attaches the file rather than scanning the view's text.

    Text-scanning for the nickname would flag this view, whose 'shop' is an
    alias and not a schema at all.
    """
    csv_at(root / "sales.csv", "sku,qty\na,3\n")
    registry.attach(str(root / "sales.csv"), "shop")
    registry.query("shop", "CREATE VIEW shop.totals AS SELECT shop.qty FROM sales shop")

    saved = registry.save("shop", str(root / "keep.db"))
    registry.detach("shop")
    registry.attach(str(saved), "renamed")

    assert registry.query("renamed", "SELECT qty FROM renamed.totals")[1] == [(3,)]


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

    # Under the name it was built with, the very same file is fine.
    registry.attach(str(poisoned), "shop")
    assert registry.query("shop", "SELECT qty FROM shop.v")[1] == [(3,)]
