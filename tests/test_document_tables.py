"""A JSON, YAML or XML document holding several tables lands all of them.

The rule a workbook already follows: a datasource is a database, so every table
the file holds lands in it, each under a name of its own. A workbook names its
tables by sheet; a document names them by **key path** — the keys from the
document's root down to the array, joined by ``.`` — and that path goes through
the same cleaning and de-duplication a sheet name does, so ``meta.orders``
lands as ``meta_orders``, and a second table cleaning up to a name already held
gets ``_1``.

What must not move: a document with one candidate table still loads as one
table named after the file, with the note it always had.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from fastmcp import Client

from localdata_mcp import config as config_module
from localdata_mcp import server as server_module
from localdata_mcp.config import Config
from localdata_mcp.loader import LoadError, Workspace, read_file
from localdata_mcp.slots import Registry, SlotError


@pytest.fixture(autouse=True)
def root(tmp_path):
    shared = tmp_path / "root"
    shared.mkdir()
    config_module.use(Config(roots=(shared,)))
    return shared


@pytest.fixture()
def workspace():
    ws = Workspace.in_memory()
    ws.attach_memory("main")
    yield ws
    ws.close()


@pytest.fixture()
def registry(root):
    reg = Registry()
    yield reg
    reg.close()


def write_json(path: Path, document: object) -> Path:
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def call(tool: str, **arguments):
    """Call a tool over the protocol and return its decoded payload."""

    async def _run():
        async with Client(server_module.mcp) as client:
            result = await client.call_tool(tool, arguments)
            if result.structured_content is not None:
                return result.structured_content
            return json.loads(result.content[0].text)

    return asyncio.run(_run())


CUSTOMERS = [{"name": "Acme", "city": "Bern"}, {"name": "Globex", "city": "Basel"}]
ORDERS = [{"id": 1, "total": 10.5}, {"id": 2, "total": 4.0}, {"id": 3, "total": 7.25}]


# ---------------------------------------------------------------------------
# Every candidate lands, named from its key path
# ---------------------------------------------------------------------------


def test_two_top_level_arrays_land_as_two_tables(workspace, root):
    source = write_json(root / "multi.json", {"customers": CUSTOMERS, "orders": ORDERS})

    landed = workspace.load_file(str(source), "main")

    assert [info.name for info in landed] == ["customers", "orders"]
    assert [info.row_count for info in landed] == [2, 3]
    _, rows = workspace.query("main", "SELECT sum(total) FROM orders")
    assert rows[0][0] == pytest.approx(21.75)
    _, rows = workspace.query("main", "SELECT name FROM customers ORDER BY name")
    assert rows == [("Acme",), ("Globex",)]


def test_the_note_says_which_table_came_from_which_key(workspace, root):
    source = write_json(
        root / "multi.json",
        {"version": 2, "customers": CUSTOMERS, "meta": {"orders": ORDERS}},
    )

    read = read_file(source)

    assert [table.name for table in read.tables] == ["customers", "meta.orders"]
    (note,) = read.notes
    assert "multi.json holds 2 tables" in note
    assert "'customers' as customers" in note
    assert "'meta.orders' as meta_orders" in note
    # What was left out is named too, as the one-table note always did.
    assert "version" in note
    # The pick-one wording is gone, from the refusal and from everything else.
    assert "which one you want" not in note


def test_a_nested_candidate_is_found_and_named_by_its_path(workspace, root):
    source = write_json(
        root / "nested.json",
        {"customers": CUSTOMERS, "meta": {"page": 1, "results": {"orders": ORDERS}}},
    )

    landed = workspace.load_file(str(source), "main")

    assert [info.name for info in landed] == ["customers", "meta_results_orders"]
    assert landed[1].row_count == 3


def test_one_nested_candidate_is_one_table_named_after_the_file(workspace, root):
    """One candidate is not a choice, at any depth: it loads as the file's table."""
    source = write_json(root / "dump.json", {"count": 3, "data": {"items": ORDERS}})

    (info,) = workspace.load_file(str(source), "main")

    assert info.name == "dump"
    assert info.row_count == 3
    (note,) = info.notes
    assert "the array under 'data.items' was the only table in it" in note
    assert "(count)" in note


def test_paths_that_clean_up_alike_each_get_a_table(registry, root):
    """``a_b`` and ``a.b`` are both ``a_b``; the later one in the file is ``a_b_1``."""
    source = write_json(root / "clash.json", {"a_b": CUSTOMERS, "a": {"b": ORDERS}})

    attached = registry.attach(str(source), "clash")

    assert attached.slot.tables == ("a_b", "a_b_1")
    assert attached.sheets == ("a_b", "a.b")
    (clash,) = attached.warnings
    assert "'a_b' as a_b" in clash and "'a.b' as a_b_1" in clash
    _, rows = registry.query("clash", "SELECT count(*) FROM a_b_1")
    assert rows[0][0] == 3


def test_the_mapping_in_the_note_follows_a_collision(root):
    """The note names the table each path really landed as, suffix included."""
    source = write_json(root / "clash.json", {"a_b": CUSTOMERS, "a": {"b": ORDERS}})

    (note,) = read_file(source).notes

    assert "'a_b' as a_b" in note and "'a.b' as a_b_1" in note


def test_an_array_inside_a_row_is_a_column_not_a_table(workspace, root):
    """Only objects are walked. A list of objects inside a row is that row's value."""
    source = write_json(
        root / "groups.json",
        {
            "groups": [{"name": "g1", "members": [{"who": "Ada"}]}],
            "other": [{"x": 1}],
        },
    )

    landed = workspace.load_file(str(source), "main")

    assert [info.name for info in landed] == ["groups", "other"]
    _, rows = workspace.query("main", "SELECT members FROM groups")
    assert rows[0][0] == '[{"who": "Ada"}]'


def test_a_nested_value_note_names_the_table_it_is_about(workspace, root):
    source = write_json(
        root / "multi.json",
        {"customers": [{"name": "Acme", "tags": ["a", "b"]}], "orders": ORDERS},
    )

    read = read_file(source)

    nested = [note for note in read.notes if "Nested values" in note]
    assert nested == [
        "Nested values in tags of 'customers' were stored as JSON text, because "
        "SQL has no nested type. Read into them with json_extract(column, "
        "'$.key'); the text is exactly what was in the file, so nothing was lost."
    ]


def test_empty_arrays_are_still_not_candidates(workspace, root):
    source = write_json(
        root / "api.json", {"data": ORDERS, "errors": [], "meta": {"warnings": []}}
    )

    (info,) = workspace.load_file(str(source), "main")

    assert info.name == "api"
    assert info.row_count == 3


def test_a_document_with_no_table_anywhere_is_still_refused(workspace, root):
    source = write_json(root / "record.json", {"name": "Ada", "meta": {"age": 36}})

    with pytest.raises(LoadError) as raised:
        workspace.load_file(str(source), "main")

    assert "holds no table" in str(raised.value)
    assert "array of objects" in str(raised.value)


def test_a_deeply_nested_candidate_does_not_exhaust_the_stack(workspace, root):
    """The walk is iterative: depth the parser accepted is depth the walk survives."""
    document: object = {"rows": ORDERS}
    for _ in range(500):
        document = {"level": document}
    source = write_json(root / "deep.json", document)

    (info,) = workspace.load_file(str(source), "main")

    assert info.row_count == 3


def test_yaml_lands_every_candidate_too(workspace, root):
    source = root / "multi.yaml"
    source.write_text(
        "customers:\n  - name: Acme\n  - name: Globex\n"
        "meta:\n  orders:\n    - id: 1\n    - id: 2\n    - id: 3\n",
        encoding="utf-8",
    )

    landed = workspace.load_file(str(source), "main")

    assert [info.name for info in landed] == ["customers", "meta_orders"]
    assert [info.row_count for info in landed] == [2, 3]


# ---------------------------------------------------------------------------
# One candidate: exactly as it was
# ---------------------------------------------------------------------------


def test_one_top_level_candidate_keeps_its_name_and_its_note_verbatim(root):
    source = write_json(
        root / "wrapped.json", {"count": 3, "orders": ORDERS, "page": {"n": 1}}
    )

    read = read_file(source)

    (table,) = read.tables
    assert table.name is None
    assert read.notes == (
        "wrapped.json is an object rather than an array, and the array under "
        "'orders' was the only table in it — that is what was loaded. The other "
        "keys (count, page) are not part of this table.",
    )


def test_a_top_level_array_still_has_nothing_to_say(root):
    source = write_json(root / "plain.json", ORDERS)

    read = read_file(source)

    assert read.notes == ()
    assert read.tables[0].name is None


# ---------------------------------------------------------------------------
# XML: every repeated element is a table
# ---------------------------------------------------------------------------


def test_two_repeated_xml_elements_land_as_two_tables(workspace, root):
    source = root / "kinds.xml"
    source.write_text(
        "<data><generated>today</generated>"
        "<employee><name>Ada</name></employee><employee><name>Grace</name></employee>"
        "<department><floor>3</floor></department>"
        "<department><floor>1</floor></department><department><floor>2</floor>"
        "</department></data>",
        encoding="utf-8",
    )

    landed = workspace.load_file(str(source), "main")

    assert [info.name for info in landed] == ["employee", "department"]
    assert [info.row_count for info in landed] == [2, 3]
    _, rows = workspace.query("main", "SELECT sum(floor) FROM department")
    assert rows[0][0] == 6
    note = next(n for n in landed[0].notes if "holds 2 tables" in n)
    assert "<employee> as employee" in note and "<department> as department" in note
    assert "<generated>" in note


def test_xml_nested_note_names_the_table(root):
    source = root / "kinds.xml"
    source.write_text(
        "<data><a><x><y>1</y></x></a><a><x><y>2</y></x></a>"
        "<b><z>1</z></b><b><z>2</z></b></data>",
        encoding="utf-8",
    )

    notes = read_file(source).notes

    assert any("Nested elements in x of <a>" in note for note in notes)


# ---------------------------------------------------------------------------
# Through the verbs
# ---------------------------------------------------------------------------


def test_create_still_refuses_a_document_of_two_tables_naming_them(registry, root):
    registry.attach(str(write_json(root / "base.json", ORDERS)), "shop")
    source = write_json(
        root / "multi.json", {"customers": CUSTOMERS, "meta": {"orders": ORDERS}}
    )

    with pytest.raises(SlotError) as raised:
        registry.create_table("shop", source=str(source))

    message = str(raised.value)
    assert "customers, meta.orders" in message
    assert "arriving as customers, meta_orders" in message


def test_attach_over_the_wire_shows_each_key_path_beside_its_table(root):
    server_module._reset()
    try:
        source = write_json(
            root / "multi.json", {"customers": CUSTOMERS, "meta": {"orders": ORDERS}}
        )

        attached = call("attach", database=str(source))

        assert attached["ok"] is True
        assert [(e["sheet"], e["table"], e["rows"]) for e in attached["loaded"]] == [
            ("customers", "customers", 2),
            ("meta.orders", "meta_orders", 3),
        ]
        mapping = [w for w in attached["warnings"] if "holds 2 tables" in w]
        # Said once, not once per table that came out of the file.
        assert len(mapping) == 1
    finally:
        server_module._reset()


def test_the_memory_budget_still_measures_a_document_of_several_tables(root):
    """The datasource is one database; outgrowing the budget moves all of it."""
    config_module.use(Config(roots=(root,), memory_budget_mb=1))
    registry = Registry()
    try:
        big = [{"id": i, "label": f"x{i}", "amount": i * 3} for i in range(30_000)]
        source = write_json(root / "big.json", {"alpha": big, "beta": {"rows": big}})
        registry.attach(str(source), "big")
        counts = [
            registry.query("big", f"SELECT count(*), sum(amount) FROM {t}")[1]
            for t in ("alpha", "beta_rows")
        ]

        moved = registry.relieve_memory()

        assert [slot.nickname for slot in moved] == ["big"]
        assert [
            registry.query("big", f"SELECT count(*), sum(amount) FROM {t}")[1]
            for t in ("alpha", "beta_rows")
        ] == counts
        assert counts[0][0][0] == 30_000
    finally:
        registry.close()
