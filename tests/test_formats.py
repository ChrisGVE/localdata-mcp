"""The one format table, and the claims it is allowed to make.

``FORMATS`` replaced four tables that could disagree with each other. Most of
what is worth testing here is therefore not "does the table have the right
entries" — a hand-written expectation of that is a copy of the table sitting in
a test file, which catches nothing — but that the *views* cannot drift from it
and that an entry which contradicts itself is refused at construction.
"""

from __future__ import annotations

import pytest

from localdata_mcp import export as export_module
from localdata_mcp import loader as loader_module
from localdata_mcp import formats as formats_module
from localdata_mcp.formats import (
    DELIMITED,
    FORMATS,
    READERS,
    STREAMED,
    WRITERS,
    Format,
)


def test_every_format_can_be_done_at_least_one_way():
    """An entry that neither reads nor writes is knowledge about nothing."""
    for suffix, fmt in FORMATS.items():
        assert fmt.reader is not None or fmt.writer is not None, suffix


def test_a_format_that_can_be_done_neither_way_is_refused():
    """The positive control for the test above.

    Without this, that assertion passes on a table where the constructor had
    quietly stopped enforcing anything.
    """
    with pytest.raises(ValueError, match="neither a reader nor a writer"):
        Format(suffix=".nothing")


def test_a_delimiter_cannot_be_declared_for_a_format_nothing_reads():
    """`delimited` is the round-trip property, so it needs both halves.

    A write-only format cannot be read back at the separator it was written
    with, which is the whole content of the claim.
    """
    with pytest.raises(ValueError, match="missing a reader"):
        Format(suffix=".oneway", writer=lambda columns, rows, path: 0, delimited=True)


def test_a_delimiter_cannot_be_declared_for_a_format_nothing_writes():
    """The other half, which only became reachable once the separator was declared.

    While the suffix supplied the separator it supplied it to both sides at
    once, so this could not be got wrong. Now that each side resolves its own —
    declared, else the format's standard — a format declaring `delimited` with
    no writer is a format whose files this server could read at a separator it
    has no way of ever having written at.
    """
    with pytest.raises(ValueError, match="missing a writer"):
        Format(suffix=".oneway", reader=lambda path: None, delimited=True)


def test_streaming_cannot_be_declared_for_a_format_nothing_reads():
    with pytest.raises(ValueError, match="streamed"):
        Format(suffix=".oneway", writer=lambda columns, rows, path: 0, streamed=True)


def test_the_key_and_the_suffix_field_are_the_same_string():
    """Said twice per entry, so it is worth pinning that they agree."""
    for suffix, fmt in FORMATS.items():
        assert fmt.suffix == suffix


def test_the_views_name_nothing_the_table_does_not():
    """A view that can gain a member of its own is a second table again."""
    for name, view in [
        ("READERS", READERS),
        ("WRITERS", WRITERS),
        ("DELIMITED", DELIMITED),
        ("STREAMED", STREAMED),
    ]:
        assert set(view) <= set(FORMATS), f"{name} has a suffix FORMATS does not"


def test_the_views_hold_exactly_the_formats_that_declare_the_capability():
    assert set(READERS) == {s for s, f in FORMATS.items() if f.reader is not None}
    assert set(WRITERS) == {s for s, f in FORMATS.items() if f.writer is not None}
    assert set(DELIMITED) == {s for s, f in FORMATS.items() if f.delimited}
    assert set(STREAMED) == {s for s, f in FORMATS.items() if f.streamed}


def test_both_sides_are_looking_at_the_one_table():
    """The point of the merge, stated as identity rather than equality.

    Equality would pass just as happily against copies, which is the state
    `localdata#98` was filed about.
    """
    assert loader_module.READERS is READERS
    assert export_module.WRITERS is WRITERS
    assert loader_module.DELIMITED is DELIMITED
    assert export_module.DELIMITED is DELIMITED


def test_a_delimited_format_can_be_read_back_at_the_separator_it_was_written_with():
    """`delimited` governs both directions, so it may not be write-only or
    read-only in practice — every entry claiming it must have both callables."""
    for suffix in DELIMITED:
        fmt = FORMATS[suffix]
        assert fmt.reader is not None, suffix
        assert fmt.writer is not None, suffix


def _engine_bound_into(fn):
    """The (module, extra) a workbook reader or writer closed over.

    Read off the closure by free-variable name rather than by cell order, which
    Python does not promise. Returns None for anything that is not one of the
    two workbook factories.
    """
    if fn is None or not fn.__closure__:
        return None
    cells = dict(zip(fn.__code__.co_freevars, fn.__closure__))
    if not {"module", "extra"} <= cells.keys():
        return None
    return tuple(cells[name].cell_contents for name in ("module", "extra"))


def test_a_workbook_is_read_and_written_by_the_same_library():
    """The whole content of #99, asserted rather than commented.

    Reader and writer are built from one declaration in the table, so this can
    only fail if someone spells the engine twice again. It matters because the
    failure it guards is silent: both formats are Zip archives, and pandas reads
    a workbook back by sniffing its contents rather than trusting the suffix, so
    a `.ods` written by openpyxl round-trips the right rows out of the wrong
    file.
    """
    checked = 0
    for suffix, fmt in FORMATS.items():
        reading = _engine_bound_into(fmt.reader)
        writing = _engine_bound_into(fmt.writer)
        if reading is None or writing is None:
            continue
        assert reading == writing, (
            f"{suffix} reads with {reading}, writes with {writing}"
        )
        checked += 1
    assert checked, "no workbook format carried an engine — the probe stopped working"


def test_every_streamed_format_is_one_whose_rows_arrive_in_order():
    """Columnar files and workbooks cannot be handed out a chunk at a time.

    Pinned because `streamed` is the one flag whose cost of being wrong is
    silent: a format wrongly declared streamable would be chunked by a reader
    that has to materialise it anyway.
    """
    assert not (STREAMED & {".parquet", ".feather", ".orc"})
    assert not (STREAMED & {".xlsx", ".xlsm", ".xls", ".ods", ".numbers"})


# ---------------------------------------------------------------------------
# Character-separated text is one format under three names
# ---------------------------------------------------------------------------


def test_the_three_delimited_suffixes_are_one_declaration():
    """`.csv`, `.tsv` and `.txt` share one reader, one writer and one set of flags.

    Identity rather than equality, and for the reason identity is used
    everywhere else in this module: three entries that happen to agree today is
    exactly the arrangement that let them disagree tomorrow. The one thing that
    differs between them is the separator a published standard gives each — a
    comma for `.csv`, a tab for `.tsv`, none for `.txt` — and that is a field of
    its own, tested in test_standard_delimiter.py.
    """
    entries = [FORMATS[suffix] for suffix in (".csv", ".tsv", ".txt")]
    first = entries[0]
    for other in entries[1:]:
        assert other.reader is first.reader
        assert other.writer is first.writer
        assert other.delimited is first.delimited is True
        assert other.streamed is first.streamed is True


def test_every_delimited_suffix_resolves_its_separator_in_one_place():
    """No suffix carries a reader or writer bound to a separator of its own.

    The table's reader and writer for delimited text both go through
    `formats.resolve_separator`, which reads the standard off the entry. A
    reader bound to `','` for one suffix would be a second place a default
    lives. Derived from `FORMATS` rather than from a list written here, so a
    fourth delimited suffix added later is covered without anyone remembering to.
    """
    for suffix in DELIMITED:
        reader = FORMATS[suffix].reader
        writer = FORMATS[suffix].writer
        assert reader is formats_module.read_delimited_text, suffix
        assert writer is formats_module.write_delimited_text, suffix


@pytest.mark.parametrize("suffix", [".txt"])
def test_reading_delimited_text_without_a_separator_is_refused(tmp_path, suffix):
    """Where no standard separator exists, and the refusal names the parameter.

    `.csv` and `.tsv` read at their standard when nothing is declared, so the
    refusal is left only for `.txt`, which has none. Named rather than derived
    from the table: derived, a `.txt` that gained a default would empty the
    list and the test would skip instead of failing.
    """
    path = tmp_path / f"data{suffix}"
    path.write_text("a;b\n1;2\n")

    with pytest.raises(loader_module.LoadError, match="no published standard"):
        loader_module.read_file(path)
    with pytest.raises(loader_module.LoadError, match="delimiter"):
        loader_module.read_source(path)


@pytest.mark.parametrize("suffix", sorted(DELIMITED))
def test_the_declared_separator_is_the_one_used_whatever_the_suffix(tmp_path, suffix):
    """The positive control: the refusals above are about the absence, not the file.

    Without this, every assertion above would pass just as happily against a
    reader that had stopped working altogether.
    """
    path = tmp_path / f"data{suffix}"
    path.write_text("a;b\n1;2\n")

    read = loader_module.read_file(path, delimiter=";")
    assert list(read.tables[0].frame.columns) == ["a", "b"]

    measured = loader_module.read_source(path, delimiter=";")
    assert [c.name for c in measured.tables[0].columns] == ["a", "b"]
