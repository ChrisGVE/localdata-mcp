"""A file read in chunks must land exactly as the same file read whole.

The load path reads a delimited file twice — once to measure it, once to insert
it — instead of building the whole frame first. That is only worth having if
nothing about the result depends on where the chunk boundaries fell, and this is
where that is either true or found out.

**Every test here loads the same file both ways and compares**: the declared
types, the column descriptions, the notes, the row count and every value. The
failure this guards against is not a crash. It is a streamed load that quietly
differs from a materialised one — a column declared from the first chunk that
the fifth does not fit, a date column spelled two ways, a text column sized to
the widest value the first chunk happened to hold — and every one of those
produces a table that looks perfectly ordinary and is wrong.

**The chunk size is parametrised down to one row.** With the shipped 50,000 the
whole fixture corpus fits in a single chunk, so the accumulation across chunks —
which is the entire mechanism — would never run and every test here would pass
against a measurement that simply used the first chunk. Sizes of 1, 2 and 3 put
a boundary between every pair of adjacent rows in the small fixtures; the large
ones cross many.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from localdata_mcp import config as config_module
from localdata_mcp import loader as loader_module
from localdata_mcp.config import Config
from localdata_mcp.loader import LoadError, TableInfo, Workspace, read_file, read_source

ASSETS = Path(__file__).parent / "assets"

#: Boundaries that fall between every pair of rows in a small fixture, and many
#: times over in a large one. One is the extreme: every row is its own chunk, so
#: every measurement is a fold over as many parts as the file has rows.
CHUNK_SIZES = [1, 2, 3, 7, 5_000]

#: The delimited fixtures — the formats that stream. `empty.csv` is left out
#: because it has no table to compare; it gets its own test below.
STREAMED_ASSETS = [
    "simple.csv",
    "messy_mixed_types.csv",
    "no_header.csv",
    "truncated.csv",
    "large_dataset.csv",
    "mixed_tabs.tsv",
    "pipes.txt",
]

#: Files neither path can read. They belong in this module all the same: a
#: refusal that differs between the two paths is a second error message to keep
#: in step, and `latin1_encoded.csv` is refused for the encoding before either
#: reader sees a column.
UNREADABLE = ["empty.csv", "latin1_encoded.csv"]

#: Files built for the chunk boundary rather than harvested. Each one puts the
#: value that decides a whole-column measurement *late*, so a measurement taken
#: from the first chunk gets it wrong. They are the cases the corpus cannot
#: cover, because a harvested file has no reason to be adversarial about where
#: its awkward row sits.
LATE_DECIDERS = {
    # Integers until the last row, which is not one. Measured per chunk, this
    # column is declared INTEGER and then meets `3a`.
    "junk_last.csv": "id,v\n1,1\n2,2\n3,3\n4,3a\n",
    # The same, with the junk first instead, so the widening is exercised in
    # both directions.
    "junk_first.csv": "id,v\n1,3a\n2,1\n3,2\n4,3\n",
    # Leading zeroes. Inferred as integers they measure one character wide;
    # the column is text because of the last row, and the values are three.
    "zero_padded.csv": "id,v\n1,007\n2,008\n3,009\n4,x\n",
    # A date column that is all midnight until it is not: written per chunk it
    # comes out `2024-03-01` in one and `2024-03-04T09:30:00Z` in the next.
    "midnight_then_time.csv": (
        "id,d\n1,2024-03-01\n2,2024-03-02\n3,2024-03-03\n4,2024-03-04T09:30:00\n"
    ),
    # And the other way round, so the merge is exercised from both sides.
    "time_then_midnight.csv": (
        "id,d\n1,2024-03-04T09:30:00\n2,2024-03-01\n3,2024-03-02\n4,2024-03-03\n"
    ),
    # Fractional seconds arriving late, which widens the spelling for every
    # value in the column including the ones already written — and arriving
    # first, which is the order that catches a fold keeping only the last chunk.
    "fraction_last.csv": (
        "id,d\n1,2024-03-01T09:30:00\n2,2024-03-02T09:30:00\n"
        "3,2024-03-03T09:30:00\n4,2024-03-04T09:30:00.500000\n"
    ),
    "fraction_first.csv": (
        "id,d\n1,2024-03-01T09:30:00.500000\n2,2024-03-02T09:30:00\n"
        "3,2024-03-03T09:30:00\n4,2024-03-04T09:30:00\n"
    ),
    # Canonical throughout, in two spellings — which is not one spelling, so
    # `standardize` settles it on the merged one and the streamed read must
    # settle it the same way (#75). The deciding value sits in the middle here
    # and last in the twin below, so the width union is exercised with it in a
    # chunk of its own and in the final chunk.
    "canonical_mixed.csv": (
        "id,d\n1,2024-03-01\n2,2024-03-02T10:00:00Z\n3,2024-03-03\n"
    ),
    "canonical_mixed_last.csv": (
        "id,d\n1,2024-03-01\n2,2024-03-02\n3,2024-03-03T10:00:00Z\n"
    ),
    # A gap in an integer column, in the last row only. The gap is what makes
    # the column REAL rather than INTEGER, and it arrives last.
    "gap_last.csv": "id,v\n1,1\n2,2\n3,3\n4,\n",
    # A gap in a boolean column, which stops it being boolean at all.
    "bool_gap_last.csv": "id,v\ntrue\nfalse\ntrue\n\n".replace("id,v\n", "v\n"),
    # Booleans throughout, so the family that has no missing marker is
    # exercised without a gap as well as with one.
    "bool_clean.csv": "v\ntrue\nfalse\ntrue\nfalse\n",
    # The widest text value in the last row, which is what sizes the column.
    "widest_last.csv": "id,v\n1,a\n2,bb\n3,ccc\n4,dddddddddddddddddddd\n",
    # Every value missing, which is a column with no evidence in it at all.
    "all_null.csv": "id,v\n1,\n2,\n3,\n",
    # Non-numeric values spread across chunks, so the examples carried into the
    # mixed-column report are accumulated rather than taken from one chunk.
    "junk_spread.csv": "id,v\n1,1\n2,na\n3,3\n4,none\n5,5\n6,missing\n7,7\n",
    # Ambiguous dates, which must stay text and be reported — the report's
    # examples being another thing accumulated across chunks.
    "ambiguous_dates.csv": (
        "id,d\n1,30.11.2023\n2,05.01.2024\n3,04.07.2024\n4,01.03.2025\n"
    ),
    # One row. There is nothing to fold, and the fold must still be right.
    "single_row.csv": "id,v\n1,alpha\n",
    # A header and nothing else. The measurements come from no values at all.
    "header_only.csv": "id,v\n",
    # Duplicate and awkward headers, which are made unique from the header
    # alone and so must survive being read from the first chunk.
    "dupe_headers.csv": "a,a,b c\n1,2,3\n4,5,6\n",
}

#: Built the same way, but for a named test rather than for the sweep — this
#: one is refused by both paths, so there is no landing to compare.
BUILT_ELSEWHERE = {
    "wide_int_last.csv": "id,v\n1,1\n2,2\n3,99999999999999999999\n",
}


@pytest.fixture(autouse=True)
def root(tmp_path):
    """The asset corpus and the built fixtures, in one readable directory."""
    shared = tmp_path / "root"
    shared.mkdir()
    for asset in ASSETS.iterdir():
        (shared / asset.name).write_bytes(asset.read_bytes())
    for name, body in {**LATE_DECIDERS, **BUILT_ELSEWHERE}.items():
        (shared / name).write_text(body)
    config_module.use(Config(roots=(shared,)))
    return shared


@pytest.fixture()
def workspace():
    """One workspace holding both loads, so the comparison is like for like."""
    space = Workspace.in_memory()
    space.attach_memory("whole")
    space.attach_memory("parts")
    yield space
    space.close()


@pytest.fixture()
def chunked(request, monkeypatch):
    """Read the file in chunks of the parametrised size."""
    monkeypatch.setattr(loader_module, "_READ_CHUNK", request.param)
    return request.param


def separator_of(path: Path) -> dict[str, str]:
    """The declared separator for this fixture, as keyword arguments.

    The server takes no separator from a suffix any more — it has to be declared
    on every read of character-separated text — but this corpus was written to
    the usual convention, so the tests say which character that is here, once,
    instead of at every call. The mapping deliberately reproduces what the
    suffixes used to imply, so each test still reads its fixture the way it was
    written and goes on meaning what it meant.

    Empty for a format that has no separator, because passing one there is
    refused rather than ignored — which is itself a test in this module.
    """
    suffix = path.suffix.lower()
    if suffix not in loader_module.DELIMITED:
        return {}
    return {"delimiter": "\t" if suffix == ".tsv" else ","}


def load_both(space: Workspace, path: Path) -> tuple[TableInfo, TableInfo]:
    """Load one file materialised and streamed, into two tags of one workspace."""
    read = read_file(path, **separator_of(path))
    assert len(read.tables) == 1, "a delimited file holds one table"
    materialised = space.insert_frame(
        read.tables[0].frame, "t", source=str(path), tag="whole", notes=read.notes
    )
    measured = read_source(path, **separator_of(path))
    assert len(measured.tables) == 1
    streamed = space.insert_source(
        measured.tables[0], "t", source=str(path), tag="parts", notes=measured.notes
    )
    return materialised, streamed


def rows(space: Workspace, tag: str) -> list[tuple]:
    """Every value of the table, in the order the file put them in."""
    _, found = space.query(tag, "SELECT * FROM t ORDER BY rowid")
    return found


def assert_identical(space: Workspace, path: Path) -> TableInfo:
    """The whole comparison, and the reason this module exists."""
    materialised, streamed = load_both(space, path)

    assert streamed.notes == materialised.notes
    assert streamed.row_count == materialised.row_count
    # `ColumnInfo` compares by value, so this covers the declared type, the
    # storage classes the values actually landed under, the numeric split and
    # its examples, the temporal standard and the unparsed-date examples — every
    # measurement, in one assertion, rather than a list of them that a new field
    # could quietly fall off.
    assert streamed.columns == materialised.columns
    assert rows(space, "parts") == rows(space, "whole")
    return streamed


# ---------------------------------------------------------------------------
# The corpus, both ways, at every chunk size
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("chunked", CHUNK_SIZES, indirect=True)
@pytest.mark.parametrize("name", STREAMED_ASSETS)
def test_a_harvested_file_lands_the_same_read_either_way(
    workspace, root, chunked, name
):
    assert_identical(workspace, root / name)


@pytest.mark.parametrize("chunked", CHUNK_SIZES, indirect=True)
@pytest.mark.parametrize("name", sorted(LATE_DECIDERS))
def test_a_file_whose_last_row_decides_it_lands_the_same_read_either_way(
    workspace, root, chunked, name
):
    """The cases where a measurement taken from the first chunk is wrong."""
    assert_identical(workspace, root / name)


# ---------------------------------------------------------------------------
# What the comparison would let through if it were the only test
# ---------------------------------------------------------------------------
#
# An equivalence sweep says the two paths agree; it does not say what they
# agree on. These name the answers, so that a change making both paths wrong in
# the same way is still caught.


@pytest.mark.parametrize("chunked", [1, 2, 5_000], indirect=True)
def test_a_column_whose_last_row_is_junk_is_text_throughout(workspace, root, chunked):
    assert_identical(workspace, root / "junk_last.csv")
    _, found = workspace.query("parts", "SELECT v FROM t ORDER BY rowid")
    assert [row[0] for row in found] == ["1", "2", "3", "3a"]


@pytest.mark.parametrize("chunked", [1, 2, 5_000], indirect=True)
def test_leading_zeroes_survive_a_column_that_turns_out_to_be_text(
    workspace, root, chunked
):
    """Inferred per chunk these are integers, and `007` comes back as `7`."""
    described = assert_identical(workspace, root / "zero_padded.csv")
    _, found = workspace.query("parts", "SELECT v FROM t ORDER BY rowid")
    assert [row[0] for row in found] == ["007", "008", "009", "x"]
    # And the column is sized for what it holds, not for what an inferred
    # integer would have measured — the number Oracle's VARCHAR2 is built from.
    assert next(c for c in described.columns if c.name == "v").declared_type == "TEXT"


#: The same column in both orders, because a fold that simply keeps the *last*
#: chunk's answer is right whenever the deciding value happens to come last.
#: That is not a hypothetical: the reversion drill ran this test against a scan
#: that overwrote the spelling instead of merging it, and the deciding-row-last
#: fixture passed. Only the reversed pair fails it.
SPELLINGS = [
    (
        "midnight_then_time.csv",
        [
            "2024-03-01T00:00:00Z",
            "2024-03-02T00:00:00Z",
            "2024-03-03T00:00:00Z",
            "2024-03-04T09:30:00Z",
        ],
    ),
    (
        "time_then_midnight.csv",
        [
            "2024-03-04T09:30:00Z",
            "2024-03-01T00:00:00Z",
            "2024-03-02T00:00:00Z",
            "2024-03-03T00:00:00Z",
        ],
    ),
    (
        "fraction_last.csv",
        [
            "2024-03-01T09:30:00.000000Z",
            "2024-03-02T09:30:00.000000Z",
            "2024-03-03T09:30:00.000000Z",
            "2024-03-04T09:30:00.500000Z",
        ],
    ),
    (
        "fraction_first.csv",
        [
            "2024-03-01T09:30:00.500000Z",
            "2024-03-02T09:30:00.000000Z",
            "2024-03-03T09:30:00.000000Z",
            "2024-03-04T09:30:00.000000Z",
        ],
    ),
]


@pytest.mark.parametrize("chunked", [1, 2, 5_000], indirect=True)
@pytest.mark.parametrize("name,expected", SPELLINGS, ids=[n for n, _ in SPELLINGS])
def test_a_date_column_is_written_in_one_spelling_throughout(
    workspace, root, chunked, name, expected
):
    """Decided per chunk, this column is date-only in one and timed in the next."""
    assert_identical(workspace, root / name)
    _, found = workspace.query("parts", "SELECT d FROM t ORDER BY rowid")
    assert [row[0] for row in found] == expected


CANONICAL_MIXED = [
    (
        "canonical_mixed.csv",
        ["2024-03-01T00:00:00Z", "2024-03-02T10:00:00Z", "2024-03-03T00:00:00Z"],
    ),
    (
        "canonical_mixed_last.csv",
        ["2024-03-01T00:00:00Z", "2024-03-02T00:00:00Z", "2024-03-03T10:00:00Z"],
    ),
]


@pytest.mark.parametrize("chunked", [1, 2, 5_000], indirect=True)
@pytest.mark.parametrize(
    "name,expected", CANONICAL_MIXED, ids=[n for n, _ in CANONICAL_MIXED]
)
def test_a_column_canonical_in_two_spellings_is_settled_on_one(
    workspace, root, chunked, name, expected
):
    """Canonical at every value is not canonical in one spelling (#75).

    This column passed `is_canonical` and was left in both spellings, where
    ``.`` sorting below ``Z`` put a later instant first. `standardize` now
    rewrites it, and the chunked read must reach the same place — which it can
    only do by unioning the widths it saw, since each chunk here is uniform on
    its own and the column is not.
    """
    assert_identical(workspace, root / name)
    _, found = workspace.query("parts", "SELECT d FROM t ORDER BY rowid")
    assert [row[0] for row in found] == expected


@pytest.mark.parametrize("chunked", [1, 2, 5_000], indirect=True)
def test_junk_spread_across_chunks_is_counted_and_exampled_once(
    workspace, root, chunked
):
    """The mixed-column report is a fold, and its examples come from everywhere."""
    described = assert_identical(workspace, root / "junk_spread.csv")
    column = next(c for c in described.columns if c.name == "v")
    assert column.is_mixed
    assert column.numeric_values == 4
    assert column.non_numeric_values == 3
    assert set(column.non_numeric_examples) == {"na", "none", "missing"}


@pytest.mark.parametrize("chunked", [1, 2, 5_000], indirect=True)
def test_a_gap_arriving_last_widens_the_column_that_held_integers(
    workspace, root, chunked
):
    """There is no gap in an integer dtype, so the column becomes the one there is."""
    described = assert_identical(workspace, root / "gap_last.csv")
    assert next(c for c in described.columns if c.name == "v").declared_type == "REAL"


# ---------------------------------------------------------------------------
# The ordinal a chunked read has to get right
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("chunked", [1, 2, 3, 5_000], indirect=True)
def test_the_surrogate_key_counts_rows_of_the_file_not_of_the_chunk(
    workspace, root, chunked, monkeypatch
):
    """Where a backend demands a key, its value is the row's place in the file.

    Only YDB demands one, so the fact is borrowed here rather than the dialect:
    what is under test is this module's *response* to a backend stating it, and
    that response has to be the same for any backend that ever does. Counted
    from a chunk's own offset the ordinals would restart at every boundary, and
    with `_READ_CHUNK` at one every row would be row zero.
    """
    entry = workspace.entry("parts")
    monkeypatch.setattr(type(entry.backend), "requires_primary_key", lambda self: True)

    measured = read_source(root / "junk_spread.csv", delimiter=",")
    landed = workspace.insert_source(
        measured.tables[0], "t", source="test", tag="parts", notes=measured.notes
    )

    assert landed.columns[0].name == "_row"
    _, found = workspace.query("parts", "SELECT _row FROM t ORDER BY _row")
    assert [row[0] for row in found] == [0, 1, 2, 3, 4, 5, 6]


# ---------------------------------------------------------------------------
# Refusals, which must also be the same either way
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("chunked", [1, 5_000], indirect=True)
@pytest.mark.parametrize("name", UNREADABLE)
def test_an_unreadable_file_is_refused_in_the_same_words_either_way(
    root, chunked, name
):
    """A refusal that differs by path is a second error message to maintain."""
    path = root / name
    with pytest.raises(LoadError) as whole:
        read_file(path, **separator_of(path))
    with pytest.raises(LoadError) as parts:
        read_source(path, **separator_of(path))
    assert str(parts.value) == str(whole.value)


@pytest.mark.parametrize("chunked", [1, 2, 5_000], indirect=True)
def test_an_integer_too_wide_to_store_is_refused_by_both_paths(
    workspace, root, chunked
):
    """Reproduced rather than improved on — see issue #72.

    `read_csv` reads a column of integers too wide for int64 as Python ints in
    an object column, and binding refuses the value for being unrepresentable.
    The column is *declared* text and text would hold the digits losslessly, so
    the refusal is arguably wrong — but it is the behaviour every other format
    has, and a file that one read path refuses and the other accepts is worse
    than either answer. Whichever way #72 is settled, it is settled for both.
    """
    path = root / "wide_int_last.csv"
    read = read_file(path, **separator_of(path))
    with pytest.raises(LoadError) as whole:
        workspace.insert_frame(read.tables[0].frame, "t", source=str(path), tag="whole")
    measured = read_source(path, **separator_of(path))
    with pytest.raises(LoadError) as parts:
        workspace.insert_source(measured.tables[0], "t", source=str(path), tag="parts")
    assert str(parts.value) == str(whole.value)


@pytest.mark.parametrize("chunked", [1, 5_000], indirect=True)
def test_a_delimiter_is_refused_for_a_format_that_has_none_either_way(root, chunked):
    path = root / "records.json"
    with pytest.raises(LoadError) as whole:
        read_file(path, delimiter=";")
    with pytest.raises(LoadError) as parts:
        read_source(path, delimiter=";")
    assert str(parts.value) == str(whole.value)


@pytest.mark.parametrize("chunked", [1, 5_000], indirect=True)
def test_an_explicit_delimiter_reaches_the_chunked_reader(workspace, root, chunked):
    """The separator is not the extension's, and both passes must use the given one."""
    read = read_file(root / "semicolons.csv", delimiter=";")
    materialised = workspace.insert_frame(
        read.tables[0].frame, "t", source="test", tag="whole", notes=read.notes
    )
    measured = read_source(root / "semicolons.csv", delimiter=";")
    streamed = workspace.insert_source(
        measured.tables[0], "t", source="test", tag="parts", notes=measured.notes
    )

    assert [c.name for c in streamed.columns] == ["name", "role", "salary"]
    assert streamed.columns == materialised.columns
    assert rows(workspace, "parts") == rows(workspace, "whole")


# ---------------------------------------------------------------------------
# The formats that do not stream
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("chunked", [1], indirect=True)
@pytest.mark.parametrize(
    "name", ["records.json", "records.jsonl", "workbook.xlsx", "employees.xml"]
)
def test_a_format_that_is_parsed_whole_still_reaches_the_same_insert(
    workspace, root, chunked, name
):
    """Measured from the frame instead of from chunks, and inserted identically.

    The point of the shared path: a format nobody can stream is one chunk rather
    than a second way of loading, so nothing downstream knows which it got.
    """
    path = root / name
    read = read_file(path, **separator_of(path))
    measured = read_source(path, **separator_of(path))

    assert len(measured.tables) == len(read.tables)
    assert measured.notes == read.notes
    for table, expected in zip(measured.tables, read.tables):
        assert table.name == expected.name
        assert [c.name for c in table.columns] == list(
            loader_module._unique_columns(list(expected.frame.columns))
        )
