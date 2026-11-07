"""The wrong-separator warning also fires when the misread file splits (#152).

``readers.fat_column_note`` first caught one shape: a file read at the wrong
separator loads as a single column whose header still holds the real one. A
header that happens to contain the separator tried splits instead, into several
wrong columns, and nothing warned:

    region;price, eur;units        read at ','  ->  'region;price' | ' eur;units'

The second signature is that **every** column name contains the same other
common delimiter. "Every", not "some" or "most": a correctly read ``;`` file
with one column called ``price, eur`` must stay quiet, and a threshold would be
the sniffing this project refuses. It is still a warning, and nothing re-reads.

The honest edge stays open and is pinned here so that it is visible: a header
where one field lacks the other separator (``x,y;z`` read at ``,`` gives ``x``
and ``y;z``) is still missed.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from test_server import call

from localdata_mcp import config as config_module
from localdata_mcp import loader as loader_module
from localdata_mcp import server as server_module
from localdata_mcp.config import Config

#: The repro from #152: semicolon-separated, with a comma inside one header name.
SEMICOLONS_WITH_A_COMMA_NAME = "region;price, eur;units\nWest;1,5;7\nEast;2,0;9\n"


@pytest.fixture()
def root(tmp_path):
    """An allowed root and a fresh registry for every test."""
    allowed = (tmp_path / "root").resolve()
    allowed.mkdir()
    config_module.use(Config(roots=(allowed,)))
    server_module._reset()
    yield allowed
    server_module._reset()


def _repro(directory: Path) -> Path:
    path = directory / "semi_comma_header.csv"
    path.write_text(SEMICOLONS_WITH_A_COMMA_NAME)
    return path


# ---------------------------------------------------------------------------
# The loader: read whole and read in chunks
# ---------------------------------------------------------------------------


def test_a_split_misread_warns_and_names_the_character_every_name_holds(tmp_path):
    read = loader_module.read_file(_repro(tmp_path), delimiter=",")

    assert list(read.tables[0].frame.columns) == ["region;price", " eur;units"]
    (note,) = read.notes
    assert "';'" in note
    assert "'region;price, eur;units'" in note  # the header line, checkable
    assert "sanitised" in note
    assert "delimiter set to the right character" in note
    assert "Nothing here guesses it" in note
    assert "single column" not in note


def test_the_chunked_path_gives_the_same_warning(tmp_path):
    path = _repro(tmp_path)

    whole = loader_module.read_file(path, delimiter=",")
    chunked = loader_module.read_source(path, delimiter=",")

    assert chunked.notes == whole.notes
    assert len(chunked.notes) == 1


def test_a_name_holding_another_delimiter_alone_does_not_warn(tmp_path):
    """The false-positive guard: the same file read correctly, at ';'.

    One column is called ``price, eur``. Its comma is a character in a name, not
    a sign of a misread, because the other two names hold none.
    """
    read = loader_module.read_file(_repro(tmp_path), delimiter=";")

    assert list(read.tables[0].frame.columns) == ["region", "price, eur", "units"]
    assert read.notes == ()


def test_three_columns_each_holding_a_pipe_warn_naming_the_pipe(tmp_path):
    path = tmp_path / "pipes.csv"
    path.write_text("a|x,b|y,c|z\n1|2,3|4,5|6\n")

    read = loader_module.read_file(path, delimiter=",")

    assert len(read.tables[0].frame.columns) == 3
    (note,) = read.notes
    assert "'|'" in note
    assert "';'" not in note


def test_two_characters_every_name_holds_are_both_named(tmp_path):
    """Nothing chooses between them: that would be the guess the note refuses."""
    path = tmp_path / "both.csv"
    path.write_text("a;b|c,d;e|f\n1;2|3,4;5|6\n")

    (note,) = loader_module.read_file(path, delimiter=",").notes

    assert "contains ';' and '|'" in note
    assert "one of those characters" in note


def test_a_header_where_one_field_lacks_the_other_separator_is_still_missed(
    tmp_path,
):
    """The documented edge, pinned so that closing it is a visible change.

    ``x`` holds no ``;``, so not every name does, and the strict signature does
    not fire. Widening it to "some" or "most" names would reopen the false
    positive above.
    """
    path = tmp_path / "edge.csv"
    path.write_text("x,y;z\n1,2;3\n")

    read = loader_module.read_file(path, delimiter=",")

    assert list(read.tables[0].frame.columns) == ["x", "y;z"]
    assert read.notes == ()


# ---------------------------------------------------------------------------
# Over the wire: the warning an agent reads, and its next step
# ---------------------------------------------------------------------------


def test_attach_warns_on_the_split_misread(root):
    attached = call("attach", database=str(_repro(root)))

    assert attached["ok"] is True, attached
    names = [c["name"] for c in attached["loaded"][0]["columns"]]
    assert len(names) == 2
    assert all(";" not in n for n in names)  # why the note quotes the header
    (warning,) = attached["warnings"]
    assert "';'" in warning
    assert "delimiter" in warning


def test_the_split_warning_s_next_step_works(root):
    """Detach and attach again at the character the warning named: real columns."""
    path = _repro(root)
    first = call("attach", database=str(path), nickname="prices")
    assert len(first["warnings"]) == 1
    call("detach", nickname="prices")

    again = call("attach", database=str(path), nickname="prices", delimiter=";")

    assert [c["name"] for c in again["loaded"][0]["columns"]] == [
        "region",
        "price_eur",
        "units",
    ]
    assert "warnings" not in again
