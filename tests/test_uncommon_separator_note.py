"""The wrong-separator warning when the separator used is not a common one (#156).

``readers.fat_column_note`` names the separator a file was read at. It used to
spell that separator by looking it up in the map of common delimiters (``;``,
tab, ``|``, ``,``), which holds only the candidates the note may name. A caller
who declared any other character, ``:`` say, on a file whose header holds one of
the four, hit a ``KeyError`` there, and attach refused the file with
``Could not read x.txt: ':'`` instead of loading it and warning.

An uncommon separator is now spelled as its ``repr``, so ``:`` reads ``':'``.
The candidates the note names are unchanged.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from test_server import call

from localdata_mcp import config as config_module
from localdata_mcp import loader as loader_module
from localdata_mcp import readers as readers_module
from localdata_mcp import server as server_module
from localdata_mcp.config import Config

#: The repro from #156: semicolon-separated, read at a declared ':'.
SEMICOLONS = "a;b;c\n1;2;3\n"

#: Every header field holds ';', and ':' splits the line into three of them.
COLON_SPLIT_SEMICOLONS = "a;x:b;y:c;z\n1;2:3;4:5;6\n"


@pytest.fixture()
def root(tmp_path):
    """An allowed root and a fresh registry for every test."""
    allowed = (tmp_path / "root").resolve()
    allowed.mkdir()
    config_module.use(Config(roots=(allowed,)))
    server_module._reset()
    yield allowed
    server_module._reset()


def _write(directory: Path, name: str, text: str) -> Path:
    path = directory / name
    path.write_text(text)
    return path


# ---------------------------------------------------------------------------
# The two note functions, called directly
# ---------------------------------------------------------------------------


def test_the_single_column_note_spells_an_uncommon_separator_as_its_repr():
    (note,) = readers_module._single_column_note("a;b;c", ":", "x.txt")

    assert "x.txt was read at ':' and loaded as a single column" in note
    assert "which contains ';'" in note
    assert "when it is read at ':'." in note


def test_the_split_header_note_spells_an_uncommon_separator_as_its_repr():
    (note,) = readers_module._split_header_note(["a;x", "b;y", "c;z"], ":", "x.txt")

    assert "x.txt was read at ':' and loaded as 3 columns" in note
    assert "'a;x:b;y:c;z'" in note  # the names joined again at the one used
    assert "contains ';'" in note
    assert "when it is read at ':'." in note


def test_a_common_separator_keeps_its_own_spelling():
    """The map's spelling still wins where it has one: a tab reads 'a tab'."""
    (note,) = readers_module._single_column_note("a;b;c", "\t", "x.tsv")

    assert "x.tsv was read at a tab and loaded" in note


# ---------------------------------------------------------------------------
# Over the wire: attach loads and warns rather than refusing
# ---------------------------------------------------------------------------


def test_attach_at_an_uncommon_separator_warns_on_a_single_column(root):
    path = _write(root, "x.txt", SEMICOLONS)

    attached = call("attach", database=str(path), delimiter=":")

    assert attached["ok"] is True, attached
    assert len(attached["loaded"][0]["columns"]) == 1
    (warning,) = attached["warnings"]
    assert "read at ':'" in warning
    assert "contains ';'" in warning
    assert "delimiter set to the right character" in warning


def test_attach_at_an_uncommon_separator_warns_on_a_split_header(root):
    path = _write(root, "y.txt", COLON_SPLIT_SEMICOLONS)

    attached = call("attach", database=str(path), delimiter=":")

    assert attached["ok"] is True, attached
    assert len(attached["loaded"][0]["columns"]) == 3
    (warning,) = attached["warnings"]
    assert "read at ':'" in warning
    assert "loaded as 3 columns" in warning
    assert "contains ';'" in warning


def test_the_chunked_path_gives_the_same_warning_at_an_uncommon_separator(
    tmp_path,
):
    path = _write(tmp_path, "x.txt", SEMICOLONS)

    whole = loader_module.read_file(path, delimiter=":")
    chunked = loader_module.read_source(path, delimiter=":")

    assert len(whole.notes) == 1
    assert chunked.notes == whole.notes
