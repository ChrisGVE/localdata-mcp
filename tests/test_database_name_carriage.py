"""A database name a URL cannot carry on SQLAlchemy 2.0 is refused, not misread (#140).

File: ``tests/test_database_name_carriage.py``. Exercises
:mod:`localdata_mcp.urls`, and the two places :mod:`localdata_mcp.slots` hands it
a URL a caller typed: ``attach`` of a server or file URL, and ``attach`` of a
``sqlite:`` URL.

SQLAlchemy 2.1 unquotes a URL's database component when it parses one, so
``a%3Fb`` names the database ``a?b``. 2.0 does not, and passes ``a%3Fb`` on to
the driver as it stands: a different database, with nothing said. Installed
with the ``cockroachdb`` extra the package resolves to 2.0.55, so that is a
version this server really runs on. There, a ``?`` or ``@`` in a database name
is refused with the reason and the remedy.

The project's own lock is 2.1.6, so the 2.0 behaviour is exercised here by
telling the helper which version it is looking at. That is sound because the
helper reads the name the same way under both parsers: 2.0 leaves the escape in
and the helper decodes it, 2.1 has already decoded it. The same refusal was
also run for real against an installed 2.0.55, outside this suite.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3

import pytest
import sqlalchemy
from fastmcp import Client
from sqlalchemy.engine import make_url

from localdata_mcp import config as config_module
from localdata_mcp import server as server_module
from localdata_mcp import urls
from localdata_mcp.config import Config

#: URLs whose database name holds a ``?`` or an ``@``, each spelled the way
#: SQLAlchemy 2.1 reads it, paired with the name 2.1 reads out of it.
UNCARRIED_ON_2_0 = [
    pytest.param("postgresql://u:pw@db.example.com:5432/a%3Fb", "a?b", id="?-pw"),
    pytest.param("postgresql://u@db.example.com:5432/a%3Fb", "a?b", id="?-no-pw"),
    pytest.param("postgresql://u:pw@db.example.com:5432/a%40b", "a@b", id="@-pw"),
    pytest.param("postgresql://u@db.example.com:5432/a%40b", "a@b", id="@-no-pw"),
    pytest.param("postgresql://u:pw@db.example.com/a@b", "a@b", id="raw-@"),
    pytest.param(
        "firebird+firebird://u:pw@db.example.com:3050//srv/x%3Fy.fdb",
        "/srv/x?y.fdb",
        id="firebird-path",
    ),
]


# -- the helper ---------------------------------------------------------------


@pytest.mark.parametrize(("raw", "name"), UNCARRIED_ON_2_0)
def test_a_question_mark_or_at_sign_is_refused_on_sqlalchemy_2_0(raw, name):
    with pytest.raises(urls.DatabaseNameNotCarried) as refused:
        urls.refuse_uncarried_database(make_url(raw), installed="2.0.55")

    message = str(refused.value)
    assert name in message
    assert "2.0.55" in message
    assert "SQLAlchemy 2.1 or later" in message
    assert "cockroachdb" in message
    assert "pw" not in message.replace(name, "")


@pytest.mark.parametrize(("raw", "name"), UNCARRIED_ON_2_0)
def test_the_same_name_round_trips_on_sqlalchemy_2_1(raw, name):
    url = make_url(raw)

    urls.refuse_uncarried_database(url, installed="2.1.6")

    assert url.database == name
    rendered = make_url(url.render_as_string(hide_password=False))
    assert rendered.database == name
    assert rendered.host == "db.example.com"


@pytest.mark.parametrize(
    "database", ["plain", "a#b", "a:b", "a b", "n#me:with/parts", None]
)
def test_a_name_without_either_character_passes_on_2_0(database):
    url = sqlalchemy.engine.URL.create(
        "postgresql", username="u", host="db.example.com", database=database
    )

    urls.refuse_uncarried_database(url, installed="2.0.55")


@pytest.mark.parametrize(
    ("installed", "refused"),
    [
        ("2.0.0", True),
        ("2.0.55", True),
        ("1.4.54", True),
        ("2.1.0b3", False),
        ("2.1.6", False),
        ("2.10.0", False),
        ("10.0.0", False),
    ],
)
def test_the_version_is_compared_as_numbers(installed, refused):
    # "10.0.0" sorts before "2.1" as a string; as a release it is far after.
    url = make_url("postgresql://u@db.example.com:5432/a%3Fb")

    if refused:
        with pytest.raises(urls.DatabaseNameNotCarried):
            urls.refuse_uncarried_database(url, installed=installed)
    else:
        urls.refuse_uncarried_database(url, installed=installed)


def test_a_version_that_does_not_parse_is_an_error_not_a_pass():
    url = make_url("postgresql://u@db.example.com:5432/a%3Fb")

    with pytest.raises(ValueError, match="not-a-version"):
        urls.refuse_uncarried_database(url, installed="not-a-version")


def test_by_default_the_helper_reads_the_installed_sqlalchemy():
    assert urls.installed_sqlalchemy() == sqlalchemy.__version__


# -- through the server ---------------------------------------------------------


@pytest.fixture
def root(tmp_path, monkeypatch):
    """An allowed root holding a file called ``a?b.db`` and a decoy beside it.

    The decoy is named ``a%3Fb.db``: the file 2.0 would open for the URL
    ``sqlite:///…/a%3Fb.db``, which 2.1 reads as ``a?b.db``.
    """
    work = tmp_path / "root"
    work.mkdir()
    monkeypatch.chdir(work)
    config_module.use(Config(roots=(work,)))
    for name, table in (("a?b.db", "meant"), ("a%3Fb.db", "decoy")):
        with sqlite3.connect(work / name) as con:
            con.execute(f"CREATE TABLE {table} (x INTEGER)")
    server_module._reset()
    yield work
    server_module._reset()


@pytest.fixture
def on_2_0(monkeypatch):
    """Make the helper see SQLAlchemy 2.0.55, the version the extra resolves to."""
    monkeypatch.setattr(urls, "installed_sqlalchemy", lambda: "2.0.55")


def call(tool: str, **arguments):
    async def run():
        async with Client(server_module.mcp) as client:
            result = await client.call_tool(tool, arguments)
            if result.structured_content is not None:
                return result.structured_content
            return json.loads(result.content[0].text)

    return asyncio.run(run())


def table_names(answer) -> list[str]:
    return [t["name"] if isinstance(t, dict) else t for t in answer["tables"]]


@pytest.mark.parametrize(("raw", "name"), UNCARRIED_ON_2_0)
def test_attach_refuses_the_url_before_the_network_gate(root, on_2_0, raw, name):
    answer = call("attach", database=raw)

    assert answer["ok"] is False
    assert name in answer["error"]
    assert "SQLAlchemy 2.1 or later" in answer["error"]
    # Refused for the name, not for the closed network it never reached.
    assert "network" not in answer["error"]
    assert ":pw@" not in answer["error"]


@pytest.mark.parametrize("scheme", ["sqlite", "duckdb"])
def test_attach_refuses_a_file_url_and_points_at_the_path(root, on_2_0, scheme):
    answer = call("attach", database=f"{scheme}:///{root}/a%3Fb.db")

    assert answer["ok"] is False
    assert "a?b.db" in answer["error"]
    assert "path" in answer["error"]
    assert "decoy" not in json.dumps(answer)


def test_the_bare_path_still_attaches_on_2_0(root, on_2_0):
    # No URL string is parsed for a path, so 2.0 carries the name: measured on
    # a real 2.0.55 install as well.
    answer = call("attach", database=str(root / "a?b.db"))

    assert answer["ok"] is True, answer
    assert table_names(answer) == ["meant"]


def test_on_2_1_the_file_url_opens_the_file_it_names(root):
    answer = call("attach", database=f"sqlite:///{root}/a%3Fb.db")

    assert answer["ok"] is True, answer
    assert table_names(answer) == ["meant"]


def test_on_2_1_a_server_url_reaches_the_network_gate_unrefused(root):
    answer = call("attach", database="postgresql://u@db.example.com:5432/a%3Fb")

    assert answer["ok"] is False
    assert "network" in answer["error"]
    assert "SQLAlchemy 2.1 or later" not in answer["error"]
