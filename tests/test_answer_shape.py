"""Every verb answers a wrong input the same way, and none of them raises.

The server's own instructions make one promise about failure, and they make it
once for the whole surface: *"a refusal is an ordinary answer rather than an
error: it comes back as ``{"ok": false, "error": "…"}`` … Branch on 'ok'."* An
agent reads that once and then branches on ``ok`` for the rest of the session.
A verb that raises instead does not merely answer differently — it answers in a
shape the agent was told would not occur, and the exception reaches the client
as a protocol error with no ``ok`` in it at all.

That promise had never been tested **across** the verbs. Each verb had tests for
its own refusals, so each was internally consistent, and the one check that
compares them to each other was missing — which is how ``update`` came to escape
with a ``LoadError`` on an unattached nickname while all seven of the siblings it
had then returned a refusal for the same input (issue #92). The defect was
introduced by a fix to
another verb, and a suite of 731 passing tests did not see it, because no test
drove two verbs at the same wrong input and compared the answers.

So this file is one table rather than a set of cases: **a wrong input down one
axis, every verb that accepts it across the other**, asserting the same shape in
every cell. A verb added later, or a check bypassed later, fails here.

These drive the tool functions directly rather than over the MCP protocol. The
shape is decided in the function body, and calling it directly is what lets the
test say *nothing raised* — over the protocol an escaping exception is converted
into an error result, which is the very outcome being tested for. The protocol
round trip is covered in ``test_server.py``.

Each condition installs its own positive control: the healthy call is made first
and asserted to succeed. Without it, a fixture that quietly broke — a root that
does not match the file's real path is enough — would make every verb refuse for
the wrong reason, and the table would pass while testing nothing.
"""

from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path
from typing import Any

import pytest
from fastmcp import Client

from localdata_mcp import config as config_module
from localdata_mcp import server as server_module
from localdata_mcp.config import Config, ConfigError

VERBS = {
    "attach": server_module.attach,
    "detach": server_module.detach,
    "directory": server_module.directory,
    "query": server_module.query,
    "create": server_module.create,
    "update": server_module.update,
    "drop": server_module.drop,
    "save": server_module.save,
    "stats": server_module.stats,
}


def test_the_table_covers_the_whole_surface():
    """The guard that makes this file's promise true rather than aspirational.

    The docstring above says a verb added later fails here. It did not: ``stats``
    shipped as the ninth verb and every column in this file went on passing,
    because :data:`VERBS` is written by hand and the one assertion that checks a
    column against it compares the table to *itself*. A self-referential guard
    cannot notice an absence.

    So the surface is read from the server rather than restated here, and the
    table is asserted against it. Adding a verb without giving it a row now fails
    at this line, which is what the paragraph above always claimed happened.
    """

    async def _names():
        async with Client(server_module.mcp) as client:
            return {tool.name for tool in await client.list_tools()}

    assert set(VERBS) == asyncio.run(_names())


@pytest.fixture
def root(tmp_path, monkeypatch):
    """A fresh registry over a directory the server is allowed to reach.

    ``resolve()`` is not decoration: on macOS ``tmp_path`` is under ``/var``,
    which is a symlink to ``/private/var``, and a root spelled the unresolved way
    puts every file in it outside the boundary. That is a fixture failure that
    presents as a passing refusal, which is what the positive controls catch.
    """
    monkeypatch.delenv(config_module.PATH_ENV_VAR, raising=False)
    workspace = (tmp_path / "root").resolve()
    workspace.mkdir()
    config_module.use(Config(roots=(workspace,)))
    server_module._reset()
    yield workspace
    server_module._reset()


def refusal(verb: str, **arguments: Any) -> str:
    """Call a verb, require the refusal shape, and return the reason.

    A raised exception fails here rather than propagating, so the report names
    the verb and the input instead of showing a traceback from inside the
    loader.
    """
    try:
        answer = VERBS[verb](**arguments)
    except Exception as exc:  # noqa: BLE001 - the escape is what is under test
        raise AssertionError(
            f"{verb}({arguments}) raised {type(exc).__name__}: {exc}. Every verb "
            f"answers a refusal as a payload; the server's instructions tell the "
            f"caller to branch on 'ok', and an exception has no 'ok' in it."
        ) from exc
    assert isinstance(answer, dict), f"{verb} answered {type(answer).__name__}"
    assert answer.get("ok") is False, f"{verb} was not a refusal: {answer!r}"
    assert (
        isinstance(answer.get("error"), str) and answer["error"]
    ), f"{verb} refused without a reason: {answer!r}"
    return answer["error"]


def succeeds(verb: str, **arguments: Any) -> dict[str, Any]:
    """The positive control: a healthy call, asserted to work."""
    answer = VERBS[verb](**arguments)
    assert answer.get("ok") is True, (
        f"positive control failed — {verb} was expected to succeed here and "
        f"answered {answer!r}. Every refusal below would be for the wrong reason."
    )
    return answer


def people_csv(root: Path) -> Path:
    path = root / "people.csv"
    path.write_text("name,age\nada,36\ngrace,45\n")
    return path


def other_csv(root: Path) -> Path:
    path = root / "other.csv"
    path.write_text("key,value\n1,2\n")
    return path


# ---------------------------------------------------------------------------
# The table
# ---------------------------------------------------------------------------


def test_a_nickname_that_is_not_attached_is_refused_by_every_verb_that_takes_one(root):
    """The column that found #92, and the one an agent hits first.

    A nickname goes stale in ordinary use — the slot was evicted to make room,
    or the agent used the name it asked for rather than the one ``attach``
    returned. When #92 was found, seven verbs said so and ``update`` raised.
    """
    source = str(other_csv(root))
    exports = str(root / "out.csv")
    saved = str(root / "out.db")

    cells = [
        ("detach", {"nickname": "nope"}),
        ("directory", {"nickname": "nope"}),
        ("directory", {"nickname": "nope", "table": "people"}),
        ("query", {"nickname": "nope", "sql": "SELECT 1"}),
        ("query", {"nickname": "nope", "sql": "SELECT 1", "path": exports}),
        ("create", {"nickname": "nope", "type": "table", "source": source}),
        (
            "create",
            {"nickname": "nope", "type": "index", "table": "t", "columns": ["a"]},
        ),
        ("update", {"nickname": "nope", "type": "table", "name": "t", "to": "u"}),
        ("drop", {"nickname": "nope", "type": "table", "name": "t"}),
        ("drop", {"nickname": "nope", "type": "index", "name": "t_a"}),
        ("save", {"nickname": "nope", "path": saved}),
        ("stats", {"nickname": "nope", "table": "people"}),
    ]

    for verb, arguments in cells:
        reason = refusal(verb, **arguments)
        # The same fact, said the same way, whichever verb was asked: an agent
        # that learns to read one of these has learned to read all of them.
        assert "nope" in reason, f"{verb} did not name the nickname: {reason}"
        assert "attached" in reason.lower(), f"{verb} said something else: {reason}"


def test_a_table_that_is_not_there_is_refused_by_every_verb_that_names_one(root):
    csv = people_csv(root)
    succeeds("attach", database=str(csv), nickname="live")

    cells = [
        ("directory", {"nickname": "live", "table": "ghost"}),
        ("query", {"nickname": "live", "sql": "SELECT * FROM ghost"}),
        (
            "create",
            {
                "nickname": "live",
                "type": "index",
                "table": "ghost",
                "columns": ["name"],
            },
        ),
        ("update", {"nickname": "live", "type": "table", "name": "ghost", "to": "u"}),
        ("drop", {"nickname": "live", "type": "table", "name": "ghost"}),
        ("stats", {"nickname": "live", "table": "ghost"}),
    ]

    for verb, arguments in cells:
        reason = refusal(verb, **arguments)
        assert "ghost" in reason, f"{verb} did not name the table: {reason}"


def test_an_index_and_a_column_that_are_not_there_are_refused_by_name(root):
    csv = people_csv(root)
    succeeds("attach", database=str(csv), nickname="live")

    reason = refusal("drop", nickname="live", type="index", name="people_name")
    assert "people_name" in reason

    reason = refusal(
        "create", nickname="live", type="index", table="people", columns=["ghost"]
    )
    assert "ghost" in reason
    # The names that would have worked, because a caller who got here does not
    # know what is in the table.
    assert "name" in reason and "age" in reason

    # ``stats`` names columns too, and refuses an unknown one the same way.
    reason = refusal("stats", nickname="live", table="people", columns=["ghost"])
    assert "ghost" in reason
    assert "name" in reason and "age" in reason


def test_a_slot_attached_read_only_refuses_every_verb_that_would_change_it(root):
    database = root / "readonly.db"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE t (a INTEGER)")
        connection.execute("INSERT INTO t VALUES (1)")

    succeeds("attach", database=str(database), nickname="ro")
    # The control that matters for this column specifically: the slot is live
    # and readable, so a refusal below is about write access and nothing else.
    succeeds("directory", nickname="ro")

    cells = [
        ("create", {"nickname": "ro", "type": "table", "source": str(other_csv(root))}),
        ("create", {"nickname": "ro", "type": "index", "table": "t", "columns": ["a"]}),
        ("update", {"nickname": "ro", "type": "table", "name": "t", "to": "u"}),
        ("drop", {"nickname": "ro", "type": "table", "name": "t"}),
    ]

    for verb, arguments in cells:
        reason = refusal(verb, **arguments)
        assert "read-only" in reason, f"{verb} said something else: {reason}"
        # Each one has to say what to do instead, or the agent retries the same
        # call: the grant belongs to an attach, so the route is detach first.
        assert "Detach it first" in reason, f"{verb} left no way forward: {reason}"

    # `query` refuses a write for a different reason — it never writes, whatever
    # the slot allows — and it too is an answer rather than an exception.
    reason = refusal("query", nickname="ro", sql="INSERT INTO t VALUES (2)")
    assert "reads" in reason


def test_a_type_the_verb_does_not_have_is_refused_with_the_types_it_does(root):
    csv = people_csv(root)
    succeeds("attach", database=str(csv), nickname="live")

    cells = [
        ("create", {"nickname": "live", "type": "view"}),
        ("drop", {"nickname": "live", "type": "view", "name": "people"}),
        ("update", {"nickname": "live", "type": "view", "name": "people", "to": "u"}),
    ]

    for verb, arguments in cells:
        reason = refusal(verb, **arguments)
        assert "view" in reason, f"{verb} did not name the type it got: {reason}"
        assert "table" in reason, f"{verb} did not name the types it has: {reason}"


def test_an_unexpected_failure_below_is_still_an_answer_at_every_verb(
    root, monkeypatch
):
    """The envelope has to be total, or the promise is one an agent cannot use.

    The columns above cover the wrong inputs that are known and refused by name.
    This one covers everything else: a driver that raises something new, a
    filesystem that fails mid-call, a bug. ``query`` already answered those as
    payloads while every other verb of the day — seven of them — let them out as
    protocol errors: one promise made by the shipped instructions, stated
    separately underneath by each verb that had to keep it.

    The type name is asserted, not just the shape. An unexpected failure must
    not be *disguised* as an ordinary refusal: the caller cannot fix it and the
    maintainer needs to know it happened, so it comes back saying what it was.
    """
    csv = people_csv(root)
    succeeds("attach", database=str(csv), nickname="live")

    class Boom(RuntimeError):
        pass

    def explode(*args: Any, **kwargs: Any):
        raise Boom("the layer below came apart")

    monkeypatch.setattr(server_module, "_session", explode)

    cells = [
        ("attach", {"database": str(csv)}),
        ("detach", {"nickname": "live"}),
        ("directory", {}),
        ("query", {"nickname": "live", "sql": "SELECT 1"}),
        (
            "create",
            {"nickname": "live", "type": "table", "source": str(other_csv(root))},
        ),
        ("update", {"nickname": "live", "type": "table", "name": "people", "to": "u"}),
        ("drop", {"nickname": "live", "type": "table", "name": "people"}),
        ("save", {"nickname": "live", "path": str(root / "out.db")}),
        ("stats", {"nickname": "live", "table": "people"}),
    ]
    assert {verb for verb, _ in cells} == set(VERBS), "a verb is missing from the table"

    for verb, arguments in cells:
        reason = refusal(verb, **arguments)
        assert "Boom" in reason, f"{verb} hid what went wrong: {reason}"
        assert "came apart" in reason, f"{verb} dropped the message: {reason}"


def test_a_configuration_that_will_not_be_run_under_stops_the_server_starting(
    tmp_path, monkeypatch
):
    """The other half of #92, and the one a catch-all would have got wrong.

    A mistyped setting used to be discovered inside the first tool call that
    happened to read the configuration, because :func:`config.active` loads on
    first use. Once the registry existed every verb reached it — measured at the
    time against the eight verbs there were, all eight raising ``ConfigError``,
    including the two an agent calls first.

    Answering those as refusals would be the wrong fix, and ``config.py`` says
    why in its own module docstring: *"Failing to start is the louder, better
    outcome — the message reaches the client's server log."* A server that
    starts under a configuration it could not read is one whose path boundary
    is not what the user believes they set. So the fix is to read it at startup,
    and the assertion is that the process does not get as far as serving.
    """
    config_module.reset()
    server_module._reset()
    broken = tmp_path / "config.toml"
    broken.write_text("path_limitted = false\n")
    monkeypatch.setenv(config_module.PATH_ENV_VAR, str(broken))

    served = False

    def serve() -> None:
        nonlocal served
        served = True

    monkeypatch.setattr(server_module.mcp, "run", serve)

    with pytest.raises(ConfigError) as raised:
        server_module.main()

    assert not served, "the server started under a configuration it could not read"
    # The message has to name the file and the mistyped key, because it arrives
    # in a log rather than in an answer, with no follow-up question possible.
    assert str(broken) in str(raised.value)
    assert "path_limitted" in str(raised.value)


def test_a_healthy_configuration_starts_the_server(tmp_path, monkeypatch):
    """The positive control for the test above, and for the startup read itself.

    Without it, a ``main()`` that raised on every configuration would pass the
    previous test, and so would one that never reached ``mcp.run`` at all.
    """
    config_module.reset()
    server_module._reset()
    fine = tmp_path / "config.toml"
    fine.write_text("[paths]\npath_limited = false\n")
    monkeypatch.setenv(config_module.PATH_ENV_VAR, str(fine))

    served = False

    def serve() -> None:
        nonlocal served
        served = True

    monkeypatch.setattr(server_module.mcp, "run", serve)

    server_module.main()

    assert served
    assert config_module.active().path_limited is False
