"""``query`` sends the caller's SQL with no bound parameters, ever.

SQL Server's read gate (#122) relies on this. ``SHOWPLAN_XML`` gives no plan
for a statement with bound parameters, so ``MSSQLBackend``'s gate does not
check one, on the ground that only this server's own statements carry them.
If ``query`` ever passes caller-supplied values, that ground is gone and a
write could reach SQL Server unchecked. This test fails first, so the gate is
reconsidered then rather than discovered later.
"""

from __future__ import annotations

import sqlite3

from sqlalchemy import event

from localdata_mcp.loader import Workspace


def test_the_callers_statement_reaches_the_driver_with_no_parameters(tmp_path):
    path = tmp_path / "t.db"
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE t (a INTEGER)")
        conn.execute("INSERT INTO t VALUES (1)")
    workspace = Workspace()
    workspace.attach_file("t", path)
    sent: list[tuple[str, object]] = []
    engine = workspace.entry("t").engines.read

    @event.listens_for(engine, "before_cursor_execute")
    def _record(conn, cursor, statement, parameters, context, executemany):  # noqa: ANN001, ANN202 - SQLAlchemy's signature
        sent.append((statement, parameters))

    sql = "SELECT a FROM t WHERE a = 1"
    workspace.query("t", sql)

    mine = [parameters for statement, parameters in sent if statement == sql]
    assert len(mine) == 1, sent
    assert not mine[0], mine
