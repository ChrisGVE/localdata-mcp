"""testbench/fixtures/db_fixtures.py — NFR-504 DB-fixture logic (pure).

The database side of the collect-and-build fixture provisioning. The batteries
self-seed their own tables through any reachable relational endpoint (the base
battery's `_seed_sql`, the security battery's DSN rows), so what they need from
a fixture backend is a reachable, empty database reached by the right DSN — not
preloaded data. This module owns three decisions around that:

  * the fixture-backend registry — which backend each ``LOCALDATA_TEST_*_DSN``
    names, and whether it is a core (per-PR) or extras (nightly) target;
  * ``discover`` — which backends actually have a DSN set this run;
  * ``wait_until_ready`` / ``load_and_verify`` — the readiness poll (the CI
    step the batteries race without) and the load-and-verify probe that proves
    a backend can hold fixture data (the only fixture-test the extras engines
    get, since no battery has an mssql/oracle row).

Pure by design: it takes injected connect/clock/sleep callables and performs no
docker or process control — that (and all printing) lives in
``scripts/build_db_fixtures.py``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

TIER_CORE = "core"
TIER_EXTRAS = "extras"

# The one fixture table the load-verify probe uses. Deliberately NOT the
# batteries' `base_t`: the batteries create their own on a fresh connection, so
# a name of our own avoids colliding with (or being mistaken for) their seed.
PROBE_TABLE = "localdata_fixture_probe"
PROBE_ROWS: tuple[tuple[int, str], ...] = ((1, "a"), (2, "b"))


@dataclass(frozen=True)
class Backend:
    """One fixture backend: the engine family, the DSN env var that turns it
    on, and the CI tier that provisions it."""

    name: str
    dsn_env: str
    tier: str


# FR-102/FR-103 relational fixtures. SQLite/DuckDB are file-based (no server to
# provision) and the kv/graph/tree stores are embedded SQLite — none appear
# here; this registry is only the server backends a container brings up. mssql
# and oracle are extras-gated (their drivers ship in the `mssql`/`enterprise`
# extras), fixture-tested nightly.
BACKENDS: tuple[Backend, ...] = (
    Backend("postgresql", "LOCALDATA_TEST_POSTGRES_DSN", TIER_CORE),
    Backend("mysql", "LOCALDATA_TEST_MYSQL_DSN", TIER_CORE),
    Backend("mssql", "LOCALDATA_TEST_MSSQL_DSN", TIER_EXTRAS),
    Backend("oracle", "LOCALDATA_TEST_ORACLE_DSN", TIER_EXTRAS),
)


def backends_for_tier(tier: str | None) -> tuple[Backend, ...]:
    """The registry, optionally filtered to one tier (None ⇒ all)."""
    if tier is None:
        return BACKENDS
    return tuple(backend for backend in BACKENDS if backend.tier == tier)


def discover(
    environ: Mapping[str, str],
    tier: str | None = None,
) -> list[tuple[Backend, str]]:
    """The (backend, dsn) pairs whose DSN env var is set and non-empty."""
    found: list[tuple[Backend, str]] = []
    for backend in backends_for_tier(tier):
        dsn = environ.get(backend.dsn_env, "").strip()
        if dsn:
            found.append((backend, dsn))
    return found


def wait_until_ready(
    connect: Callable[[], object],
    *,
    timeout_s: float,
    interval_s: float,
    now: Callable[[], float],
    sleep: Callable[[float], None],
) -> bool:
    """Poll ``connect`` until it returns without raising, or the timeout.

    ``connect`` opens and immediately closes a throwaway connection (a raise
    means not-ready-yet). Returns True on the first success, False once
    ``timeout_s`` has elapsed. Clock and sleep are injected so the poll loop is
    testable without a real database or wall-clock waiting.
    """
    deadline = now() + timeout_s
    while True:
        try:
            handle = connect()
        except Exception:  # noqa: BLE001 — any driver error means "not yet"
            handle = None
        if handle is not None:
            _close(handle)
            return True
        if now() >= deadline:
            return False
        sleep(interval_s)


def load_and_verify(
    execute: Callable[[str], object], read: Callable[[str], Sequence[object]]
) -> None:
    """Provision and verify the probe fixture through injected DB callables.

    Creates the probe table, inserts the canonical rows, reads them back and
    asserts they round-trip, then drops the table — proving the backend both
    accepts and returns fixture data (NFR-504 "loads fixture data"), leaving
    nothing behind. ``execute`` runs a statement; ``read`` runs a query and
    returns its rows. Raises ``FixtureVerificationError`` on any mismatch.
    """
    execute(f"DROP TABLE IF EXISTS {PROBE_TABLE}")
    execute(f"CREATE TABLE {PROBE_TABLE} (id INTEGER, label VARCHAR(8))")
    for identifier, label in PROBE_ROWS:
        execute(
            f"INSERT INTO {PROBE_TABLE} (id, label) VALUES ({identifier}, '{label}')"
        )
    rows = read(f"SELECT id, label FROM {PROBE_TABLE} ORDER BY id")
    got = tuple((int(row[0]), str(row[1])) for row in rows)  # type: ignore[index]
    if got != PROBE_ROWS:
        raise FixtureVerificationError(
            f"probe fixture did not round-trip: expected {PROBE_ROWS}, got {got}"
        )
    execute(f"DROP TABLE IF EXISTS {PROBE_TABLE}")


class FixtureVerificationError(RuntimeError):
    """The load-and-verify probe read back something other than it wrote."""


def _close(handle: object) -> None:
    closer = getattr(handle, "close", None)
    if callable(closer):
        closer()
