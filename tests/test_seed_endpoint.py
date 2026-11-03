"""The seed script's logic, offline: what it loads, how, and what it refuses.

``scripts/seed_endpoint.py`` loads the standard assets into a running endpoint
for an ad-hoc session (task 35, #139). Its database is a URL in real use, but
everything it does past building that URL is the server's own ``attach`` and
``create``. So the same code is run here against a SQLite file, which needs no
container.

The expected row counts are the files' own, counted from the files, not from
anything the server reports.
"""

from __future__ import annotations

import csv
import importlib.util
import sqlite3
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
ASSETS = REPO / "tests" / "assets"


def _load_seed_module():
    spec = importlib.util.spec_from_file_location(
        "seed_endpoint", REPO / "scripts" / "seed_endpoint.py"
    )
    module = importlib.util.module_from_spec(spec)
    # Registered before it runs: a dataclass looks its own module up in
    # sys.modules while it is being defined.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


seed_endpoint = _load_seed_module()


def _data_rows(asset) -> int:
    with (ASSETS / asset.file).open(newline="") as handle:
        return sum(1 for _ in csv.reader(handle, delimiter=asset.delimiter)) - 1


def _sqlite_counts(path: Path) -> dict[str, int]:
    with sqlite3.connect(path) as conn:
        names = [
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        ]
        return {
            name: conn.execute(f'SELECT count(*) FROM "{name}"').fetchone()[0]
            for name in names
        }


@pytest.fixture
def database(tmp_path) -> Path:
    """An empty SQLite database. Not an empty *file*: connecting alone writes no
    header, and the server rightly refuses a file whose header names no format."""
    path = tmp_path / "seeded.sqlite"
    with sqlite3.connect(path) as conn:
        conn.execute("PRAGMA user_version = 1")
    return path


def test_the_standard_assets_are_existing_fixtures():
    """Reused, not invented: every asset the seed loads is in tests/assets."""
    assert seed_endpoint.ASSETS
    for asset in seed_endpoint.ASSETS:
        assert (ASSETS / asset.file).is_file(), asset.file


def test_every_asset_lands_with_every_row(database):
    landed = seed_endpoint.seed(str(database), allowed=(database.parent,))

    counts = _sqlite_counts(database)
    for asset in seed_endpoint.ASSETS:
        assert counts[asset.table] == _data_rows(asset), asset.table
    assert [result.table for result in landed] == [
        asset.table for asset in seed_endpoint.ASSETS
    ]
    assert all(result.created for result in landed)


def test_seeding_twice_leaves_one_copy_and_says_so(database):
    """A second seed must not fail on a table that is already there, and must not
    double it. Nothing is overwritten either: what is there may be an ad-hoc
    session's own work by now.
    """
    seed_endpoint.seed(str(database), allowed=(database.parent,))
    again = seed_endpoint.seed(str(database), allowed=(database.parent,))

    assert not any(result.created for result in again)
    counts = _sqlite_counts(database)
    for asset in seed_endpoint.ASSETS:
        assert counts[asset.table] == _data_rows(asset), asset.table


def test_a_refused_attach_is_an_error_that_names_the_reason(tmp_path):
    with pytest.raises(seed_endpoint.SeedError, match="attach"):
        seed_endpoint.seed(str(tmp_path / "no-such.sqlite"), allowed=(tmp_path,))


def test_the_service_resolves_to_its_own_credentialed_target():
    target = seed_endpoint.target_for("localdata-test-postgres")
    assert target.auth is None
    assert target.service == "localdata-test-postgres"

    short = seed_endpoint.target_for("postgres")
    assert short == target


def test_an_authentication_service_resolves_to_a_mode_it_serves():
    target = seed_endpoint.target_for("localdata-test-postgres-tls")
    assert target.service == "localdata-test-postgres-tls"
    assert target.auth is not None


def test_a_helper_or_unknown_service_is_refused():
    for name in ("localdata-test-tls-ca", "localdata-test-no-such-thing"):
        with pytest.raises(seed_endpoint.SeedError, match="no tests|no target"):
            seed_endpoint.target_for(name)
