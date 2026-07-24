"""tests/v3/test_persistence_ephemeral.py — E5.3 acceptance.

Read-only default over real tmp_path files, operator rw grant keyed by
canonical path/prefix, canonical identity through symlinks, and the §5
never-outlives-the-call guarantee. The grant field itself is exercised
through NX-2's merge in test_config_merge_security-style assertions:
a project layer cannot mint an ephemeral write grant.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from sqlalchemy import text

from localdata_mcp.nexus.config.merge import merge_sources
from localdata_mcp.nexus.config.provenance import Layer, LayerSource
from localdata_mcp.nexus.persistence.ephemeral import (
    EphemeralFileConnection,
    EphemeralOpenRefusedError,
    ephemeral_for,
    rw_granted,
)


def seeded_sqlite(path: Path) -> None:
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE t (x INTEGER)")
    connection.execute("INSERT INTO t VALUES (7)")
    connection.commit()
    connection.close()


class TestPostureDefault:
    def test_read_only_is_the_unconditional_default(self, tmp_path: Path) -> None:
        db = tmp_path / "data.db"
        seeded_sqlite(db)
        ephemeral = ephemeral_for(db, "sqlite", write_grants=())
        assert ephemeral.posture == "read_only"
        with ephemeral.open() as connection:
            assert connection.execute(text("SELECT x FROM t")).scalar() == 7

    def test_read_only_open_refuses_writes(self, tmp_path: Path) -> None:
        db = tmp_path / "data.db"
        seeded_sqlite(db)
        ephemeral = ephemeral_for(db, "sqlite", write_grants=())
        with ephemeral.open() as connection:
            with pytest.raises(Exception, match="(?i)readonly|query_only"):
                connection.execute(text("INSERT INTO t VALUES (8)"))


class TestOperatorGrant:
    def test_exact_canonical_path_grant_opens_read_write(self, tmp_path: Path) -> None:
        db = tmp_path / "data.db"
        seeded_sqlite(db)
        ephemeral = ephemeral_for(db, "sqlite", write_grants=(str(db),))
        assert ephemeral.posture == "read_write"
        with ephemeral.open() as connection:
            connection.execute(text("INSERT INTO t VALUES (8)"))
            connection.commit()

    def test_contained_prefix_grant_opens_read_write(self, tmp_path: Path) -> None:
        db = tmp_path / "nested" / "data.db"
        db.parent.mkdir()
        seeded_sqlite(db)
        ephemeral = ephemeral_for(db, "sqlite", write_grants=(str(tmp_path),))
        assert ephemeral.posture == "read_write"

    def test_unrelated_grant_stays_read_only(self, tmp_path: Path) -> None:
        db = tmp_path / "data.db"
        seeded_sqlite(db)
        elsewhere = tmp_path / "other"
        ephemeral = ephemeral_for(db, "sqlite", write_grants=(str(elsewhere),))
        assert ephemeral.posture == "read_only"

    def test_grant_comparison_is_canonical_both_sides(self, tmp_path: Path) -> None:
        real_dir = tmp_path / "real"
        real_dir.mkdir()
        db = real_dir / "data.db"
        seeded_sqlite(db)
        link = tmp_path / "alias"
        link.symlink_to(real_dir)
        # Grant names the symlinked alias; the path opened is the real one.
        assert rw_granted(db.resolve(), (str(link),))


class TestCanonicalIdentity:
    def test_identity_is_the_resolved_path(self, tmp_path: Path) -> None:
        db = tmp_path / "data.db"
        seeded_sqlite(db)
        relative_ish = tmp_path / "." / "data.db"
        ephemeral = ephemeral_for(relative_ish, "sqlite", write_grants=())
        assert ephemeral.canonical_path == db.resolve()


class TestPerCallLifetime:
    def test_connection_never_outlives_the_call(self, tmp_path: Path) -> None:
        db = tmp_path / "data.db"
        seeded_sqlite(db)
        ephemeral = ephemeral_for(db, "sqlite", write_grants=())
        with ephemeral.open() as connection:
            live = connection
        with pytest.raises(Exception):
            live.execute(text("SELECT x FROM t"))

    def test_duckdb_ephemeral_round_trip(self, tmp_path: Path) -> None:
        import duckdb

        path = tmp_path / "data.duckdb"
        seed = duckdb.connect(str(path))
        seed.execute("CREATE TABLE t (x INTEGER)")
        seed.execute("INSERT INTO t VALUES (7)")
        seed.close()
        ephemeral = ephemeral_for(path, "duckdb", write_grants=())
        assert ephemeral.posture == "read_only"
        with ephemeral.open() as connection:
            assert connection.execute("SELECT x FROM t").fetchone() == (7,)


class TestGrantIsIntroductionGated:
    """The NX-2 field behaves like allowed_paths: operator layers may
    mint entries, a project layer may only narrow (§5's introduction
    rule — a mintable rw grant would reopen the NFR-114 hole)."""

    def test_operator_layer_may_mint_a_grant(self) -> None:
        result = merge_sources(
            [
                LayerSource(
                    name="user-file",
                    layer=Layer.USER,
                    values={"security": {"ephemeral_write_paths": ["/srv/scratch"]}},
                ),
            ]
        )
        assert result.model.security.ephemeral_write_paths == ("/srv/scratch",)
        assert result.refusals == ()

    def test_project_layer_cannot_mint_a_grant(self) -> None:
        result = merge_sources(
            [
                LayerSource(
                    name="project-file",
                    layer=Layer.PROJECT,
                    values={"security": {"ephemeral_write_paths": ["/srv/scratch"]}},
                ),
            ]
        )
        assert result.model.security.ephemeral_write_paths == ()
        assert any(
            "ephemeral_write_paths" in str(refusal) for refusal in result.refusals
        )


class TestAtomicContainOpenCr024:
    """CR-024: the ephemeral open re-validates the canonical path with
    O_NOFOLLOW, so a final component that is a symlink (standing in for a
    post-containment swap) is refused before the engine opens it."""

    def test_open_refuses_a_symlinked_canonical_path(self, tmp_path: Path) -> None:
        real = tmp_path / "real.db"
        seeded_sqlite(real)
        link = tmp_path / "link.db"
        link.symlink_to(real)
        # Construct with the symlink AS the canonical path (ephemeral_for
        # would resolve it away) — the O_NOFOLLOW guard refuses to follow.
        ephemeral = EphemeralFileConnection(
            canonical_path=link, engine_kind="sqlite", posture="read_only"
        )
        with pytest.raises(OSError):  # ELOOP or EphemeralOpenRefusedError
            with ephemeral.open():
                pass

    def test_open_succeeds_on_a_real_regular_file(self, tmp_path: Path) -> None:
        db = tmp_path / "data.db"
        seeded_sqlite(db)
        ephemeral = ephemeral_for(db, "sqlite", write_grants=())
        with ephemeral.open() as connection:
            assert connection.execute(text("SELECT x FROM t")).scalar() == 7

    def test_refusal_class_is_a_permission_error(self) -> None:
        assert issubclass(EphemeralOpenRefusedError, PermissionError)
