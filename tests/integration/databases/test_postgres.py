"""PostgreSQL integration tests.

Requires:
  - docker-compose.test.yml postgres container running, OR
  - TEST_POSTGRES_URL environment variable pointing at an accessible instance.
"""

import os

import pytest

pytestmark = [pytest.mark.integration]

_SKIP_REASON = "No PostgreSQL container available (set TEST_POSTGRES_URL or start docker-compose.test.yml)"


def _pg_available() -> bool:
    """Return True when the PostgreSQL test container is reachable."""
    url = os.environ.get(
        "TEST_POSTGRES_URL",
        "postgresql://testuser:testpass@localhost:15432/testdb",
    )
    try:
        from sqlalchemy import create_engine, text

        engine = create_engine(url, connect_args={"connect_timeout": 3})
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        engine.dispose()
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _pg_available(), reason=_SKIP_REASON)
class TestPostgresIntegration:
    """Basic smoke tests for the PostgreSQL connector."""

    def test_connect(self, postgres_url):
        """Verify connection to PostgreSQL."""
        from sqlalchemy import create_engine, text

        engine = create_engine(postgres_url)
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            assert result.scalar() == 1
        engine.dispose()

    def test_create_and_query_table(self, postgres_url):
        """Create a table, insert data, and read it back."""
        from sqlalchemy import create_engine, text

        engine = create_engine(postgres_url)
        with engine.connect() as conn:
            conn.execute(text("DROP TABLE IF EXISTS _ld_test"))
            conn.execute(
                text("CREATE TABLE _ld_test (id SERIAL PRIMARY KEY, val TEXT)")
            )
            conn.execute(text("INSERT INTO _ld_test (val) VALUES ('hello')"))
            conn.commit()
            row = conn.execute(text("SELECT val FROM _ld_test")).fetchone()
            assert row[0] == "hello"
            conn.execute(text("DROP TABLE _ld_test"))
            conn.commit()
        engine.dispose()
