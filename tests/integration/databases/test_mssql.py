"""MS SQL Server integration tests.

Requires:
  - docker-compose.test.yml mssql container running, OR
  - TEST_MSSQL_URL environment variable pointing at an accessible instance.
"""

import os

import pytest

pytestmark = [pytest.mark.integration]

_SKIP_REASON = "No MS SQL Server container available (set TEST_MSSQL_URL or start docker-compose.test.yml)"


def _mssql_available() -> bool:
    """Return True when the MS SQL Server test container is reachable."""
    url = os.environ.get(
        "TEST_MSSQL_URL",
        "mssql+pymssql://sa:TestPass123!@localhost:11433/master",
    )
    try:
        from sqlalchemy import create_engine, text

        engine = create_engine(url, connect_args={"login_timeout": 3})
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        engine.dispose()
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _mssql_available(), reason=_SKIP_REASON)
class TestMSSQLIntegration:
    """Basic smoke tests for the MS SQL Server connector."""

    def test_connect(self, mssql_url):
        """Verify connection to MS SQL Server."""
        from sqlalchemy import create_engine, text

        engine = create_engine(mssql_url)
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            assert result.scalar() == 1
        engine.dispose()

    def test_create_and_query_table(self, mssql_url):
        """Create a table, insert data, and read it back."""
        from sqlalchemy import create_engine, text

        engine = create_engine(mssql_url)
        with engine.connect() as conn:
            conn.execute(
                text("IF OBJECT_ID('_ld_test', 'U') IS NOT NULL DROP TABLE _ld_test")
            )
            conn.execute(
                text(
                    "CREATE TABLE _ld_test (id INT IDENTITY PRIMARY KEY, val NVARCHAR(255))"
                )
            )
            conn.execute(text("INSERT INTO _ld_test (val) VALUES ('hello')"))
            conn.commit()
            row = conn.execute(text("SELECT val FROM _ld_test")).fetchone()
            assert row[0] == "hello"
            conn.execute(text("DROP TABLE _ld_test"))
            conn.commit()
        engine.dispose()
