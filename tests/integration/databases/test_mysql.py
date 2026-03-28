"""MySQL integration tests.

Requires:
  - docker-compose.test.yml mysql container running, OR
  - TEST_MYSQL_URL environment variable pointing at an accessible instance.
"""

import os

import pytest

pytestmark = [pytest.mark.integration]

_SKIP_REASON = (
    "No MySQL container available (set TEST_MYSQL_URL or start docker-compose.test.yml)"
)


def _mysql_available() -> bool:
    """Return True when the MySQL test container is reachable."""
    url = os.environ.get(
        "TEST_MYSQL_URL",
        "mysql+mysqlconnector://testuser:testpass@localhost:13306/testdb",
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


@pytest.mark.skipif(not _mysql_available(), reason=_SKIP_REASON)
class TestMySQLIntegration:
    """Basic smoke tests for the MySQL connector."""

    def test_connect(self, mysql_url):
        """Verify connection to MySQL."""
        from sqlalchemy import create_engine, text

        engine = create_engine(mysql_url)
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            assert result.scalar() == 1
        engine.dispose()

    def test_create_and_query_table(self, mysql_url):
        """Create a table, insert data, and read it back."""
        from sqlalchemy import create_engine, text

        engine = create_engine(mysql_url)
        with engine.connect() as conn:
            conn.execute(text("DROP TABLE IF EXISTS _ld_test"))
            conn.execute(
                text(
                    "CREATE TABLE _ld_test (id INT AUTO_INCREMENT PRIMARY KEY, val VARCHAR(255))"
                )
            )
            conn.execute(text("INSERT INTO _ld_test (val) VALUES ('hello')"))
            conn.commit()
            row = conn.execute(text("SELECT val FROM _ld_test")).fetchone()
            assert row[0] == "hello"
            conn.execute(text("DROP TABLE _ld_test"))
            conn.commit()
        engine.dispose()
