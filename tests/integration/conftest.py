"""Shared fixtures for integration tests."""

import os

import pytest

from .data_generator import TestDataGenerator
from .database_setup import create_csv_test_file, create_sqlite_test_db

try:
    import duckdb  # noqa: F401

    _has_duckdb = True
except ImportError:
    _has_duckdb = False


def pytest_configure(config):
    config.addinivalue_line("markers", "integration: integration tests")
    config.addinivalue_line("markers", "large_data: tests with large datasets")
    config.addinivalue_line("markers", "docker: tests requiring Docker containers")
    config.addinivalue_line("markers", "postgres: PostgreSQL integration tests")
    config.addinivalue_line("markers", "mysql: MySQL integration tests")
    config.addinivalue_line("markers", "mssql: MS SQL Server integration tests")
    config.addinivalue_line("markers", "mongodb: MongoDB integration tests")
    config.addinivalue_line("markers", "redis: Redis integration tests")
    config.addinivalue_line("markers", "elasticsearch: Elasticsearch integration tests")
    config.addinivalue_line("markers", "influxdb: InfluxDB integration tests")
    config.addinivalue_line("markers", "neo4j: Neo4j integration tests")
    config.addinivalue_line("markers", "couchdb: CouchDB integration tests")


@pytest.fixture(autouse=True)
def mock_mcp_framework():
    """Override the root conftest autouse mock so integration tests use the real server."""
    yield None


@pytest.fixture(autouse=True)
def _disable_path_restriction(monkeypatch):
    """Disable path restriction for integration tests (tmp_path is outside cwd)."""
    monkeypatch.setenv("LOCALDATA_SECURITY_RESTRICT_PATHS", "false")


@pytest.fixture(scope="session")
def data_generator():
    return TestDataGenerator(seed=42)


@pytest.fixture(scope="session")
def small_sqlite_db():
    """SQLite with 1,000 rows."""
    path = create_sqlite_test_db(rows=1000)
    yield path
    os.unlink(path)


@pytest.fixture(scope="session")
def large_sqlite_db():
    """SQLite with 500,000 rows for stress testing."""
    path = create_sqlite_test_db(rows=500_000)
    yield path
    os.unlink(path)


@pytest.fixture(scope="session")
def small_duckdb_db():
    """DuckDB with 1,000 rows. Skipped if duckdb is not installed."""
    if not _has_duckdb:
        pytest.skip("duckdb not installed")
    from .database_setup import create_duckdb_test_db

    path = create_duckdb_test_db(rows=1000)
    yield path
    os.unlink(path)


@pytest.fixture(scope="session")
def large_duckdb_db():
    """DuckDB with 500,000 rows. Skipped if duckdb is not installed."""
    if not _has_duckdb:
        pytest.skip("duckdb not installed")
    from .database_setup import create_duckdb_test_db

    path = create_duckdb_test_db(rows=500_000)
    yield path
    os.unlink(path)


@pytest.fixture(scope="session")
def small_csv_file():
    """CSV with 1,000 rows."""
    path = create_csv_test_file(rows=1000)
    yield path
    os.unlink(path)


@pytest.fixture
def tmp_dir(tmp_path):
    """Temporary directory for test file creation."""
    return tmp_path


# Database URL fixtures
@pytest.fixture
def postgres_url():
    return os.environ.get(
        "TEST_POSTGRES_URL",
        "postgresql://test:test@localhost:15432/testdb",
    )


@pytest.fixture
def mysql_url():
    return os.environ.get(
        "TEST_MYSQL_URL",
        "mysql+mysqlconnector://test:test@localhost:13306/testdb",
    )


@pytest.fixture
def mssql_url():
    return os.environ.get(
        "TEST_MSSQL_URL",
        "mssql+pymssql://sa:TestPass123!@localhost:11433/master",
    )
