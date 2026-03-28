"""Shared fixtures for integration tests."""

import os

import pytest

from .data_generator import TestDataGenerator
from .database_setup import create_csv_test_file, create_sqlite_test_db


def pytest_configure(config):
    config.addinivalue_line("markers", "integration: integration tests")
    config.addinivalue_line("markers", "large_data: tests with large datasets")
    config.addinivalue_line("markers", "docker: tests requiring Docker containers")


@pytest.fixture(autouse=True)
def mock_mcp_framework():
    """Override the root conftest autouse mock so integration tests use the real server."""
    yield None


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
