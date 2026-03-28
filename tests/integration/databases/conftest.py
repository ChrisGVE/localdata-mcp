"""Shared fixtures for database integration tests.

Default URLs point at the containers defined in docker-compose.test.yml.
Override via environment variables for custom setups or CI service containers.
"""

import os

import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "integration: marks tests as integration tests (deselect with '-m \"not integration\"')",
    )


@pytest.fixture
def postgres_url():
    """Connection URL for the PostgreSQL test container."""
    return os.environ.get(
        "TEST_POSTGRES_URL",
        "postgresql://testuser:testpass@localhost:15432/testdb",
    )


@pytest.fixture
def mysql_url():
    """Connection URL for the MySQL test container."""
    return os.environ.get(
        "TEST_MYSQL_URL",
        "mysql+mysqlconnector://testuser:testpass@localhost:13306/testdb",
    )


@pytest.fixture
def mssql_url():
    """Connection URL for the MS SQL Server test container."""
    return os.environ.get(
        "TEST_MSSQL_URL",
        "mssql+pymssql://sa:TestPass123!@localhost:11433/master",
    )
