"""Integration tests for MS SQL Server support.

These tests require a running SQL Server instance (e.g., via Docker) and are
skipped by default.  Run with ``pytest -m integration`` after starting:

    docker run -e ACCEPT_EULA=Y -e SA_PASSWORD='YourStrong!Passw0rd' \
        -p 1433:1433 -d mcr.microsoft.com/mssql/server:2022-latest
"""

import pytest

pytestmark = pytest.mark.integration


@pytest.mark.skip(reason="Requires SQL Server Docker container")
class TestMSSQLIntegration:
    """End-to-end tests against a live SQL Server instance."""

    def test_connect_and_query(self):
        """Integration test: connect to SQL Server and execute a query."""
        pass

    def test_error_classification(self):
        """Integration test: verify error classification with real errors."""
        pass
