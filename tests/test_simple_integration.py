"""
Simple integration test to validate basic functionality works before complex tests.
"""

import json
import pytest
from unittest.mock import patch, Mock
from tests.test_helpers import create_test_manager
import pandas as pd


def test_basic_list_databases():
    """Test that list_databases works correctly."""
    manager = create_test_manager()
    result = manager.list_databases()
    
    # Should return JSON
    response = json.loads(result)
    assert response["message"] == "No databases are currently connected."
    assert response["databases"] == []


def test_connect_database_with_mocked_files():
    """Test database connection with properly mocked file operations."""
    from tests.mock_helpers import mock_csv_connection
    
    manager = create_test_manager()
    
    # Use the mock helper for comprehensive mocking
    with mock_csv_connection():
        # Test connection
        result = manager.connect_database("test_csv", "csv", "/tmp/test.csv")
        
        # Verify it returns proper JSON
        response = json.loads(result)
        assert response["success"] is True
        assert response["message"] == "Successfully connected to database 'test_csv'"
        assert response["connection_info"]["name"] == "test_csv"
        assert response["connection_info"]["db_type"] == "csv"


def test_execute_query_basic():
    """Test execute query with mocked connection."""
    from tests.mock_helpers import mock_csv_connection, mock_database_query
    
    manager = create_test_manager()
    
    # First establish a connection using mocks
    with mock_csv_connection():
        # Connect database
        manager.connect_database("test_csv", "csv", "/tmp/test.csv")
        
        # Now test query execution
        query_result_df = pd.DataFrame({
            'id': [1, 2],
            'name': ['Alice', 'Bob']
        })
        
        with mock_database_query(query_result_df):
            result = manager.execute_query("test_csv", "SELECT * FROM data_table LIMIT 2")
            
            # Parse result
            response = json.loads(result)
            assert "data" in response
            assert len(response["data"]) == 2
            assert response["data"][0]["name"] == "Alice"


if __name__ == "__main__":
    pytest.main([__file__])