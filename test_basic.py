#!/usr/bin/env python3
"""Basic test of the database functionality."""

import sys
import os
sys.path.insert(0, 'src')

from localdata_mcp.localdata_mcp import DatabaseManager

def test_basic_functionality():
    """Test basic CSV loading functionality."""
    
    manager = DatabaseManager()
    
    # Test CSV connection
    result = manager.connect_database.__wrapped__(manager, "test_db", "csv", "simple_test.csv")
    print(f"Connect result: {result}")
    
    # Test describe database
    db_info = manager.describe_database.__wrapped__(manager, "test_db")
    print(f"\nDatabase info: {db_info}")
    
    # Test query
    query_result = manager.execute_query.__wrapped__(manager, "test_db", "SELECT * FROM data_table LIMIT 3")
    print(f"\nQuery result: {query_result}")
    
    # Clean up
    manager.disconnect_database.__wrapped__(manager, "test_db")
    print("\nBasic test completed successfully!")

if __name__ == "__main__":
    test_basic_functionality()