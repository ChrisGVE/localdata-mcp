#!/usr/bin/env python3
"""Test complete database integration with multi-sheet Excel files."""

import sys
sys.path.insert(0, 'src')
import json

from localdata_mcp.localdata_mcp import DatabaseManager

def test_database_integration():
    """Test complete database integration."""
    
    manager = DatabaseManager()
    
    try:
        # Test connecting to multi-sheet Excel file
        print("Testing connection to multi-sheet Excel file...")
        engine = manager._create_engine_from_file("simple_test.xlsx", "excel")
        
        # Test database inspection
        from sqlalchemy import inspect
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        print(f"Tables found in database: {tables}")
        
        # Test querying each table
        import pandas as pd
        for table in tables:
            df = pd.read_sql_query(f"SELECT * FROM {table}", engine)
            print(f"\nTable '{table}':")
            print(f"  Rows: {len(df)}")
            print(f"  Columns: {list(df.columns)}")
            print(f"  Sample: {df.head(2).to_dict('records')}")
            
        # Test table metadata
        for table in tables:
            columns = inspector.get_columns(table)
            print(f"\nTable '{table}' schema:")
            for col in columns:
                print(f"  {col['name']}: {col['type']}")
                
        engine.dispose()
        print("\nDatabase integration test completed successfully!")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_database_integration()