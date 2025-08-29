#!/usr/bin/env python3
"""Test database tools integration with multi-sheet Excel files."""

import sys
sys.path.insert(0, 'src')
import json

from localdata_mcp.localdata_mcp import DatabaseManager

def test_database_tools():
    """Test all database tools with multi-sheet Excel files."""
    
    manager = DatabaseManager()
    
    try:
        print("=== Testing Multi-Sheet Database Tools Integration ===\n")
        
        # Test 1: Connect to multi-sheet Excel file
        print("1. Testing connection to multi-sheet Excel file...")
        engine = manager._create_engine_from_file("simple_test.xlsx", "excel")
        manager.connections["test_excel"] = engine
        manager.query_history["test_excel"] = []
        print("   ✓ Connection successful")
        
        # Test 2: Inspect database structure 
        print("\n2. Testing database inspection...")
        from sqlalchemy import inspect
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        print(f"   Tables found: {tables}")
        expected_tables = ['Employees', 'Product_Catalog', 'Sales_Marketing']
        assert set(tables) == set(expected_tables), f"Expected {expected_tables}, got {tables}"
        print("   ✓ All expected tables found")
        
        # Test 3: Table metadata for each sheet
        print("\n3. Testing table metadata for each sheet...")
        for table in tables:
            table_info = manager._get_table_metadata(inspector, table)
            print(f"   Table '{table}':")
            print(f"     Columns: {[col['name'] for col in table_info['columns']]}")
            print(f"     Row count: {table_info.get('size', 'unknown')}")
            assert len(table_info['columns']) > 0, f"Table {table} should have columns"
        print("   ✓ All table metadata retrieved successfully")
        
        # Test 4: Query each table 
        print("\n4. Testing queries on each sheet table...")
        import pandas as pd
        for table in tables:
            df = pd.read_sql_query(f"SELECT * FROM {table} LIMIT 2", engine)
            print(f"   Table '{table}': {len(df)} rows sample retrieved")
            assert len(df) > 0, f"Table {table} should have data"
        print("   ✓ All tables queryable")
        
        # Test 5: Sheet-specific connection
        print("\n5. Testing connection to specific sheet...")
        specific_engine = manager._create_engine_from_file("simple_test.xlsx", "excel", "Sales & Marketing")
        specific_inspector = inspect(specific_engine)
        specific_tables = specific_inspector.get_table_names()
        print(f"   Tables from specific sheet connection: {specific_tables}")
        assert specific_tables == ['Sales_Marketing'], f"Expected ['Sales_Marketing'], got {specific_tables}"
        print("   ✓ Sheet-specific connection working")
        
        # Test 6: SQL identifier safety
        print("\n6. Testing SQL identifier safety...")
        for table in tables:
            # Test that table names are SQL-safe
            safe_name = manager._safe_table_identifier(table)
            print(f"   Table '{table}' -> Safe identifier: {safe_name}")
            # Should not raise an exception
        print("   ✓ All table names are SQL-safe")
        
        # Test 7: Integration with existing tools would work
        print("\n7. Testing integration with database tools...")
        
        # Simulate describe_database
        db_info = {
            "name": "test_excel",
            "tables": []
        }
        for table_name in tables:
            table_info = manager._get_table_metadata(inspector, table_name)
            # Add row count like describe_database does
            with engine.connect() as conn:
                safe_table_name = manager._safe_table_identifier(table_name)
                from sqlalchemy import text
                result = conn.execute(text(f"SELECT COUNT(*) FROM {safe_table_name}"))
                row_count = result.scalar()
            table_info["size"] = row_count
            db_info["tables"].append(table_info)
            
        print(f"   Database info structure: {len(db_info['tables'])} tables with metadata")
        for table in db_info['tables']:
            print(f"     {table['name']}: {table['size']} rows, {len(table['columns'])} columns")
        print("   ✓ Database tools integration ready")
        
        # Cleanup
        engine.dispose()
        specific_engine.dispose()
        
        print("\n=== All Multi-Sheet Database Tools Tests Passed! ===")
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = test_database_tools()
    sys.exit(0 if success else 1)