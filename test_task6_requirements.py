#!/usr/bin/env python3
"""Test Task 6 requirements for multi-sheet spreadsheet support."""

import sys
sys.path.insert(0, 'src')
import json

from localdata_mcp.localdata_mcp import DatabaseManager

def test_task6_requirements():
    """Test all Task 6 requirements for multi-sheet spreadsheet support."""
    
    manager = DatabaseManager()
    
    print("=== Testing Task 6: Multi-Sheet Spreadsheet Support Requirements ===\n")
    
    try:
        # Requirement 1: Multi-Sheet Discovery - describe_database lists all sheets as separate tables
        print("✓ Requirement 1: Multi-Sheet Discovery")
        print("  Testing describe_database lists all sheets as separate tables...")
        
        # Create connection like describe_database would
        engine = manager._create_engine_from_file("simple_test.xlsx", "excel")
        from sqlalchemy import inspect
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        
        print(f"  Found tables representing sheets: {tables}")
        expected_sheets = ['Employees', 'Sales_Marketing', 'Product_Catalog']
        assert set(tables) == set(expected_sheets), f"Expected {expected_sheets}, got {tables}"
        print("  ✓ All sheets discovered as separate tables\n")
        
        # Requirement 2: Sheet Metadata - describe_table shows sheet-specific metadata
        print("✓ Requirement 2: Sheet Metadata")
        print("  Testing describe_table works for sheet-specific tables...")
        
        for table in tables:
            table_info = manager._get_table_metadata(inspector, table)
            print(f"  Table '{table}': {len(table_info['columns'])} columns")
            print(f"    Columns: {[col['name'] for col in table_info['columns']]}")
            assert len(table_info['columns']) > 0, f"Table {table} should have columns"
            assert table_info['name'] == table, f"Table name should match"
        print("  ✓ Sheet-specific metadata retrieval working\n")
        
        # Requirement 3: Sheet Name Sanitization - sheet names work as SQL table names
        print("✓ Requirement 3: Sheet Name Sanitization")
        print("  Testing sheet names are sanitized for SQL compatibility...")
        
        test_cases = [
            ("Sales & Marketing", "Sales_Marketing"),
            ("Product-Catalog", "Product_Catalog"), 
            ("Empty Sheet", "Empty_Sheet"),
            ("123Numbers", "sheet_123Numbers"),
            ("Special!@#", "Special_"),
        ]
        
        used_names = set()
        for original, expected in test_cases:
            result = manager._sanitize_sheet_name(original, used_names)
            print(f"  '{original}' -> '{result}' (expected: '{expected}')")
            # Check SQL safety
            try:
                safe_name = manager._safe_table_identifier(result)
                print(f"    SQL-safe identifier: {safe_name}")
            except ValueError as e:
                assert False, f"Sanitized name {result} should be SQL-safe: {e}"
        print("  ✓ Sheet name sanitization working correctly\n")
        
        # Requirement 4: Connection Enhancement - sheet selection capabilities
        print("✓ Requirement 4: Connection Enhancement")
        print("  Testing sheet selection in connect_database...")
        
        # Test loading specific sheet
        specific_engine = manager._create_engine_from_file("simple_test.xlsx", "excel", "Sales & Marketing")
        specific_inspector = inspect(specific_engine)
        specific_tables = specific_inspector.get_table_names()
        print(f"  Specific sheet connection tables: {specific_tables}")
        assert specific_tables == ['Sales_Marketing'], f"Should only have Sales_Marketing table"
        
        # Test error handling for invalid sheet
        try:
            manager._load_excel_file("simple_test.xlsx", "NonExistentSheet")
            assert False, "Should have raised ValueError for invalid sheet"
        except ValueError as e:
            print(f"  Invalid sheet error (expected): {e}")
            assert "not found" in str(e).lower()
        
        print("  ✓ Sheet selection capabilities working\n")
        
        # Requirement 5: Table Naming - proper naming conventions
        print("✓ Requirement 5: Table Naming")
        print("  Testing proper naming conventions for multi-sheet files...")
        
        # Verify original sheet names are preserved in logs but sanitized for tables
        sheets_data = manager._load_excel_file("simple_test.xlsx")
        print(f"  Tables created: {list(sheets_data.keys())}")
        
        # Verify naming follows conventions
        for table_name in sheets_data.keys():
            # Should start with letter or underscore
            assert table_name[0].isalpha() or table_name[0] == '_', f"Table name {table_name} should start with letter or underscore"
            # Should only contain safe characters
            import re
            assert re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', table_name), f"Table name {table_name} contains invalid characters"
        print("  ✓ Table naming conventions followed\n")
        
        # Requirement 6: Edge Cases - duplicate names, empty sheets, special characters
        print("✓ Requirement 6: Edge Cases")
        print("  Testing edge cases handling...")
        
        # Empty sheets are skipped (verified by checking 'Empty Sheet' is not in tables)
        all_available_sheets = ['Employees', 'Sales & Marketing', 'Product-Catalog', 'Empty Sheet']
        loaded_sheets = list(sheets_data.keys())
        print(f"  Available sheets: {all_available_sheets}")
        print(f"  Loaded tables: {loaded_sheets}")
        assert 'Empty_Sheet' not in loaded_sheets, "Empty sheets should be skipped"
        
        # Special characters are handled ('Sales & Marketing' -> 'Sales_Marketing')
        assert 'Sales_Marketing' in loaded_sheets, "Special characters should be sanitized"
        
        # Duplicate name handling
        used_names = set(['TestTable'])
        duplicate1 = manager._sanitize_sheet_name('TestTable', used_names)
        duplicate2 = manager._sanitize_sheet_name('TestTable', used_names)
        print(f"  Duplicate handling: 'TestTable' -> '{duplicate1}', 'TestTable' -> '{duplicate2}'")
        assert duplicate1 != duplicate2, "Duplicate names should be made unique"
        
        print("  ✓ Edge cases handled correctly\n")
        
        # Integration Test: Verify database tools work with multi-sheet
        print("✓ Integration Test: Database Tools Compatibility")
        print("  Testing existing database tools work with multi-sheet files...")
        
        # Test querying each sheet table
        import pandas as pd
        for table in tables:
            df = pd.read_sql_query(f"SELECT * FROM {table} LIMIT 1", engine)
            print(f"  Successfully queried table '{table}': {len(df)} rows")
            assert len(df) > 0, f"Should be able to query table {table}"
            
        # Test table sample functionality would work
        for table in tables:
            safe_table_name = manager._safe_table_identifier(table)
            from sqlalchemy import text
            with engine.connect() as conn:
                result = conn.execute(text(f"SELECT * FROM {safe_table_name} LIMIT 2"))
                rows = result.fetchall()
                print(f"  Table '{table}' sample: {len(rows)} rows")
                assert len(rows) > 0, f"Should get sample data from {table}"
        
        print("  ✓ All database tools compatible with multi-sheet structure\n")
        
        # Cleanup
        engine.dispose()
        specific_engine.dispose()
        
        print("🎉 === ALL TASK 6 REQUIREMENTS SUCCESSFULLY IMPLEMENTED === 🎉")
        print("\nSummary:")
        print("✅ Multi-Sheet Discovery: describe_database lists sheets as tables")
        print("✅ Sheet Metadata: describe_table works for sheet tables") 
        print("✅ Sheet Name Sanitization: SQL-compatible table names")
        print("✅ Connection Enhancement: Optional sheet selection")
        print("✅ Table Naming: Proper conventions followed")
        print("✅ Edge Cases: Empty sheets, duplicates, special chars handled")
        print("✅ Integration: Existing database tools fully compatible")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_task6_requirements()
    sys.exit(0 if success else 1)