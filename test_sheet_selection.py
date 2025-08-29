#!/usr/bin/env python3
"""Test sheet selection functionality."""

import sys
sys.path.insert(0, 'src')

from localdata_mcp.localdata_mcp import DatabaseManager

def test_sheet_selection():
    """Test sheet selection functionality."""
    
    manager = DatabaseManager()
    
    try:
        print("Testing loading all sheets (default behavior)...")
        all_sheets_data = manager._load_excel_file("simple_test.xlsx")
        print(f"All sheets loaded: {list(all_sheets_data.keys())}")
        
        print("\nTesting loading specific sheet...")
        specific_sheet_data = manager._load_excel_file("simple_test.xlsx", "Sales & Marketing")
        print(f"Specific sheet loaded: {list(specific_sheet_data.keys())}")
        print(f"Table name: {list(specific_sheet_data.keys())[0]}")
        print(f"Data shape: {specific_sheet_data[list(specific_sheet_data.keys())[0]].shape}")
        
        print("\nTesting invalid sheet name...")
        try:
            invalid_sheet_data = manager._load_excel_file("simple_test.xlsx", "NonExistentSheet")
        except ValueError as e:
            print(f"Expected error: {e}")
            
        print("\nTesting database creation with sheet selection...")
        engine = manager._create_engine_from_file("simple_test.xlsx", "excel", "Employees")
        
        # Check tables created
        from sqlalchemy import inspect
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        print(f"Tables created with sheet selection: {tables}")
        
        engine.dispose()
        print("\nSheet selection tests completed successfully!")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_sheet_selection()