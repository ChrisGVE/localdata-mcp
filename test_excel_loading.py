#!/usr/bin/env python3
"""Test Excel loading functionality directly."""

import sys
sys.path.insert(0, 'src')

from localdata_mcp.localdata_mcp import DatabaseManager

def test_excel_loading():
    """Test Excel loading functionality."""
    
    manager = DatabaseManager()
    
    try:
        # Test multi-sheet Excel loading
        print("Testing multi-sheet Excel loading...")
        sheets_data = manager._load_excel_file("simple_test.xlsx")
        
        print(f"Loaded {len(sheets_data)} sheets:")
        for table_name, df in sheets_data.items():
            print(f"  - Table '{table_name}': {len(df)} rows, {len(df.columns)} columns")
            print(f"    Columns: {list(df.columns)}")
            print(f"    Sample data: {df.head(2).to_dict('records')}")
            print()
        
        # Test single-sheet Excel loading  
        print("Testing single-sheet Excel loading...")
        single_sheets_data = manager._load_excel_file("single_sheet.xlsx")
        
        print(f"Loaded {len(single_sheets_data)} sheets:")
        for table_name, df in single_sheets_data.items():
            print(f"  - Table '{table_name}': {len(df)} rows, {len(df.columns)} columns")
            print(f"    Columns: {list(df.columns)}")
            print()
            
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_excel_loading()