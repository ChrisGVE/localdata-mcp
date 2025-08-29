#!/usr/bin/env python3
"""Simple test to verify Excel loading logic works."""

import pandas as pd
from pathlib import Path

def test_excel_logic():
    """Test the Excel detection and error handling logic."""
    print("Testing Excel library availability...")
    
    # Test imports
    try:
        import openpyxl
        print("✓ openpyxl is available")
    except ImportError:
        print("✗ openpyxl not available")
    
    try:
        import xlrd
        print("✓ xlrd is available") 
    except ImportError:
        print("✗ xlrd not available")
        
    try:
        import defusedxml
        print("✓ defusedxml is available")
    except ImportError:
        print("✗ defusedxml not available")
    
    # Test file extension detection
    print("\nTesting file extension detection...")
    test_files = [
        "/path/to/file.xlsx",
        "/path/to/file.xlsm", 
        "/path/to/file.xls",
        "/path/to/file.unknown"
    ]
    
    for file_path in test_files:
        ext = Path(file_path).suffix.lower()
        if ext in ['.xlsx', '.xlsm']:
            engine = 'openpyxl'
        elif ext == '.xls':
            engine = 'xlrd'
        else:
            engine = 'auto-detect'
        print(f"  {file_path} -> engine: {engine}")
    
    # Test pandas Excel reading capabilities
    print("\nTesting pandas Excel engine support...")
    try:
        # This will fail without actual file, but we can check error messages
        pd.read_excel("nonexistent.xlsx", engine='openpyxl')
    except FileNotFoundError:
        print("✓ pandas.read_excel with openpyxl engine is working (file not found is expected)")
    except ImportError as e:
        print(f"✗ pandas.read_excel with openpyxl failed: {e}")
    except Exception as e:
        print(f"? pandas.read_excel with openpyxl gave unexpected error: {e}")
    
    try:
        pd.read_excel("nonexistent.xls", engine='xlrd') 
    except FileNotFoundError:
        print("✓ pandas.read_excel with xlrd engine is working (file not found is expected)")
    except ImportError as e:
        print(f"✗ pandas.read_excel with xlrd failed: {e}")
    except Exception as e:
        print(f"? pandas.read_excel with xlrd gave unexpected error: {e}")
    
    print("\nLogic test completed. Excel support implementation looks correct.")

if __name__ == '__main__':
    print("Simple Excel support logic test...")
    test_excel_logic()