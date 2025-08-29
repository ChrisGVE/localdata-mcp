#!/usr/bin/env python3
"""Test the sheet name sanitization function."""

import sys
sys.path.insert(0, 'src')

from localdata_mcp.localdata_mcp import DatabaseManager

def test_sanitization():
    """Test the sheet name sanitization function."""
    
    manager = DatabaseManager()
    
    # Test cases
    test_cases = [
        ("Sheet1", "Sheet1"),
        ("Sales & Marketing", "Sales_Marketing"),
        ("Product-Catalog", "Product_Catalog"),
        ("Empty Sheet", "Empty_Sheet"),
        ("123-Numbers", "sheet_123_Numbers"),
        ("Special!@#$%^&*()", "Special_"),
        ("", "sheet_unnamed"),
        ("Sheet   with   spaces", "Sheet_with_spaces"),
        ("Sheet1", "Sheet1_1"),  # Duplicate test
    ]
    
    used_names = set()
    print("Testing sheet name sanitization:")
    print("-" * 50)
    
    for original, expected in test_cases:
        result = manager._sanitize_sheet_name(original, used_names)
        status = "✓" if result == expected else "✗"
        print(f"{status} '{original}' -> '{result}' (expected: '{expected}')")
        
        # For duplicate test, add the first result to used_names
        if original == "Sheet1" and len(used_names) == 0:
            used_names.add(result)
    
    print("-" * 50)
    print(f"Used names: {sorted(used_names)}")

if __name__ == "__main__":
    test_sanitization()