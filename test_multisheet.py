#!/usr/bin/env python3
"""Test script to create multi-sheet Excel files and test the multi-sheet functionality."""

import pandas as pd
import os
from pathlib import Path

def create_test_excel_file():
    """Create a test Excel file with multiple sheets."""
    
    # Create test data for different sheets
    sheet1_data = pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
        'age': [25, 30, 35, 28, 32],
        'department': ['IT', 'HR', 'Finance', 'IT', 'Marketing']
    })
    
    sheet2_data = pd.DataFrame({
        'product_id': [101, 102, 103, 104],
        'product_name': ['Laptop', 'Mouse', 'Keyboard', 'Monitor'],
        'price': [999.99, 25.50, 75.00, 299.99],
        'category': ['Electronics', 'Accessories', 'Accessories', 'Electronics']
    })
    
    sheet3_data = pd.DataFrame({
        'order_id': [1001, 1002, 1003, 1004, 1005],
        'customer_name': ['John Doe', 'Jane Smith', 'Bob Wilson', 'Alice Brown', 'Charlie Davis'],
        'order_date': pd.date_range('2024-01-01', periods=5),
        'total_amount': [1500.00, 325.50, 100.00, 299.99, 75.00]
    })
    
    # Empty sheet for testing
    empty_sheet = pd.DataFrame()
    
    # Create Excel file with multiple sheets
    test_file = Path('test_multisheet.xlsx')
    
    with pd.ExcelWriter(test_file, engine='openpyxl') as writer:
        sheet1_data.to_excel(writer, sheet_name='Employees', index=False)
        sheet2_data.to_excel(writer, sheet_name='Products & Inventory', index=False)  # Test special characters
        sheet3_data.to_excel(writer, sheet_name='Order History', index=False)  # Test spaces
        sheet2_data.to_excel(writer, sheet_name='Products & Inventory Copy', index=False)  # Test duplicate-like names
        empty_sheet.to_excel(writer, sheet_name='Empty Sheet', index=False)  # Test empty sheet
        sheet1_data.to_excel(writer, sheet_name='123-Numbers-Start', index=False)  # Test numbers at start
    
    print(f"Created test Excel file: {test_file.absolute()}")
    return test_file

def create_test_ods_file():
    """Create a test ODS file with multiple sheets."""
    
    # Create test data for different sheets
    sheet1_data = pd.DataFrame({
        'country': ['USA', 'Canada', 'Mexico', 'Brazil'],
        'capital': ['Washington D.C.', 'Ottawa', 'Mexico City', 'Brasília'],
        'population_millions': [331, 38, 128, 213],
        'continent': ['North America', 'North America', 'North America', 'South America']
    })
    
    sheet2_data = pd.DataFrame({
        'language': ['English', 'Spanish', 'French', 'German', 'Italian'],
        'speakers_millions': [1500, 500, 280, 100, 65],
        'family': ['Germanic', 'Romance', 'Romance', 'Germanic', 'Romance']
    })
    
    # Create ODS file with multiple sheets
    test_file = Path('test_multisheet.ods')
    
    try:
        with pd.ExcelWriter(test_file, engine='odf') as writer:
            sheet1_data.to_excel(writer, sheet_name='Countries', index=False)
            sheet2_data.to_excel(writer, sheet_name='Languages & Speakers', index=False)  # Test special characters
            sheet1_data.to_excel(writer, sheet_name='Additional Data', index=False)  # Test spaces
        
        print(f"Created test ODS file: {test_file.absolute()}")
        return test_file
    except ImportError:
        print("ODS support not available - odfpy not installed")
        return None

if __name__ == "__main__":
    # Create test files
    excel_file = create_test_excel_file()
    ods_file = create_test_ods_file()
    
    print("\nTest files created successfully!")
    print("You can now test the multi-sheet functionality with these files.")