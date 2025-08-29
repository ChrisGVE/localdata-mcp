#!/usr/bin/env python3
"""Create a simple test Excel file with multiple sheets."""

import pandas as pd
from pathlib import Path

def create_simple_excel():
    """Create a simple Excel file with multiple sheets."""
    
    # Sheet 1: Employee data
    employees = pd.DataFrame({
        'id': [1, 2, 3],
        'name': ['Alice', 'Bob', 'Charlie'],
        'department': ['IT', 'HR', 'Finance']
    })
    
    # Sheet 2: Sales data  
    sales = pd.DataFrame({
        'month': ['Jan', 'Feb', 'Mar'],
        'revenue': [10000, 12000, 11000],
        'expenses': [5000, 6000, 5500]
    })
    
    # Sheet 3: With special characters in name
    products = pd.DataFrame({
        'product': ['Laptop', 'Mouse'],
        'price': [999, 25]
    })
    
    # Create Excel file
    excel_path = Path('simple_test.xlsx')
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        employees.to_excel(writer, sheet_name='Employees', index=False)
        sales.to_excel(writer, sheet_name='Sales & Marketing', index=False)  # Special characters
        products.to_excel(writer, sheet_name='Product-Catalog', index=False)  # Hyphen
        # Empty sheet
        pd.DataFrame().to_excel(writer, sheet_name='Empty Sheet', index=False)
    
    print(f"Created Excel file: {excel_path.absolute()}")
    
    # Also create a single sheet for comparison
    single_path = Path('single_sheet.xlsx')
    with pd.ExcelWriter(single_path, engine='openpyxl') as writer:
        employees.to_excel(writer, sheet_name='Sheet1', index=False)
    
    print(f"Created single-sheet Excel file: {single_path.absolute()}")

if __name__ == "__main__":
    create_simple_excel()