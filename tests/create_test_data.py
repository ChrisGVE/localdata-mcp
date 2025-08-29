#!/usr/bin/env python3
"""Script to create test data files for spreadsheet format testing."""

import os
import pandas as pd
import json
import configparser
from pathlib import Path
import xml.etree.ElementTree as ET

# Test data directory
DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

# Sample data for testing
sample_data = {
    'Name': ['Alice Johnson', 'Bob Smith', 'Carol Williams', 'David Brown', 'Eve Davis'],
    'Age': [28, 35, 42, 31, 26],
    'Department': ['Engineering', 'Sales', 'Marketing', 'Engineering', 'HR'],
    'Salary': [75000, 65000, 70000, 80000, 55000],
    'Start_Date': pd.to_datetime(['2020-01-15', '2019-03-22', '2021-07-10', '2020-11-05', '2022-02-28']),
    'Active': [True, True, False, True, True]
}

df_main = pd.DataFrame(sample_data)

# Create multi-sheet data
sheet1_data = df_main.copy()
sheet2_data = pd.DataFrame({
    'Product': ['Widget A', 'Widget B', 'Widget C', 'Gadget X', 'Gadget Y'],
    'Price': [19.99, 29.99, 39.99, 49.99, 59.99],
    'Category': ['Widgets', 'Widgets', 'Widgets', 'Gadgets', 'Gadgets'],
    'In_Stock': [True, False, True, True, False]
})
sheet3_data = pd.DataFrame({
    'Region': ['North', 'South', 'East', 'West', 'Central'],
    'Sales_Q1': [125000, 98000, 110000, 87000, 95000],
    'Sales_Q2': [132000, 105000, 118000, 92000, 101000]
})

def create_excel_files():
    """Create Excel test files."""
    print("Creating Excel files...")
    
    # Modern Excel file (.xlsx)
    xlsx_path = DATA_DIR / "test_data.xlsx"
    df_main.to_excel(xlsx_path, index=False)
    print(f"Created {xlsx_path}")
    
    # Multi-sheet Excel file
    multi_xlsx_path = DATA_DIR / "multi_sheet.xlsx"
    with pd.ExcelWriter(multi_xlsx_path, engine='openpyxl') as writer:
        sheet1_data.to_excel(writer, sheet_name='Employees', index=False)
        sheet2_data.to_excel(writer, sheet_name='Products', index=False)
        sheet3_data.to_excel(writer, sheet_name='Regional Sales', index=False)
        # Create empty sheet for testing
        pd.DataFrame().to_excel(writer, sheet_name='Empty Sheet', index=False)
    print(f"Created {multi_xlsx_path}")
    
    # Excel with special characters in sheet names
    special_xlsx_path = DATA_DIR / "special_sheets.xlsx"
    with pd.ExcelWriter(special_xlsx_path, engine='openpyxl') as writer:
        sheet1_data.to_excel(writer, sheet_name='Sheet-With-Hyphens', index=False)
        sheet2_data.to_excel(writer, sheet_name='Sheet With Spaces', index=False)
        sheet3_data.to_excel(writer, sheet_name='Sheet@#$%Special', index=False)
    print(f"Created {special_xlsx_path}")

def create_ods_files():
    """Create ODS test files."""
    print("Creating ODS files...")
    
    try:
        # Single sheet ODS
        ods_path = DATA_DIR / "test_data.ods"
        df_main.to_excel(ods_path, engine='odf', index=False)
        print(f"Created {ods_path}")
        
        # Multi-sheet ODS
        multi_ods_path = DATA_DIR / "multi_sheet.ods"
        with pd.ExcelWriter(multi_ods_path, engine='odf') as writer:
            sheet1_data.to_excel(writer, sheet_name='Employees', index=False)
            sheet2_data.to_excel(writer, sheet_name='Products', index=False)
            sheet3_data.to_excel(writer, sheet_name='Regional_Sales', index=False)
        print(f"Created {multi_ods_path}")
        
    except ImportError:
        print("Warning: odfpy not available, skipping ODS file creation")
    except Exception as e:
        print(f"Warning: Could not create ODS files: {e}")

def create_xml_files():
    """Create XML test files."""
    print("Creating XML files...")
    
    # Simple structured XML
    root = ET.Element("data")
    
    for _, row in df_main.iterrows():
        person = ET.SubElement(root, "person")
        ET.SubElement(person, "name").text = str(row['Name'])
        ET.SubElement(person, "age").text = str(row['Age'])
        ET.SubElement(person, "department").text = str(row['Department'])
        ET.SubElement(person, "salary").text = str(row['Salary'])
        ET.SubElement(person, "start_date").text = str(row['Start_Date'])
        ET.SubElement(person, "active").text = str(row['Active'])
    
    xml_path = DATA_DIR / "structured_data.xml"
    tree = ET.ElementTree(root)
    tree.write(xml_path, encoding='utf-8', xml_declaration=True)
    print(f"Created {xml_path}")

def create_ini_files():
    """Create INI test files."""
    print("Creating INI files...")
    
    config = configparser.ConfigParser()
    
    config['database'] = {
        'host': 'localhost',
        'port': '5432',
        'name': 'testdb',
        'user': 'admin'
    }
    
    config['cache'] = {
        'enabled': 'true',
        'ttl': '3600',
        'max_size': '1000'
    }
    
    config['logging'] = {
        'level': 'INFO',
        'file': 'app.log',
        'max_files': '5'
    }
    
    ini_path = DATA_DIR / "config.ini"
    with open(ini_path, 'w') as f:
        config.write(f)
    print(f"Created {ini_path}")

def create_tsv_files():
    """Create TSV test files."""
    print("Creating TSV files...")
    
    tsv_path = DATA_DIR / "test_data.tsv"
    df_main.to_csv(tsv_path, sep='\t', index=False)
    print(f"Created {tsv_path}")

def create_analytical_format_files():
    """Create Parquet, Feather, and Arrow files."""
    print("Creating analytical format files...")
    
    try:
        # Parquet file
        parquet_path = DATA_DIR / "analytical_data.parquet"
        df_main.to_parquet(parquet_path, engine='pyarrow')
        print(f"Created {parquet_path}")
        
        # Feather file
        feather_path = DATA_DIR / "analytical_data.feather"
        df_main.to_feather(feather_path)
        print(f"Created {feather_path}")
        
        # Arrow IPC file (same format as Feather in pandas)
        arrow_path = DATA_DIR / "analytical_data.arrow"
        df_main.to_feather(arrow_path)
        print(f"Created {arrow_path}")
        
    except ImportError:
        print("Warning: pyarrow not available, skipping analytical format file creation")
    except Exception as e:
        print(f"Warning: Could not create analytical format files: {e}")

def create_edge_case_files():
    """Create files for edge case testing."""
    print("Creating edge case files...")
    
    # Empty CSV for testing empty files
    empty_csv_path = DATA_DIR / "empty.csv"
    pd.DataFrame().to_csv(empty_csv_path, index=False)
    print(f"Created {empty_csv_path}")
    
    # CSV with special characters
    special_data = pd.DataFrame({
        'Column With Spaces': ['Value 1', 'Value 2'],
        'Column-With-Hyphens': ['Data A', 'Data B'],
        'Column@Special#Chars': ['Test 1', 'Test 2']
    })
    special_csv_path = DATA_DIR / "special_characters.csv"
    special_data.to_csv(special_csv_path, index=False)
    print(f"Created {special_csv_path}")
    
    # Large dataset for performance testing (but not too large)
    large_data = pd.DataFrame({
        'ID': range(10000),
        'Value': [f'Item_{i}' for i in range(10000)],
        'Number': [i * 1.5 for i in range(10000)],
        'Category': [f'Category_{i % 10}' for i in range(10000)]
    })
    large_csv_path = DATA_DIR / "large_dataset.csv"
    large_data.to_csv(large_csv_path, index=False)
    print(f"Created {large_csv_path}")

def create_corrupted_files():
    """Create corrupted files for security testing."""
    print("Creating corrupted files...")
    
    # Corrupted Excel file (invalid content)
    corrupted_xlsx_path = DATA_DIR / "corrupted.xlsx"
    with open(corrupted_xlsx_path, 'w') as f:
        f.write("This is not a valid Excel file content")
    print(f"Created {corrupted_xlsx_path}")
    
    # Malicious XML with external entity (for XXE testing)
    malicious_xml_path = DATA_DIR / "malicious.xml"
    malicious_xml_content = '''<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE data [
    <!ENTITY xxe SYSTEM "file:///etc/passwd">
]>
<data>
    <record>
        <name>&xxe;</name>
        <value>test</value>
    </record>
</data>'''
    with open(malicious_xml_path, 'w') as f:
        f.write(malicious_xml_content)
    print(f"Created {malicious_xml_path}")

def main():
    """Create all test data files."""
    print(f"Creating test data files in {DATA_DIR}")
    
    create_excel_files()
    create_ods_files()
    create_xml_files()
    create_ini_files()
    create_tsv_files()
    create_analytical_format_files()
    create_edge_case_files()
    create_corrupted_files()
    
    print(f"\nTest data files created successfully!")
    print(f"Files created in: {DATA_DIR}")
    
    # List all created files
    files = list(DATA_DIR.glob("*"))
    files.sort()
    print("\nCreated files:")
    for file in files:
        size = file.stat().st_size
        print(f"  {file.name} ({size} bytes)")

if __name__ == "__main__":
    main()