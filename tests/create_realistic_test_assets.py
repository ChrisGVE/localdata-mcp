#!/usr/bin/env python3
"""Create comprehensive realistic test assets for edge case testing."""

import json
import os
import tempfile
import pandas as pd
import numpy as np
import yaml
import configparser
from pathlib import Path
from typing import Dict, Any
import xml.etree.ElementTree as ET

# Try to import optional dependencies
try:
    import toml
    TOML_AVAILABLE = True
except ImportError:
    TOML_AVAILABLE = False

try:
    import openpyxl
    OPENPYXL_AVAILABLE = True
except ImportError:
    OPENPYXL_AVAILABLE = False


def create_messy_csv_data():
    """Create CSV files with real-world edge cases."""
    data_dir = Path(__file__).parent / "assets"
    data_dir.mkdir(exist_ok=True)
    
    # 1. Mixed data types CSV with encoding issues
    messy_data = [
        ['id', 'name', 'value', 'date', 'notes'],
        [1, 'John Doe', '123.45', '2023-01-15', 'Normal entry'],
        [2, 'María García', '456abc', '2023/02/20', 'Contains unicode'],
        ['3a', 'Bob Smith', '', '15-Mar-2023', 'Empty value cell'],
        [4, 'User "Special"', 'N/A', 'invalid-date', 'Contains quotes'],
        [5, 'Line\nBreak', '999.99', '2023-04-01T10:30:00', 'Line break in data'],
        ['', '', '', '', ''],  # Empty row
        [6, 'Final,Entry', '123,456.78', '2023-05-15', 'Contains commas'],
        [7, '🚀 Emoji User', '-123.45', '2023-06-01', 'Unicode emoji'],
    ]
    
    with open(data_dir / "messy_mixed_types.csv", "w", encoding="utf-8") as f:
        for row in messy_data:
            f.write(",".join([f'"{str(item)}"' if "," in str(item) or '"' in str(item) else str(item) for item in row]) + "\n")
    
    # 2. CSV with no headers
    no_header_data = [
        [1, 'Product A', 99.99, 'Electronics'],
        [2, 'Product B', 149.50, 'Books'],
        [3, 'Product C', 'invalid_price', 'Clothing'],
        [4, 'Product D', 75.25, ''],
    ]
    
    with open(data_dir / "no_header.csv", "w", encoding="utf-8") as f:
        for row in no_header_data:
            f.write(",".join([str(item) for item in row]) + "\n")
    
    # 3. CSV with encoding issues (Latin-1)
    latin1_data = "id,name,description\n1,Café,Français\n2,Niño,Español\n3,Müller,Deutsch\n"
    with open(data_dir / "latin1_encoded.csv", "w", encoding="latin-1") as f:
        f.write(latin1_data)
    
    # 4. Large CSV for memory testing
    large_rows = []
    large_rows.append(['id', 'category', 'value', 'description', 'timestamp'])
    for i in range(1000):  # Moderate size for CI testing
        large_rows.append([
            i, f'category_{i % 10}', 
            round(np.random.normal(100, 20), 2),
            f'Description for item {i} with some text content',
            f'2023-{(i % 12) + 1:02d}-{(i % 28) + 1:02d}'
        ])
    
    with open(data_dir / "large_dataset.csv", "w", encoding="utf-8") as f:
        for row in large_rows:
            f.write(",".join([str(item) for item in row]) + "\n")
    
    print(f"✓ Created CSV test assets in {data_dir}")


def create_excel_test_files():
    """Create Excel files with edge cases."""
    if not OPENPYXL_AVAILABLE:
        print("⚠️  Skipping Excel files - openpyxl not available")
        return
    
    data_dir = Path(__file__).parent / "assets"
    data_dir.mkdir(exist_ok=True)
    
    # Multi-sheet Excel with edge cases
    with pd.ExcelWriter(data_dir / "multi_sheet_messy.xlsx", engine='openpyxl') as writer:
        # Sheet 1: Mixed data types
        df1 = pd.DataFrame({
            'id': [1, 2, '3a', 4, None],
            'name': ['Alice', 'Bob & Co.', 'Charlie-Smith', '', 'David🚀'],
            'value': [123.45, 'not_a_number', None, 0, -456.78],
            'active': [True, False, 'maybe', '', 1]
        })
        df1.to_excel(writer, sheet_name='Mixed Data', index=False)
        
        # Sheet 2: Dates and times in various formats
        df2 = pd.DataFrame({
            'date_iso': ['2023-01-15', '2023/02/20', '15-Mar-2023', 'invalid', ''],
            'time_12h': ['10:30 AM', '2:45 PM', 'noon', '25:99', ''],
            'datetime_mixed': ['2023-01-15 10:30:00', '20/02/2023 14:45', 'Today', '', None]
        })
        df2.to_excel(writer, sheet_name='Date-Time Issues', index=False)
        
        # Sheet 3: Empty sheet (should be skipped)
        df3 = pd.DataFrame()
        df3.to_excel(writer, sheet_name='Empty Sheet', index=False)
        
        # Sheet 4: Wide sheet with many columns
        wide_data = {f'col_{i}': [f'value_{i}_{j}' for j in range(5)] for i in range(20)}
        df4 = pd.DataFrame(wide_data)
        df4.to_excel(writer, sheet_name='Wide-Data', index=False)
    
    print(f"✓ Created Excel test assets")


def create_json_test_files():
    """Create JSON files with edge cases."""
    data_dir = Path(__file__).parent / "assets"
    data_dir.mkdir(exist_ok=True)
    
    # 1. Nested JSON structure
    nested_json = {
        "users": [
            {"id": 1, "name": "Alice", "profile": {"age": 30, "city": "New York"}},
            {"id": 2, "name": "Bob", "profile": {"age": "unknown", "city": ""}},
            {"id": 3, "name": "Charlie", "profile": None},
            {"id": 4, "profile": {"age": 25}}  # Missing name
        ]
    }
    
    with open(data_dir / "nested_structure.json", "w") as f:
        json.dump(nested_json, f, indent=2)
    
    # 2. Malformed JSON (missing comma)
    malformed_json = '{\n  "id": 1\n  "name": "test"\n  "value": 123\n}'
    with open(data_dir / "malformed.json", "w") as f:
        f.write(malformed_json)
    
    # 3. Array of objects with inconsistent fields
    inconsistent_json = [
        {"id": 1, "name": "Product A", "price": 99.99, "category": "Electronics"},
        {"id": 2, "title": "Product B", "cost": "150", "type": "Books"},  # Different field names
        {"id": 3, "name": "Product C", "price": None, "category": "", "extra": "field"},
        {"name": "Product D", "price": -50}  # Missing id
    ]
    
    with open(data_dir / "inconsistent_fields.json", "w") as f:
        json.dump(inconsistent_json, f, indent=2)
    
    print("✓ Created JSON test assets")


def create_yaml_test_files():
    """Create YAML files with edge cases."""
    data_dir = Path(__file__).parent / "assets"
    data_dir.mkdir(exist_ok=True)
    
    # 1. Complex YAML structure
    yaml_data = {
        'config': {
            'database': {
                'host': 'localhost',
                'port': 5432,
                'settings': {
                    'timeout': 30,
                    'retries': 3
                }
            },
            'features': ['auth', 'logging', 'metrics'],
            'enabled': True,
            'version': 1.5
        },
        'users': [
            {'name': 'admin', 'roles': ['admin', 'user']},
            {'name': 'guest', 'roles': []}
        ]
    }
    
    with open(data_dir / "complex_config.yaml", "w") as f:
        yaml.dump(yaml_data, f, default_flow_style=False)
    
    # 2. YAML with special characters and unicode
    unicode_yaml = {
        'messages': {
            'en': 'Hello, World!',
            'es': '¡Hola, Mundo!',
            'fr': 'Bonjour le monde!',
            'emoji': '🌍 Hello 🚀'
        },
        'special_chars': 'Contains "quotes" and \'apostrophes\' and: colons',
        'multiline': 'This is a\nmultiline string\nwith line breaks'
    }
    
    with open(data_dir / "unicode_special.yaml", "w", encoding="utf-8") as f:
        yaml.dump(unicode_yaml, f, default_flow_style=False, allow_unicode=True)
    
    print("✓ Created YAML test assets")


def create_xml_test_files():
    """Create XML files with edge cases."""
    data_dir = Path(__file__).parent / "assets"
    data_dir.mkdir(exist_ok=True)
    
    # 1. Structured XML data
    root = ET.Element("catalog")
    
    for i in range(1, 4):
        product = ET.SubElement(root, "product", id=str(i))
        
        name = ET.SubElement(product, "name")
        name.text = f"Product {i}"
        
        price = ET.SubElement(product, "price", currency="USD")
        price.text = str(99.99 + i)
        
        description = ET.SubElement(product, "description")
        description.text = f"Description for product {i} with <special> characters & entities"
        
        # Add some attributes and nested elements
        specs = ET.SubElement(product, "specifications")
        spec1 = ET.SubElement(specs, "spec", name="weight")
        spec1.text = f"{i * 1.5}kg"
        
        spec2 = ET.SubElement(specs, "spec", name="color")
        spec2.text = ["Red", "Blue", "Green"][i-1]
    
    tree = ET.ElementTree(root)
    tree.write(data_dir / "structured_catalog.xml", encoding="utf-8", xml_declaration=True)
    
    # 2. XML with mixed content and CDATA
    mixed_xml = '''<?xml version="1.0" encoding="UTF-8"?>
<data>
    <item id="1">
        <title>Sample Title</title>
        <content><![CDATA[This contains <b>HTML</b> and "quotes" and 'apostrophes']]></content>
        <metadata>
            <created>2023-01-15T10:30:00Z</created>
            <author>John Doe</author>
        </metadata>
    </item>
    <item id="2">
        <title>Another Item</title>
        <content>Simple text content</content>
        <metadata>
            <created>invalid-date</created>
            <author></author>
        </metadata>
    </item>
</data>'''
    
    with open(data_dir / "mixed_content.xml", "w", encoding="utf-8") as f:
        f.write(mixed_xml)
    
    print("✓ Created XML test assets")


def create_ini_test_files():
    """Create INI files with edge cases."""
    data_dir = Path(__file__).parent / "assets"
    data_dir.mkdir(exist_ok=True)
    
    # Complex INI configuration - disable interpolation to handle special characters
    config = configparser.ConfigParser(interpolation=None)
    
    # Default section
    config['DEFAULT'] = {
        'debug': 'true',
        'timeout': '30'
    }
    
    # Database section
    config['database'] = {
        'host': 'localhost',
        'port': '5432',
        'name': 'myapp',
        'user': 'admin',
        'password': 'secret!@#$%'
    }
    
    # Features section with special values
    config['features'] = {
        'feature1': 'enabled',
        'feature2': 'disabled',
        'feature3': '',  # Empty value
        'feature4': 'value with spaces',
        'feature5': 'value; with semicolon',
        'feature6': 'value = with equals'
    }
    
    # Section with unicode
    config['international'] = {
        'english': 'Hello',
        'spanish': 'Hola',
        'french': 'Bonjour',
        'emoji': '🌍'
    }
    
    with open(data_dir / "complex_config.ini", "w", encoding="utf-8") as f:
        config.write(f)
    
    print("✓ Created INI test assets")


def create_toml_test_files():
    """Create TOML files with edge cases."""
    if not TOML_AVAILABLE:
        print("⚠️  Skipping TOML files - toml library not available")
        return
    
    data_dir = Path(__file__).parent / "assets"
    data_dir.mkdir(exist_ok=True)
    
    toml_data = {
        'title': 'Test Configuration',
        'database': {
            'server': '192.168.1.1',
            'ports': [8001, 8001, 8002],
            'connection_max': 5000,
            'enabled': True
        },
        'servers': {
            'alpha': {
                'ip': '10.0.0.1',
                'dc': 'eqdc10'
            },
            'beta': {
                'ip': '10.0.0.2',
                'dc': 'eqdc10'
            }
        },
        'clients': {
            'data': [['gamma', 'delta'], [1, 2]],
            'hosts': ['alpha', 'omega']
        },
        'unicode_test': {
            'string': 'Hello 🌍',
            'multiline': '''
Line 1
Line 2 with "quotes"
Line 3 with 'apostrophes'
'''
        }
    }
    
    with open(data_dir / "complex_config.toml", "w", encoding="utf-8") as f:
        toml.dump(toml_data, f)
    
    print("✓ Created TOML test assets")


def create_tsv_test_files():
    """Create TSV (Tab-separated values) files with edge cases."""
    data_dir = Path(__file__).parent / "assets"
    data_dir.mkdir(exist_ok=True)
    
    # TSV with mixed data and tab characters in data
    tsv_data = [
        ['id', 'name', 'description', 'value'],
        ['1', 'Product A', 'Description with\ttab', '99.99'],
        ['2', 'Product B', 'Normal description', 'invalid_number'],
        ['3', 'Product\tC', 'Contains tab in name', ''],
        ['', '', '', ''],  # Empty row
        ['4', '"Quoted Name"', 'Description', '-123.45']
    ]
    
    with open(data_dir / "mixed_tabs.tsv", "w", encoding="utf-8") as f:
        for row in tsv_data:
            f.write("\t".join(row) + "\n")
    
    print("✓ Created TSV test assets")


def create_parquet_test_files():
    """Create Parquet files (if pyarrow available)."""
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        print("⚠️  Skipping Parquet files - pyarrow not available")
        return
    
    data_dir = Path(__file__).parent / "assets"
    data_dir.mkdir(exist_ok=True)
    
    # Create DataFrame with mixed data types
    df = pd.DataFrame({
        'id': range(1, 101),
        'name': [f'User_{i}' for i in range(1, 101)],
        'age': np.random.randint(18, 80, 100),
        'salary': np.random.normal(50000, 15000, 100),
        'active': np.random.choice([True, False], 100),
        'join_date': pd.date_range('2020-01-01', periods=100, freq='D'),
        'category': np.random.choice(['A', 'B', 'C', 'D'], 100),
        'score': np.random.uniform(0, 100, 100)
    })
    
    # Add some null values for edge case testing
    df.loc[df.sample(10).index, 'salary'] = None
    df.loc[df.sample(5).index, 'name'] = None
    
    df.to_parquet(data_dir / "mixed_types.parquet", index=False)
    
    print("✓ Created Parquet test assets")


def create_corrupted_files():
    """Create deliberately corrupted files for error handling tests."""
    data_dir = Path(__file__).parent / "assets"
    data_dir.mkdir(exist_ok=True)
    
    # Truncated CSV
    with open(data_dir / "truncated.csv", "w") as f:
        f.write("id,name,value\n1,Alice,123\n2,Bob,")  # Incomplete last row
    
    # Binary file with .csv extension (should fail)
    with open(data_dir / "fake.csv", "wb") as f:
        f.write(b'\x89PNG\r\n\x1a\n' + b'This is not a CSV file' * 10)
    
    # Empty files
    open(data_dir / "empty.csv", "w").close()
    open(data_dir / "empty.json", "w").close()
    open(data_dir / "empty.xml", "w").close()
    
    print("✓ Created corrupted/edge case files")


def main():
    """Create all realistic test assets."""
    print("Creating comprehensive realistic test assets...")
    
    # Create all test assets
    create_messy_csv_data()
    create_excel_test_files()
    create_json_test_files()
    create_yaml_test_files()
    create_xml_test_files()
    create_ini_test_files()
    create_toml_test_files()
    create_tsv_test_files()
    create_parquet_test_files()
    create_corrupted_files()
    
    assets_dir = Path(__file__).parent / "assets"
    assets_count = len(list(assets_dir.glob("*")))
    print(f"\n✅ Created {assets_count} realistic test asset files in {assets_dir}")
    print("\nTest assets created for:")
    print("- Mixed data types and encoding issues")
    print("- Malformed and corrupted data")
    print("- Unicode and special characters")
    print("- Large datasets for memory testing")
    print("- Multi-sheet spreadsheets")
    print("- Various date/time formats")
    print("- Empty and edge case files")


if __name__ == "__main__":
    main()