"""
Helper functions for mocking common operations in tests.
These provide reusable mocking patterns for file operations, database connections, etc.
"""

from unittest.mock import patch, Mock
import pandas as pd
import contextlib


@contextlib.contextmanager  
def mock_csv_connection(sample_data=None, file_path=None):
    """Context manager to mock CSV file connection operations."""
    if sample_data is None:
        sample_data = pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
            'value': [100, 200, 300, 400, 500]
        })
    
    if file_path is None:
        file_path = '/tmp/test.csv'
    
    with patch('localdata_mcp.localdata_mcp.DatabaseManager._sanitize_path') as mock_sanitize, \
         patch('localdata_mcp.localdata_mcp.DatabaseManager._get_file_size') as mock_get_size, \
         patch('localdata_mcp.localdata_mcp.DatabaseManager._is_large_file') as mock_is_large, \
         patch('pandas.read_csv', return_value=sample_data), \
         patch('sqlalchemy.create_engine') as mock_create_engine, \
         patch('pandas.DataFrame.to_sql'):
        
        # Configure path sanitization and file operations
        mock_sanitize.return_value = file_path
        mock_get_size.return_value = 1024  # 1KB file
        mock_is_large.return_value = False  # Small file
        
        # Configure SQLAlchemy engine mock
        mock_engine = Mock()
        mock_engine.dialect.name = 'sqlite'
        mock_create_engine.return_value = mock_engine
        
        yield mock_engine


@contextlib.contextmanager
def mock_json_connection(sample_data=None, file_path=None):
    """Context manager to mock JSON file connection operations."""
    if sample_data is None:
        sample_data = pd.DataFrame([
            {'id': 1, 'name': 'Item1', 'category': 'A', 'price': 10.99},
            {'id': 2, 'name': 'Item2', 'category': 'B', 'price': 25.50},
            {'id': 3, 'name': 'Item3', 'category': 'A', 'price': 15.75}
        ])
    
    if file_path is None:
        file_path = '/tmp/test.json'
    
    with patch('localdata_mcp.localdata_mcp.DatabaseManager._sanitize_path') as mock_sanitize, \
         patch('localdata_mcp.localdata_mcp.DatabaseManager._get_file_size') as mock_get_size, \
         patch('localdata_mcp.localdata_mcp.DatabaseManager._is_large_file') as mock_is_large, \
         patch('pandas.read_json', return_value=sample_data), \
         patch('sqlalchemy.create_engine') as mock_create_engine, \
         patch('pandas.DataFrame.to_sql'):
        
        # Configure path sanitization and file operations
        mock_sanitize.return_value = file_path
        mock_get_size.return_value = 1024  # 1KB file
        mock_is_large.return_value = False  # Small file
        
        # Configure SQLAlchemy engine mock  
        mock_engine = Mock()
        mock_engine.dialect.name = 'sqlite'
        mock_create_engine.return_value = mock_engine
        
        yield mock_engine


@contextlib.contextmanager
def mock_excel_connection(sample_sheets=None, file_path=None):
    """Context manager to mock Excel file connection operations."""
    if sample_sheets is None:
        sample_sheets = {
            'Sheet1': pd.DataFrame({
                'id': [1, 2, 3],
                'name': ['Alice', 'Bob', 'Charlie'],
                'age': [25, 30, 35]
            }),
            'Data': pd.DataFrame({
                'metric': ['cpu', 'memory', 'disk'],
                'value': [75.5, 82.1, 45.8]
            })
        }
    
    if file_path is None:
        file_path = '/tmp/test.xlsx'
    
    with patch('localdata_mcp.localdata_mcp.DatabaseManager._sanitize_path') as mock_sanitize, \
         patch('localdata_mcp.localdata_mcp.DatabaseManager._get_file_size') as mock_get_size, \
         patch('localdata_mcp.localdata_mcp.DatabaseManager._is_large_file') as mock_is_large, \
         patch('localdata_mcp.localdata_mcp.DatabaseManager._load_excel_file', return_value=sample_sheets), \
         patch('sqlalchemy.create_engine') as mock_create_engine, \
         patch('pandas.DataFrame.to_sql'):
        
        # Configure path sanitization and file operations
        mock_sanitize.return_value = file_path
        mock_get_size.return_value = 1024  # 1KB file
        mock_is_large.return_value = False  # Small file
        
        # Configure SQLAlchemy engine mock
        mock_engine = Mock()
        mock_engine.dialect.name = 'sqlite' 
        mock_create_engine.return_value = mock_engine
        
        yield mock_engine


@contextlib.contextmanager
def mock_database_query(query_result=None):
    """Context manager to mock database query operations."""
    if query_result is None:
        query_result = pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['Alice', 'Bob', 'Charlie'],
            'score': [95, 87, 92]
        })
    
    with patch('pandas.read_sql_query', return_value=query_result):
        yield query_result


@contextlib.contextmanager
def mock_large_dataset_query(rows=1000):
    """Context manager to mock large dataset queries for testing chunking."""
    large_data = pd.DataFrame({
        'id': range(1, rows + 1),
        'name': [f'User_{i}' for i in range(1, rows + 1)],
        'value': [i * 10 for i in range(1, rows + 1)]
    })
    
    with patch('pandas.read_sql_query', return_value=large_data):
        yield large_data


@contextlib.contextmanager
def mock_database_inspection():
    """Context manager to mock database inspection operations."""
    with patch('sqlalchemy.inspect') as mock_inspect:
        mock_inspector = Mock()
        
        # Configure table information
        mock_inspector.get_table_names.return_value = ['data_table', 'users', 'products']
        mock_inspector.get_columns.return_value = [
            {'name': 'id', 'type': 'INTEGER', 'nullable': False, 'autoincrement': True},
            {'name': 'name', 'type': 'VARCHAR(255)', 'nullable': True, 'autoincrement': False},
            {'name': 'value', 'type': 'FLOAT', 'nullable': True, 'autoincrement': False}
        ]
        mock_inspector.get_foreign_keys.return_value = []
        mock_inspector.get_pk_constraint.return_value = {'constrained_columns': ['id']}
        mock_inspector.get_indexes.return_value = []
        mock_inspector.get_table_options.return_value = {}
        mock_inspector.get_server_version_info.return_value = (3, 39, 4)
        mock_inspector.default_schema_name = 'main'
        mock_inspector.get_schema_names.return_value = ['main']
        
        mock_inspect.return_value = mock_inspector
        yield mock_inspector


@contextlib.contextmanager
def mock_memory_pressure():
    """Context manager to mock high memory pressure scenarios."""
    with patch('psutil.virtual_memory') as mock_memory:
        # Configure high memory usage (90%)
        mock_memory.return_value = Mock(
            total=16 * 1024**3,  # 16GB
            available=1.6 * 1024**3,  # Only 1.6GB available  
            percent=90.0,  # 90% used - triggers low memory
            used=14.4 * 1024**3
        )
        yield mock_memory


@contextlib.contextmanager 
def mock_normal_memory():
    """Context manager to mock normal memory conditions."""
    with patch('psutil.virtual_memory') as mock_memory:
        mock_memory.return_value = Mock(
            total=16 * 1024**3,  # 16GB
            available=8 * 1024**3,  # 8GB available
            percent=50.0,  # 50% used
            used=8 * 1024**3
        )
        yield mock_memory


@contextlib.contextmanager
def mock_corrupted_file():
    """Context manager to mock corrupted file scenarios."""
    with patch('pandas.read_csv', side_effect=pd.errors.ParserError("Corrupted file")), \
         patch('pandas.read_json', side_effect=ValueError("Invalid JSON")), \
         patch('pandas.read_excel', side_effect=Exception("Corrupted Excel file")):
        yield


def create_sample_dataframes():
    """Create a collection of sample DataFrames for testing."""
    return {
        'csv_data': pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
            'age': [25, 30, 35, 28, 32],
            'salary': [50000, 60000, 70000, 55000, 65000]
        }),
        'json_data': pd.DataFrame([
            {'id': 1, 'item': 'Laptop', 'price': 999.99, 'in_stock': True},
            {'id': 2, 'item': 'Mouse', 'price': 29.99, 'in_stock': False},
            {'id': 3, 'item': 'Keyboard', 'price': 79.99, 'in_stock': True}
        ]),
        'large_data': pd.DataFrame({
            'user_id': range(1, 151),  # 150 rows for chunking tests
            'username': [f'user_{i}' for i in range(1, 151)],
            'score': [i * 2.5 for i in range(1, 151)],
            'active': [i % 2 == 0 for i in range(1, 151)]
        })
    }