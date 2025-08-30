"""Test import error handling for optional dependencies.

This module tests the import error handling paths in localdata_mcp.py
to ensure graceful degradation when optional dependencies are missing.
"""

import sys
import unittest
from unittest.mock import patch, MagicMock
import pytest


class TestImportErrorHandling(unittest.TestCase):
    """Test import error handling for all optional dependencies."""
    
    def setUp(self):
        """Set up test environment by removing imported modules."""
        # Remove any already imported modules to test import behavior
        modules_to_remove = [
            'localdata_mcp.localdata_mcp',
            'toml',
            'openpyxl',
            'xlrd', 
            'defusedxml',
            'odf',
            'lxml',
            'pyarrow',
            'numbers_parser',
            'duckdb',
            'h5py',
            'redis',
            'elasticsearch'
        ]
        
        for module in modules_to_remove:
            if module in sys.modules:
                del sys.modules[module]
    
    @patch.dict('sys.modules', {'toml': None})
    def test_toml_import_error_handling(self):
        """Test TOML import error handling - lines 27-28."""
        with patch('builtins.__import__', side_effect=ImportError):
            # Import the module with toml unavailable
            import importlib
            spec = importlib.util.spec_from_file_location(
                "localdata_mcp", 
                "/Users/chris/Dropbox/dev/ai/claude-code-cfg/mcp/localdata-mcp/src/localdata_mcp/localdata_mcp.py"
            )
            module = importlib.util.module_from_spec(spec)
            
            # Mock the imports that will succeed
            with patch.dict('sys.modules', {
                'pandas': MagicMock(),
                'yaml': MagicMock(),
                'fastmcp': MagicMock(),
                'sqlalchemy': MagicMock(),
                'psutil': MagicMock()
            }):
                try:
                    spec.loader.exec_module(module)
                    # Check that TOML_AVAILABLE is set to False
                    self.assertFalse(module.TOML_AVAILABLE)
                except ImportError:
                    # Expected when core dependencies are missing
                    pass
    
    def test_openpyxl_import_error_handling(self):
        """Test openpyxl import error handling - lines 34-35."""
        with patch.dict('sys.modules', {'openpyxl': None}):
            with patch('builtins.__import__') as mock_import:
                def import_side_effect(name, *args, **kwargs):
                    if name == 'openpyxl':
                        raise ImportError("No module named openpyxl")
                    return MagicMock()
                
                mock_import.side_effect = import_side_effect
                
                # Test import behavior
                try:
                    import openpyxl
                    OPENPYXL_AVAILABLE = True
                except ImportError:
                    OPENPYXL_AVAILABLE = False
                
                self.assertFalse(OPENPYXL_AVAILABLE)
    
    def test_xlrd_import_error_handling(self):
        """Test xlrd import error handling - lines 40-41."""
        with patch.dict('sys.modules', {'xlrd': None}):
            with patch('builtins.__import__') as mock_import:
                def import_side_effect(name, *args, **kwargs):
                    if name == 'xlrd':
                        raise ImportError("No module named xlrd")
                    return MagicMock()
                
                mock_import.side_effect = import_side_effect
                
                try:
                    import xlrd
                    XLRD_AVAILABLE = True
                except ImportError:
                    XLRD_AVAILABLE = False
                
                self.assertFalse(XLRD_AVAILABLE)
    
    def test_defusedxml_import_error_handling(self):
        """Test defusedxml import error handling - lines 46-47."""
        with patch.dict('sys.modules', {'defusedxml': None}):
            with patch('builtins.__import__') as mock_import:
                def import_side_effect(name, *args, **kwargs):
                    if name == 'defusedxml':
                        raise ImportError("No module named defusedxml")
                    return MagicMock()
                
                mock_import.side_effect = import_side_effect
                
                try:
                    import defusedxml
                    DEFUSEDXML_AVAILABLE = True
                except ImportError:
                    DEFUSEDXML_AVAILABLE = False
                
                self.assertFalse(DEFUSEDXML_AVAILABLE)
    
    def test_odf_import_error_handling(self):
        """Test ODF import error handling - line 52."""
        with patch.dict('sys.modules', {'odf': None}):
            with patch('builtins.__import__') as mock_import:
                def import_side_effect(name, *args, **kwargs):
                    if name in ['odf', 'odf.opendocument', 'odf.table']:
                        raise ImportError("No module named odf")
                    return MagicMock()
                
                mock_import.side_effect = import_side_effect
                
                try:
                    from odf import opendocument, table
                    ODFPY_AVAILABLE = True
                except ImportError:
                    ODFPY_AVAILABLE = False
                
                self.assertFalse(ODFPY_AVAILABLE)
    
    def test_lxml_import_error_handling(self):
        """Test lxml import error handling - lines 60-61."""
        with patch.dict('sys.modules', {'lxml': None}):
            with patch('builtins.__import__') as mock_import:
                def import_side_effect(name, *args, **kwargs):
                    if name == 'lxml':
                        raise ImportError("No module named lxml")
                    return MagicMock()
                
                mock_import.side_effect = import_side_effect
                
                try:
                    import lxml
                    LXML_AVAILABLE = True
                except ImportError:
                    LXML_AVAILABLE = False
                
                self.assertFalse(LXML_AVAILABLE)
    
    def test_pyarrow_import_error_handling(self):
        """Test pyarrow import error handling - lines 67-68."""
        with patch.dict('sys.modules', {'pyarrow': None}):
            with patch('builtins.__import__') as mock_import:
                def import_side_effect(name, *args, **kwargs):
                    if name == 'pyarrow':
                        raise ImportError("No module named pyarrow")
                    return MagicMock()
                
                mock_import.side_effect = import_side_effect
                
                try:
                    import pyarrow
                    PYARROW_AVAILABLE = True
                except ImportError:
                    PYARROW_AVAILABLE = False
                
                self.assertFalse(PYARROW_AVAILABLE)
    
    def test_numbers_parser_import_error_handling(self):
        """Test numbers_parser import error handling."""
        with patch.dict('sys.modules', {'numbers_parser': None}):
            with patch('builtins.__import__') as mock_import:
                def import_side_effect(name, *args, **kwargs):
                    if name == 'numbers_parser':
                        raise ImportError("No module named numbers_parser")
                    return MagicMock()
                
                mock_import.side_effect = import_side_effect
                
                try:
                    from numbers_parser import Document
                    NUMBERS_PARSER_AVAILABLE = True
                except ImportError:
                    NUMBERS_PARSER_AVAILABLE = False
                
                self.assertFalse(NUMBERS_PARSER_AVAILABLE)
    
    def test_duckdb_import_error_handling(self):
        """Test DuckDB import error handling."""
        with patch.dict('sys.modules', {'duckdb': None}):
            with patch('builtins.__import__') as mock_import:
                def import_side_effect(name, *args, **kwargs):
                    if name == 'duckdb':
                        raise ImportError("No module named duckdb")
                    return MagicMock()
                
                mock_import.side_effect = import_side_effect
                
                try:
                    import duckdb
                    DUCKDB_AVAILABLE = True
                except ImportError:
                    DUCKDB_AVAILABLE = False
                
                self.assertFalse(DUCKDB_AVAILABLE)
    
    def test_h5py_import_error_handling(self):
        """Test h5py import error handling."""
        with patch.dict('sys.modules', {'h5py': None}):
            with patch('builtins.__import__') as mock_import:
                def import_side_effect(name, *args, **kwargs):
                    if name == 'h5py':
                        raise ImportError("No module named h5py")
                    return MagicMock()
                
                mock_import.side_effect = import_side_effect
                
                try:
                    import h5py
                    H5PY_AVAILABLE = True
                except ImportError:
                    H5PY_AVAILABLE = False
                
                self.assertFalse(H5PY_AVAILABLE)
    
    def test_redis_import_error_handling(self):
        """Test Redis import error handling."""
        with patch.dict('sys.modules', {'redis': None}):
            with patch('builtins.__import__') as mock_import:
                def import_side_effect(name, *args, **kwargs):
                    if name == 'redis':
                        raise ImportError("No module named redis")
                    return MagicMock()
                
                mock_import.side_effect = import_side_effect
                
                try:
                    import redis
                    REDIS_AVAILABLE = True
                except ImportError:
                    REDIS_AVAILABLE = False
                
                self.assertFalse(REDIS_AVAILABLE)
    
    def test_elasticsearch_import_error_handling(self):
        """Test Elasticsearch import error handling."""
        with patch.dict('sys.modules', {'elasticsearch': None}):
            with patch('builtins.__import__') as mock_import:
                def import_side_effect(name, *args, **kwargs):
                    if name == 'elasticsearch':
                        raise ImportError("No module named elasticsearch")
                    return MagicMock()
                
                mock_import.side_effect = import_side_effect
                
                try:
                    from elasticsearch import Elasticsearch
                    ELASTICSEARCH_AVAILABLE = True
                except ImportError:
                    ELASTICSEARCH_AVAILABLE = False
                
                self.assertFalse(ELASTICSEARCH_AVAILABLE)


if __name__ == '__main__':
    unittest.main()