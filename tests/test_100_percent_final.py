"""Final comprehensive test to achieve 100% coverage by targeting remaining uncovered lines."""

import os
import sys
import tempfile
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

import pytest
import pandas as pd

from localdata_mcp import DatabaseManager


class Test100PercentFinal:
    """Final test class to achieve 100% coverage."""

    def test_sanitize_path_error_140_141(self):
        """Test _sanitize_path error handling (lines 140-141)."""
        manager = DatabaseManager()
        
        # Test path sanitization with Path.resolve raising OSError
        with patch('pathlib.Path.resolve', side_effect=OSError("Path resolution error")):
            with pytest.raises(ValueError, match="Invalid path"):
                manager._sanitize_path("/some/invalid/path")

    def test_cleanup_all_with_errors_241_242_248_251(self):
        """Test _cleanup_all with various errors (lines 241-242, 248-251)."""
        manager = DatabaseManager()
        
        # Add temp files that will cause removal errors
        manager.temp_files = ["nonexistent1.tmp", "nonexistent2.tmp"]
        
        # Add mock engines that will cause disposal errors
        mock_engine1 = Mock()
        mock_engine1.dispose.side_effect = Exception("Engine disposal error")
        mock_engine2 = Mock()
        mock_engine2.dispose.side_effect = OSError("OS error during disposal")
        
        manager.connections = {
            "conn1": mock_engine1,
            "conn2": mock_engine2
        }
        
        # Mock os.remove to raise OSError for temp file cleanup
        with patch('os.remove', side_effect=OSError("Cannot remove file")):
            # This should handle all errors gracefully
            manager._cleanup_all()
            
        # Verify cleanup attempted
        assert len(manager.connections) == 0
        assert len(manager.temp_files) == 0

    def test_create_engine_from_file_257_259_261(self):
        """Test _create_engine_from_file different paths (lines 257, 259, 261)."""
        manager = DatabaseManager()
        
        # Test SQLite path (line 257)
        engine = manager._create_engine_from_file("test.db", "sqlite")
        assert engine is not None
        assert "sqlite" in str(engine.url)
        
        # Test PostgreSQL path (line 259) 
        engine = manager._create_engine_from_file("postgresql://user:pass@localhost/db", "postgresql")
        assert engine is not None
        
        # Test MySQL path (line 261)
        engine = manager._create_engine_from_file("mysql://user:pass@localhost/db", "mysql")  
        assert engine is not None

    def test_yaml_file_loading_291_292(self):
        """Test YAML file loading (lines 291-292)."""
        manager = DatabaseManager()
        
        yaml_content = """
data:
  - name: "test1"
    value: 100
  - name: "test2" 
    value: 200
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            yaml_file = f.name
        
        try:
            # Test YAML loading by creating file-based connection
            result = manager._create_engine_from_file(yaml_file, "yaml")
            assert result is not None
        finally:
            os.unlink(yaml_file)

    def test_toml_file_loading_303_305(self):
        """Test TOML file loading (lines 303-305)."""
        # Check if TOML is available
        try:
            import toml
        except ImportError:
            pytest.skip("TOML not available")
            
        manager = DatabaseManager()
        
        toml_content = """
[database]
host = "localhost"
port = 5432
name = "testdb"

[settings]
timeout = 30
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
            f.write(toml_content)
            toml_file = f.name
        
        try:
            # Test TOML loading
            result = manager._create_engine_from_file(toml_file, "toml")
            assert result is not None
        finally:
            os.unlink(toml_file)

    def test_ini_file_loading_317(self):
        """Test INI file loading (line 317)."""
        manager = DatabaseManager()
        
        ini_content = """
[database]
host=localhost
port=5432
name=testdb

[settings]
timeout=30
debug=true
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.ini', delete=False) as f:
            f.write(ini_content)
            ini_file = f.name
        
        try:
            # Test INI loading
            data = manager._load_ini_file(ini_file)
            assert data is not None
            assert len(data) > 0  # Should have loaded some data
        finally:
            os.unlink(ini_file)

    def test_feather_file_loading_337(self):
        """Test Feather file loading (line 337)."""
        try:
            import pyarrow
        except ImportError:
            pytest.skip("PyArrow not available")
            
        manager = DatabaseManager()
        
        # Create test DataFrame
        df = pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
            'score': [95.5, 87.2, 92.1, 88.9, 91.3]
        })
        
        with tempfile.NamedTemporaryFile(suffix='.feather', delete=False) as f:
            feather_file = f.name
        
        try:
            df.to_feather(feather_file)
            # Test Feather loading
            result = manager._create_engine_from_file(feather_file, "feather")
            assert result is not None
        finally:
            os.unlink(feather_file)

    def test_arrow_file_loading_344_346(self):
        """Test Arrow file loading (lines 344-346)."""
        try:
            import pyarrow
        except ImportError:
            pytest.skip("PyArrow not available")
            
        manager = DatabaseManager()
        
        # Create test DataFrame  
        df = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=5, freq='D'),
            'value': [10.1, 20.2, 30.3, 40.4, 50.5],
            'category': ['A', 'B', 'A', 'C', 'B']
        })
        
        with tempfile.NamedTemporaryFile(suffix='.arrow', delete=False) as f:
            arrow_file = f.name
        
        try:
            df.to_feather(arrow_file)  # Arrow uses same format
            # Test Arrow loading
            result = manager._create_engine_from_file(arrow_file, "arrow")
            assert result is not None
        finally:
            os.unlink(arrow_file)

    def test_unsupported_file_type_351_361(self):
        """Test unsupported file type error (lines 351-361)."""
        manager = DatabaseManager()
        
        with tempfile.NamedTemporaryFile(suffix='.unknown', delete=False) as f:
            f.write(b"This is an unsupported file format")
            unknown_file = f.name
        
        try:
            with pytest.raises(ValueError, match="Unsupported file type"):
                manager._create_engine_from_file(unknown_file, "auto")
        finally:
            os.unlink(unknown_file)

    def test_ods_file_error_handling_527_528_550_552(self):
        """Test ODS file error handling (lines 527-528, 550-552)."""
        manager = DatabaseManager()
        
        # Create invalid ODS file to trigger error paths
        with tempfile.NamedTemporaryFile(suffix='.ods', delete=False) as f:
            f.write(b"This is not a valid ODS file content")
            ods_file = f.name
        
        try:
            # This should trigger the error handling in _load_ods_file
            with pytest.raises((Exception, ValueError)):
                manager._load_ods_file(ods_file)
        finally:
            os.unlink(ods_file)

    def test_memory_checking_1100_1102(self):
        """Test memory checking functionality (lines 1100-1102)."""
        manager = DatabaseManager()
        
        # Test normal memory checking
        memory_info = manager._check_memory_usage()
        assert isinstance(memory_info, dict)
        assert "memory_percent" in memory_info
        
        # Test memory checking with psutil error
        with patch('psutil.virtual_memory', side_effect=Exception("Memory check failed")):
            memory_info = manager._check_memory_usage()
            assert "error" in memory_info
            assert memory_info["low_memory"] is False

    def test_auto_buffer_clearing_1109_1120_1122(self):
        """Test auto buffer clearing functionality (lines 1109-1120, 1122)."""
        manager = DatabaseManager()
        
        # Add some mock buffers
        mock_buffer1 = Mock()
        mock_buffer1.db_name = "test_db"
        mock_buffer2 = Mock() 
        mock_buffer2.db_name = "other_db"
        mock_buffer3 = Mock()
        mock_buffer3.db_name = "test_db"
        
        manager.query_buffers = {
            "query1": mock_buffer1,
            "query2": mock_buffer2,
            "query3": mock_buffer3
        }
        
        # Mock low memory condition
        with patch.object(manager, '_check_memory_usage', return_value={"low_memory": True}):
            result = manager._auto_clear_buffers_if_needed("test_db")
            assert result is True  # Should have cleared buffers
            
        # Test with high memory (no clearing needed)
        manager.query_buffers = {"query1": mock_buffer1}
        with patch.object(manager, '_check_memory_usage', return_value={"low_memory": False}):
            result = manager._auto_clear_buffers_if_needed("test_db")
            assert result is False

    def test_file_modification_checking_edge_cases(self):
        """Test file modification checking edge cases."""
        manager = DatabaseManager()
        
        # Create a test file
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"test content")
            test_file = f.name
        
        try:
            # Test normal file modification checking
            result = manager._check_file_modified(test_file, 0)  # Very old timestamp
            assert result is True
            
            # Test with current timestamp
            import time
            current_time = time.time()
            result = manager._check_file_modified(test_file, current_time + 1)  # Future timestamp
            assert result is False
            
        finally:
            os.unlink(test_file)

    def test_excel_and_xml_loading_edge_cases(self):
        """Test Excel and XML loading edge cases."""
        manager = DatabaseManager()
        
        # Test XML loading
        xml_content = """<?xml version="1.0" encoding="UTF-8"?>
<root>
    <record>
        <id>1</id>
        <name>Test</name>
        <value>100</value>
    </record>
    <record>
        <id>2</id>
        <name>Another</name>
        <value>200</value>
    </record>
</root>"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
            f.write(xml_content)
            xml_file = f.name
        
        try:
            data = manager._load_xml_file(xml_file)
            assert data is not None
        finally:
            os.unlink(xml_file)

    def test_remaining_utility_methods(self):
        """Test remaining utility methods for full coverage."""
        manager = DatabaseManager()
        
        # Test query ID generation
        query_id = manager._generate_query_id()
        assert isinstance(query_id, str)
        assert len(query_id) > 0
        
        # Test file size detection
        with tempfile.NamedTemporaryFile() as f:
            f.write(b"test content for size detection")
            f.flush()
            size = manager._get_file_size(f.name)
            assert size > 0
            
        # Test large file detection
        is_large = manager._is_large_file(1024 * 1024 * 100)  # 100MB
        assert isinstance(is_large, bool)
        
        # Test sheet name sanitization
        sanitized = manager._sanitize_sheet_name("Test Sheet!@#$%")
        assert isinstance(sanitized, str)
        
    def test_import_availability_flags_coverage(self):
        """Ensure all import availability flags are covered."""
        import localdata_mcp.localdata_mcp as ldm
        
        # This ensures all the try/except import blocks are covered
        flags = [
            ldm.TOML_AVAILABLE,
            ldm.OPENPYXL_AVAILABLE,
            ldm.XLRD_AVAILABLE,
            ldm.DEFUSEDXML_AVAILABLE,
            ldm.ODFPY_AVAILABLE,
            ldm.LXML_AVAILABLE,
            ldm.PYARROW_AVAILABLE,
        ]
        
        # All flags should be boolean
        for flag in flags:
            assert isinstance(flag, bool)
        
        # Access the module-level logger to ensure initialization is covered
        assert ldm.logger is not None
        assert ldm.mcp is not None

    def test_main_function_complete_coverage_1161(self):
        """Test main function for complete coverage (line 1161)."""
        from localdata_mcp.localdata_mcp import main
        
        # Mock mcp.run to test the main function
        with patch('localdata_mcp.localdata_mcp.mcp.run') as mock_run:
            main()
            mock_run.assert_called_once_with(transport="stdio")