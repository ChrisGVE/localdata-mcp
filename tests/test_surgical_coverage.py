"""
Surgical test approach to hit specific uncovered lines directly.
Target only the exact missing lines with minimal, focused tests.
"""
import pytest
import os
import tempfile
import json
from unittest.mock import Mock, patch, MagicMock, mock_open
from pathlib import Path
import pandas as pd
from localdata_mcp.localdata_mcp import DatabaseManager


class TestSurgicalCoverage:
    """Direct tests for specific uncovered lines."""
    
    def setup_method(self):
        self.manager = DatabaseManager()
    
    def test_lines_140_141_path_errors(self):
        """Hit lines 140-141: Path resolution errors"""
        # Line 140-141: Path.resolve() raises OSError/ValueError
        with patch.object(Path, 'resolve', side_effect=OSError("Bad path")):
            with pytest.raises(ValueError, match="Invalid path"):
                self.manager._sanitize_path("/bad/path")
        
        with patch.object(Path, 'resolve', side_effect=ValueError("Invalid")):
            with pytest.raises(ValueError, match="Invalid path"):
                self.manager._sanitize_path("/bad/path")
    
    def test_lines_241_242_cleanup_errors(self):
        """Hit lines 241-242: Cleanup OSError handling"""
        self.manager.temp_files = ["/fake/file"]
        
        with patch('os.path.exists', return_value=True), \
             patch('os.unlink', side_effect=OSError("Permission denied")):
            # Should not raise, just continue (line 242: pass)
            self.manager._cleanup_all()
    
    def test_lines_257_259_261_direct_engines(self):
        """Hit lines 257, 259, 261: Direct database connections"""
        with patch('sqlalchemy.create_engine') as mock_create:
            # Line 257: SQLite
            self.manager._get_engine("sqlite", "/path/to/db")
            mock_create.assert_called_with("sqlite:///path/to/db")
            
            # Line 259: PostgreSQL  
            self.manager._get_engine("postgresql", "postgresql://conn")
            mock_create.assert_called_with("postgresql://conn")
            
            # Line 261: MySQL
            self.manager._get_engine("mysql", "mysql://conn") 
            mock_create.assert_called_with("mysql://conn")
    
    def test_lines_351_361_temp_file_creation(self):
        """Hit lines 351-361: Temp file creation for large files"""
        
        # Create large mock DataFrame
        mock_df = Mock()
        mock_df.memory_usage.return_value.sum.return_value = 60 * 1024 * 1024  # 60MB
        
        with patch('tempfile.mkstemp', return_value=(99, "/tmp/temp.db")) as mock_temp, \
             patch('os.close') as mock_close, \
             patch('sqlalchemy.create_engine') as mock_engine, \
             patch.object(self.manager, '_load_data_file', return_value=mock_df), \
             patch.object(self.manager, '_sanitize_path', return_value="/file.csv"):
            
            self.manager._create_engine_from_file("/file.csv", "csv")
            
            # Verify temp file creation (lines 351-361)
            mock_temp.assert_called_with(suffix=".sqlite", prefix="db_client_")
            mock_close.assert_called_with(99)
            mock_engine.assert_called_with("sqlite:///tmp/temp.db")
    
    def test_lines_527_528_empty_ods_sheet(self):
        """Hit lines 527-528: ODS empty sheet handling"""
        
        empty_df = pd.DataFrame()
        good_df = pd.DataFrame({'a': [1, 2]})
        
        with patch('pandas.ExcelFile') as mock_file, \
             patch('pandas.read_excel', side_effect=[empty_df, good_df]) as mock_read, \
             patch('localdata_mcp.localdata_mcp.logger') as mock_logger:
            
            mock_file.return_value.sheet_names = ['Empty', 'Good']
            
            result = self.manager._load_ods_file("/test.ods")
            
            # Line 527: Warning for empty sheet
            mock_logger.warning.assert_called_with("Sheet 'Empty' is empty, skipping")
            assert len(result) == 1
    
    def test_lines_550_552_ods_sheet_exception(self):
        """Hit lines 550-552: ODS sheet loading exception"""
        
        good_df = pd.DataFrame({'a': [1, 2]})
        
        with patch('pandas.ExcelFile') as mock_file, \
             patch('pandas.read_excel', side_effect=[Exception("Read failed"), good_df]), \
             patch('localdata_mcp.localdata_mcp.logger') as mock_logger:
            
            mock_file.return_value.sheet_names = ['Bad', 'Good']
            
            result = self.manager._load_ods_file("/test.ods")
            
            # Line 551: Warning for failed sheet
            mock_logger.warning.assert_called_with("Failed to load sheet 'Bad': Read failed")
    
    def test_line_555_no_readable_ods_data(self):
        """Hit line 555: No readable ODS data error"""
        
        with patch('pandas.ExcelFile') as mock_file, \
             patch('pandas.read_excel', side_effect=Exception("All failed")):
            
            mock_file.return_value.sheet_names = ['Sheet1']
            
            with pytest.raises(ValueError, match="contains no readable data"):
                self.manager._load_ods_file("/test.ods")
    
    def test_lines_764_766_disconnect_errors(self):
        """Hit lines 764-766: Database disconnection errors"""
        
        mock_engine = Mock()
        mock_engine.dispose.side_effect = Exception("Disconnect failed")
        self.manager.connections['test'] = mock_engine
        self.manager.db_types['test'] = 'sqlite'
        
        result = self.manager.disconnect_database('test')
        assert "An error occurred while disconnecting" in result
    
    def test_lines_1009_1020_describe_database(self):
        """Hit lines 1009-1020: Database description with table metadata"""
        
        mock_engine = Mock()
        mock_inspector = Mock()
        mock_connection = Mock()
        mock_result = Mock()
        mock_result.scalar.return_value = 42
        
        with patch('sqlalchemy.inspect', return_value=mock_inspector), \
             patch.object(self.manager, '_get_table_metadata', return_value={'name': 'table1'}), \
             patch.object(self.manager, '_safe_table_identifier', return_value='table1'):
            
            mock_inspector.get_schema_names.return_value = ['main']
            mock_inspector.get_table_names.return_value = ['table1']
            mock_engine.connect.return_value.__enter__.return_value = mock_connection
            mock_connection.execute.return_value = mock_result
            
            self.manager.connections['test'] = mock_engine
            self.manager.db_types['test'] = 'sqlite'
            
            result = self.manager.describe_database('test')
            
            # Verify table size was queried (lines 1013-1017)
            assert '"size": 42' in result or '42' in result
    
    def test_line_1161_main_function(self):
        """Hit line 1161: main() function call"""
        
        with patch('localdata_mcp.localdata_mcp.mcp.run') as mock_run:
            from localdata_mcp.localdata_mcp import main
            main()
            mock_run.assert_called_with(transport="stdio")
    
    def test_main_module_execution_path(self):
        """Test execution when run as main module"""
        
        # Test the __main__ execution path
        import localdata_mcp.localdata_mcp as module
        
        # Patch the main function and simulate __name__ == "__main__"
        with patch.object(module, 'main') as mock_main:
            with patch.object(module, '__name__', '__main__'):
                # Re-execute the module-level code
                exec(compile(open(module.__file__).read(), module.__file__, 'exec'))
    
    def test_additional_missing_lines(self):
        """Cover any other specific missing lines"""
        
        # Test unsupported database type
        with pytest.raises(ValueError, match="Unsupported db_type"):
            self.manager._get_engine("invalid_type", "path")
    
    def test_engine_disposal_errors(self):
        """Test engine disposal error handling in cleanup"""
        mock_engine = Mock()
        mock_engine.dispose.side_effect = Exception("Disposal error") 
        
        self.manager.connections['test'] = mock_engine
        self.manager.db_types['test'] = 'sqlite'
        
        # Should not raise exception (error is caught)
        self.manager._cleanup_all()
        assert len(self.manager.connections) == 0