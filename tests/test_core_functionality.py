"""Test core functionality of LocalData MCP without MCP decorators.

This tests the underlying database functionality directly, bypassing MCP tool decorators.
"""

import json
import os
import tempfile
import time
from pathlib import Path
from unittest.mock import patch, MagicMock
import pandas as pd
import pytest


class TestDatabaseManager:
    """Test DatabaseManager core functionality."""
    
    def setup_method(self):
        """Create a fresh DatabaseManager instance for each test."""
        # Import here to avoid MCP initialization issues
        from localdata_mcp.localdata_mcp import DatabaseManager
        
        # Create manager but don't initialize MCP
        self.manager = DatabaseManager.__new__(DatabaseManager)
        self.manager.__init__()
    
    @pytest.fixture
    def test_csv_file(self):
        """Create a test CSV file in the current directory."""
        df = pd.DataFrame({
            'id': range(1, 151),  # 150 rows to test chunking
            'name': [f'user_{i}' for i in range(1, 151)],
            'value': [i * 10 for i in range(1, 151)]
        })
        
        # Create file in current directory to pass security checks
        file_path = f"test_data_{int(time.time())}.csv"
        df.to_csv(file_path, index=False)
        yield file_path
        
        # Cleanup
        if os.path.exists(file_path):
            os.unlink(file_path)
    
    @pytest.fixture
    def test_json_file(self):
        """Create a test JSON file in the current directory."""
        data = [
            {'category': 'A', 'count': 10, 'active': True},
            {'category': 'B', 'count': 20, 'active': False},
            {'category': 'C', 'count': 15, 'active': True}
        ]
        
        # Create file in current directory to pass security checks
        file_path = f"test_data_{int(time.time())}.json"
        with open(file_path, 'w') as f:
            json.dump(data, f)
        yield file_path
        
        # Cleanup
        if os.path.exists(file_path):
            os.unlink(file_path)


class TestCoreConnectionFunctionality(TestDatabaseManager):
    """Test core connection functionality."""
    
    def test_basic_connection_and_sql_flavor_detection(self, test_csv_file):
        """Test connecting to CSV and SQL flavor detection."""
        
        # Test CSV connection
        try:
            engine = self.manager._get_engine("csv", test_csv_file)
            assert engine is not None
            
            # Test SQL flavor detection
            flavor = self.manager._get_sql_flavor("csv", engine)
            assert flavor == "SQLite"
            
            # Test connection tracking
            assert "test_csv" not in self.manager.connections
            
            # Manually add to connections to test tracking
            self.manager.connections["test_csv"] = engine
            self.manager.db_types["test_csv"] = "csv"
            self.manager.query_history["test_csv"] = []
            
            assert "test_csv" in self.manager.connections
            assert self.manager.db_types["test_csv"] == "csv"
            
        finally:
            # Cleanup
            if "test_csv" in self.manager.connections:
                self.manager.connections["test_csv"].dispose()
                del self.manager.connections["test_csv"]
                del self.manager.db_types["test_csv"] 
                del self.manager.query_history["test_csv"]
    
    def test_json_connection(self, test_json_file):
        """Test connecting to JSON file."""
        try:
            engine = self.manager._get_engine("json", test_json_file)
            assert engine is not None
            
            flavor = self.manager._get_sql_flavor("json", engine)
            assert flavor == "SQLite"
            
        finally:
            if engine:
                engine.dispose()
    
    def test_unsupported_database_type(self):
        """Test error handling for unsupported database types."""
        with pytest.raises(ValueError, match="Unsupported db_type"):
            self.manager._get_engine("unsupported_type", "dummy_path")
    
    def test_path_sanitization_security(self):
        """Test path sanitization for security."""
        dangerous_paths = [
            "../etc/passwd",
            "/etc/passwd",
            "../../secrets.txt"
        ]
        
        for path in dangerous_paths:
            with pytest.raises(ValueError, match="outside the allowed directory|File not found"):
                self.manager._sanitize_path(path)
    
    def test_safe_table_identifier(self):
        """Test SQL injection prevention in table names."""
        # Valid table names
        valid_names = ["users", "data_table", "_private", "table123"]
        for name in valid_names:
            result = self.manager._safe_table_identifier(name)
            assert result is not None
        
        # Invalid table names
        invalid_names = [
            "users; DROP TABLE users; --",
            "'; DELETE FROM data; --",
            "users/*comment*/"
        ]
        for name in invalid_names:
            with pytest.raises(ValueError, match="Invalid table name"):
                self.manager._safe_table_identifier(name)


class TestCoreQueryFunctionality(TestDatabaseManager):
    """Test core query functionality including chunking and memory management."""
    
    def test_query_id_generation(self):
        """Test query ID generation."""
        query_id1 = self.manager._generate_query_id("test_db", "SELECT * FROM table")
        time.sleep(1)  # Ensure different timestamp
        query_id2 = self.manager._generate_query_id("test_db", "SELECT * FROM table")
        query_id3 = self.manager._generate_query_id("other_db", "SELECT * FROM table")
        query_id4 = self.manager._generate_query_id("test_db", "SELECT COUNT(*) FROM table")
        
        # IDs should be unique due to timestamp or different query
        assert query_id1 != query_id2  # Different timestamp
        assert query_id1 != query_id4  # Different query hash
        
        # IDs should contain database name
        assert "test_db" in query_id1
        assert "other_db" in query_id3
        
        # IDs should have expected format: {db}_{timestamp}_{4char_hash}
        # For database names with underscores, check the pattern differently
        assert query_id1.startswith("test_db_")
        assert query_id3.startswith("other_db_")
        
        # Extract the hash part (last 4 characters after final underscore)
        hash_part = query_id1.split('_')[-1]
        assert len(hash_part) == 4
        
        # Timestamp part should be second to last when split by underscore
        timestamp_part = query_id1.split('_')[-2]
        assert timestamp_part.isdigit()
    
    @patch('localdata_mcp.localdata_mcp.psutil.virtual_memory')
    def test_memory_usage_checking(self, mock_memory):
        """Test memory usage checking functionality."""
        # Mock normal memory usage
        mock_memory.return_value = MagicMock(
            total=8 * 1024**3,  # 8GB
            available=4 * 1024**3,  # 4GB available
            percent=50  # Normal usage
        )
        
        memory_info = self.manager._check_memory_usage()
        
        assert "total_gb" in memory_info
        assert "available_gb" in memory_info
        assert "used_percent" in memory_info
        assert "low_memory" in memory_info
        assert memory_info["used_percent"] == 50
        assert memory_info["low_memory"] is False
        
        # Mock high memory usage
        mock_memory.return_value = MagicMock(
            total=8 * 1024**3,  # 8GB
            available=1 * 1024**3,  # 1GB available 
            percent=90  # High usage
        )
        
        memory_info = self.manager._check_memory_usage()
        assert memory_info["low_memory"] is True
    
    def test_buffer_cleanup_functionality(self):
        """Test query buffer cleanup."""
        # Create mock buffers
        from localdata_mcp.localdata_mcp import QueryBuffer
        
        current_time = time.time()
        
        # Recent buffer (should not be cleaned up)
        recent_buffer = QueryBuffer(
            query_id="recent_123",
            db_name="test_db",
            query="SELECT * FROM table",
            results=pd.DataFrame({'col': [1, 2, 3]}),
            timestamp=current_time
        )
        
        # Old buffer (should be cleaned up)
        old_buffer = QueryBuffer(
            query_id="old_456",
            db_name="test_db", 
            query="SELECT * FROM table",
            results=pd.DataFrame({'col': [1, 2, 3]}),
            timestamp=current_time - 700  # 11+ minutes old
        )
        
        self.manager.query_buffers["recent_123"] = recent_buffer
        self.manager.query_buffers["old_456"] = old_buffer
        
        # Force cleanup
        self.manager.last_cleanup = 0  # Force cleanup to run
        self.manager._cleanup_expired_buffers()
        
        # Recent buffer should remain, old should be cleaned up
        assert "recent_123" in self.manager.query_buffers
        assert "old_456" not in self.manager.query_buffers
    
    def test_auto_buffer_clearing_logic(self):
        """Test automatic buffer clearing when memory is low."""
        # Create test buffers for different databases
        from localdata_mcp.localdata_mcp import QueryBuffer
        
        buffer1 = QueryBuffer(
            query_id="db1_query",
            db_name="db1",
            query="SELECT * FROM table",
            results=pd.DataFrame({'col': [1, 2, 3]}),
            timestamp=time.time()
        )
        
        buffer2 = QueryBuffer(
            query_id="db2_query", 
            db_name="db2",
            query="SELECT * FROM table", 
            results=pd.DataFrame({'col': [1, 2, 3]}),
            timestamp=time.time()
        )
        
        self.manager.query_buffers["db1_query"] = buffer1
        self.manager.query_buffers["db2_query"] = buffer2
        
        # Mock high memory usage
        with patch('localdata_mcp.localdata_mcp.psutil.virtual_memory') as mock_memory:
            mock_memory.return_value = MagicMock(
                total=8 * 1024**3,
                available=1 * 1024**3,
                percent=90  # High usage
            )
            
            # Clear buffers for db1 only
            cleared = self.manager._auto_clear_buffers_if_needed("db1")
            
            assert cleared is True
            assert "db1_query" not in self.manager.query_buffers  # Should be cleared
            assert "db2_query" in self.manager.query_buffers  # Should remain
    
    def test_file_modification_checking(self):
        """Test file modification time checking for buffers."""
        from localdata_mcp.localdata_mcp import QueryBuffer
        
        # Create a temporary file in current directory
        temp_path = f"test_file_{int(time.time())}.txt"
        with open(temp_path, 'wb') as f:
            f.write(b"test content")
        
        try:
            # Get initial mtime
            initial_mtime = os.path.getmtime(temp_path)
            
            # Create buffer with file info
            buffer = QueryBuffer(
                query_id="test_query",
                db_name="test_db",
                query="SELECT * FROM table",
                results=pd.DataFrame({'col': [1, 2, 3]}),
                timestamp=time.time(),
                source_file_path=temp_path,
                source_file_mtime=initial_mtime
            )
            
            # File should not be modified yet
            assert not self.manager._check_file_modified(buffer)
            
            # Modify the file
            time.sleep(0.1)  # Ensure different timestamp
            with open(temp_path, 'a') as f:
                f.write("modified content")
            
            # Now file should be detected as modified
            assert self.manager._check_file_modified(buffer)
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


class TestIntegrationWorkflow(TestDatabaseManager):
    """Test complete integration workflows."""
    
    def test_csv_query_workflow(self, test_csv_file):
        """Test complete CSV query workflow."""
        try:
            # Create engine
            engine = self.manager._get_engine("csv", test_csv_file)
            
            # Set up connection tracking
            self.manager.connections["test_db"] = engine
            self.manager.db_types["test_db"] = "csv"
            self.manager.query_history["test_db"] = []
            
            # Execute query using pandas (simulating the actual query execution)
            df = pd.read_sql_query("SELECT COUNT(*) as total FROM data_table", engine)
            
            # Verify results
            assert len(df) == 1
            assert df.iloc[0]['total'] == 150  # Our test data has 150 rows
            
            # Test large query that would trigger buffering
            large_df = pd.read_sql_query("SELECT * FROM data_table", engine)
            assert len(large_df) == 150
            
            # Test query history tracking (manual simulation)
            self.manager.query_history["test_db"].append("SELECT COUNT(*) as total FROM data_table")
            self.manager.query_history["test_db"].append("SELECT * FROM data_table")
            
            assert len(self.manager.query_history["test_db"]) == 2
            
        finally:
            # Cleanup
            if "test_db" in self.manager.connections:
                self.manager.connections["test_db"].dispose()
                del self.manager.connections["test_db"]
                del self.manager.db_types["test_db"]
                del self.manager.query_history["test_db"]
    
    def test_json_query_workflow(self, test_json_file):
        """Test complete JSON query workflow."""
        try:
            # Create engine
            engine = self.manager._get_engine("json", test_json_file)
            
            # Set up connection tracking
            self.manager.connections["json_db"] = engine
            self.manager.db_types["json_db"] = "json"
            self.manager.query_history["json_db"] = []
            
            # Execute query
            df = pd.read_sql_query("SELECT COUNT(*) as total FROM data_table", engine)
            
            # Verify results (JSON test data has 3 records)
            assert len(df) == 1
            assert df.iloc[0]['total'] == 3
            
            # Test category query
            categories_df = pd.read_sql_query("SELECT category FROM data_table WHERE active = 1", engine)
            assert len(categories_df) == 2  # A and C are active
            
        finally:
            # Cleanup
            if "json_db" in self.manager.connections:
                self.manager.connections["json_db"].dispose()
                del self.manager.connections["json_db"]
                del self.manager.db_types["json_db"]
                del self.manager.query_history["json_db"]


class TestErrorHandlingAndSecurity(TestDatabaseManager):
    """Test error handling and security features."""
    
    def test_connection_limit_logic(self):
        """Test connection limit enforcement logic."""
        # Connection semaphore is initialized with 10 permits
        initial_permits = self.manager.connection_semaphore._value
        assert initial_permits == 10
        
        # Acquire permits to simulate connections
        acquired = []
        for i in range(10):
            if self.manager.connection_semaphore.acquire(blocking=False):
                acquired.append(i)
        
        # Should have acquired all 10
        assert len(acquired) == 10
        
        # Next acquire should fail
        assert not self.manager.connection_semaphore.acquire(blocking=False)
        
        # Release permits
        for _ in acquired:
            self.manager.connection_semaphore.release()
    
    def test_resource_cleanup_logic(self):
        """Test resource cleanup functionality."""
        # Create temporary files and add them to tracking
        temp_files = []
        for i in range(3):
            temp_path = f"cleanup_test_{i}_{int(time.time())}.txt"
            with open(temp_path, 'w') as f:
                f.write("test content")
            temp_files.append(temp_path)
        
        # Add to manager's tracking
        self.manager.temp_files.extend(temp_files)
        
        # Verify files exist
        for temp_file in temp_files:
            assert os.path.exists(temp_file)
        
        # Run cleanup
        self.manager._cleanup_all()
        
        # Files should be deleted and list should be empty
        for temp_file in temp_files:
            assert not os.path.exists(temp_file)
        assert len(self.manager.temp_files) == 0
    
    def test_large_file_detection(self):
        """Test large file detection logic."""
        # Create a small test file in current directory
        temp_path = f"large_file_test_{int(time.time())}.txt"
        with open(temp_path, 'w') as f:
            f.write("small content")
        
        try:
            # Should not be large with default threshold
            assert not self.manager._is_large_file(temp_path)
            
            # Should be large with very small threshold (file is ~13 bytes, so use 0.0001 MB)
            assert self.manager._is_large_file(temp_path, threshold_mb=0.00001)
            
            # Test file size getting
            size = self.manager._get_file_size(temp_path)
            assert size > 0
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_sheet_name_sanitization(self):
        """Test Excel/ODS sheet name sanitization."""
        test_cases = [
            ("Normal Sheet", "Normal_Sheet"),
            ("Sheet With Spaces", "Sheet_With_Spaces"),
            ("Sheet-With-Hyphens", "Sheet_With_Hyphens"),
            ("Sheet.With.Dots", "Sheet_With_Dots"),
            ("Sheet@#$%Special", "Sheet_Special"),  # Consecutive special chars become single underscore
            ("123Numbers", "sheet_123Numbers"),  # Starts with number
            ("", "sheet_unnamed"),  # Empty name
            ("  ", "sheet_unnamed"),  # Whitespace only
        ]
        
        for input_name, expected in test_cases:
            result = self.manager._sanitize_sheet_name(input_name)
            assert result == expected
        
        # Test uniqueness handling
        used_names = {"existing_sheet"}
        result1 = self.manager._sanitize_sheet_name("test", used_names)
        result2 = self.manager._sanitize_sheet_name("test", used_names)
        
        assert result1 == "test"
        assert result2 == "test_1"  # Should be made unique


if __name__ == "__main__":
    pytest.main([__file__, "-v"])