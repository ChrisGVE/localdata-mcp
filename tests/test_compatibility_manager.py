"""
Test suite for CompatibilityManager - Backward Compatibility Layer
================================================================

Tests ensure seamless migration from v1.3.0 while maintaining API compatibility.
"""

import json
import os
import tempfile
import unittest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import warnings
import logging

import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.localdata_mcp.compatibility_manager import (
    CompatibilityManager, 
    DeprecationInfo,
    get_compatibility_manager
)


class TestCompatibilityManager(unittest.TestCase):
    """Test the CompatibilityManager functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.compatibility_manager = CompatibilityManager()
        
        # Capture warnings for testing
        self.warning_list = []
        def custom_warning_handler(message, category, filename, lineno, file=None, line=None):
            self.warning_list.append((str(message), category))
        
        self.old_showwarning = warnings.showwarning
        warnings.showwarning = custom_warning_handler
        
        # Setup logging capture
        self.log_messages = []
        self.log_handler = logging.Handler()
        self.log_handler.emit = lambda record: self.log_messages.append(record.getMessage())
        
    def tearDown(self):
        """Clean up after tests."""
        warnings.showwarning = self.old_showwarning
        # Reset the manager's warning tracking
        self.compatibility_manager._deprecation_warnings_shown.clear()
        
    def test_deprecation_info_structure(self):
        """Test DeprecationInfo dataclass structure."""
        info = DeprecationInfo(
            feature="test_feature",
            version_deprecated="1.3.1",
            version_removed="1.4.0",
            replacement="new_feature",
            migration_guide="Use new_feature instead",
            impact_level="medium"
        )
        
        self.assertEqual(info.feature, "test_feature")
        self.assertEqual(info.version_deprecated, "1.3.1")
        self.assertEqual(info.version_removed, "1.4.0")
        self.assertEqual(info.replacement, "new_feature")
        self.assertEqual(info.migration_guide, "Use new_feature instead")
        self.assertEqual(info.impact_level, "medium")
        
    def test_warn_deprecated_shows_warning_once(self):
        """Test that deprecation warnings are shown only once per session."""
        # First call should show warning
        self.compatibility_manager.warn_deprecated('env_only_config')
        
        # Second call should not show warning again
        self.compatibility_manager.warn_deprecated('env_only_config')
        
        # Should only be called once
        self.assertIn('env_only_config', self.compatibility_manager._deprecation_warnings_shown)
        
    def test_warn_deprecated_unknown_feature(self):
        """Test warning for unknown deprecated feature."""
        with self.assertLogs(level='WARNING') as log:
            self.compatibility_manager.warn_deprecated('unknown_feature')
            
        self.assertIn('Unknown deprecated feature key: unknown_feature', log.output[0])
        
    @patch.dict(os.environ, {
        'POSTGRES_HOST': 'localhost',
        'POSTGRES_PORT': '5432',
        'POSTGRES_USER': 'testuser',
        'POSTGRES_PASSWORD': 'testpass',
        'POSTGRES_DB': 'testdb'
    })
    def test_detect_legacy_configuration_postgres(self):
        """Test detection of legacy PostgreSQL environment variables."""
        legacy_config = self.compatibility_manager.detect_legacy_configuration()
        
        self.assertTrue(len(legacy_config['detected']) > 0)
        
        # Find PostgreSQL pattern
        postgres_pattern = None
        for pattern in legacy_config['detected']:
            if pattern['pattern'] == 'Legacy Environment Variables':
                postgres_pattern = pattern
                break
                
        self.assertIsNotNone(postgres_pattern)
        self.assertIn('POSTGRES_HOST', postgres_pattern['variables'])
        self.assertIn('POSTGRES_PORT', postgres_pattern['variables'])
        self.assertIn('POSTGRES_USER', postgres_pattern['variables'])
        self.assertIn('POSTGRES_PASSWORD', postgres_pattern['variables'])
        self.assertIn('POSTGRES_DB', postgres_pattern['variables'])
        self.assertEqual(postgres_pattern['impact'], 'medium')
        
    @patch.dict(os.environ, {
        'MYSQL_HOST': 'localhost',
        'MYSQL_PORT': '3306',
        'SQLITE_PATH': '/tmp/test.db'
    })
    def test_detect_legacy_configuration_multiple_dbs(self):
        """Test detection of multiple legacy database configurations."""
        legacy_config = self.compatibility_manager.detect_legacy_configuration()
        
        self.assertTrue(len(legacy_config['detected']) > 0)
        
        # Should detect legacy variables
        pattern = legacy_config['detected'][0]
        self.assertEqual(pattern['pattern'], 'Legacy Environment Variables')
        
        # Should contain variables from different database types
        variables = pattern['variables']
        self.assertIn('MYSQL_HOST', variables)
        self.assertIn('MYSQL_PORT', variables)
        self.assertIn('SQLITE_PATH', variables)
        
    @patch.dict(os.environ, {}, clear=True)
    def test_detect_legacy_configuration_none(self):
        """Test detection when no legacy configuration exists."""
        legacy_config = self.compatibility_manager.detect_legacy_configuration()
        
        self.assertEqual(len(legacy_config['detected']), 0)
        self.assertFalse(legacy_config['migration_required'])
        
    @patch.dict(os.environ, {'POSTGRES_HOST': 'localhost'})
    @patch('pathlib.Path.exists')
    def test_detect_legacy_configuration_migration_required(self, mock_exists):
        """Test detection when migration is required (no YAML config)."""
        # Mock that no YAML config files exist
        mock_exists.return_value = False
        
        legacy_config = self.compatibility_manager.detect_legacy_configuration()
        
        self.assertTrue(legacy_config['migration_required'])
        self.assertTrue(len(legacy_config['suggestions']) > 0)
        
    def test_check_api_compatibility_execute_query_legacy(self):
        """Test API compatibility check for legacy execute_query usage."""
        # Test legacy 2-parameter call
        compatibility_info = self.compatibility_manager.check_api_compatibility(
            'execute_query', 
            ('test_db', 'SELECT * FROM users'),
            {}
        )
        
        self.assertTrue(compatibility_info['compatible'])
        self.assertTrue(len(compatibility_info['warnings']) > 0)
        self.assertIn('only name and query parameters', compatibility_info['warnings'][0])
        
    def test_check_api_compatibility_execute_query_modern(self):
        """Test API compatibility check for modern execute_query usage."""
        # Test modern call with all parameters
        compatibility_info = self.compatibility_manager.check_api_compatibility(
            'execute_query',
            ('test_db', 'SELECT * FROM users'),
            {'chunk_size': 1000, 'enable_analysis': True}
        )
        
        self.assertTrue(compatibility_info['compatible'])
        # Should have no warnings for modern usage
        self.assertEqual(len(compatibility_info['warnings']), 0)
        
    def test_check_api_compatibility_analysis_disabled(self):
        """Test API compatibility when analysis is explicitly disabled."""
        compatibility_info = self.compatibility_manager.check_api_compatibility(
            'execute_query',
            ('test_db', 'SELECT * FROM users'),
            {'enable_analysis': False}
        )
        
        self.assertTrue(compatibility_info['compatible'])
        # Should not generate warnings in compatibility check (warnings come from warn_deprecated)
        
    def test_migrate_legacy_response_format_v130(self):
        """Test response format migration for v1.3.0 compatibility."""
        # Test response with metadata structure
        new_response = {
            'data': '{"users": [{"id": 1, "name": "John"}]}',
            'metadata': {'rows': 1, 'columns': 2}
        }
        
        migrated = self.compatibility_manager.migrate_legacy_response_format(
            new_response, 'v1.3.0'
        )
        
        # Should extract just the data portion
        self.assertEqual(migrated, '{"users": [{"id": 1, "name": "John"}]}')
        
    def test_migrate_legacy_response_format_result_structure(self):
        """Test response format migration with result structure."""
        new_response = {
            'result': '{"users": [{"id": 1, "name": "John"}]}',
            'analysis': {'estimated_rows': 1}
        }
        
        migrated = self.compatibility_manager.migrate_legacy_response_format(
            new_response, 'v1.3.0'
        )
        
        # Should extract the result portion
        self.assertEqual(migrated, '{"users": [{"id": 1, "name": "John"}]}')
        
    def test_migrate_legacy_response_format_simple(self):
        """Test response format migration with simple response."""
        simple_response = '{"users": [{"id": 1, "name": "John"}]}'
        
        migrated = self.compatibility_manager.migrate_legacy_response_format(
            simple_response, 'v1.3.0'
        )
        
        # Should return as-is for simple responses
        self.assertEqual(migrated, simple_response)
        
    @patch.dict(os.environ, {'POSTGRES_HOST': 'localhost'})
    def test_get_compatibility_status(self):
        """Test getting comprehensive compatibility status."""
        status = self.compatibility_manager.get_compatibility_status()
        
        self.assertEqual(status['version'], '1.3.1')
        self.assertFalse(status['compatibility_mode'])
        self.assertTrue(status['legacy_config_detected'])
        self.assertIn('legacy_patterns', status)
        self.assertIn('recommendations', status)
        
    def test_enable_compatibility_mode(self):
        """Test enabling/disabling compatibility mode."""
        # Initially disabled
        self.assertFalse(self.compatibility_manager._compatibility_mode)
        
        # Enable
        with self.assertLogs(level='INFO') as log:
            self.compatibility_manager.enable_compatibility_mode(True)
            
        self.assertTrue(self.compatibility_manager._compatibility_mode)
        self.assertIn('Compatibility mode enabled', log.output[0])
        
        # Disable
        with self.assertLogs(level='INFO') as log:
            self.compatibility_manager.enable_compatibility_mode(False)
            
        self.assertFalse(self.compatibility_manager._compatibility_mode)
        self.assertIn('Compatibility mode disabled', log.output[0])
        
    @patch.dict(os.environ, {
        'POSTGRES_HOST': 'localhost',
        'POSTGRES_PORT': '5432',
        'POSTGRES_USER': 'testuser',
        'POSTGRES_PASSWORD': 'testpass',
        'POSTGRES_DB': 'testdb'
    })
    def test_create_migration_script(self):
        """Test migration script generation."""
        script = self.compatibility_manager.create_migration_script()
        
        # Should be a valid Python script
        self.assertIn('#!/usr/bin/env python3', script)
        self.assertIn('def migrate_configuration():', script)
        self.assertIn('import os', script)
        self.assertIn('import yaml', script)
        
        # Should contain PostgreSQL configuration migration
        self.assertIn('postgresql', script)
        self.assertIn('POSTGRES_HOST', script)
        self.assertIn('POSTGRES_PORT', script)
        
        # Should write YAML configuration
        self.assertIn('yaml.dump', script)
        self.assertIn('localdata.yaml', script)
        
    def test_create_migration_script_with_file(self):
        """Test migration script generation with file output."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            script_path = f.name
            
        try:
            with self.assertLogs(level='INFO') as log:
                script = self.compatibility_manager.create_migration_script(script_path)
                
            # Should log successful write
            self.assertIn(f'Migration script written to {script_path}', log.output[0])
            
            # File should exist and contain script
            with open(script_path, 'r') as f:
                file_content = f.read()
                
            self.assertEqual(script, file_content)
            self.assertIn('#!/usr/bin/env python3', file_content)
            
        finally:
            # Clean up
            if os.path.exists(script_path):
                os.unlink(script_path)
                
    def test_global_compatibility_manager_singleton(self):
        """Test that get_compatibility_manager returns singleton instance."""
        manager1 = get_compatibility_manager()
        manager2 = get_compatibility_manager()
        
        # Should be the same instance
        self.assertIs(manager1, manager2)
        self.assertIsInstance(manager1, CompatibilityManager)
        

class TestCompatibilityManagerIntegration(unittest.TestCase):
    """Integration tests for CompatibilityManager with LocalData MCP components."""
    
    def setUp(self):
        """Set up integration test fixtures."""
        self.compatibility_manager = CompatibilityManager()
        
    def test_deprecated_features_definitions(self):
        """Test that all expected deprecated features are defined."""
        expected_features = [
            'env_only_config',
            'single_database_env_vars',
            'execute_query_no_analysis'
        ]
        
        for feature in expected_features:
            self.assertIn(feature, self.compatibility_manager._deprecated_features)
            
        # Test structure of each feature
        for feature_key, info in self.compatibility_manager._deprecated_features.items():
            self.assertIsInstance(info, DeprecationInfo)
            self.assertIsInstance(info.feature, str)
            self.assertIsInstance(info.version_deprecated, str)
            self.assertIsInstance(info.replacement, str)
            self.assertIsInstance(info.migration_guide, str)
            self.assertIn(info.impact_level, ['low', 'medium', 'high', 'critical'])
            
    @patch.dict(os.environ, {
        'POSTGRES_HOST': 'localhost',
        'MYSQL_HOST': 'localhost',
        'SQLITE_PATH': '/tmp/test.db',
        'MONGODB_URL': 'mongodb://localhost:27017/test'
    })
    def test_all_database_types_detected(self):
        """Test that all supported database types are detected in legacy config."""
        legacy_config = self.compatibility_manager.detect_legacy_configuration()
        
        self.assertTrue(len(legacy_config['detected']) > 0)
        
        variables = legacy_config['detected'][0]['variables']
        
        # Should detect all database types
        self.assertIn('POSTGRES_HOST', variables)
        self.assertIn('MYSQL_HOST', variables)
        self.assertIn('SQLITE_PATH', variables)
        self.assertIn('MONGODB_URL', variables)
        
    def test_migration_script_all_database_types(self):
        """Test migration script generation with all database types."""
        # Patch environment to include all database types
        with patch.dict(os.environ, {
            'POSTGRES_HOST': 'pg-host',
            'POSTGRES_PORT': '5432',
            'POSTGRES_USER': 'pguser',
            'POSTGRES_PASSWORD': 'pgpass',
            'POSTGRES_DB': 'pgdb',
            'MYSQL_HOST': 'mysql-host',
            'MYSQL_PORT': '3306',
            'MYSQL_USER': 'mysqluser',
            'MYSQL_PASSWORD': 'mysqlpass',
            'MYSQL_DB': 'mysqldb',
            'SQLITE_PATH': '/tmp/test.sqlite',
            'DUCKDB_PATH': '/tmp/test.duckdb',
            'MONGODB_URL': 'mongodb://localhost:27017/testdb'
        }):
            script = self.compatibility_manager.create_migration_script()
            
            # Should contain all database types
            self.assertIn('postgresql', script)
            self.assertIn('mysql', script)
            self.assertIn('sqlite', script)
            self.assertIn('duckdb', script)
            self.assertIn('mongodb', script)
            
            # Should contain specific environment variables
            self.assertIn('POSTGRES_HOST', script)
            self.assertIn('MYSQL_HOST', script)
            self.assertIn('SQLITE_PATH', script)
            self.assertIn('DUCKDB_PATH', script)
            self.assertIn('MONGODB_URL', script)
            

if __name__ == '__main__':
    # Setup logging for tests
    logging.basicConfig(level=logging.INFO)
    
    # Run the tests
    unittest.main(verbosity=2)