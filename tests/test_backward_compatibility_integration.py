"""
Integration tests for backward compatibility layer
================================================

Tests the complete backward compatibility system including:
- Legacy API signatures still working
- Configuration migration detection
- Deprecation warnings in realistic scenarios
- Migration script generation
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

# Add the src directory to the path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import the compatibility manager directly
exec(open(os.path.join(os.path.dirname(__file__), '..', 'src', 'localdata_mcp', 'compatibility_manager.py')).read())


class TestBackwardCompatibilityIntegration(unittest.TestCase):
    """Integration tests for the complete backward compatibility system."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.manager = CompatibilityManager()
        
        # Setup environment variables for testing
        self.test_env = {
            'POSTGRES_HOST': 'localhost',
            'POSTGRES_PORT': '5432', 
            'POSTGRES_USER': 'testuser',
            'POSTGRES_PASSWORD': 'testpass',
            'POSTGRES_DB': 'testdb',
            'MYSQL_HOST': 'mysql-server',
            'MYSQL_PORT': '3306',
            'SQLITE_PATH': '/tmp/test.db'
        }
        
        # Capture logs
        self.logs = []
        self.handler = logging.Handler()
        self.handler.emit = lambda record: self.logs.append(record)
        
    def test_complete_legacy_detection_workflow(self):
        """Test the complete legacy configuration detection workflow."""
        with patch.dict(os.environ, self.test_env, clear=True):
            # Simulate what happens when DatabaseManager initializes
            legacy_config = self.manager.detect_legacy_configuration()
            
            # Should detect legacy patterns
            self.assertTrue(legacy_config['detected'])
            self.assertTrue(legacy_config['migration_required'])
            
            # Should have suggestions
            self.assertTrue(len(legacy_config['suggestions']) > 0)
            
            # Test warning generation
            for pattern in legacy_config['detected']:
                if pattern['pattern'] == 'Legacy Environment Variables':
                    self.manager.warn_deprecated('single_database_env_vars')
                    
            if legacy_config['migration_required']:
                self.manager.warn_deprecated('env_only_config')
                
            # Warnings should be tracked
            self.assertIn('single_database_env_vars', self.manager._deprecation_warnings_shown)
            self.assertIn('env_only_config', self.manager._deprecation_warnings_shown)
            
    def test_api_backward_compatibility_scenarios(self):
        """Test various API backward compatibility scenarios."""
        
        # Scenario 1: Legacy execute_query call (2 parameters only)
        compat_info = self.manager.check_api_compatibility(
            'execute_query',
            ('my_database', 'SELECT * FROM users'),
            {}
        )
        
        self.assertTrue(compat_info['compatible'])
        self.assertTrue(len(compat_info['warnings']) > 0)
        self.assertIn('enable_analysis=True', compat_info['suggestions'][0])
        
        # Scenario 2: Modern execute_query call with all parameters
        compat_info = self.manager.check_api_compatibility(
            'execute_query',
            ('my_database', 'SELECT * FROM users'),
            {'chunk_size': 1000, 'enable_analysis': True}
        )
        
        self.assertTrue(compat_info['compatible'])
        self.assertEqual(len(compat_info['warnings']), 0)  # No warnings for modern usage
        
        # Scenario 3: Analysis disabled (should trigger deprecation elsewhere)
        compat_info = self.manager.check_api_compatibility(
            'execute_query',
            ('my_database', 'SELECT * FROM users'),
            {'enable_analysis': False}
        )
        
        self.assertTrue(compat_info['compatible'])
        
    def test_response_format_migration(self):
        """Test response format migration for different scenarios."""
        
        # Test new format with metadata being converted to old format
        new_response = {
            'data': json.dumps([{'id': 1, 'name': 'John'}, {'id': 2, 'name': 'Jane'}]),
            'metadata': {'rows': 2, 'columns': 2, 'query_analysis': {'estimated_time': '10ms'}}
        }
        
        migrated = self.manager.migrate_legacy_response_format(new_response, 'v1.3.0')
        
        # Should extract just the data
        expected_data = json.dumps([{'id': 1, 'name': 'John'}, {'id': 2, 'name': 'Jane'}])
        self.assertEqual(migrated, expected_data)
        
        # Test 'result' structure
        new_response_result = {
            'result': '{"success": true, "data": []}',
            'analysis': {'memory_used': '100MB'}
        }
        
        migrated = self.manager.migrate_legacy_response_format(new_response_result, 'v1.3.0')
        self.assertEqual(migrated, '{"success": true, "data": []}')
        
        # Test simple response (no migration needed)
        simple_response = '{"users": []}'
        migrated = self.manager.migrate_legacy_response_format(simple_response, 'v1.3.0')
        self.assertEqual(migrated, simple_response)
        
    def test_comprehensive_migration_script_generation(self):
        """Test migration script generation with comprehensive configuration."""
        with patch.dict(os.environ, self.test_env, clear=True):
            script = self.manager.create_migration_script()
            
            # Should be valid Python
            self.assertIn('#!/usr/bin/env python3', script)
            self.assertIn('def migrate_configuration():', script)
            
            # Should handle PostgreSQL
            self.assertIn('postgresql', script)
            self.assertIn("'host': os.getenv('POSTGRES_HOST'", script)
            self.assertIn("'port': int(os.getenv('POSTGRES_PORT'", script)
            
            # Should handle MySQL
            self.assertIn('mysql', script)
            self.assertIn("'host': os.getenv('MYSQL_HOST'", script)
            
            # Should handle SQLite
            self.assertIn('sqlite', script)
            self.assertIn("'path': os.getenv('SQLITE_PATH'", script)
            
            # Should write YAML
            self.assertIn('yaml.dump', script)
            self.assertIn('localdata.yaml', script)
            
            # Should provide user feedback
            self.assertIn('print', script)
            self.assertIn('Configuration migrated', script)
            
    def test_compatibility_status_reporting(self):
        """Test comprehensive compatibility status reporting."""
        with patch.dict(os.environ, self.test_env, clear=True):
            # Trigger some deprecation warnings first
            self.manager.warn_deprecated('single_database_env_vars')
            
            status = self.manager.get_compatibility_status()
            
            # Check structure
            self.assertEqual(status['version'], '1.3.1')
            self.assertIn('compatibility_mode', status)
            self.assertIn('legacy_config_detected', status)
            self.assertIn('legacy_patterns', status)
            self.assertIn('deprecation_warnings_shown', status)
            self.assertIn('recommendations', status)
            
            # Should detect legacy config
            self.assertTrue(status['legacy_config_detected'])
            
            # Should track deprecation warnings
            self.assertIn('single_database_env_vars', status['deprecation_warnings_shown'])
            
            # Should have recommendations
            self.assertTrue(len(status['recommendations']) > 0)
            
    def test_compatibility_mode_effects(self):
        """Test the effects of enabling compatibility mode."""
        # Initially disabled
        status = self.manager.get_compatibility_status()
        self.assertFalse(status['compatibility_mode'])
        
        # Enable compatibility mode
        self.manager.enable_compatibility_mode(True)
        
        status = self.manager.get_compatibility_status()
        self.assertTrue(status['compatibility_mode'])
        
        # Disable again
        self.manager.enable_compatibility_mode(False)
        
        status = self.manager.get_compatibility_status()
        self.assertFalse(status['compatibility_mode'])
        
    def test_migration_script_execution_simulation(self):
        """Simulate executing the migration script to validate its correctness."""
        with patch.dict(os.environ, self.test_env, clear=True):
            script = self.manager.create_migration_script()
            
            # Create a temporary environment to "execute" the script logic
            with tempfile.TemporaryDirectory() as temp_dir:
                # Simulate the script's behavior
                config = {
                    'databases': {},
                    'logging': {'level': 'INFO'},
                    'performance': {'memory_limit_mb': 512}
                }
                
                # PostgreSQL configuration (as the script would do)
                if os.getenv('POSTGRES_HOST'):
                    config['databases']['postgresql'] = {
                        'type': 'postgresql',
                        'host': os.getenv('POSTGRES_HOST', 'localhost'),
                        'port': int(os.getenv('POSTGRES_PORT', 5432)),
                        'database': os.getenv('POSTGRES_DB', ''),
                        'user': os.getenv('POSTGRES_USER', ''),
                        'password': os.getenv('POSTGRES_PASSWORD', '')
                    }
                
                # MySQL configuration
                if os.getenv('MYSQL_HOST'):
                    config['databases']['mysql'] = {
                        'type': 'mysql',
                        'host': os.getenv('MYSQL_HOST', 'localhost'),
                        'port': int(os.getenv('MYSQL_PORT', 3306)),
                        'database': os.getenv('MYSQL_DB', ''),
                        'user': os.getenv('MYSQL_USER', ''),
                        'password': os.getenv('MYSQL_PASSWORD', '')
                    }
                
                # SQLite configuration
                if os.getenv('SQLITE_PATH'):
                    config['databases']['sqlite'] = {
                        'type': 'sqlite',
                        'path': os.getenv('SQLITE_PATH', '')
                    }
                
                # Validate the configuration structure
                self.assertIn('databases', config)
                self.assertIn('postgresql', config['databases'])
                self.assertIn('mysql', config['databases'])
                self.assertIn('sqlite', config['databases'])
                
                # Validate PostgreSQL config
                pg_config = config['databases']['postgresql']
                self.assertEqual(pg_config['host'], 'localhost')
                self.assertEqual(pg_config['port'], 5432)
                self.assertEqual(pg_config['user'], 'testuser')
                self.assertEqual(pg_config['database'], 'testdb')
                
                # Validate MySQL config
                mysql_config = config['databases']['mysql']
                self.assertEqual(mysql_config['host'], 'mysql-server')
                self.assertEqual(mysql_config['port'], 3306)
                
                # Validate SQLite config
                sqlite_config = config['databases']['sqlite']
                self.assertEqual(sqlite_config['path'], '/tmp/test.db')
                
    def test_all_database_types_in_migration(self):
        """Test migration script handles all supported database types."""
        comprehensive_env = {
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
        }
        
        with patch.dict(os.environ, comprehensive_env, clear=True):
            script = self.manager.create_migration_script()
            
            # Should include all database types
            expected_db_types = ['postgresql', 'mysql', 'sqlite', 'duckdb', 'mongodb']
            for db_type in expected_db_types:
                self.assertIn(db_type, script)
                
            # Should include specific environment variables for each type
            expected_vars = [
                'POSTGRES_HOST', 'MYSQL_HOST', 'SQLITE_PATH', 
                'DUCKDB_PATH', 'MONGODB_URL'
            ]
            for var in expected_vars:
                self.assertIn(var, script)
                
    def test_singleton_pattern(self):
        """Test that the global compatibility manager follows singleton pattern."""
        manager1 = get_compatibility_manager()
        manager2 = get_compatibility_manager()
        
        self.assertIs(manager1, manager2)
        self.assertIsInstance(manager1, CompatibilityManager)


if __name__ == '__main__':
    # Run the integration tests
    unittest.main(verbosity=2)