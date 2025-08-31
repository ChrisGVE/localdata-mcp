"""Integration tests for complete SecurityManager pipeline.

Tests the full end-to-end security workflow from enhanced database tools
through all security layers including Task 1 (QueryParser), Task 6 (TimeoutManager),
and Task 10 (SecurityManager) integration.
"""

import json
import pytest
import time
from unittest.mock import Mock, patch, MagicMock

from src.localdata_mcp.enhanced_database_tools import EnhancedDatabaseTools
from src.localdata_mcp.security_manager import (
    SecurityManager, SecurityConfig, SecurityEventType, SecurityThreatLevel, 
    get_security_manager
)
from src.localdata_mcp.config_manager import DatabaseConfig, DatabaseType


class TestSecurityIntegration:
    """Test complete security integration with enhanced database tools."""
    
    @pytest.fixture
    def mock_database_config(self):
        """Mock database configuration for testing."""
        return DatabaseConfig(
            name="test_db",
            type=DatabaseType.SQLITE,
            connection_string=":memory:",
            enabled=True,
            max_connections=5,
            connection_timeout=30,
            query_timeout=300
        )
    
    @pytest.fixture
    def enhanced_tools(self):
        """Enhanced database tools instance for testing."""
        return EnhancedDatabaseTools()
    
    def test_complete_security_pipeline_valid_query(self, enhanced_tools, mock_database_config):
        """Test complete security pipeline with a valid query."""
        with patch('src.localdata_mcp.enhanced_database_tools.get_enhanced_connection_manager') as mock_conn_mgr, \
             patch('src.localdata_mcp.enhanced_database_tools.get_config_manager') as mock_config_mgr, \
             patch('psutil.virtual_memory') as mock_memory, \
             patch('psutil.cpu_percent') as mock_cpu:
            
            # Setup mocks
            mock_config_mgr.return_value.get_database_config.return_value = mock_database_config
            
            # Mock engine and connection manager
            mock_engine = Mock()
            mock_conn_mgr.return_value.get_engine.return_value = mock_engine
            mock_conn_mgr.return_value.managed_query_execution.return_value.__enter__ = Mock(return_value={
                'start_time': time.time(),
                'query_id': 'test_123'
            })
            mock_conn_mgr.return_value.managed_query_execution.return_value.__exit__ = Mock(return_value=None)
            
            # Mock resource monitoring
            mock_memory.return_value = Mock(used=100 * 1024 * 1024)  # 100MB
            mock_cpu.return_value = 30.0  # 30%
            
            # Mock streaming executor
            with patch('src.localdata_mcp.enhanced_database_tools.StreamingQueryExecutor') as mock_executor_class:
                mock_executor = Mock()
                mock_executor_class.return_value = mock_executor
                
                # Mock successful query execution
                import pandas as pd
                mock_df = pd.DataFrame({'id': [1, 2, 3], 'name': ['Alice', 'Bob', 'Charlie']})
                mock_executor.execute_streaming.return_value = (
                    mock_df, 
                    {'total_rows_processed': 3}
                )
                
                # Mock create_streaming_source
                with patch('src.localdata_mcp.enhanced_database_tools.create_streaming_source') as mock_streaming_source:
                    mock_streaming_source.return_value = Mock()
                    
                    # Execute valid query
                    query = "SELECT id, name FROM users WHERE active = 1 LIMIT 10"
                    result = enhanced_tools.execute_enhanced_query("test_db", query)
                    
                    # Verify no errors
                    assert 'error' not in result
                    assert 'data' in result
                    assert 'security' in result
                    
                    # Verify security metadata
                    security = result['security']
                    assert 'fingerprint' in security
                    assert security['threat_level'] == 'low'
                    assert 'basic_sql_validation' in security['checks_performed']
                    assert 'rate_limiting' in security['checks_performed']
                    assert 'resource_limits' in security['checks_performed']
                    assert 'complexity_analysis' in security['checks_performed']
                    assert 'attack_pattern_detection' in security['checks_performed']
                    
                    # Verify data was returned
                    assert len(result['data']) == 3
                    assert result['metadata']['total_rows'] == 3
    
    def test_complete_security_pipeline_malicious_query(self, enhanced_tools, mock_database_config):
        """Test complete security pipeline blocks malicious queries."""
        with patch('src.localdata_mcp.enhanced_database_tools.get_config_manager') as mock_config_mgr, \
             patch('psutil.virtual_memory') as mock_memory, \
             patch('psutil.cpu_percent') as mock_cpu:
            
            # Setup mocks
            mock_config_mgr.return_value.get_database_config.return_value = mock_database_config
            mock_memory.return_value = Mock(used=100 * 1024 * 1024)
            mock_cpu.return_value = 30.0
            
            # Test UNION injection attack
            malicious_query = "SELECT * FROM users UNION SELECT password FROM admin_users"
            result = enhanced_tools.execute_enhanced_query("test_db", malicious_query)
            
            # Verify query is blocked
            assert 'error' in result
            assert 'Advanced Security Error' in result['error']
            assert 'Suspicious patterns detected' in result['error']
            assert 'security_metadata' in result
            assert result['security_metadata']['threat_level'].value == 'critical'
    
    def test_complete_security_pipeline_basic_sql_violation(self, enhanced_tools, mock_database_config):
        """Test that basic SQL violations are caught before advanced security."""
        with patch('src.localdata_mcp.enhanced_database_tools.get_config_manager') as mock_config_mgr:
            
            # Setup mocks
            mock_config_mgr.return_value.get_database_config.return_value = mock_database_config
            
            # Test non-SELECT query (should be blocked by basic validation)
            insert_query = "INSERT INTO users (name) VALUES ('hacker')"
            result = enhanced_tools.execute_enhanced_query("test_db", insert_query)
            
            # Verify query is blocked by basic validation
            assert 'error' in result
            assert 'Basic Security Error' in result['error']
            # Should not reach advanced security validation
            assert 'security_metadata' not in result
    
    def test_security_rate_limiting_integration(self, enhanced_tools, mock_database_config):
        """Test rate limiting integration in complete pipeline."""
        with patch('src.localdata_mcp.enhanced_database_tools.get_config_manager') as mock_config_mgr, \
             patch('psutil.virtual_memory') as mock_memory, \
             patch('psutil.cpu_percent') as mock_cpu:
            
            # Setup restrictive security config for testing
            security_config = SecurityConfig(
                queries_per_minute=2,  # Very low limit for testing
                queries_per_hour=10,
                burst_limit=1
            )
            
            # Replace security manager with test config
            with patch('src.localdata_mcp.enhanced_database_tools.get_security_manager') as mock_security_mgr:
                test_security_manager = SecurityManager(security_config)
                mock_security_mgr.return_value = test_security_manager
                
                mock_config_mgr.return_value.get_database_config.return_value = mock_database_config
                mock_memory.return_value = Mock(used=100 * 1024 * 1024)
                mock_cpu.return_value = 30.0
                
                connection_id = "test_rate_limit_connection"
                query = "SELECT id FROM users LIMIT 1"
                
                # First 2 queries should succeed
                for i in range(2):
                    result = enhanced_tools.execute_enhanced_query(
                        "test_db", query, connection_id=connection_id
                    )
                    # May still have errors due to missing engine, but not rate limit errors
                    if 'error' in result:
                        assert 'Rate limit exceeded' not in result['error']
                
                # Third query should be rate limited
                result = enhanced_tools.execute_enhanced_query(
                    "test_db", query, connection_id=connection_id
                )
                
                assert 'error' in result
                assert 'Rate limit exceeded' in result['error']
                assert 'security_metadata' in result
                assert result['security_metadata']['threat_level'].value == 'medium'
    
    def test_security_resource_limits_integration(self, enhanced_tools, mock_database_config):
        """Test resource limits integration in complete pipeline."""
        with patch('src.localdata_mcp.enhanced_database_tools.get_config_manager') as mock_config_mgr, \
             patch('psutil.virtual_memory') as mock_memory, \
             patch('psutil.cpu_percent') as mock_cpu:
            
            mock_config_mgr.return_value.get_database_config.return_value = mock_database_config
            
            # Mock high memory usage (exceeds default 500MB threshold)
            mock_memory.return_value = Mock(used=600 * 1024 * 1024)  # 600MB
            mock_cpu.return_value = 30.0
            
            query = "SELECT id FROM users LIMIT 1"
            result = enhanced_tools.execute_enhanced_query("test_db", query)
            
            # Should be blocked due to memory threshold
            assert 'error' in result
            assert 'memory threshold exceeded' in result['error'].lower()
            assert 'security_metadata' in result
            assert result['security_metadata']['threat_level'].value == 'high'
    
    def test_security_query_complexity_integration(self, enhanced_tools, mock_database_config):
        """Test query complexity limits integration."""
        with patch('src.localdata_mcp.enhanced_database_tools.get_config_manager') as mock_config_mgr, \
             patch('psutil.virtual_memory') as mock_memory, \
             patch('psutil.cpu_percent') as mock_cpu:
            
            # Setup restrictive complexity config
            security_config = SecurityConfig(
                max_joins=2,  # Low limit for testing
                max_subqueries=1
            )
            
            with patch('src.localdata_mcp.enhanced_database_tools.get_security_manager') as mock_security_mgr:
                test_security_manager = SecurityManager(security_config)
                mock_security_mgr.return_value = test_security_manager
                
                mock_config_mgr.return_value.get_database_config.return_value = mock_database_config
                mock_memory.return_value = Mock(used=100 * 1024 * 1024)
                mock_cpu.return_value = 30.0
                
                # Complex query that exceeds join limit
                complex_query = """
                    SELECT u.id, p.name, r.role, d.department
                    FROM users u 
                    JOIN profiles p ON u.id = p.user_id
                    JOIN user_roles r ON u.id = r.user_id  
                    JOIN departments d ON r.dept_id = d.id
                """
                
                result = enhanced_tools.execute_enhanced_query("test_db", complex_query)
                
                # Should be blocked due to complexity
                assert 'error' in result
                assert 'Too many joins' in result['error']
                assert 'security_metadata' in result
                assert result['security_metadata']['threat_level'].value == 'medium'
    
    def test_security_audit_logging_integration(self, enhanced_tools, mock_database_config):
        """Test that security events are properly logged during pipeline execution."""
        with patch('src.localdata_mcp.enhanced_database_tools.get_config_manager') as mock_config_mgr, \
             patch('psutil.virtual_memory') as mock_memory, \
             patch('psutil.cpu_percent') as mock_cpu:
            
            mock_config_mgr.return_value.get_database_config.return_value = mock_database_config
            mock_memory.return_value = Mock(used=100 * 1024 * 1024)
            mock_cpu.return_value = 30.0
            
            # Get security manager to check events
            security_manager = get_security_manager()
            initial_event_count = len(security_manager.get_security_events())
            
            # Execute a malicious query
            malicious_query = "SELECT * FROM users; DROP TABLE admin;"
            result = enhanced_tools.execute_enhanced_query("test_db", malicious_query)
            
            # Verify query was blocked
            assert 'error' in result
            
            # Verify security events were logged
            events = security_manager.get_security_events()
            assert len(events) > initial_event_count
            
            # Find the injection attempt event
            injection_events = [e for e in events if e.event_type == SecurityEventType.INJECTION_ATTEMPT]
            assert len(injection_events) > 0
            
            event = injection_events[0]
            assert event.threat_level == SecurityThreatLevel.CRITICAL
            assert event.database_name == "test_db"
            assert "stacked_queries" in str(event.attack_pattern).lower()
    
    def test_security_fingerprint_consistency(self, enhanced_tools, mock_database_config):
        """Test that query fingerprints are consistent across executions."""
        with patch('src.localdata_mcp.enhanced_database_tools.get_config_manager') as mock_config_mgr, \
             patch('psutil.virtual_memory') as mock_memory, \
             patch('psutil.cpu_percent') as mock_cpu:
            
            mock_config_mgr.return_value.get_database_config.return_value = mock_database_config
            mock_memory.return_value = Mock(used=100 * 1024 * 1024)
            mock_cpu.return_value = 30.0
            
            query1 = "SELECT id FROM users WHERE name = 'test'"
            query2 = "SELECT   id   FROM users   WHERE name = 'test'  "  # Same but different whitespace
            
            # Execute both queries (both should fail due to no engine, but we get security metadata)
            result1 = enhanced_tools.execute_enhanced_query("test_db", query1)
            result2 = enhanced_tools.execute_enhanced_query("test_db", query2)
            
            # Both should have same fingerprint (normalized)
            if 'security_metadata' in result1 and 'security_metadata' in result2:
                fingerprint1 = result1['security_metadata']['fingerprint']
                fingerprint2 = result2['security_metadata']['fingerprint']
                assert fingerprint1 == fingerprint2
    
    def test_security_mcp_tools_integration(self):
        """Test security-related MCP tools integration."""
        from src.localdata_mcp.enhanced_database_tools import (
            get_security_statistics, get_security_events, validate_query_security_standalone
        )
        
        # Test security statistics tool
        stats_result = get_security_statistics()
        stats_data = json.loads(stats_result)
        
        assert 'event_statistics' in stats_data
        assert 'rate_limiting' in stats_data
        assert 'resource_monitoring' in stats_data
        assert 'configuration' in stats_data
        
        # Test security events tool
        events_result = get_security_events(limit=10)
        events_data = json.loads(events_result)
        
        assert 'total_events' in events_data
        assert 'events' in events_data
        
        # Test standalone validation tool
        with patch('psutil.virtual_memory') as mock_memory, \
             patch('psutil.cpu_percent') as mock_cpu:
            
            mock_memory.return_value = Mock(used=100 * 1024 * 1024)
            mock_cpu.return_value = 30.0
            
            # Valid query
            valid_result = validate_query_security_standalone(
                "SELECT id FROM users", "test_db", "test_connection"
            )
            valid_data = json.loads(valid_result)
            
            assert valid_data['is_valid'] is True
            assert valid_data['error_message'] is None
            assert 'security_metadata' in valid_data
            
            # Malicious query
            malicious_result = validate_query_security_standalone(
                "SELECT * FROM users UNION SELECT password FROM admin", "test_db", "test_connection"
            )
            malicious_data = json.loads(malicious_result)
            
            assert malicious_data['is_valid'] is False
            assert malicious_data['error_message'] is not None
            assert 'Suspicious patterns detected' in malicious_data['error_message']
            assert malicious_data['security_metadata']['threat_level'] == 'critical'


class TestSecurityManagerLifecycle:
    """Test SecurityManager lifecycle and resource management."""
    
    def test_security_manager_singleton(self):
        """Test security manager singleton behavior."""
        manager1 = get_security_manager()
        manager2 = get_security_manager()
        
        assert manager1 is manager2
        assert isinstance(manager1, SecurityManager)
    
    def test_security_manager_initialization_with_config(self):
        """Test security manager initialization with custom config."""
        from src.localdata_mcp.security_manager import initialize_security_manager
        
        config = SecurityConfig(
            queries_per_minute=30,
            audit_enabled=False,
            enable_pattern_detection=False
        )
        
        manager = initialize_security_manager(config)
        
        assert manager.config.queries_per_minute == 30
        assert manager.config.audit_enabled is False
        assert manager.config.enable_pattern_detection is False
        
        # Should replace global instance
        global_manager = get_security_manager()
        assert global_manager is manager
    
    def test_security_manager_resource_cleanup(self):
        """Test security manager resource cleanup."""
        from src.localdata_mcp.security_manager import initialize_security_manager
        
        # Create manager with events
        manager = initialize_security_manager()
        
        # Generate some events
        with patch('psutil.virtual_memory') as mock_memory, \
             patch('psutil.cpu_percent') as mock_cpu:
            
            mock_memory.return_value = Mock(used=100 * 1024 * 1024)
            mock_cpu.return_value = 30.0
            
            manager.validate_query_security("SELECT id FROM users", "test_db", "test_conn")
        
        # Verify events exist
        events_before = len(manager.get_security_events())
        assert events_before > 0
        
        # Close manager
        manager.close()
        
        # Resource monitor should be stopped
        assert not manager._resource_monitor_active


if __name__ == "__main__":
    pytest.main([__file__, "-v"])