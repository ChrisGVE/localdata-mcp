"""Comprehensive tests for SecurityManager enterprise-grade security system.

Tests cover all security features including:
- Query fingerprinting and audit logging
- Rate limiting and resource protection
- Attack pattern detection and threat classification
- Integration with existing query parser and timeout systems
- OWASP SQL injection test cases
"""

import hashlib
import pytest
import time
import threading
from unittest.mock import Mock, patch, MagicMock

from src.localdata_mcp.security_manager import (
    SecurityManager, SecurityConfig, SecurityEvent, SecurityEventType,
    SecurityThreatLevel, AttackPattern, QueryComplexity, RateLimitState,
    get_security_manager, initialize_security_manager, validate_query_security
)
from src.localdata_mcp.query_parser import SQLSecurityError


class TestSecurityConfig:
    """Test security configuration validation."""
    
    def test_default_config(self):
        """Test default security configuration."""
        config = SecurityConfig()
        assert config.queries_per_minute == 60
        assert config.queries_per_hour == 1000
        assert config.burst_limit == 10
        assert config.max_query_length == 10000
        assert config.audit_enabled is True
        assert config.enable_pattern_detection is True
    
    def test_custom_config(self):
        """Test custom security configuration."""
        config = SecurityConfig(
            queries_per_minute=30,
            queries_per_hour=500,
            burst_limit=5,
            max_query_length=5000,
            audit_enabled=False
        )
        assert config.queries_per_minute == 30
        assert config.queries_per_hour == 500
        assert config.audit_enabled is False
    
    def test_config_validation(self):
        """Test configuration validation."""
        with pytest.raises(ValueError, match="queries_per_minute must be positive"):
            SecurityConfig(queries_per_minute=0)
        
        with pytest.raises(ValueError, match="burst_limit must be positive"):
            SecurityConfig(burst_limit=-1)
        
        with pytest.raises(ValueError, match="max_query_length must be positive"):
            SecurityConfig(max_query_length=0)


class TestSecurityManager:
    """Test SecurityManager core functionality."""
    
    @pytest.fixture
    def security_config(self):
        """Test security configuration."""
        return SecurityConfig(
            queries_per_minute=5,
            queries_per_hour=50,
            burst_limit=2,
            max_query_length=1000,
            max_joins=3,
            max_subqueries=2,
            audit_enabled=True,
            enable_pattern_detection=True,
            block_suspicious_patterns=True
        )
    
    @pytest.fixture
    def security_manager(self, security_config):
        """Test security manager instance."""
        return SecurityManager(security_config)
    
    def test_initialization(self, security_manager):
        """Test security manager initialization."""
        assert security_manager.config is not None
        assert security_manager.query_parser is not None
        assert security_manager.timeout_manager is not None
        assert security_manager.connection_manager is not None
        assert len(security_manager._attack_patterns) > 0
    
    def test_query_fingerprinting(self, security_manager):
        """Test query fingerprinting functionality."""
        query1 = "SELECT * FROM users WHERE id = 1"
        query2 = "SELECT   *   FROM users   WHERE id = 1  "  # Different whitespace
        query3 = "SELECT * FROM users WHERE id = 2"  # Different content
        
        fingerprint1 = security_manager.create_query_fingerprint(query1)
        fingerprint2 = security_manager.create_query_fingerprint(query2)
        fingerprint3 = security_manager.create_query_fingerprint(query3)
        
        # Same normalized query should have same fingerprint
        assert fingerprint1 == fingerprint2
        
        # Different queries should have different fingerprints
        assert fingerprint1 != fingerprint3
        
        # Fingerprints should be 16-character hex strings
        assert len(fingerprint1) == 16
        assert all(c in '0123456789abcdef' for c in fingerprint1)
    
    def test_query_fingerprinting_with_comments(self, security_manager):
        """Test fingerprinting removes comments correctly."""
        query1 = "SELECT * FROM users WHERE id = 1"
        query2 = "SELECT * FROM users /* comment */ WHERE id = 1"
        query3 = "SELECT * FROM users -- line comment\n WHERE id = 1"
        
        fingerprint1 = security_manager.create_query_fingerprint(query1)
        fingerprint2 = security_manager.create_query_fingerprint(query2)
        fingerprint3 = security_manager.create_query_fingerprint(query3)
        
        # All should normalize to the same fingerprint
        assert fingerprint1 == fingerprint2 == fingerprint3
    
    def test_query_complexity_analysis(self, security_manager):
        """Test query complexity analysis."""
        # Simple query
        simple_query = "SELECT id FROM users"
        complexity = security_manager.analyze_query_complexity(simple_query)
        
        assert complexity.length == len(simple_query)
        assert complexity.joins == 0
        assert complexity.subqueries == 0
        assert complexity.unions == 0
        
        # Complex query
        complex_query = """
        SELECT u.id, p.name FROM users u 
        INNER JOIN profiles p ON u.id = p.user_id
        LEFT JOIN settings s ON u.id = s.user_id
        WHERE u.active = 1 AND p.verified = 1
        UNION
        SELECT u2.id, 'default' FROM users u2 WHERE u2.id NOT IN (SELECT user_id FROM profiles)
        """
        complexity = security_manager.analyze_query_complexity(complex_query)
        
        assert complexity.joins >= 2
        assert complexity.subqueries >= 1
        assert complexity.unions >= 1
        assert complexity.complexity_score > 0
        assert complexity.tables >= 2
    
    def test_attack_pattern_detection(self, security_manager):
        """Test SQL injection attack pattern detection."""
        test_cases = {
            AttackPattern.UNION_INJECTION: "SELECT * FROM users UNION SELECT password FROM admin",
            AttackPattern.TIME_BASED_BLIND: "SELECT * FROM users WHERE id = 1 AND sleep(5)",
            AttackPattern.BOOLEAN_BLIND: "SELECT * FROM users WHERE id = 1 AND 1=1",
            AttackPattern.INFORMATION_EXTRACTION: "SELECT * FROM information_schema.tables",
            AttackPattern.COMMENT_INJECTION: "SELECT * FROM users WHERE id = 1 /* comment */",
            AttackPattern.STACKED_QUERIES: "SELECT * FROM users; DROP TABLE admin;",
            AttackPattern.FUNCTION_ABUSE: "SELECT load_file('/etc/passwd')",
        }
        
        for expected_pattern, query in test_cases.items():
            detected_patterns = security_manager.detect_attack_patterns(query)
            assert expected_pattern in detected_patterns, f"Failed to detect {expected_pattern.value} in query: {query}"
    
    def test_rate_limiting_minute_limit(self, security_manager):
        """Test rate limiting per minute."""
        connection_id = "test_connection"
        database_name = "test_db"
        
        # Should allow queries up to the limit
        for i in range(security_manager.config.queries_per_minute):
            allowed, error = security_manager.check_rate_limits(connection_id, database_name)
            assert allowed, f"Query {i+1} should be allowed"
            assert error is None
        
        # Next query should be blocked
        allowed, error = security_manager.check_rate_limits(connection_id, database_name)
        assert not allowed
        assert "Rate limit exceeded" in error
        assert "queries per minute" in error
    
    def test_rate_limiting_burst_limit(self, security_manager):
        """Test burst rate limiting."""
        connection_id = "test_connection"
        database_name = "test_db"
        
        # Rapid queries should trigger burst limit
        start_time = time.time()
        for i in range(security_manager.config.burst_limit + 2):
            allowed, error = security_manager.check_rate_limits(connection_id, database_name)
            if i <= security_manager.config.burst_limit:
                assert allowed or "Burst limit exceeded" in (error or "")
            else:
                assert not allowed
                assert "Burst limit exceeded" in error
            
            # Small delay but still within burst window
            time.sleep(0.1)
    
    def test_rate_limiting_cleanup(self, security_manager):
        """Test rate limiting cleanup of old queries."""
        connection_id = "test_connection"
        database_name = "test_db"
        
        rate_state = security_manager._rate_limits[connection_id]
        current_time = time.time()
        
        # Add some old queries
        rate_state.queries_this_minute.extend([current_time - 120, current_time - 90])
        rate_state.queries_this_hour.extend([current_time - 4000, current_time - 3700])
        
        assert len(rate_state.queries_this_minute) == 2
        assert len(rate_state.queries_this_hour) == 2
        
        # Cleanup should remove old queries
        rate_state.cleanup_old_queries(current_time)
        
        assert len(rate_state.queries_this_minute) == 0
        assert len(rate_state.queries_this_hour) == 0
    
    @patch('psutil.virtual_memory')
    @patch('psutil.cpu_percent')
    def test_resource_limits(self, mock_cpu, mock_memory, security_manager):
        """Test resource limit checking."""
        database_name = "test_db"
        
        # Mock normal resource usage
        mock_memory.return_value = Mock(used=100 * 1024 * 1024)  # 100MB
        mock_cpu.return_value = 30.0  # 30%
        
        allowed, error = security_manager.check_resource_limits(database_name)
        assert allowed
        assert error is None
        
        # Mock high memory usage
        mock_memory.return_value = Mock(used=600 * 1024 * 1024)  # 600MB (exceeds 500MB threshold)
        
        allowed, error = security_manager.check_resource_limits(database_name)
        assert not allowed
        assert "memory threshold exceeded" in error.lower()
        
        # Reset memory, test high CPU
        mock_memory.return_value = Mock(used=100 * 1024 * 1024)
        mock_cpu.return_value = 90.0  # 90% (exceeds 80% threshold)
        
        allowed, error = security_manager.check_resource_limits(database_name)
        assert not allowed
        assert "cpu threshold exceeded" in error.lower()
    
    def test_comprehensive_security_validation_valid_query(self, security_manager):
        """Test comprehensive security validation for valid query."""
        query = "SELECT id, name FROM users WHERE active = 1 LIMIT 10"
        database_name = "test_db"
        connection_id = "test_connection"
        
        with patch('psutil.virtual_memory') as mock_memory, \
             patch('psutil.cpu_percent') as mock_cpu:
            
            mock_memory.return_value = Mock(used=100 * 1024 * 1024)
            mock_cpu.return_value = 30.0
            
            is_valid, error_msg, metadata = security_manager.validate_query_security(
                query, database_name, connection_id
            )
            
            assert is_valid
            assert error_msg is None
            assert metadata['fingerprint'] is not None
            assert metadata['threat_level'] == SecurityThreatLevel.LOW
            assert 'basic_sql_validation' in metadata['checks_performed']
            assert 'rate_limiting' in metadata['checks_performed']
            assert 'resource_limits' in metadata['checks_performed']
            assert 'complexity_analysis' in metadata['checks_performed']
    
    def test_comprehensive_security_validation_malicious_query(self, security_manager):
        """Test comprehensive security validation for malicious query."""
        query = "SELECT * FROM users UNION SELECT password FROM admin"
        database_name = "test_db"
        connection_id = "test_connection"
        
        with patch('psutil.virtual_memory') as mock_memory, \
             patch('psutil.cpu_percent') as mock_cpu:
            
            mock_memory.return_value = Mock(used=100 * 1024 * 1024)
            mock_cpu.return_value = 30.0
            
            is_valid, error_msg, metadata = security_manager.validate_query_security(
                query, database_name, connection_id
            )
            
            assert not is_valid
            assert "Suspicious patterns detected" in error_msg
            assert metadata['threat_level'] == SecurityThreatLevel.CRITICAL
            assert AttackPattern.UNION_INJECTION.value in metadata['attack_patterns']
    
    def test_comprehensive_security_validation_complex_query(self, security_manager):
        """Test validation for overly complex query."""
        # Create a query that exceeds complexity limits
        query = "SELECT * FROM " + " INNER JOIN ".join([f"table{i}" for i in range(10)]) + " WHERE 1=1"
        database_name = "test_db"
        connection_id = "test_connection"
        
        with patch('psutil.virtual_memory') as mock_memory, \
             patch('psutil.cpu_percent') as mock_cpu:
            
            mock_memory.return_value = Mock(used=100 * 1024 * 1024)
            mock_cpu.return_value = 30.0
            
            is_valid, error_msg, metadata = security_manager.validate_query_security(
                query, database_name, connection_id
            )
            
            assert not is_valid
            assert "Too many joins" in error_msg
            assert metadata['threat_level'] == SecurityThreatLevel.MEDIUM
    
    def test_secure_query_execution_context(self, security_manager):
        """Test secure query execution context manager."""
        query = "SELECT id FROM users LIMIT 1"
        database_name = "test_db"
        connection_id = "test_connection"
        
        with patch('psutil.virtual_memory') as mock_memory, \
             patch('psutil.cpu_percent') as mock_cpu:
            
            mock_memory.return_value = Mock(used=100 * 1024 * 1024)
            mock_cpu.return_value = 30.0
            
            with security_manager.secure_query_execution(query, database_name, connection_id) as context:
                assert context['query_fingerprint'] is not None
                assert context['database_name'] == database_name
                assert context['connection_id'] == connection_id
                assert context['security_metadata'] is not None
                
                # Simulate some work
                time.sleep(0.01)
            
            # Context should have execution metadata after completion
            assert 'execution_time' in context['execution_metadata']
            assert context['execution_metadata']['success'] is True
    
    def test_secure_query_execution_context_with_exception(self, security_manager):
        """Test secure query execution context manager with exception."""
        query = "SELECT id FROM users LIMIT 1"
        database_name = "test_db"
        connection_id = "test_connection"
        
        with patch('psutil.virtual_memory') as mock_memory, \
             patch('psutil.cpu_percent') as mock_cpu:
            
            mock_memory.return_value = Mock(used=100 * 1024 * 1024)
            mock_cpu.return_value = 30.0
            
            with pytest.raises(ValueError):
                with security_manager.secure_query_execution(query, database_name, connection_id) as context:
                    raise ValueError("Test exception")
            
            # Context should have execution metadata after exception
            assert 'execution_time' in context['execution_metadata']
            assert context['execution_metadata']['success'] is False
            assert 'error' in context['execution_metadata']
    
    def test_secure_query_execution_blocked_query(self, security_manager):
        """Test secure query execution with blocked query."""
        query = "INSERT INTO users (name) VALUES ('hacker')"
        database_name = "test_db"
        connection_id = "test_connection"
        
        with pytest.raises(SQLSecurityError, match="Security validation failed"):
            with security_manager.secure_query_execution(query, database_name, connection_id):
                pass
    
    def test_security_event_logging(self, security_manager):
        """Test security event logging functionality."""
        # Generate some security events
        query = "SELECT * FROM users UNION SELECT password FROM admin"
        database_name = "test_db"
        connection_id = "test_connection"
        
        with patch('psutil.virtual_memory') as mock_memory, \
             patch('psutil.cpu_percent') as mock_cpu:
            
            mock_memory.return_value = Mock(used=100 * 1024 * 1024)
            mock_cpu.return_value = 30.0
            
            # This should generate security events
            security_manager.validate_query_security(query, database_name, connection_id)
        
        # Check that events were logged
        events = security_manager.get_security_events()
        assert len(events) > 0
        
        # Find the injection attempt event
        injection_events = [e for e in events if e.event_type == SecurityEventType.INJECTION_ATTEMPT]
        assert len(injection_events) > 0
        
        event = injection_events[0]
        assert event.threat_level == SecurityThreatLevel.CRITICAL
        assert event.attack_pattern == AttackPattern.UNION_INJECTION
        assert event.database_name == database_name
        assert event.connection_id == connection_id
    
    def test_security_event_filtering(self, security_manager):
        """Test security event filtering."""
        # Generate multiple types of events
        with patch('psutil.virtual_memory') as mock_memory, \
             patch('psutil.cpu_percent') as mock_cpu:
            
            mock_memory.return_value = Mock(used=100 * 1024 * 1024)
            mock_cpu.return_value = 30.0
            
            # Valid query (audit log event)
            security_manager.validate_query_security(
                "SELECT id FROM users", "test_db", "connection1"
            )
            
            # Malicious query (injection attempt event)
            security_manager.validate_query_security(
                "SELECT * FROM users UNION SELECT 1", "test_db", "connection2"
            )
        
        # Test filtering by event type
        injection_events = security_manager.get_security_events(
            event_types=[SecurityEventType.INJECTION_ATTEMPT]
        )
        assert all(e.event_type == SecurityEventType.INJECTION_ATTEMPT for e in injection_events)
        
        # Test filtering by threat level
        critical_events = security_manager.get_security_events(
            threat_levels=[SecurityThreatLevel.CRITICAL]
        )
        assert all(e.threat_level == SecurityThreatLevel.CRITICAL for e in critical_events)
        
        # Test limit
        limited_events = security_manager.get_security_events(limit=1)
        assert len(limited_events) <= 1
    
    def test_security_statistics(self, security_manager):
        """Test security statistics generation."""
        # Generate some events first
        with patch('psutil.virtual_memory') as mock_memory, \
             patch('psutil.cpu_percent') as mock_cpu:
            
            mock_memory.return_value = Mock(used=200 * 1024 * 1024)
            mock_cpu.return_value = 50.0
            
            # Valid query
            security_manager.validate_query_security(
                "SELECT id FROM users", "test_db", "connection1"
            )
            
            # Rate limit test
            for _ in range(security_manager.config.queries_per_minute + 1):
                security_manager.check_rate_limits("connection2", "test_db")
        
        stats = security_manager.get_security_statistics()
        
        # Check structure
        assert 'event_statistics' in stats
        assert 'rate_limiting' in stats
        assert 'resource_monitoring' in stats
        assert 'configuration' in stats
        
        # Check event statistics
        assert 'total_events' in stats['event_statistics']
        assert 'event_types' in stats['event_statistics']
        assert 'threat_levels' in stats['event_statistics']
        
        # Check rate limiting stats
        assert 'active_connections' in stats['rate_limiting']
        assert 'config' in stats['rate_limiting']
        
        # Check resource monitoring
        assert 'current_memory_mb' in stats['resource_monitoring']
        assert 'current_cpu_percent' in stats['resource_monitoring']
        
        # Check configuration
        assert stats['configuration']['audit_enabled'] == security_manager.config.audit_enabled


class TestOWASPSQLInjectionCases:
    """Test OWASP SQL injection test cases for comprehensive coverage."""
    
    @pytest.fixture
    def security_manager(self):
        """Security manager for OWASP testing."""
        config = SecurityConfig(
            queries_per_minute=100,
            enable_pattern_detection=True,
            block_suspicious_patterns=True
        )
        return SecurityManager(config)
    
    def test_union_based_injection(self, security_manager):
        """Test UNION-based SQL injection detection."""
        malicious_queries = [
            "SELECT * FROM users WHERE id = 1 UNION SELECT username, password FROM admin",
            "SELECT name FROM products WHERE id = 1 UNION ALL SELECT credit_card FROM payments",
            "' UNION SELECT null, username, password FROM admin--",
            "1' UNION SELECT @@version, user()--"
        ]
        
        for query in malicious_queries:
            detected_patterns = security_manager.detect_attack_patterns(query)
            assert AttackPattern.UNION_INJECTION in detected_patterns, f"Failed to detect UNION injection in: {query}"
    
    def test_time_based_blind_injection(self, security_manager):
        """Test time-based blind SQL injection detection."""
        malicious_queries = [
            "SELECT * FROM users WHERE id = 1 AND (SELECT COUNT(*) FROM information_schema.tables WHERE sleep(5))",
            "1'; WAITFOR DELAY '0:0:5'--",
            "1' AND (SELECT * FROM (SELECT COUNT(*), CONCAT(version(), FLOOR(RAND(0)*2)) x FROM information_schema.tables GROUP BY x) a) AND sleep(5)--",
            "1' AND pg_sleep(5)--",
            "1' AND BENCHMARK(5000000, MD5('test'))--"
        ]
        
        for query in malicious_queries:
            detected_patterns = security_manager.detect_attack_patterns(query)
            assert AttackPattern.TIME_BASED_BLIND in detected_patterns, f"Failed to detect time-based blind injection in: {query}"
    
    def test_boolean_based_blind_injection(self, security_manager):
        """Test boolean-based blind SQL injection detection."""
        malicious_queries = [
            "SELECT * FROM users WHERE id = 1 AND 1=1",
            "SELECT * FROM users WHERE id = 1 AND 1=2",
            "1' AND '1'='1",
            "1' OR '1'='1",
            "1' AND 'abc'='abc"
        ]
        
        for query in malicious_queries:
            detected_patterns = security_manager.detect_attack_patterns(query)
            assert AttackPattern.BOOLEAN_BLIND in detected_patterns, f"Failed to detect boolean-based blind injection in: {query}"
    
    def test_error_based_injection(self, security_manager):
        """Test error-based SQL injection detection."""
        malicious_queries = [
            "SELECT * FROM users WHERE id = CAST('abc' AS INT)",
            "1' AND EXTRACTVALUE(1, CONCAT(0x7e, (SELECT version()), 0x7e))--",
            "1' AND UPDATEXML(1, CONCAT(0x7e, (SELECT user()), 0x7e), 1)--",
            "' AND (SELECT * FROM (SELECT COUNT(*), CONCAT(version(), FLOOR(RAND(0)*2)) x FROM information_schema.tables GROUP BY x) a)--"
        ]
        
        for query in malicious_queries:
            detected_patterns = security_manager.detect_attack_patterns(query)
            assert AttackPattern.ERROR_BASED in detected_patterns, f"Failed to detect error-based injection in: {query}"
    
    def test_information_schema_exploitation(self, security_manager):
        """Test information schema exploitation detection."""
        malicious_queries = [
            "SELECT * FROM information_schema.tables",
            "SELECT column_name FROM information_schema.columns WHERE table_name = 'users'",
            "SELECT * FROM sys.tables",
            "SELECT * FROM sys.columns",
            "SELECT name FROM sqlite_master WHERE type = 'table'",
            "SELECT * FROM pg_tables"
        ]
        
        for query in malicious_queries:
            detected_patterns = security_manager.detect_attack_patterns(query)
            assert AttackPattern.INFORMATION_EXTRACTION in detected_patterns, f"Failed to detect information extraction in: {query}"
    
    def test_stacked_queries_injection(self, security_manager):
        """Test stacked queries injection detection."""
        malicious_queries = [
            "SELECT * FROM users; DROP TABLE admin;",
            "1'; INSERT INTO admin (username, password) VALUES ('hacker', 'password'); --",
            "1'; UPDATE users SET password = 'hacked' WHERE id = 1; --",
            "SELECT id FROM users; DELETE FROM logs; --"
        ]
        
        for query in malicious_queries:
            detected_patterns = security_manager.detect_attack_patterns(query)
            assert AttackPattern.STACKED_QUERIES in detected_patterns, f"Failed to detect stacked queries in: {query}"
    
    def test_function_abuse_injection(self, security_manager):
        """Test function abuse injection detection."""
        malicious_queries = [
            "SELECT load_file('/etc/passwd')",
            "SELECT * FROM users INTO OUTFILE '/var/www/dump.txt'",
            "SELECT * FROM users INTO DUMPFILE '/var/www/shell.php'",
            "EXEC xp_cmdshell('dir')",
            "SELECT eval('malicious_code')"
        ]
        
        for query in malicious_queries:
            detected_patterns = security_manager.detect_attack_patterns(query)
            assert AttackPattern.FUNCTION_ABUSE in detected_patterns, f"Failed to detect function abuse in: {query}"
    
    def test_comment_injection_detection(self, security_manager):
        """Test SQL comment injection detection."""
        malicious_queries = [
            "SELECT * FROM users WHERE id = 1 -- AND password = 'secret'",
            "SELECT * FROM users WHERE id = 1 /* bypass password check */ OR 1=1",
            "SELECT * FROM users WHERE id = 1 # ignore the rest",
            "1' OR '1'='1' --",
            "1' OR '1'='1' /*"
        ]
        
        for query in malicious_queries:
            detected_patterns = security_manager.detect_attack_patterns(query)
            assert AttackPattern.COMMENT_INJECTION in detected_patterns, f"Failed to detect comment injection in: {query}"
    
    def test_comprehensive_owasp_validation(self, security_manager):
        """Test comprehensive validation of OWASP attack patterns."""
        owasp_test_cases = [
            # Classic SQL injection
            ("1' OR '1'='1", [AttackPattern.BOOLEAN_BLIND, AttackPattern.COMMENT_INJECTION]),
            
            # Union-based information disclosure
            ("' UNION SELECT username, password FROM admin--", [AttackPattern.UNION_INJECTION, AttackPattern.COMMENT_INJECTION]),
            
            # Time-based blind injection
            ("1' AND sleep(5)--", [AttackPattern.TIME_BASED_BLIND, AttackPattern.COMMENT_INJECTION]),
            
            # Information schema exploitation
            ("' UNION SELECT table_name FROM information_schema.tables--", 
             [AttackPattern.UNION_INJECTION, AttackPattern.INFORMATION_EXTRACTION, AttackPattern.COMMENT_INJECTION]),
            
            # Stacked queries with data manipulation
            ("1'; DROP TABLE users; --", [AttackPattern.STACKED_QUERIES, AttackPattern.COMMENT_INJECTION]),
            
            # File system access
            ("1' UNION SELECT load_file('/etc/passwd')--", 
             [AttackPattern.UNION_INJECTION, AttackPattern.FUNCTION_ABUSE, AttackPattern.COMMENT_INJECTION])
        ]
        
        with patch('psutil.virtual_memory') as mock_memory, \
             patch('psutil.cpu_percent') as mock_cpu:
            
            mock_memory.return_value = Mock(used=100 * 1024 * 1024)
            mock_cpu.return_value = 30.0
            
            for malicious_query, expected_patterns in owasp_test_cases:
                # Validate that security manager blocks these
                is_valid, error_msg, metadata = security_manager.validate_query_security(
                    f"SELECT * FROM users WHERE id = {malicious_query}",
                    "test_db",
                    "test_connection"
                )
                
                assert not is_valid, f"Should block malicious query: {malicious_query}"
                assert metadata['threat_level'] in [SecurityThreatLevel.CRITICAL, SecurityThreatLevel.HIGH], \
                    f"Should classify as high/critical threat: {malicious_query}"
                
                # Check that expected patterns are detected
                detected_patterns = security_manager.detect_attack_patterns(malicious_query)
                for expected_pattern in expected_patterns:
                    assert expected_pattern in detected_patterns, \
                        f"Should detect {expected_pattern.value} in: {malicious_query}"


class TestGlobalSecurityManagerFunctions:
    """Test global security manager functions."""
    
    def test_get_security_manager_singleton(self):
        """Test global security manager singleton."""
        manager1 = get_security_manager()
        manager2 = get_security_manager()
        
        assert manager1 is manager2
        assert isinstance(manager1, SecurityManager)
    
    def test_initialize_security_manager(self):
        """Test security manager initialization."""
        config = SecurityConfig(queries_per_minute=30)
        manager = initialize_security_manager(config)
        
        assert isinstance(manager, SecurityManager)
        assert manager.config.queries_per_minute == 30
        
        # Should replace global instance
        global_manager = get_security_manager()
        assert global_manager is manager
    
    def test_validate_query_security_convenience_function(self):
        """Test convenience function for query validation."""
        with patch('psutil.virtual_memory') as mock_memory, \
             patch('psutil.cpu_percent') as mock_cpu:
            
            mock_memory.return_value = Mock(used=100 * 1024 * 1024)
            mock_cpu.return_value = 30.0
            
            # Valid query
            is_valid, error = validate_query_security(
                "SELECT id FROM users", "test_db", "connection1"
            )
            assert is_valid
            assert error is None
            
            # Invalid query
            is_valid, error = validate_query_security(
                "INSERT INTO users VALUES (1)", "test_db", "connection1"
            )
            assert not is_valid
            assert error is not None


class TestSecurityIntegration:
    """Test integration with existing security systems."""
    
    def test_integration_with_query_parser(self):
        """Test integration with existing QueryParser from Task 1."""
        security_manager = SecurityManager()
        
        # Test that basic SQL validation is called
        invalid_query = "INSERT INTO users VALUES (1)"
        is_valid, error, metadata = security_manager.validate_query_security(
            invalid_query, "test_db"
        )
        
        assert not is_valid
        assert 'basic_sql_validation' in metadata['checks_performed']
        # Should be blocked by basic SQL validation before advanced checks
        assert metadata['threat_level'] == SecurityThreatLevel.HIGH
    
    def test_integration_with_timeout_manager(self):
        """Test integration with TimeoutManager from Task 6."""
        security_manager = SecurityManager()
        
        # Timeout manager should be accessible
        assert security_manager.timeout_manager is not None
        
        # SecurityManager should work with timeout system
        # (Full integration testing would require actual query execution)
    
    def test_integration_with_connection_manager(self):
        """Test integration with EnhancedConnectionManager."""
        security_manager = SecurityManager()
        
        # Connection manager should be accessible
        assert security_manager.connection_manager is not None
        
        # Test resource checking uses connection manager data
        # (Full integration testing would require actual connections)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])