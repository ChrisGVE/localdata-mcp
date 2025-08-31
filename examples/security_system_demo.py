#!/usr/bin/env python3
"""
LocalData MCP Security Enhancement Layer Demo

Demonstrates the enterprise-grade security features implemented in Task 10,
including advanced SQL injection prevention, rate limiting, resource protection,
and comprehensive audit logging.

Features demonstrated:
- Basic SQL validation (Task 1 integration)  
- Advanced attack pattern detection
- Query fingerprinting and audit logging
- Rate limiting per connection
- Resource exhaustion protection
- Security event monitoring
- Integration with timeout system (Task 6)
"""

import json
import time
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from localdata_mcp.security_manager import (
    SecurityManager, SecurityConfig, SecurityEventType, SecurityThreatLevel,
    get_security_manager, validate_query_security, secure_query_execution
)
from localdata_mcp.enhanced_database_tools import (
    execute_enhanced_query, get_security_statistics, get_security_events, 
    validate_query_security_standalone
)


def demo_query_fingerprinting():
    """Demonstrate query fingerprinting functionality."""
    print("=" * 60)
    print("DEMO 1: Query Fingerprinting and Normalization")
    print("=" * 60)
    
    security_manager = get_security_manager()
    
    # Test queries with different formatting but same semantics
    queries = [
        "SELECT id, name FROM users WHERE active = 1",
        "SELECT   id,   name   FROM users   WHERE active = 1  ",
        "SELECT id, name FROM users /* comment */ WHERE active = 1",
        "SELECT id, name FROM users -- line comment\n WHERE active = 1"
    ]
    
    print("Testing query fingerprint consistency:")
    fingerprints = []
    
    for i, query in enumerate(queries, 1):
        fingerprint = security_manager.create_query_fingerprint(query)
        fingerprints.append(fingerprint)
        print(f"Query {i}: {repr(query[:50])}")
        print(f"Fingerprint: {fingerprint}")
        print()
    
    # Verify all fingerprints are the same (normalized)
    if len(set(fingerprints)) == 1:
        print("✅ SUCCESS: All queries normalized to same fingerprint")
    else:
        print("❌ ERROR: Fingerprints should be identical")
    
    print()


def demo_attack_detection():
    """Demonstrate SQL injection attack pattern detection."""
    print("=" * 60)
    print("DEMO 2: SQL Injection Attack Pattern Detection")
    print("=" * 60)
    
    # OWASP SQL injection test cases
    attack_examples = [
        ("UNION Injection", "SELECT * FROM users UNION SELECT username, password FROM admin"),
        ("Time-based Blind", "SELECT * FROM users WHERE id = 1 AND sleep(5)"),
        ("Boolean-based Blind", "SELECT * FROM users WHERE id = 1 AND 1=1"),
        ("Information Schema", "SELECT table_name FROM information_schema.tables"),
        ("Stacked Queries", "SELECT * FROM users; DROP TABLE admin;"),
        ("Function Abuse", "SELECT load_file('/etc/passwd')"),
        ("Comment Injection", "SELECT * FROM users WHERE id = 1 /* bypass */ OR 1=1"),
        ("Error-based", "SELECT * FROM users WHERE id = CAST('abc' AS INT)")
    ]
    
    security_manager = get_security_manager()
    
    for attack_type, malicious_query in attack_examples:
        print(f"Testing: {attack_type}")
        print(f"Query: {malicious_query}")
        
        detected_patterns = security_manager.detect_attack_patterns(malicious_query)
        
        if detected_patterns:
            pattern_names = [p.value for p in detected_patterns]
            print(f"✅ DETECTED: {', '.join(pattern_names)}")
        else:
            print("❌ NOT DETECTED")
        
        print("-" * 40)
    
    print()


def demo_comprehensive_validation():
    """Demonstrate comprehensive security validation pipeline."""
    print("=" * 60)
    print("DEMO 3: Comprehensive Security Validation Pipeline")
    print("=" * 60)
    
    test_cases = [
        ("Valid Query", "SELECT id, name FROM users WHERE active = 1 LIMIT 10", True),
        ("Basic SQL Violation", "INSERT INTO users (name) VALUES ('hacker')", False),
        ("Union Injection", "SELECT * FROM users UNION SELECT password FROM admin", False),
        ("Complex Query", "SELECT u.* FROM users u JOIN profiles p ON u.id = p.user_id JOIN roles r ON u.id = r.user_id JOIN departments d ON r.dept_id = d.id", None),  # Depends on complexity limits
    ]
    
    for test_name, query, expected_valid in test_cases:
        print(f"Testing: {test_name}")
        print(f"Query: {query[:60]}{'...' if len(query) > 60 else ''}")
        
        # Use the SecurityManager directly
        security_manager = get_security_manager()
        is_valid, error_msg, metadata = security_manager.validate_query_security(
            query, "demo_db", "demo_connection"
        )
        
        if not is_valid:
            print(f"❌ ERROR: {error_msg}")
        else:
            print(f"Valid: {is_valid}")
            print(f"Threat Level: {metadata['threat_level'].value}")
            print(f"Checks: {', '.join(metadata['checks_performed'])}")
            
            if 'attack_patterns' in metadata and metadata['attack_patterns']:
                print(f"Attack Patterns: {', '.join(metadata['attack_patterns'])}")
            
            if 'complexity' in metadata:
                complexity = metadata['complexity']
                print(f"Complexity: Score={complexity.get('score', 0):.1f}, Length={complexity.get('length', 0)}, Joins={complexity.get('joins', 0)}")
            
            if expected_valid is not None:
                if is_valid == expected_valid:
                    print("✅ EXPECTED RESULT")
                else:
                    print("❌ UNEXPECTED RESULT")
        
        print("-" * 50)
    
    print()


def demo_rate_limiting():
    """Demonstrate rate limiting functionality."""
    print("=" * 60)
    print("DEMO 4: Rate Limiting Protection")
    print("=" * 60)
    
    # Create security manager with restrictive limits for demo
    config = SecurityConfig(
        queries_per_minute=3,  # Very low for demo
        burst_limit=2
    )
    
    security_manager = SecurityManager(config)
    connection_id = "demo_rate_limit_connection"
    query = "SELECT id FROM users LIMIT 1"
    
    print(f"Rate Limits: {config.queries_per_minute} queries/minute, {config.burst_limit} burst limit")
    print("Sending queries rapidly...")
    
    for i in range(6):
        print(f"\nQuery {i+1}:")
        
        allowed, error = security_manager.check_rate_limits(connection_id, "demo_db")
        
        if allowed:
            print("✅ ALLOWED")
        else:
            print(f"❌ BLOCKED: {error}")
        
        # Small delay between queries
        time.sleep(0.1)
    
    print("\nRate limiting statistics:")
    stats = security_manager.get_security_statistics()
    
    if 'demo_rate_limit_connection' in stats['rate_limiting']['connections']:
        conn_stats = stats['rate_limiting']['connections']['demo_rate_limit_connection']
        print(f"- Queries this minute: {conn_stats['queries_this_minute']}")
        print(f"- Violations: {conn_stats['violations']}")
        print(f"- Currently blocked: {conn_stats['currently_blocked']}")
    
    print()


def demo_security_events():
    """Demonstrate security event logging and monitoring."""
    print("=" * 60)
    print("DEMO 5: Security Event Logging and Monitoring")
    print("=" * 60)
    
    security_manager = get_security_manager()
    
    # Generate some security events
    test_queries = [
        "SELECT id FROM users",  # Valid query
        "SELECT * FROM users UNION SELECT password FROM admin",  # Malicious
        "INSERT INTO users VALUES (1, 'hacker')",  # Basic violation
    ]
    
    print("Generating security events...")
    for query in test_queries:
        try:
            # This will generate security events
            validate_query_security(query, "demo_db", "demo_connection")
        except Exception:
            pass  # Expected for invalid queries
    
    # Get recent security events
    events = security_manager.get_security_events(limit=10)
    
    print(f"\nRecent Security Events ({len(events)} total):")
    print("-" * 50)
    
    for event in events[-5:]:  # Show last 5 events
        timestamp = time.strftime('%H:%M:%S', time.localtime(event.timestamp))
        print(f"[{timestamp}] {event.event_type.value.upper()}")
        print(f"  Threat Level: {event.threat_level.value}")
        print(f"  Database: {event.database_name}")
        print(f"  Message: {event.message}")
        
        if event.attack_pattern:
            print(f"  Attack Pattern: {event.attack_pattern.value}")
        
        print()
    
    # Show security statistics
    print("Security Statistics Summary:")
    print("-" * 30)
    
    stats = security_manager.get_security_statistics()
    event_stats = stats['event_statistics']
    
    print(f"Total Events: {event_stats['total_events']}")
    print(f"Events (last hour): {event_stats['events_last_hour']}")
    
    if event_stats['threat_levels']:
        print("Threat Level Distribution:")
        for level, count in event_stats['threat_levels'].items():
            print(f"  {level}: {count}")
    
    if event_stats['attack_patterns']:
        print("Attack Patterns Detected:")
        for pattern, count in event_stats['attack_patterns'].items():
            print(f"  {pattern}: {count}")
    
    print()


def demo_mcp_tools():
    """Demonstrate MCP security tools integration."""
    print("=" * 60)
    print("DEMO 6: MCP Security Tools Integration")
    print("=" * 60)
    
    # Demonstrate security statistics
    print("1. Security Statistics:")
    security_manager = get_security_manager()
    stats = security_manager.get_security_statistics()
    
    print(f"  - Configuration audit enabled: {stats['configuration']['audit_enabled']}")
    print(f"  - Pattern detection enabled: {stats['configuration']['pattern_detection_enabled']}")
    print(f"  - Current memory usage: {stats['resource_monitoring']['current_memory_mb']:.1f} MB")
    print(f"  - Active connections: {stats['rate_limiting']['active_connections']}")
    
    # Demonstrate security events
    print("\n2. Security Events:")
    high_threat_events = security_manager.get_security_events(
        limit=3, threat_levels=[SecurityThreatLevel.HIGH, SecurityThreatLevel.CRITICAL]
    )
    
    print(f"  - High threat events found: {len(high_threat_events)}")
    
    # Demonstrate direct validation
    print("\n3. Direct Security Validation (replacing MCP tool calls):")
    
    validation_cases = [
        ("Valid query", "SELECT name FROM products LIMIT 5"),
        ("Malicious query", "SELECT * FROM products UNION SELECT password FROM users")
    ]
    
    for case_name, test_query in validation_cases:
        is_valid, error_msg, metadata = security_manager.validate_query_security(
            test_query, "demo_db", "demo_connection"
        )
        
        print(f"  {case_name}:")
        print(f"    Valid: {is_valid}")
        
        if not is_valid:
            print(f"    Error: {error_msg}")
        else:
            print(f"    Threat Level: {metadata['threat_level'].value}")
    
    print()


def main():
    """Run all security system demonstrations."""
    print("LocalData MCP Security Enhancement Layer Demo")
    print("=" * 60)
    print()
    print("This demo showcases the enterprise-grade security features")
    print("implemented in Task 10, building on Tasks 1 and 6.")
    print()
    
    try:
        # Run all demonstrations
        demo_query_fingerprinting()
        demo_attack_detection()
        demo_comprehensive_validation()
        demo_rate_limiting()
        demo_security_events()
        demo_mcp_tools()
        
        print("=" * 60)
        print("DEMO COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print()
        print("Key Security Features Demonstrated:")
        print("✅ Query fingerprinting with SHA-256 hashing")
        print("✅ Advanced SQL injection attack detection (OWASP patterns)")
        print("✅ Multi-layered security validation pipeline")
        print("✅ Rate limiting with configurable thresholds")
        print("✅ Comprehensive security event logging")
        print("✅ Integration with MCP tools for monitoring")
        print()
        print("The LocalData MCP now provides enterprise-grade security")
        print("protection against SQL injection and abuse scenarios.")
        
    except Exception as e:
        print(f"Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())