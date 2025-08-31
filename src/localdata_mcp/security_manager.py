"""Security Enhancement Layer for LocalData MCP v1.3.1.

This module provides enterprise-grade security measures beyond basic SQL validation:
- Advanced SQL injection prevention with attack pattern detection
- Query fingerprinting and comprehensive audit logging
- Rate limiting per connection with configurable thresholds
- Resource exhaustion protection and monitoring
- Integration with existing query timeout and connection management systems

Security Approach:
- Defense in depth: multiple layers of protection
- Real-time threat detection and response
- Comprehensive audit trail for security analysis
- Integration with existing security, timeout, and connection systems
"""

import hashlib
import json
import logging
import psutil
import re
import threading
import time
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Deque, Dict, List, Optional, Set, Tuple, Union

# Import existing security and management systems
from .query_parser import QueryParser, SQLSecurityError, get_query_parser
from .timeout_manager import get_timeout_manager, TimeoutReason
from .connection_manager import get_enhanced_connection_manager

logger = logging.getLogger(__name__)


class SecurityThreatLevel(Enum):
    """Security threat severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SecurityEventType(Enum):
    """Types of security events."""
    QUERY_BLOCKED = "query_blocked"
    INJECTION_ATTEMPT = "injection_attempt"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    SUSPICIOUS_PATTERN = "suspicious_pattern"
    PARAMETER_VIOLATION = "parameter_violation"
    COMPLEXITY_VIOLATION = "complexity_violation"
    AUDIT_LOG = "audit_log"


class AttackPattern(Enum):
    """Known SQL injection attack patterns."""
    UNION_INJECTION = "union_injection"
    TIME_BASED_BLIND = "time_based_blind"
    BOOLEAN_BLIND = "boolean_blind"
    ERROR_BASED = "error_based"
    INFORMATION_EXTRACTION = "information_extraction"
    COMMENT_INJECTION = "comment_injection"
    STACKED_QUERIES = "stacked_queries"
    FUNCTION_ABUSE = "function_abuse"


@dataclass
class SecurityConfig:
    """Security configuration settings."""
    # Rate limiting
    queries_per_minute: int = 60
    queries_per_hour: int = 1000
    burst_limit: int = 10
    
    # Query complexity limits
    max_query_length: int = 10000
    max_joins: int = 10
    max_subqueries: int = 5
    max_functions: int = 20
    
    # Resource protection
    memory_threshold_mb: float = 500.0
    cpu_threshold_percent: float = 80.0
    max_concurrent_queries: int = 10
    
    # Audit settings
    audit_enabled: bool = True
    audit_all_queries: bool = True
    audit_failed_only: bool = False
    retain_audit_days: int = 30
    
    # Threat detection
    enable_pattern_detection: bool = True
    enable_anomaly_detection: bool = True
    block_suspicious_patterns: bool = True
    
    def __post_init__(self):
        """Validate configuration."""
        if self.queries_per_minute <= 0:
            raise ValueError("queries_per_minute must be positive")
        if self.burst_limit <= 0:
            raise ValueError("burst_limit must be positive")
        if self.max_query_length <= 0:
            raise ValueError("max_query_length must be positive")


@dataclass
class SecurityEvent:
    """Security event record."""
    timestamp: float
    event_type: SecurityEventType
    threat_level: SecurityThreatLevel
    database_name: str
    connection_id: Optional[str]
    query_fingerprint: str
    query_text: Optional[str]
    attack_pattern: Optional[AttackPattern]
    message: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for logging."""
        return {
            'timestamp': self.timestamp,
            'event_type': self.event_type.value,
            'threat_level': self.threat_level.value,
            'database_name': self.database_name,
            'connection_id': self.connection_id,
            'query_fingerprint': self.query_fingerprint,
            'query_text': self.query_text,
            'attack_pattern': self.attack_pattern.value if self.attack_pattern else None,
            'message': self.message,
            'metadata': self.metadata
        }


@dataclass 
class RateLimitState:
    """Rate limiting state for a connection."""
    queries_this_minute: Deque[float] = field(default_factory=deque)
    queries_this_hour: Deque[float] = field(default_factory=deque)
    burst_count: int = 0
    last_query_time: float = 0.0
    violations: int = 0
    blocked_until: float = 0.0
    
    def cleanup_old_queries(self, current_time: float):
        """Remove queries outside the time windows."""
        # Clean minute window
        while self.queries_this_minute and current_time - self.queries_this_minute[0] > 60:
            self.queries_this_minute.popleft()
        
        # Clean hour window
        while self.queries_this_hour and current_time - self.queries_this_hour[0] > 3600:
            self.queries_this_hour.popleft()


@dataclass
class QueryComplexity:
    """Analysis of query complexity."""
    length: int = 0
    joins: int = 0
    subqueries: int = 0
    functions: int = 0
    tables: int = 0
    conditions: int = 0
    unions: int = 0
    complexity_score: float = 0.0
    
    def calculate_score(self):
        """Calculate overall complexity score."""
        self.complexity_score = (
            self.length * 0.001 +
            self.joins * 2.0 +
            self.subqueries * 3.0 +
            self.functions * 1.0 +
            self.unions * 5.0 +
            self.conditions * 0.5
        )


class SecurityManager:
    """Enterprise-grade security management system."""
    
    def __init__(self, config: Optional[SecurityConfig] = None):
        """Initialize the security manager.
        
        Args:
            config: Security configuration. Uses defaults if None.
        """
        self.config = config or SecurityConfig()
        self.query_parser = get_query_parser()
        self.timeout_manager = get_timeout_manager()
        self.connection_manager = get_enhanced_connection_manager()
        
        # Rate limiting state
        self._rate_limits: Dict[str, RateLimitState] = defaultdict(RateLimitState)
        self._rate_limit_lock = threading.RLock()
        
        # Security event storage
        self._security_events: Deque[SecurityEvent] = deque(maxlen=10000)
        self._events_lock = threading.Lock()
        
        # Attack pattern detection
        self._attack_patterns = self._compile_attack_patterns()
        
        # Resource monitoring
        self._resource_monitor_active = True
        self._start_resource_monitoring()
        
        logger.info("SecurityManager initialized with enterprise-grade protection")
    
    def _compile_attack_patterns(self) -> Dict[AttackPattern, re.Pattern]:
        """Compile regex patterns for attack detection."""
        patterns = {
            AttackPattern.UNION_INJECTION: re.compile(
                r'\b(?:union\s+(?:all\s+)?select|union\s+select)\b',
                re.IGNORECASE
            ),
            AttackPattern.TIME_BASED_BLIND: re.compile(
                r'\b(?:sleep\s*\(|waitfor\s+delay|benchmark\s*\(|pg_sleep\s*\()\b',
                re.IGNORECASE
            ),
            AttackPattern.BOOLEAN_BLIND: re.compile(
                r'\b(?:and\s+\d+=\d+|or\s+\d+=\d+|and\s+true|or\s+false)\b',
                re.IGNORECASE
            ),
            AttackPattern.ERROR_BASED: re.compile(
                r'\b(?:cast\s*\(.*as\s+int\)|extractvalue\s*\(|updatexml\s*\()\b',
                re.IGNORECASE
            ),
            AttackPattern.INFORMATION_EXTRACTION: re.compile(
                r'\b(?:information_schema|sys\.tables|sys\.columns|pg_tables|sqlite_master)\b',
                re.IGNORECASE
            ),
            AttackPattern.COMMENT_INJECTION: re.compile(
                r'(?:--[^\r\n]*|/\*.*?\*/|#[^\r\n]*)',
                re.DOTALL | re.IGNORECASE
            ),
            AttackPattern.STACKED_QUERIES: re.compile(
                r';\s*(?:insert|update|delete|drop|create|alter)\b',
                re.IGNORECASE
            ),
            AttackPattern.FUNCTION_ABUSE: re.compile(
                r'\b(?:load_file\s*\(|into\s+outfile|into\s+dumpfile|exec\s*\(|eval\s*\()',
                re.IGNORECASE
            )
        }
        return patterns
    
    def create_query_fingerprint(self, query: str) -> str:
        """Create a cryptographic fingerprint of the query.
        
        Args:
            query: SQL query string
            
        Returns:
            str: SHA-256 hexdigest of the normalized query
        """
        # Normalize query for consistent fingerprinting
        normalized = re.sub(r'\s+', ' ', query.strip().lower())
        normalized = re.sub(r'--.*$', '', normalized, flags=re.MULTILINE)
        normalized = re.sub(r'/\*.*?\*/', '', normalized, flags=re.DOTALL)
        
        # Create fingerprint
        fingerprint = hashlib.sha256(normalized.encode('utf-8')).hexdigest()[:16]
        return fingerprint
    
    def analyze_query_complexity(self, query: str) -> QueryComplexity:
        """Analyze query complexity for resource limit enforcement.
        
        Args:
            query: SQL query string
            
        Returns:
            QueryComplexity: Detailed complexity analysis
        """
        complexity = QueryComplexity()
        query_lower = query.lower()
        
        # Basic metrics
        complexity.length = len(query)
        complexity.joins = len(re.findall(r'\b(?:inner\s+join|left\s+join|right\s+join|full\s+join|join)\b', query_lower))
        complexity.subqueries = len(re.findall(r'\(\s*select\b', query_lower))
        complexity.functions = len(re.findall(r'\b\w+\s*\(', query_lower))
        complexity.unions = len(re.findall(r'\bunion\b', query_lower))
        complexity.conditions = len(re.findall(r'\b(?:where|and|or|having)\b', query_lower))
        
        # Count unique table references
        table_pattern = re.compile(r'\bfrom\s+(\w+)|join\s+(\w+)', re.IGNORECASE)
        tables = set()
        for match in table_pattern.finditer(query):
            table = match.group(1) or match.group(2)
            if table:
                tables.add(table.lower())
        complexity.tables = len(tables)
        
        # Calculate overall score
        complexity.calculate_score()
        
        return complexity
    
    def detect_attack_patterns(self, query: str) -> List[AttackPattern]:
        """Detect known attack patterns in query.
        
        Args:
            query: SQL query string
            
        Returns:
            List[AttackPattern]: List of detected attack patterns
        """
        detected_patterns = []
        
        for pattern_type, regex in self._attack_patterns.items():
            if regex.search(query):
                detected_patterns.append(pattern_type)
                
        return detected_patterns
    
    def check_rate_limits(self, connection_id: str, database_name: str) -> Tuple[bool, Optional[str]]:
        """Check if connection exceeds rate limits.
        
        Args:
            connection_id: Connection identifier
            database_name: Database name
            
        Returns:
            Tuple[bool, Optional[str]]: (is_allowed, error_message)
        """
        current_time = time.time()
        
        with self._rate_limit_lock:
            rate_state = self._rate_limits[connection_id]
            rate_state.cleanup_old_queries(current_time)
            
            # Check if currently blocked
            if current_time < rate_state.blocked_until:
                remaining = int(rate_state.blocked_until - current_time)
                return False, f"Connection blocked for {remaining} seconds due to rate limit violations"
            
            # Check minute limit
            if len(rate_state.queries_this_minute) >= self.config.queries_per_minute:
                rate_state.violations += 1
                rate_state.blocked_until = current_time + min(60, rate_state.violations * 10)
                
                self._log_security_event(
                    SecurityEvent(
                        timestamp=current_time,
                        event_type=SecurityEventType.RATE_LIMIT_EXCEEDED,
                        threat_level=SecurityThreatLevel.MEDIUM,
                        database_name=database_name,
                        connection_id=connection_id,
                        query_fingerprint="rate_limit",
                        query_text=None,
                        attack_pattern=None,
                        message=f"Minute rate limit exceeded: {len(rate_state.queries_this_minute)}/{self.config.queries_per_minute}",
                        metadata={"violations": rate_state.violations}
                    )
                )
                
                return False, f"Rate limit exceeded: {len(rate_state.queries_this_minute)}/{self.config.queries_per_minute} queries per minute"
            
            # Check hour limit
            if len(rate_state.queries_this_hour) >= self.config.queries_per_hour:
                rate_state.violations += 1
                rate_state.blocked_until = current_time + min(300, rate_state.violations * 60)
                
                self._log_security_event(
                    SecurityEvent(
                        timestamp=current_time,
                        event_type=SecurityEventType.RATE_LIMIT_EXCEEDED,
                        threat_level=SecurityThreatLevel.HIGH,
                        database_name=database_name,
                        connection_id=connection_id,
                        query_fingerprint="rate_limit",
                        query_text=None,
                        attack_pattern=None,
                        message=f"Hour rate limit exceeded: {len(rate_state.queries_this_hour)}/{self.config.queries_per_hour}",
                        metadata={"violations": rate_state.violations}
                    )
                )
                
                return False, f"Rate limit exceeded: {len(rate_state.queries_this_hour)}/{self.config.queries_per_hour} queries per hour"
            
            # Check burst limit
            if current_time - rate_state.last_query_time < 1.0:
                rate_state.burst_count += 1
            else:
                rate_state.burst_count = 0
                
            if rate_state.burst_count > self.config.burst_limit:
                rate_state.violations += 1
                rate_state.blocked_until = current_time + 5  # Short block for burst
                
                self._log_security_event(
                    SecurityEvent(
                        timestamp=current_time,
                        event_type=SecurityEventType.RATE_LIMIT_EXCEEDED,
                        threat_level=SecurityThreatLevel.MEDIUM,
                        database_name=database_name,
                        connection_id=connection_id,
                        query_fingerprint="burst_limit",
                        query_text=None,
                        attack_pattern=None,
                        message=f"Burst limit exceeded: {rate_state.burst_count}/{self.config.burst_limit} queries per second",
                        metadata={"violations": rate_state.violations}
                    )
                )
                
                return False, f"Burst limit exceeded: too many queries in short time"
            
            # Record query
            rate_state.queries_this_minute.append(current_time)
            rate_state.queries_this_hour.append(current_time)
            rate_state.last_query_time = current_time
            
            return True, None
    
    def check_resource_limits(self, database_name: str) -> Tuple[bool, Optional[str]]:
        """Check system resource limits before query execution.
        
        Args:
            database_name: Database name
            
        Returns:
            Tuple[bool, Optional[str]]: (is_allowed, error_message)
        """
        try:
            # Check memory usage
            memory_mb = psutil.virtual_memory().used / 1024 / 1024
            if memory_mb > self.config.memory_threshold_mb:
                self._log_security_event(
                    SecurityEvent(
                        timestamp=time.time(),
                        event_type=SecurityEventType.RESOURCE_EXHAUSTION,
                        threat_level=SecurityThreatLevel.HIGH,
                        database_name=database_name,
                        connection_id=None,
                        query_fingerprint="resource_limit",
                        query_text=None,
                        attack_pattern=None,
                        message=f"Memory threshold exceeded: {memory_mb:.1f}MB > {self.config.memory_threshold_mb}MB",
                        metadata={"memory_mb": memory_mb, "threshold": self.config.memory_threshold_mb}
                    )
                )
                return False, f"System memory threshold exceeded: {memory_mb:.1f}MB"
            
            # Check CPU usage
            cpu_percent = psutil.cpu_percent(interval=0.1)
            if cpu_percent > self.config.cpu_threshold_percent:
                self._log_security_event(
                    SecurityEvent(
                        timestamp=time.time(),
                        event_type=SecurityEventType.RESOURCE_EXHAUSTION,
                        threat_level=SecurityThreatLevel.MEDIUM,
                        database_name=database_name,
                        connection_id=None,
                        query_fingerprint="resource_limit",
                        query_text=None,
                        attack_pattern=None,
                        message=f"CPU threshold exceeded: {cpu_percent:.1f}% > {self.config.cpu_threshold_percent}%",
                        metadata={"cpu_percent": cpu_percent, "threshold": self.config.cpu_threshold_percent}
                    )
                )
                return False, f"System CPU threshold exceeded: {cpu_percent:.1f}%"
            
            # Check active query count using connection manager
            active_operations = self.connection_manager.get_active_operations()
            db_operations = [op for op in active_operations.values() if op['database_name'] == database_name]
            
            if len(db_operations) >= self.config.max_concurrent_queries:
                return False, f"Maximum concurrent queries exceeded: {len(db_operations)}/{self.config.max_concurrent_queries}"
            
            return True, None
            
        except Exception as e:
            logger.warning(f"Error checking resource limits: {e}")
            return True, None  # Allow query if we can't check resources
    
    def validate_query_security(self, query: str, database_name: str, 
                              connection_id: Optional[str] = None) -> Tuple[bool, Optional[str], Dict[str, Any]]:
        """Comprehensive security validation for SQL queries.
        
        Args:
            query: SQL query string
            database_name: Database name
            connection_id: Optional connection identifier
            
        Returns:
            Tuple[bool, Optional[str], Dict[str, Any]]: (is_valid, error_message, security_metadata)
        """
        start_time = time.time()
        query_fingerprint = self.create_query_fingerprint(query)
        
        security_metadata = {
            'fingerprint': query_fingerprint,
            'validation_time': 0.0,
            'threat_level': SecurityThreatLevel.LOW,
            'checks_performed': []
        }
        
        try:
            # 1. Basic SQL validation from Task 1
            security_metadata['checks_performed'].append('basic_sql_validation')
            is_valid, error_msg = self.query_parser.validate_query(query)
            if not is_valid:
                self._log_security_event(
                    SecurityEvent(
                        timestamp=start_time,
                        event_type=SecurityEventType.QUERY_BLOCKED,
                        threat_level=SecurityThreatLevel.HIGH,
                        database_name=database_name,
                        connection_id=connection_id,
                        query_fingerprint=query_fingerprint,
                        query_text=query[:200],
                        attack_pattern=None,
                        message=f"Basic SQL validation failed: {error_msg}",
                        metadata={'validation_error': error_msg}
                    )
                )
                security_metadata['threat_level'] = SecurityThreatLevel.HIGH
                return False, error_msg, security_metadata
            
            # 2. Rate limiting check
            if connection_id:
                security_metadata['checks_performed'].append('rate_limiting')
                rate_allowed, rate_error = self.check_rate_limits(connection_id, database_name)
                if not rate_allowed:
                    security_metadata['threat_level'] = SecurityThreatLevel.MEDIUM
                    return False, rate_error, security_metadata
            
            # 3. Resource limit check
            security_metadata['checks_performed'].append('resource_limits')
            resource_allowed, resource_error = self.check_resource_limits(database_name)
            if not resource_allowed:
                security_metadata['threat_level'] = SecurityThreatLevel.HIGH
                return False, resource_error, security_metadata
            
            # 4. Query complexity analysis
            security_metadata['checks_performed'].append('complexity_analysis')
            complexity = self.analyze_query_complexity(query)
            security_metadata['complexity'] = {
                'score': complexity.complexity_score,
                'length': complexity.length,
                'joins': complexity.joins,
                'subqueries': complexity.subqueries
            }
            
            # Check complexity limits
            if complexity.length > self.config.max_query_length:
                self._log_security_event(
                    SecurityEvent(
                        timestamp=start_time,
                        event_type=SecurityEventType.COMPLEXITY_VIOLATION,
                        threat_level=SecurityThreatLevel.MEDIUM,
                        database_name=database_name,
                        connection_id=connection_id,
                        query_fingerprint=query_fingerprint,
                        query_text=query[:200],
                        attack_pattern=None,
                        message=f"Query length exceeds limit: {complexity.length}/{self.config.max_query_length}",
                        metadata={'complexity': complexity.__dict__}
                    )
                )
                security_metadata['threat_level'] = SecurityThreatLevel.MEDIUM
                return False, f"Query too long: {complexity.length}/{self.config.max_query_length} characters", security_metadata
            
            if complexity.joins > self.config.max_joins:
                return False, f"Too many joins: {complexity.joins}/{self.config.max_joins}", security_metadata
                
            if complexity.subqueries > self.config.max_subqueries:
                return False, f"Too many subqueries: {complexity.subqueries}/{self.config.max_subqueries}", security_metadata
            
            # 5. Attack pattern detection
            if self.config.enable_pattern_detection:
                security_metadata['checks_performed'].append('attack_pattern_detection')
                detected_patterns = self.detect_attack_patterns(query)
                security_metadata['attack_patterns'] = [p.value for p in detected_patterns]
                
                if detected_patterns and self.config.block_suspicious_patterns:
                    for pattern in detected_patterns:
                        self._log_security_event(
                            SecurityEvent(
                                timestamp=start_time,
                                event_type=SecurityEventType.INJECTION_ATTEMPT,
                                threat_level=SecurityThreatLevel.CRITICAL,
                                database_name=database_name,
                                connection_id=connection_id,
                                query_fingerprint=query_fingerprint,
                                query_text=query[:200],
                                attack_pattern=pattern,
                                message=f"Detected attack pattern: {pattern.value}",
                                metadata={'all_patterns': [p.value for p in detected_patterns]}
                            )
                        )
                    
                    security_metadata['threat_level'] = SecurityThreatLevel.CRITICAL
                    return False, f"Suspicious patterns detected: {', '.join(p.value for p in detected_patterns)}", security_metadata
                
                elif detected_patterns:
                    # Log but don't block
                    self._log_security_event(
                        SecurityEvent(
                            timestamp=start_time,
                            event_type=SecurityEventType.SUSPICIOUS_PATTERN,
                            threat_level=SecurityThreatLevel.MEDIUM,
                            database_name=database_name,
                            connection_id=connection_id,
                            query_fingerprint=query_fingerprint,
                            query_text=query[:200],
                            attack_pattern=detected_patterns[0],
                            message=f"Suspicious patterns detected but allowed: {', '.join(p.value for p in detected_patterns)}",
                            metadata={'all_patterns': [p.value for p in detected_patterns]}
                        )
                    )
                    security_metadata['threat_level'] = SecurityThreatLevel.MEDIUM
            
            # All checks passed
            validation_time = time.time() - start_time
            security_metadata['validation_time'] = validation_time
            
            # Log successful validation if audit is enabled
            if self.config.audit_enabled and self.config.audit_all_queries:
                self._log_security_event(
                    SecurityEvent(
                        timestamp=start_time,
                        event_type=SecurityEventType.AUDIT_LOG,
                        threat_level=SecurityThreatLevel.LOW,
                        database_name=database_name,
                        connection_id=connection_id,
                        query_fingerprint=query_fingerprint,
                        query_text=query[:500],
                        attack_pattern=None,
                        message=f"Query validation passed (validation time: {validation_time:.3f}s)",
                        metadata=security_metadata.copy()
                    )
                )
            
            return True, None, security_metadata
            
        except Exception as e:
            logger.error(f"Error during security validation: {e}")
            security_metadata['validation_time'] = time.time() - start_time
            security_metadata['error'] = str(e)
            return False, f"Security validation error: {e}", security_metadata
    
    @contextmanager
    def secure_query_execution(self, query: str, database_name: str, connection_id: Optional[str] = None):
        """Context manager for secure query execution with comprehensive monitoring.
        
        Args:
            query: SQL query string
            database_name: Database name
            connection_id: Optional connection identifier
            
        Yields:
            dict: Execution context with security metadata
            
        Raises:
            SQLSecurityError: If security validation fails
        """
        start_time = time.time()
        query_fingerprint = self.create_query_fingerprint(query)
        
        # Perform security validation
        is_valid, error_msg, security_metadata = self.validate_query_security(
            query, database_name, connection_id
        )
        
        if not is_valid:
            raise SQLSecurityError(f"Security validation failed: {error_msg}")
        
        # Create execution context
        execution_context = {
            'query_fingerprint': query_fingerprint,
            'database_name': database_name,
            'connection_id': connection_id,
            'start_time': start_time,
            'security_metadata': security_metadata,
            'execution_metadata': {}
        }
        
        try:
            yield execution_context
            
            # Query executed successfully
            execution_time = time.time() - start_time
            execution_context['execution_metadata']['execution_time'] = execution_time
            execution_context['execution_metadata']['success'] = True
            
            # Log successful execution
            if self.config.audit_enabled:
                self._log_security_event(
                    SecurityEvent(
                        timestamp=start_time,
                        event_type=SecurityEventType.AUDIT_LOG,
                        threat_level=SecurityThreatLevel.LOW,
                        database_name=database_name,
                        connection_id=connection_id,
                        query_fingerprint=query_fingerprint,
                        query_text=query[:200],
                        attack_pattern=None,
                        message=f"Query executed successfully (time: {execution_time:.3f}s)",
                        metadata={
                            'security_metadata': security_metadata,
                            'execution_time': execution_time
                        }
                    )
                )
            
        except Exception as e:
            # Query execution failed
            execution_time = time.time() - start_time
            execution_context['execution_metadata']['execution_time'] = execution_time
            execution_context['execution_metadata']['success'] = False
            execution_context['execution_metadata']['error'] = str(e)
            
            # Log failed execution
            if self.config.audit_enabled:
                self._log_security_event(
                    SecurityEvent(
                        timestamp=start_time,
                        event_type=SecurityEventType.AUDIT_LOG,
                        threat_level=SecurityThreatLevel.MEDIUM,
                        database_name=database_name,
                        connection_id=connection_id,
                        query_fingerprint=query_fingerprint,
                        query_text=query[:200],
                        attack_pattern=None,
                        message=f"Query execution failed: {str(e)[:200]}",
                        metadata={
                            'security_metadata': security_metadata,
                            'execution_time': execution_time,
                            'error': str(e)
                        }
                    )
                )
            
            raise
    
    def _log_security_event(self, event: SecurityEvent):
        """Log a security event.
        
        Args:
            event: SecurityEvent to log
        """
        with self._events_lock:
            self._security_events.append(event)
            
        # Log to standard logging system
        log_level = logging.WARNING
        if event.threat_level == SecurityThreatLevel.CRITICAL:
            log_level = logging.ERROR
        elif event.threat_level == SecurityThreatLevel.HIGH:
            log_level = logging.WARNING
        elif event.threat_level == SecurityThreatLevel.LOW:
            log_level = logging.INFO
            
        logger.log(log_level, f"[SECURITY] {event.message}", extra={
            'security_event': event.to_dict()
        })
    
    def get_security_events(self, limit: Optional[int] = None, 
                           event_types: Optional[List[SecurityEventType]] = None,
                           threat_levels: Optional[List[SecurityThreatLevel]] = None) -> List[SecurityEvent]:
        """Get security events with optional filtering.
        
        Args:
            limit: Maximum number of events to return
            event_types: Filter by event types
            threat_levels: Filter by threat levels
            
        Returns:
            List[SecurityEvent]: Filtered security events
        """
        with self._events_lock:
            events = list(self._security_events)
        
        # Apply filters
        if event_types:
            events = [e for e in events if e.event_type in event_types]
            
        if threat_levels:
            events = [e for e in events if e.threat_level in threat_levels]
        
        # Sort by timestamp (newest first)
        events.sort(key=lambda e: e.timestamp, reverse=True)
        
        if limit:
            events = events[:limit]
        
        return events
    
    def get_security_statistics(self) -> Dict[str, Any]:
        """Get comprehensive security statistics.
        
        Returns:
            Dict[str, Any]: Security statistics and metrics
        """
        with self._events_lock:
            events = list(self._security_events)
        
        current_time = time.time()
        hour_ago = current_time - 3600
        day_ago = current_time - 86400
        
        # Basic statistics
        total_events = len(events)
        events_last_hour = len([e for e in events if e.timestamp >= hour_ago])
        events_last_day = len([e for e in events if e.timestamp >= day_ago])
        
        # Event type breakdown
        event_type_counts = defaultdict(int)
        for event in events:
            event_type_counts[event.event_type.value] += 1
        
        # Threat level breakdown
        threat_level_counts = defaultdict(int)
        for event in events:
            threat_level_counts[event.threat_level.value] += 1
        
        # Attack pattern breakdown
        attack_pattern_counts = defaultdict(int)
        for event in events:
            if event.attack_pattern:
                attack_pattern_counts[event.attack_pattern.value] += 1
        
        # Rate limiting statistics
        with self._rate_limit_lock:
            rate_limit_stats = {}
            for connection_id, rate_state in self._rate_limits.items():
                rate_limit_stats[connection_id] = {
                    'queries_this_minute': len(rate_state.queries_this_minute),
                    'queries_this_hour': len(rate_state.queries_this_hour),
                    'violations': rate_state.violations,
                    'blocked_until': rate_state.blocked_until,
                    'currently_blocked': current_time < rate_state.blocked_until
                }
        
        # Resource statistics
        try:
            memory_mb = psutil.virtual_memory().used / 1024 / 1024
            cpu_percent = psutil.cpu_percent()
        except:
            memory_mb = 0
            cpu_percent = 0
        
        return {
            'event_statistics': {
                'total_events': total_events,
                'events_last_hour': events_last_hour,
                'events_last_day': events_last_day,
                'event_types': dict(event_type_counts),
                'threat_levels': dict(threat_level_counts),
                'attack_patterns': dict(attack_pattern_counts)
            },
            'rate_limiting': {
                'active_connections': len(self._rate_limits),
                'connections': rate_limit_stats,
                'config': {
                    'queries_per_minute': self.config.queries_per_minute,
                    'queries_per_hour': self.config.queries_per_hour,
                    'burst_limit': self.config.burst_limit
                }
            },
            'resource_monitoring': {
                'current_memory_mb': memory_mb,
                'memory_threshold_mb': self.config.memory_threshold_mb,
                'current_cpu_percent': cpu_percent,
                'cpu_threshold_percent': self.config.cpu_threshold_percent,
                'memory_warning': memory_mb > self.config.memory_threshold_mb,
                'cpu_warning': cpu_percent > self.config.cpu_threshold_percent
            },
            'configuration': {
                'audit_enabled': self.config.audit_enabled,
                'pattern_detection_enabled': self.config.enable_pattern_detection,
                'block_suspicious_patterns': self.config.block_suspicious_patterns,
                'max_query_length': self.config.max_query_length,
                'max_joins': self.config.max_joins,
                'max_subqueries': self.config.max_subqueries
            }
        }
    
    def _start_resource_monitoring(self):
        """Start background resource monitoring thread."""
        def monitor():
            while self._resource_monitor_active:
                try:
                    # Clean up old rate limit data
                    current_time = time.time()
                    with self._rate_limit_lock:
                        for rate_state in self._rate_limits.values():
                            rate_state.cleanup_old_queries(current_time)
                    
                    # Clean up old security events
                    cutoff_time = current_time - (self.config.retain_audit_days * 86400)
                    with self._events_lock:
                        while (self._security_events and 
                               self._security_events[0].timestamp < cutoff_time):
                            self._security_events.popleft()
                    
                    time.sleep(60)  # Run cleanup every minute
                    
                except Exception as e:
                    logger.error(f"Error in resource monitoring: {e}")
                    time.sleep(10)
        
        thread = threading.Thread(target=monitor, daemon=True)
        thread.start()
        logger.info("Started security resource monitoring thread")
    
    def close(self):
        """Close the security manager and clean up resources."""
        self._resource_monitor_active = False
        logger.info("SecurityManager closed")


# Global security manager instance
_security_manager: Optional[SecurityManager] = None


def get_security_manager() -> SecurityManager:
    """Get or create global security manager instance.
    
    Returns:
        SecurityManager: Global security manager
    """
    global _security_manager
    if _security_manager is None:
        _security_manager = SecurityManager()
    return _security_manager


def initialize_security_manager(config: Optional[SecurityConfig] = None) -> SecurityManager:
    """Initialize a new global security manager instance.
    
    Args:
        config: Optional security configuration
        
    Returns:
        SecurityManager: New security manager
    """
    global _security_manager
    if _security_manager is not None:
        _security_manager.close()
    _security_manager = SecurityManager(config)
    return _security_manager


# Convenience functions for common operations

def validate_query_security(query: str, database_name: str, connection_id: Optional[str] = None) -> Tuple[bool, Optional[str]]:
    """Validate query security using global security manager.
    
    Args:
        query: SQL query string
        database_name: Database name
        connection_id: Optional connection identifier
        
    Returns:
        Tuple[bool, Optional[str]]: (is_valid, error_message)
    """
    security_manager = get_security_manager()
    is_valid, error_msg, _ = security_manager.validate_query_security(query, database_name, connection_id)
    return is_valid, error_msg


def secure_query_execution(query: str, database_name: str, connection_id: Optional[str] = None):
    """Context manager for secure query execution.
    
    Args:
        query: SQL query string
        database_name: Database name
        connection_id: Optional connection identifier
        
    Returns:
        Context manager for secure execution
    """
    security_manager = get_security_manager()
    return security_manager.secure_query_execution(query, database_name, connection_id)