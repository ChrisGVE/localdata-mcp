"""SQL Query Parser and Safety Validator for LocalData MCP.

This module provides security validation for SQL queries to ensure only SELECT
operations are allowed. It blocks all data modification operations including
INSERT, UPDATE, DELETE, DROP, CREATE, ALTER, and other potentially dangerous SQL operations.

Security Approach:
- Whitelist only SELECT statements and Common Table Expressions (WITH)
- Block all data modification and schema alteration operations
- Provide clear error messages for blocked operations
- Use regex patterns for robust detection of dangerous SQL keywords
"""

import logging
import re
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class SQLSecurityError(Exception):
    """Raised when a SQL query is blocked for security reasons."""
    pass


class QueryParser:
    """SQL Query Parser with security validation.
    
    This parser validates SQL queries and only allows SELECT operations.
    All other operations are blocked to prevent data modification or schema changes.
    """
    
    # Dangerous SQL operations that are always blocked
    BLOCKED_OPERATIONS = {
        'INSERT': 'Data insertion operations are not allowed',
        'UPDATE': 'Data modification operations are not allowed', 
        'DELETE': 'Data deletion operations are not allowed',
        'DROP': 'Schema deletion operations are not allowed',
        'CREATE': 'Schema creation operations are not allowed',
        'ALTER': 'Schema modification operations are not allowed',
        'TRUNCATE': 'Table truncation operations are not allowed',
        'REPLACE': 'Data replacement operations are not allowed',
        'MERGE': 'Data merge operations are not allowed',
        'UPSERT': 'Upsert operations are not allowed',
        'CALL': 'Procedure calls are not allowed',
        'EXEC': 'Command execution is not allowed',
        'EXECUTE': 'Command execution is not allowed',
        'GRANT': 'Permission modification is not allowed',
        'REVOKE': 'Permission modification is not allowed',
        'COMMIT': 'Transaction control is not allowed',
        'ROLLBACK': 'Transaction control is not allowed',
        'SAVEPOINT': 'Transaction control is not allowed',
        'SET': 'Configuration changes are not allowed',
        'USE': 'Database switching is not allowed',
        'PRAGMA': 'Database configuration is not allowed',
        'ATTACH': 'Database attachment is not allowed',
        'DETACH': 'Database detachment is not allowed',
        'VACUUM': 'Database maintenance operations are not allowed',
        'REINDEX': 'Index rebuilding is not allowed',
        'ANALYZE': 'Database analysis operations are not allowed'
    }
    
    # Allowed operations (whitelist approach)
    ALLOWED_OPERATIONS = {
        'SELECT': 'Data selection queries are allowed',
        'WITH': 'Common Table Expressions (CTEs) are allowed'
    }
    
    def __init__(self):
        """Initialize the query parser with security patterns."""
        # Compile regex patterns for efficient matching
        self._compile_security_patterns()
    
    def _compile_security_patterns(self) -> None:
        """Compile regex patterns for detecting SQL operations."""
        # Pattern to match SQL comments (-- and /* */)
        self.comment_pattern = re.compile(r'(?:--[^\r\n]*)|(?:/\*.*?\*/)', re.DOTALL | re.IGNORECASE)
        
        # Pattern to detect blocked operations at the start of statements
        blocked_ops = '|'.join(self.BLOCKED_OPERATIONS.keys())
        self.blocked_pattern = re.compile(
            rf'\b(?:{blocked_ops})\b',
            re.IGNORECASE
        )
        
        # Pattern to detect allowed operations at the start of statements
        allowed_ops = '|'.join(self.ALLOWED_OPERATIONS.keys())
        self.allowed_pattern = re.compile(
            rf'^\s*(?:{allowed_ops})\b',
            re.IGNORECASE
        )
        
        # Pattern to detect multiple statements (semicolon separated)
        self.multi_statement_pattern = re.compile(r';\s*(?:\S)', re.IGNORECASE)
    
    def _normalize_query(self, query: str) -> str:
        """Normalize SQL query by removing comments and extra whitespace.
        
        Args:
            query: Raw SQL query string
            
        Returns:
            Normalized query string with comments removed and whitespace cleaned
        """
        if not query or not isinstance(query, str):
            raise ValueError("Query must be a non-empty string")
        
        # Remove SQL comments
        normalized = self.comment_pattern.sub(' ', query)
        
        # Remove extra whitespace and normalize
        normalized = ' '.join(normalized.split())
        
        return normalized.strip()
    
    def _detect_blocked_operations(self, query: str) -> List[str]:
        """Detect any blocked SQL operations in the query.
        
        Args:
            query: Normalized SQL query string
            
        Returns:
            List of blocked operations found in the query
        """
        found_operations = []
        matches = self.blocked_pattern.findall(query)
        
        for match in matches:
            operation = match.upper()
            if operation in self.BLOCKED_OPERATIONS:
                found_operations.append(operation)
        
        # Remove duplicates while preserving order
        return list(dict.fromkeys(found_operations))
    
    def _is_select_operation(self, query: str) -> bool:
        """Check if the query starts with an allowed operation.
        
        Args:
            query: Normalized SQL query string
            
        Returns:
            True if query starts with SELECT or WITH, False otherwise
        """
        return bool(self.allowed_pattern.match(query))
    
    def _check_multiple_statements(self, query: str) -> bool:
        """Check if query contains multiple statements.
        
        Args:
            query: Normalized SQL query string
            
        Returns:
            True if multiple statements detected, False otherwise
        """
        return bool(self.multi_statement_pattern.search(query))
    
    def validate_query(self, query: str) -> Tuple[bool, Optional[str]]:
        """Validate SQL query for security compliance.
        
        Args:
            query: SQL query string to validate
            
        Returns:
            Tuple of (is_valid, error_message). If valid, error_message is None.
            If invalid, error_message explains why the query was blocked.
            
        Raises:
            ValueError: If query is not a valid string
        """
        try:
            # Normalize the query
            normalized_query = self._normalize_query(query)
            
            if not normalized_query:
                return False, "Empty query is not allowed"
            
            # Check for multiple statements
            if self._check_multiple_statements(normalized_query):
                return False, "Multiple SQL statements in a single query are not allowed for security reasons"
            
            # Check if query starts with allowed operations
            if not self._is_select_operation(normalized_query):
                return False, "Only SELECT queries and Common Table Expressions (WITH) are allowed"
            
            # Check for blocked operations anywhere in the query
            blocked_ops = self._detect_blocked_operations(normalized_query)
            if blocked_ops:
                blocked_list = ', '.join(blocked_ops)
                return False, f"Query contains blocked operations: {blocked_list}. Only SELECT queries are allowed."
            
            # Query passed all security checks
            return True, None
            
        except ValueError as e:
            return False, str(e)
        except Exception as e:
            logger.error(f"Unexpected error during query validation: {e}")
            return False, f"Query validation failed due to unexpected error: {e}"
    
    def parse_and_validate(self, query: str) -> str:
        """Parse and validate SQL query, raising exception if blocked.
        
        Args:
            query: SQL query string to validate
            
        Returns:
            The original query if validation passes
            
        Raises:
            SQLSecurityError: If query is blocked for security reasons
            ValueError: If query is malformed
        """
        is_valid, error_message = self.validate_query(query)
        
        if not is_valid:
            logger.warning(f"Blocked SQL query: {error_message}")
            raise SQLSecurityError(f"Security validation failed: {error_message}")
        
        logger.debug(f"SQL query validation passed: {query[:100]}{'...' if len(query) > 100 else ''}")
        return query
    
    def get_allowed_operations(self) -> Dict[str, str]:
        """Get dictionary of allowed SQL operations and their descriptions.
        
        Returns:
            Dictionary mapping operation names to descriptions
        """
        return self.ALLOWED_OPERATIONS.copy()
    
    def get_blocked_operations(self) -> Dict[str, str]:
        """Get dictionary of blocked SQL operations and their descriptions.
        
        Returns:
            Dictionary mapping operation names to descriptions
        """
        return self.BLOCKED_OPERATIONS.copy()


# Global parser instance for efficient reuse
_query_parser = None


def get_query_parser() -> QueryParser:
    """Get the global QueryParser instance (singleton pattern).
    
    Returns:
        QueryParser instance
    """
    global _query_parser
    if _query_parser is None:
        _query_parser = QueryParser()
    return _query_parser


def validate_sql_query(query: str) -> Tuple[bool, Optional[str]]:
    """Validate SQL query using the global parser instance.
    
    Args:
        query: SQL query string to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    parser = get_query_parser()
    return parser.validate_query(query)


def parse_and_validate_sql(query: str) -> str:
    """Parse and validate SQL query using the global parser instance.
    
    Args:
        query: SQL query string to validate
        
    Returns:
        The original query if validation passes
        
    Raises:
        SQLSecurityError: If query is blocked for security reasons
    """
    parser = get_query_parser()
    return parser.parse_and_validate(query)