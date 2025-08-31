"""Test suite for SQL Query Parser and Safety Validator.

This test suite validates the security features of the query parser,
ensuring that only SELECT operations are allowed while blocking all
dangerous SQL operations.
"""

import pytest
from src.localdata_mcp.query_parser import (
    QueryParser,
    SQLSecurityError,
    get_query_parser,
    validate_sql_query,
    parse_and_validate_sql,
)


class TestQueryParser:
    """Test cases for QueryParser class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.parser = QueryParser()
    
    def test_parser_initialization(self):
        """Test that parser initializes correctly."""
        assert self.parser is not None
        assert hasattr(self.parser, 'BLOCKED_OPERATIONS')
        assert hasattr(self.parser, 'ALLOWED_OPERATIONS')
        assert len(self.parser.BLOCKED_OPERATIONS) > 0
        assert len(self.parser.ALLOWED_OPERATIONS) > 0
    
    def test_singleton_parser(self):
        """Test that global parser returns same instance."""
        parser1 = get_query_parser()
        parser2 = get_query_parser()
        assert parser1 is parser2


class TestValidSelectQueries:
    """Test cases for valid SELECT queries that should pass validation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.parser = QueryParser()
    
    def test_simple_select(self):
        """Test simple SELECT query."""
        query = "SELECT * FROM users"
        is_valid, error = self.parser.validate_query(query)
        assert is_valid is True
        assert error is None
    
    def test_select_with_where(self):
        """Test SELECT with WHERE clause."""
        query = "SELECT name, email FROM users WHERE active = 1"
        is_valid, error = self.parser.validate_query(query)
        assert is_valid is True
        assert error is None
    
    def test_select_with_joins(self):
        """Test SELECT with JOIN operations."""
        query = """
        SELECT u.name, p.title 
        FROM users u 
        JOIN posts p ON u.id = p.user_id 
        WHERE u.active = 1
        """
        is_valid, error = self.parser.validate_query(query)
        assert is_valid is True
        assert error is None
    
    def test_select_with_aggregations(self):
        """Test SELECT with aggregate functions."""
        query = "SELECT COUNT(*), AVG(age), MAX(created_at) FROM users GROUP BY department"
        is_valid, error = self.parser.validate_query(query)
        assert is_valid is True
        assert error is None
    
    def test_select_with_subqueries(self):
        """Test SELECT with subqueries."""
        query = "SELECT * FROM users WHERE id IN (SELECT user_id FROM active_sessions)"
        is_valid, error = self.parser.validate_query(query)
        assert is_valid is True
        assert error is None
    
    def test_select_with_case_statement(self):
        """Test SELECT with CASE statements."""
        query = """
        SELECT name,
               CASE 
                   WHEN age < 18 THEN 'Minor'
                   WHEN age >= 18 THEN 'Adult'
                   ELSE 'Unknown'
               END as age_group
        FROM users
        """
        is_valid, error = self.parser.validate_query(query)
        assert is_valid is True
        assert error is None
    
    def test_with_cte_query(self):
        """Test WITH clause (Common Table Expressions)."""
        query = """
        WITH active_users AS (
            SELECT * FROM users WHERE active = 1
        )
        SELECT name FROM active_users WHERE age > 25
        """
        is_valid, error = self.parser.validate_query(query)
        assert is_valid is True
        assert error is None
    
    def test_recursive_cte(self):
        """Test recursive Common Table Expressions."""
        query = """
        WITH RECURSIVE employee_hierarchy AS (
            SELECT id, name, manager_id, 0 as level
            FROM employees 
            WHERE manager_id IS NULL
            UNION ALL
            SELECT e.id, e.name, e.manager_id, eh.level + 1
            FROM employees e
            JOIN employee_hierarchy eh ON e.manager_id = eh.id
        )
        SELECT * FROM employee_hierarchy ORDER BY level, name
        """
        is_valid, error = self.parser.validate_query(query)
        assert is_valid is True
        assert error is None
    
    def test_select_with_comments(self):
        """Test SELECT queries with SQL comments."""
        query = """
        -- This is a comment
        SELECT * FROM users /* inline comment */ WHERE active = 1
        /* Multi-line
           comment */
        """
        is_valid, error = self.parser.validate_query(query)
        assert is_valid is True
        assert error is None


class TestBlockedOperations:
    """Test cases for blocked SQL operations that should fail validation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.parser = QueryParser()
    
    def test_insert_blocked(self):
        """Test that INSERT operations are blocked."""
        query = "INSERT INTO users (name, email) VALUES ('John', 'john@example.com')"
        is_valid, error = self.parser.validate_query(query)
        assert is_valid is False
        assert "SELECT" in error
    
    def test_update_blocked(self):
        """Test that UPDATE operations are blocked."""
        query = "UPDATE users SET active = 0 WHERE id = 1"
        is_valid, error = self.parser.validate_query(query)
        assert is_valid is False
        assert "Only SELECT queries" in error
    
    def test_delete_blocked(self):
        """Test that DELETE operations are blocked."""
        query = "DELETE FROM users WHERE inactive = 1"
        is_valid, error = self.parser.validate_query(query)
        assert is_valid is False
        assert "Only SELECT queries" in error
    
    def test_drop_table_blocked(self):
        """Test that DROP TABLE operations are blocked."""
        query = "DROP TABLE old_users"
        is_valid, error = self.parser.validate_query(query)
        assert is_valid is False
        assert "Only SELECT queries" in error
    
    def test_create_table_blocked(self):
        """Test that CREATE TABLE operations are blocked."""
        query = "CREATE TABLE new_users (id INT, name VARCHAR(50))"
        is_valid, error = self.parser.validate_query(query)
        assert is_valid is False
        assert "Only SELECT queries" in error
    
    def test_alter_table_blocked(self):
        """Test that ALTER TABLE operations are blocked."""
        query = "ALTER TABLE users ADD COLUMN middle_name VARCHAR(50)"
        is_valid, error = self.parser.validate_query(query)
        assert is_valid is False
        assert "Only SELECT queries" in error
    
    def test_truncate_blocked(self):
        """Test that TRUNCATE operations are blocked."""
        query = "TRUNCATE TABLE logs"
        is_valid, error = self.parser.validate_query(query)
        assert is_valid is False
        assert "Only SELECT queries" in error
    
    def test_replace_blocked(self):
        """Test that REPLACE operations are blocked."""
        query = "REPLACE INTO users VALUES (1, 'Updated Name', 'email@example.com')"
        is_valid, error = self.parser.validate_query(query)
        assert is_valid is False
        assert "Only SELECT queries" in error
    
    def test_merge_blocked(self):
        """Test that MERGE operations are blocked."""
        query = "MERGE INTO users USING new_data ON users.id = new_data.id"
        is_valid, error = self.parser.validate_query(query)
        assert is_valid is False
        assert "Only SELECT queries" in error
    
    def test_call_procedure_blocked(self):
        """Test that CALL operations are blocked."""
        query = "CALL update_user_stats()"
        is_valid, error = self.parser.validate_query(query)
        assert is_valid is False
        assert "Only SELECT queries" in error
    
    def test_exec_blocked(self):
        """Test that EXEC operations are blocked."""
        query = "EXEC sp_update_users"
        is_valid, error = self.parser.validate_query(query)
        assert is_valid is False
        assert "Only SELECT queries" in error
    
    def test_grant_blocked(self):
        """Test that GRANT operations are blocked."""
        query = "GRANT SELECT ON users TO guest_user"
        is_valid, error = self.parser.validate_query(query)
        assert is_valid is False
        assert "Only SELECT queries" in error
    
    def test_revoke_blocked(self):
        """Test that REVOKE operations are blocked."""
        query = "REVOKE SELECT ON users FROM guest_user"
        is_valid, error = self.parser.validate_query(query)
        assert is_valid is False
        assert "Only SELECT queries" in error
    
    def test_pragma_blocked(self):
        """Test that PRAGMA operations are blocked."""
        query = "PRAGMA table_info(users)"
        is_valid, error = self.parser.validate_query(query)
        assert is_valid is False
        assert "Only SELECT queries" in error
    
    def test_set_blocked(self):
        """Test that SET operations are blocked."""
        query = "SET GLOBAL max_connections = 1000"
        is_valid, error = self.parser.validate_query(query)
        assert is_valid is False
        assert "Only SELECT queries" in error
    
    def test_use_database_blocked(self):
        """Test that USE operations are blocked."""
        query = "USE production_db"
        is_valid, error = self.parser.validate_query(query)
        assert is_valid is False
        assert "Only SELECT queries" in error


class TestSecurityEdgeCases:
    """Test cases for security edge cases and potential bypass attempts."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.parser = QueryParser()
    
    def test_multiple_statements_blocked(self):
        """Test that multiple statements are blocked."""
        query = "SELECT * FROM users; DROP TABLE users;"
        is_valid, error = self.parser.validate_query(query)
        assert is_valid is False
        assert "Multiple SQL statements" in error
    
    def test_select_with_hidden_insert_blocked(self):
        """Test SELECT with hidden dangerous operations."""
        query = "SELECT * FROM users WHERE id = 1; INSERT INTO logs VALUES ('hack')"
        is_valid, error = self.parser.validate_query(query)
        assert is_valid is False
        assert "Multiple SQL statements" in error
    
    def test_case_insensitive_blocking(self):
        """Test that blocking works regardless of case."""
        queries = [
            "insert into users values (1, 'test')",
            "INSERT INTO users VALUES (1, 'test')",
            "Insert Into Users Values (1, 'test')",
            "iNsErT iNtO uSeRs VaLuEs (1, 'test')"
        ]
        
        for query in queries:
            is_valid, error = self.parser.validate_query(query)
            assert is_valid is False, f"Should block: {query}"
            assert "SELECT" in error
    
    def test_comments_dont_bypass_security(self):
        """Test that comments don't allow bypassing security."""
        queries = [
            "/* SELECT */ INSERT INTO users VALUES (1, 'test')",
            "-- SELECT query\nINSERT INTO users VALUES (1, 'test')",
            "SELECT /* INSERT INTO hack */ * FROM users",  # This should pass
        ]
        
        # First two should fail
        for query in queries[:2]:
            is_valid, error = self.parser.validate_query(query)
            assert is_valid is False, f"Should block: {query}"
        
        # Last one should pass (comment doesn't affect SELECT)
        is_valid, error = self.parser.validate_query(queries[2])
        assert is_valid is True
    
    def test_whitespace_normalization(self):
        """Test that extra whitespace doesn't bypass validation."""
        queries = [
            "   SELECT   *   FROM   users   ",
            "\n\nSELECT\n\n*\n\nFROM\n\nusers\n\n",
            "\t\tSELECT\t\t*\t\tFROM\t\tusers\t\t",
        ]
        
        for query in queries:
            is_valid, error = self.parser.validate_query(query)
            assert is_valid is True, f"Should allow: {repr(query)}"
    
    def test_nested_blocked_operations(self):
        """Test that blocked operations in subqueries are caught."""
        query = "SELECT * FROM (DELETE FROM users WHERE id = 1) as subq"
        is_valid, error = self.parser.validate_query(query)
        assert is_valid is False
        assert "DELETE" in error


class TestInputValidation:
    """Test cases for input validation and error handling."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.parser = QueryParser()
    
    def test_empty_query_blocked(self):
        """Test that empty queries are blocked."""
        queries = ["", "   ", "\n\n", "\t\t"]
        
        for query in queries:
            is_valid, error = self.parser.validate_query(query)
            assert is_valid is False
            assert "Empty query" in error or "non-empty string" in error
    
    def test_none_query_blocked(self):
        """Test that None queries are blocked."""
        is_valid, error = self.parser.validate_query(None)
        assert is_valid is False
        assert "non-empty string" in error
    
    def test_non_string_query_blocked(self):
        """Test that non-string queries are blocked."""
        invalid_queries = [123, [], {}, True, object()]
        
        for query in invalid_queries:
            is_valid, error = self.parser.validate_query(query)
            assert is_valid is False
            assert "non-empty string" in error
    
    def test_sql_injection_patterns_blocked(self):
        """Test that common SQL injection patterns are blocked."""
        injection_queries = [
            "SELECT * FROM users; DROP TABLE users; --",
            "SELECT * FROM users WHERE id = 1 OR 1=1; INSERT INTO admin VALUES ('hacker');",
            "1'; DROP TABLE users; --",
        ]
        
        for query in injection_queries:
            is_valid, error = self.parser.validate_query(query)
            assert is_valid is False, f"Should block injection attempt: {query}"


class TestParseAndValidateFunction:
    """Test cases for parse_and_validate_sql function."""
    
    def test_valid_query_passes(self):
        """Test that valid queries pass through unchanged."""
        query = "SELECT * FROM users WHERE active = 1"
        result = parse_and_validate_sql(query)
        assert result == query
    
    def test_invalid_query_raises_exception(self):
        """Test that invalid queries raise SQLSecurityError."""
        query = "INSERT INTO users VALUES (1, 'test')"
        
        with pytest.raises(SQLSecurityError) as exc_info:
            parse_and_validate_sql(query)
        
        assert "Security validation failed" in str(exc_info.value)
    
    def test_global_validation_function(self):
        """Test the global validate_sql_query function."""
        # Valid query
        is_valid, error = validate_sql_query("SELECT * FROM users")
        assert is_valid is True
        assert error is None
        
        # Invalid query
        is_valid, error = validate_sql_query("DROP TABLE users")
        assert is_valid is False
        assert "Only SELECT queries" in error


class TestOperationLists:
    """Test cases for operation list methods."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.parser = QueryParser()
    
    def test_get_allowed_operations(self):
        """Test getting allowed operations list."""
        allowed = self.parser.get_allowed_operations()
        assert isinstance(allowed, dict)
        assert 'SELECT' in allowed
        assert 'WITH' in allowed
        assert len(allowed) >= 2
    
    def test_get_blocked_operations(self):
        """Test getting blocked operations list."""
        blocked = self.parser.get_blocked_operations()
        assert isinstance(blocked, dict)
        assert 'INSERT' in blocked
        assert 'UPDATE' in blocked
        assert 'DELETE' in blocked
        assert 'DROP' in blocked
        assert 'CREATE' in blocked
        assert len(blocked) > 10
    
    def test_operations_are_mutually_exclusive(self):
        """Test that allowed and blocked operations don't overlap."""
        allowed = set(self.parser.get_allowed_operations().keys())
        blocked = set(self.parser.get_blocked_operations().keys())
        
        overlap = allowed.intersection(blocked)
        assert len(overlap) == 0, f"Operations overlap: {overlap}"


class TestIntegrationScenarios:
    """Test cases for realistic integration scenarios."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.parser = QueryParser()
    
    def test_complex_analytics_query(self):
        """Test complex analytics query that should be allowed."""
        query = """
        WITH monthly_stats AS (
            SELECT 
                DATE_TRUNC('month', created_at) as month,
                COUNT(*) as total_users,
                COUNT(CASE WHEN active = true THEN 1 END) as active_users,
                AVG(age) as avg_age
            FROM users 
            WHERE created_at >= '2023-01-01'
            GROUP BY DATE_TRUNC('month', created_at)
        ),
        growth_rates AS (
            SELECT 
                month,
                total_users,
                active_users,
                avg_age,
                LAG(total_users) OVER (ORDER BY month) as prev_month_users,
                (total_users - LAG(total_users) OVER (ORDER BY month)) / 
                NULLIF(LAG(total_users) OVER (ORDER BY month), 0) * 100 as growth_rate
            FROM monthly_stats
        )
        SELECT 
            month,
            total_users,
            active_users,
            ROUND(avg_age, 2) as avg_age,
            ROUND(growth_rate, 2) as growth_rate_percent
        FROM growth_rates
        WHERE month IS NOT NULL
        ORDER BY month DESC
        LIMIT 12
        """
        
        is_valid, error = self.parser.validate_query(query)
        assert is_valid is True
        assert error is None
    
    def test_reporting_query_with_multiple_joins(self):
        """Test complex reporting query with multiple joins."""
        query = """
        SELECT 
            u.id,
            u.name,
            u.email,
            COUNT(DISTINCT o.id) as total_orders,
            SUM(oi.quantity * oi.price) as total_spent,
            AVG(r.rating) as avg_rating,
            MAX(o.created_at) as last_order_date
        FROM users u
        LEFT JOIN orders o ON u.id = o.user_id
        LEFT JOIN order_items oi ON o.id = oi.order_id
        LEFT JOIN reviews r ON u.id = r.user_id
        WHERE u.active = true
          AND u.created_at >= '2023-01-01'
        GROUP BY u.id, u.name, u.email
        HAVING COUNT(DISTINCT o.id) > 0
        ORDER BY total_spent DESC
        LIMIT 100
        """
        
        is_valid, error = self.parser.validate_query(query)
        assert is_valid is True
        assert error is None
    
    def test_data_export_simulation_blocked(self):
        """Test that data export with modification is blocked."""
        query = """
        CREATE TABLE user_export AS 
        SELECT * FROM users WHERE active = true;
        
        INSERT INTO audit_log (action, table_name, timestamp)
        VALUES ('export', 'users', NOW());
        """
        
        is_valid, error = self.parser.validate_query(query)
        assert is_valid is False
        # Should fail on multiple statements first
        assert "Multiple SQL statements" in error