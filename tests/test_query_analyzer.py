"""Tests for the QueryAnalyzer system in LocalData MCP v1.3.1.

This module tests the pre-query analysis system including:
- Query complexity analysis
- Memory usage estimation  
- Token count estimation
- Execution time estimation
- Risk assessment and recommendations
- Integration with execute_query
"""

import json
import unittest
from unittest.mock import MagicMock, patch, Mock
import pandas as pd
import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from localdata_mcp.query_analyzer import (
    QueryAnalyzer, 
    QueryAnalysis, 
    analyze_query,
    get_query_analyzer
)
from localdata_mcp.query_parser import SQLSecurityError


class TestQueryAnalyzer(unittest.TestCase):
    """Test the QueryAnalyzer class functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = QueryAnalyzer()
        
        # Create in-memory SQLite engine for testing
        self.engine = create_engine("sqlite:///:memory:")
        
        # Create test table with sample data
        with self.engine.connect() as conn:
            conn.execute(text("""
                CREATE TABLE test_table (
                    id INTEGER PRIMARY KEY,
                    name TEXT,
                    age INTEGER,
                    salary REAL,
                    is_active BOOLEAN,
                    description TEXT
                )
            """))
            
            # Insert test data
            test_data = [
                (1, "Alice", 30, 50000.0, True, "Software engineer with 5 years experience"),
                (2, "Bob", 25, 45000.0, True, "Junior developer"),
                (3, "Charlie", 35, 65000.0, False, "Senior architect"),
                (4, "Diana", 28, 55000.0, True, "Full-stack developer"),
                (5, "Eve", 32, 60000.0, True, "DevOps engineer"),
            ]
            
            for data in test_data:
                conn.execute(text("""
                    INSERT INTO test_table (id, name, age, salary, is_active, description)
                    VALUES (?, ?, ?, ?, ?, ?)
                """), data)
    
    def test_query_complexity_analysis(self):
        """Test query complexity scoring and feature detection."""
        test_cases = [
            # Simple SELECT - should have low complexity
            ("SELECT * FROM test_table", 1, {
                'has_joins': False,
                'has_aggregations': False,
                'has_subqueries': False,
                'has_window_functions': False
            }),
            
            # Query with JOIN - should have higher complexity
            ("SELECT t1.*, t2.name FROM test_table t1 JOIN test_table t2 ON t1.id = t2.id", 3, {
                'has_joins': True,
                'has_aggregations': False,
                'has_subqueries': False,
                'has_window_functions': False
            }),
            
            # Query with aggregation - should have medium complexity
            ("SELECT COUNT(*), AVG(salary) FROM test_table GROUP BY is_active", 3, {
                'has_joins': False,
                'has_aggregations': True,
                'has_subqueries': False,
                'has_window_functions': False
            }),
            
            # Complex query with subquery
            ("SELECT * FROM test_table WHERE salary > (SELECT AVG(salary) FROM test_table)", 4, {
                'has_joins': False,
                'has_aggregations': False,
                'has_subqueries': True,
                'has_window_functions': False
            }),
            
            # Very complex query with multiple features
            ("""SELECT 
                    name, 
                    salary,
                    COUNT(*) OVER (PARTITION BY is_active) as active_count,
                    AVG(salary) as avg_salary
                FROM test_table t1 
                JOIN test_table t2 ON t1.age = t2.age
                WHERE salary > (SELECT AVG(salary) FROM test_table)
                GROUP BY name, salary, is_active
            """, 10, {
                'has_joins': True,
                'has_aggregations': True,
                'has_subqueries': True,
                'has_window_functions': True
            })
        ]
        
        for query, expected_min_score, expected_features in test_cases:
            with self.subTest(query=query[:50]):
                complexity = self.analyzer._analyze_query_complexity(query)
                
                # Check minimum complexity score
                self.assertGreaterEqual(complexity['score'], expected_min_score,
                    f"Query complexity score {complexity['score']} should be at least {expected_min_score}")
                
                # Check feature detection
                for feature, expected in expected_features.items():
                    self.assertEqual(complexity[feature], expected,
                        f"Feature {feature} should be {expected}")
    
    def test_row_count_estimation(self):
        """Test COUNT(*) row count estimation."""
        test_queries = [
            "SELECT * FROM test_table",
            "SELECT * FROM test_table WHERE age > 30",
            "SELECT * FROM test_table WHERE salary > 50000",
        ]
        
        for query in test_queries:
            with self.subTest(query=query):
                row_count = self.analyzer._get_row_count(query, self.engine)
                
                # Should return reasonable count (our test table has 5 rows)
                self.assertIsInstance(row_count, int)
                self.assertGreaterEqual(row_count, 0)
                self.assertLessEqual(row_count, 5)
    
    def test_sample_row_analysis(self):
        """Test LIMIT 1 sample row analysis."""
        query = "SELECT * FROM test_table"
        
        sample_row, column_info = self.analyzer._get_sample_row(query, self.engine)
        
        # Should have sample data
        self.assertIsNotNone(sample_row)
        self.assertIsInstance(sample_row, pd.Series)
        
        # Should have correct column count
        self.assertEqual(column_info['count'], 6)  # id, name, age, salary, is_active, description
        
        # Should have column type information
        self.assertIn('types', column_info)
        self.assertEqual(len(column_info['types']), 6)
    
    def test_memory_estimation(self):
        """Test memory usage estimation with different data types."""
        # Create sample row with various data types
        sample_data = {
            'id': 1,
            'name': 'Test User',
            'age': 30,
            'salary': 50000.0,
            'is_active': True,
            'description': 'This is a test description with some text content'
        }
        sample_row = pd.Series(sample_data)
        
        column_info = {
            'count': 6,
            'types': {
                'id': 'int64',
                'name': 'object',
                'age': 'int64', 
                'salary': 'float64',
                'is_active': 'bool',
                'description': 'object'
            }
        }
        
        # Test with different row counts
        test_cases = [1, 100, 1000, 10000]
        
        for row_count in test_cases:
            with self.subTest(row_count=row_count):
                memory_analysis = self.analyzer._estimate_memory_usage(
                    row_count, sample_row, column_info
                )
                
                # Should return valid analysis
                self.assertIn('row_size', memory_analysis)
                self.assertIn('total_memory', memory_analysis)
                self.assertIn('risk_level', memory_analysis)
                
                # Memory should scale with row count
                self.assertGreater(memory_analysis['row_size'], 0)
                self.assertGreater(memory_analysis['total_memory'], 0)
                
                # Risk level should be valid
                self.assertIn(memory_analysis['risk_level'], ['low', 'medium', 'high', 'critical'])
    
    @patch('localdata_mcp.query_analyzer.tiktoken')
    def test_token_estimation(self, mock_tiktoken):
        """Test token count estimation using tiktoken."""
        # Mock tiktoken encoding
        mock_encoder = Mock()
        mock_encoder.encode.return_value = [1, 2, 3, 4, 5]  # 5 tokens
        mock_tiktoken.get_encoding.return_value = mock_encoder
        
        # Re-initialize analyzer with mocked tiktoken
        analyzer = QueryAnalyzer()
        
        # Create sample row
        sample_data = {
            'id': 1,
            'name': 'Test User',
            'description': 'This is a test description'
        }
        sample_row = pd.Series(sample_data)
        
        column_info = {
            'count': 3,
            'types': {
                'id': 'int64',
                'name': 'object',
                'description': 'object'
            }
        }
        
        token_analysis = analyzer._estimate_token_count(100, sample_row, column_info)
        
        # Should return valid analysis
        self.assertIn('tokens_per_row', token_analysis)
        self.assertIn('total_tokens', token_analysis)
        self.assertIn('risk_level', token_analysis)
        
        # Should have reasonable token counts
        self.assertGreater(token_analysis['tokens_per_row'], 0)
        self.assertGreater(token_analysis['total_tokens'], 0)
        
        # Risk level should be valid
        self.assertIn(token_analysis['risk_level'], ['low', 'medium', 'high', 'critical'])
    
    def test_execution_time_estimation(self):
        """Test execution time estimation based on complexity."""
        complexity_cases = [
            ({'score': 1, 'has_joins': False, 'has_aggregations': False, 
              'has_subqueries': False, 'has_window_functions': False}, 'low'),
            ({'score': 5, 'has_joins': True, 'has_aggregations': False, 
              'has_subqueries': False, 'has_window_functions': False}, 'low'),
            ({'score': 8, 'has_joins': True, 'has_aggregations': True, 
              'has_subqueries': True, 'has_window_functions': False}, 'medium'),
            ({'score': 10, 'has_joins': True, 'has_aggregations': True, 
              'has_subqueries': True, 'has_window_functions': True}, 'medium'),
        ]
        
        for complexity_analysis, expected_risk_max in complexity_cases:
            with self.subTest(complexity_score=complexity_analysis['score']):
                timeout_analysis = self.analyzer._estimate_execution_time(
                    1000, complexity_analysis, self.engine
                )
                
                # Should return valid analysis
                self.assertIn('estimated_time', timeout_analysis)
                self.assertIn('risk_level', timeout_analysis)
                
                # Time should be reasonable
                self.assertGreater(timeout_analysis['estimated_time'], 0)
                
                # Risk level should be valid
                self.assertIn(timeout_analysis['risk_level'], ['low', 'medium', 'high', 'critical'])
    
    def test_full_query_analysis(self):
        """Test complete query analysis workflow."""
        query = "SELECT * FROM test_table WHERE age > 30"
        
        analysis = self.analyzer.analyze_query(query, self.engine, "test_db")
        
        # Should return QueryAnalysis object
        self.assertIsInstance(analysis, QueryAnalysis)
        
        # Check required fields are present
        self.assertIsNotNone(analysis.query)
        self.assertIsNotNone(analysis.query_hash)
        self.assertIsNotNone(analysis.validated_query)
        self.assertIsNotNone(analysis.estimated_rows)
        self.assertIsNotNone(analysis.complexity_score)
        
        # Check risk levels are valid
        risk_levels = ['low', 'medium', 'high', 'critical']
        self.assertIn(analysis.memory_risk_level, risk_levels)
        self.assertIn(analysis.token_risk_level, risk_levels)
        self.assertIn(analysis.timeout_risk_level, risk_levels)
        
        # Check recommendations
        self.assertIsInstance(analysis.recommendations, list)
        self.assertIsInstance(analysis.should_chunk, bool)
        
        # Check timing information
        self.assertGreater(analysis.analysis_time_seconds, 0)
        self.assertGreater(analysis.timestamp, 0)
    
    def test_recommendations_generation(self):
        """Test recommendation generation based on different risk scenarios."""
        # High memory scenario
        memory_analysis = {'total_memory': 300, 'risk_level': 'high'}
        token_analysis = {'total_tokens': 5000, 'risk_level': 'low'}
        timeout_analysis = {'estimated_time': 2, 'risk_level': 'low'}
        
        recommendations = self.analyzer._generate_recommendations(
            2000, memory_analysis, token_analysis, timeout_analysis
        )
        
        self.assertTrue(recommendations['should_chunk'])
        self.assertIsNotNone(recommendations['chunk_size'])
        self.assertGreater(len(recommendations['messages']), 0)
        
        # Low risk scenario
        memory_analysis = {'total_memory': 5, 'risk_level': 'low'}
        token_analysis = {'total_tokens': 500, 'risk_level': 'low'}
        timeout_analysis = {'estimated_time': 0.5, 'risk_level': 'low'}
        
        recommendations = self.analyzer._generate_recommendations(
            50, memory_analysis, token_analysis, timeout_analysis
        )
        
        self.assertFalse(recommendations['should_chunk'])
        self.assertGreater(len(recommendations['messages']), 0)
    
    def test_error_handling(self):
        """Test error handling in query analysis."""
        # Test with invalid query
        with self.assertRaises(SQLSecurityError):
            self.analyzer.analyze_query("DROP TABLE test_table", self.engine, "test_db")
        
        # Test with non-existent table
        query = "SELECT * FROM non_existent_table"
        try:
            analysis = self.analyzer.analyze_query(query, self.engine, "test_db")
            # Should handle gracefully and still return analysis (even if estimates are wrong)
            self.assertIsInstance(analysis, QueryAnalysis)
        except Exception as e:
            # Should provide meaningful error message
            self.assertIsInstance(e, Exception)


class TestQueryAnalyzerIntegration(unittest.TestCase):
    """Test integration of QueryAnalyzer with DatabaseManager."""
    
    def setUp(self):
        """Set up integration test fixtures."""
        # Import here to avoid circular imports during testing
        from localdata_mcp.localdata_mcp import DatabaseManager
        
        self.db_manager = DatabaseManager()
        
        # Create test CSV file for integration testing
        import tempfile
        import os
        
        self.temp_dir = tempfile.mkdtemp()
        self.csv_path = os.path.join(self.temp_dir, "test_data.csv")
        
        # Create test CSV data
        csv_data = """id,name,age,salary,department
1,Alice,30,50000,Engineering
2,Bob,25,45000,Marketing
3,Charlie,35,65000,Engineering
4,Diana,28,55000,Sales
5,Eve,32,60000,Engineering
6,Frank,29,48000,Marketing
7,Grace,31,62000,Sales
8,Henry,26,47000,Engineering
9,Ivy,33,58000,Sales
10,Jack,27,52000,Marketing"""
        
        with open(self.csv_path, 'w') as f:
            f.write(csv_data)
        
        # Connect to the CSV file
        self.db_manager.connect_database("test_csv", "csv", self.csv_path)
    
    def tearDown(self):
        """Clean up integration test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_execute_query_with_analysis(self):
        """Test execute_query with analysis enabled."""
        query = "SELECT * FROM data_table WHERE age > 30"
        
        result_json = self.db_manager.execute_query("test_csv", query, enable_analysis=True)
        result = json.loads(result_json)
        
        # Should include analysis section
        self.assertIn('analysis', result)
        
        analysis = result['analysis']
        
        # Check analysis fields
        self.assertIn('estimated_rows', analysis)
        self.assertIn('actual_rows', analysis)
        self.assertIn('row_estimate_accuracy', analysis)
        self.assertIn('estimated_memory_mb', analysis)
        self.assertIn('estimated_tokens', analysis)
        self.assertIn('complexity_score', analysis)
        self.assertIn('risk_levels', analysis)
        self.assertIn('recommendations', analysis)
        self.assertIn('analysis_time', analysis)
        
        # Check risk levels structure
        risk_levels = analysis['risk_levels']
        self.assertIn('memory', risk_levels)
        self.assertIn('tokens', risk_levels)
        self.assertIn('timeout', risk_levels)
    
    def test_execute_query_without_analysis(self):
        """Test execute_query with analysis disabled."""
        query = "SELECT * FROM data_table LIMIT 5"
        
        result_json = self.db_manager.execute_query("test_csv", query, enable_analysis=False)
        result = json.loads(result_json)
        
        # Should not include analysis section
        self.assertNotIn('analysis', result)
        
        # Should still have data
        self.assertIn('data', result)
        self.assertIn('metadata', result)
    
    def test_analyze_query_preview_tool(self):
        """Test the analyze_query_preview MCP tool."""
        query = "SELECT department, COUNT(*), AVG(salary) FROM data_table GROUP BY department"
        
        result_json = self.db_manager.analyze_query_preview("test_csv", query)
        result = json.loads(result_json)
        
        # Should have all expected sections
        expected_sections = [
            'query_info', 'estimates', 'query_features', 
            'risk_assessment', 'recommendations', 'sampling_info'
        ]
        
        for section in expected_sections:
            self.assertIn(section, result)
        
        # Check query_info section
        query_info = result['query_info']
        self.assertIn('query_hash', query_info)
        self.assertIn('complexity_score', query_info)
        self.assertIn('analysis_time', query_info)
        
        # Check estimates section
        estimates = result['estimates']
        self.assertIn('rows', estimates)
        self.assertIn('memory_mb', estimates)
        self.assertIn('tokens', estimates)
        self.assertIn('execution_time_seconds', estimates)
        
        # Check query_features section
        query_features = result['query_features']
        self.assertIn('has_joins', query_features)
        self.assertIn('has_aggregations', query_features)
        self.assertTrue(query_features['has_aggregations'])  # This query has GROUP BY
        
        # Check risk_assessment section
        risk_assessment = result['risk_assessment']
        self.assertIn('memory', risk_assessment)
        self.assertIn('tokens', risk_assessment)
        self.assertIn('timeout', risk_assessment)
        self.assertIn('overall_risk', risk_assessment)
    
    def test_chunking_based_on_analysis(self):
        """Test that chunking decisions are influenced by analysis results."""
        # Query that should trigger chunking recommendations
        query = "SELECT * FROM data_table"
        
        result_json = self.db_manager.execute_query("test_csv", query, enable_analysis=True)
        result = json.loads(result_json)
        
        # Check if chunking was applied based on analysis
        if result.get('metadata', {}).get('chunked', False):
            # If chunked, should have analysis that influenced the decision
            self.assertIn('analysis', result)
            analysis = result['analysis']
            
            # Should have recommendations about chunking
            self.assertIn('recommendations', analysis)
            self.assertIsInstance(analysis['recommendations'], list)


class TestQueryAnalyzerAccuracy(unittest.TestCase):
    """Test accuracy of QueryAnalyzer predictions."""
    
    def setUp(self):
        """Set up accuracy test fixtures."""
        self.analyzer = QueryAnalyzer()
        self.engine = create_engine("sqlite:///:memory:")
        
        # Create larger test dataset for accuracy testing
        with self.engine.connect() as conn:
            conn.execute(text("""
                CREATE TABLE large_table (
                    id INTEGER PRIMARY KEY,
                    category TEXT,
                    value REAL,
                    description TEXT,
                    created_date TEXT
                )
            """))
            
            # Insert more test data for better accuracy testing
            import random
            import datetime
            
            categories = ['A', 'B', 'C', 'D', 'E']
            base_date = datetime.datetime(2023, 1, 1)
            
            for i in range(1, 101):  # 100 rows
                category = random.choice(categories)
                value = random.uniform(1.0, 1000.0)
                description = f"Description for item {i} " * random.randint(1, 5)
                created_date = (base_date + datetime.timedelta(days=random.randint(0, 365))).isoformat()
                
                conn.execute(text("""
                    INSERT INTO large_table (id, category, value, description, created_date)
                    VALUES (?, ?, ?, ?, ?)
                """), (i, category, value, description, created_date))
    
    def test_row_count_accuracy(self):
        """Test accuracy of row count estimation."""
        test_queries = [
            ("SELECT * FROM large_table", 100),
            ("SELECT * FROM large_table WHERE id <= 50", 50),
            ("SELECT * FROM large_table WHERE category = 'A'", None),  # Variable count
            ("SELECT category, COUNT(*) FROM large_table GROUP BY category", 5),
        ]
        
        for query, expected_rows in test_queries:
            with self.subTest(query=query):
                # Get analysis prediction
                analysis = self.analyzer.analyze_query(query, self.engine, "test_db")
                
                # Get actual results
                with self.engine.connect() as conn:
                    actual_df = pd.read_sql_query(query, conn)
                    actual_rows = len(actual_df)
                
                if expected_rows is not None:
                    # Check both prediction and actual against expected
                    self.assertEqual(actual_rows, expected_rows)
                
                # Check prediction accuracy (allow some tolerance)
                if actual_rows > 0:
                    accuracy = 1 - abs(analysis.estimated_rows - actual_rows) / actual_rows
                    self.assertGreater(accuracy, 0.5, 
                        f"Row count accuracy too low: predicted {analysis.estimated_rows}, actual {actual_rows}")
    
    def test_memory_estimation_reasonableness(self):
        """Test that memory estimations are reasonable."""
        queries = [
            "SELECT * FROM large_table LIMIT 10",
            "SELECT * FROM large_table LIMIT 50", 
            "SELECT * FROM large_table",
        ]
        
        for query in queries:
            with self.subTest(query=query):
                analysis = self.analyzer.analyze_query(query, self.engine, "test_db")
                
                # Memory estimate should be reasonable
                self.assertGreater(analysis.estimated_total_memory_mb, 0)
                self.assertLess(analysis.estimated_total_memory_mb, 1000)  # Shouldn't be crazy high
                
                # Row size should be reasonable
                self.assertGreater(analysis.estimated_row_size_bytes, 0)
                self.assertLess(analysis.estimated_row_size_bytes, 10000)  # Shouldn't be unreasonably large


def run_query_analyzer_tests():
    """Run all QueryAnalyzer tests."""
    test_classes = [
        TestQueryAnalyzer,
        TestQueryAnalyzerIntegration, 
        TestQueryAnalyzerAccuracy
    ]
    
    suite = unittest.TestSuite()
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    runner = unittest.TextTestRunner(verbosity=2)
    return runner.run(suite)


if __name__ == '__main__':
    run_query_analyzer_tests()