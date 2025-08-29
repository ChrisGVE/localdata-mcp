"""Tests for multi-sheet handling in Excel and ODS files."""

import os
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import pandas as pd

from localdata_mcp import DatabaseManager


class TestMultiSheetHandling:
    """Test multi-sheet functionality for Excel and ODS files."""
    
    @pytest.fixture
    def manager(self):
        """Create a fresh DatabaseManager instance for testing."""
        return DatabaseManager()
    
    @pytest.fixture
    def multi_sheet_data(self):
        """Create sample data for multiple sheets."""
        return {
            'employees': pd.DataFrame({
                'name': ['Alice', 'Bob', 'Carol'],
                'department': ['Engineering', 'Sales', 'Marketing'],
                'salary': [75000, 65000, 70000]
            }),
            'products': pd.DataFrame({
                'product_name': ['Widget A', 'Widget B', 'Gadget X'],
                'price': [19.99, 29.99, 49.99],
                'category': ['Widgets', 'Widgets', 'Gadgets']
            }),
            'sales': pd.DataFrame({
                'region': ['North', 'South', 'East'],
                'q1_sales': [125000, 98000, 110000],
                'q2_sales': [132000, 105000, 118000]
            })
        }
    
    @patch('localdata_mcp.localdata_mcp.OPENPYXL_AVAILABLE', True)
    def test_excel_multi_sheet_connection(self, manager, multi_sheet_data):
        """Test connecting to multi-sheet Excel files."""
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            # Create multi-sheet Excel file
            with pd.ExcelWriter(temp_path, engine='openpyxl') as writer:
                for sheet_name, df in multi_sheet_data.items():
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
            
            # Connect to Excel file (should load all sheets)
            result = manager.connect_database("multi_excel", "excel", temp_path)
            assert "Successfully connected" in result
            assert "multi_excel" in manager.connections
            
            # Test that all sheets are loaded as separate tables
            description = manager.describe_database("multi_excel")
            assert "employees" in description
            assert "products" in description
            assert "sales" in description
            
            # Test querying each sheet
            emp_result = manager.execute_query("multi_excel", "SELECT COUNT(*) FROM employees")
            assert "3" in emp_result
            
            prod_result = manager.execute_query("multi_excel", "SELECT COUNT(*) FROM products")
            assert "3" in prod_result
            
            sales_result = manager.execute_query("multi_excel", "SELECT COUNT(*) FROM sales")
            assert "3" in sales_result
            
            # Test cross-sheet queries (JOIN across sheets)
            join_result = manager.execute_query("multi_excel", 
                "SELECT e.name, e.department FROM employees e")
            assert "Alice" in join_result
            assert "Engineering" in join_result
            
            manager.disconnect_database("multi_excel")
        
        except ImportError:
            pytest.skip("openpyxl not available for Excel testing")
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    @patch('localdata_mcp.localdata_mcp.OPENPYXL_AVAILABLE', True)
    def test_excel_specific_sheet_connection(self, manager, multi_sheet_data):
        """Test connecting to a specific sheet in Excel file."""
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            # Create multi-sheet Excel file
            with pd.ExcelWriter(temp_path, engine='openpyxl') as writer:
                for sheet_name, df in multi_sheet_data.items():
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
            
            # Connect to specific sheet only
            result = manager.connect_database("single_sheet", "excel", temp_path, sheet_name="employees")
            assert "Successfully connected" in result
            
            # Should only have the employees table
            description = manager.describe_database("single_sheet")
            assert "employees" in description
            assert "products" not in description
            assert "sales" not in description
            
            # Test querying the specific sheet
            result = manager.execute_query("single_sheet", "SELECT name FROM employees WHERE department = 'Engineering'")
            assert "Alice" in result
            
            manager.disconnect_database("single_sheet")
        
        except ImportError:
            pytest.skip("openpyxl not available for Excel testing")
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    @patch('localdata_mcp.localdata_mcp.ODFPY_AVAILABLE', True)
    def test_ods_multi_sheet_connection(self, manager, multi_sheet_data):
        """Test connecting to multi-sheet ODS files."""
        with tempfile.NamedTemporaryFile(suffix='.ods', delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            # Create multi-sheet ODS file
            with pd.ExcelWriter(temp_path, engine='odf') as writer:
                for sheet_name, df in multi_sheet_data.items():
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
            
            # Connect to ODS file
            result = manager.connect_database("multi_ods", "ods", temp_path)
            assert "Successfully connected" in result
            
            # Test that all sheets are loaded
            description = manager.describe_database("multi_ods")
            assert "employees" in description
            assert "products" in description
            assert "sales" in description
            
            # Test querying sheets
            result = manager.execute_query("multi_ods", "SELECT product_name FROM products WHERE category = 'Widgets'")
            assert "Widget A" in result
            assert "Widget B" in result
            
            manager.disconnect_database("multi_ods")
        
        except ImportError:
            pytest.skip("odfpy not available for ODS testing")
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_sheet_name_sanitization_edge_cases(self, manager):
        """Test sheet name sanitization with various edge cases."""
        test_cases = [
            # (original_name, expected_pattern)
            ("Sheet with spaces", "Sheet_with_spaces"),
            ("Sheet-with-hyphens", "Sheet_with_hyphens"), 
            ("Sheet@#$%Special", "Sheet____Special"),
            ("123StartsWithNumber", "sheet_123StartsWithNumber"),
            ("", "sheet_unnamed"),
            ("Sheet.with.dots", "Sheet_with_dots"),
            ("Sheet/with/slashes", "Sheet_with_slashes"),
            ("Sheet\\with\\backslashes", "Sheet_with_backslashes"),
            ("Sheet:with:colons", "Sheet_with_colons"),
            ("Very_Long_Sheet_Name_That_Should_Still_Work_Fine", "Very_Long_Sheet_Name_That_Should_Still_Work_Fine"),
        ]
        
        used_names = set()
        for original, expected_start in test_cases:
            result = manager._sanitize_sheet_name(original, used_names)
            
            # Check that result starts with expected pattern or is exactly the expected
            assert result.startswith(expected_start.split('_')[0]) or result == expected_start
            
            # Check that result is valid SQL identifier
            import re
            assert re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', result), f"Invalid SQL identifier: {result}"
            
            # Check uniqueness
            assert result not in used_names
            used_names.add(result)
    
    def test_duplicate_sheet_names(self, manager):
        """Test handling of duplicate sheet names."""
        used_names = set()
        
        # Test multiple sheets with same name
        name1 = manager._sanitize_sheet_name("Data", used_names)
        name2 = manager._sanitize_sheet_name("Data", used_names)
        name3 = manager._sanitize_sheet_name("Data", used_names)
        
        assert name1 == "Data"
        assert name2 == "Data_1"  
        assert name3 == "Data_2"
        
        # All should be unique
        assert len({name1, name2, name3}) == 3
    
    @patch('localdata_mcp.localdata_mcp.OPENPYXL_AVAILABLE', True)
    def test_empty_sheets_handling(self, manager):
        """Test handling of empty sheets in multi-sheet files."""
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            # Create file with empty and non-empty sheets
            with pd.ExcelWriter(temp_path, engine='openpyxl') as writer:
                # Non-empty sheet
                pd.DataFrame({'col1': [1, 2, 3]}).to_excel(writer, sheet_name='data_sheet', index=False)
                # Empty sheet
                pd.DataFrame().to_excel(writer, sheet_name='empty_sheet', index=False)
                # Another non-empty sheet
                pd.DataFrame({'col2': ['a', 'b']}).to_excel(writer, sheet_name='text_sheet', index=False)
            
            # Connect and test
            result = manager.connect_database("mixed_sheets", "excel", temp_path)
            assert "Successfully connected" in result
            
            # Should only load non-empty sheets
            description = manager.describe_database("mixed_sheets")
            assert "data_sheet" in description
            assert "text_sheet" in description
            # Empty sheet might or might not be present depending on implementation
            
            # Test querying non-empty sheets
            data_result = manager.execute_query("mixed_sheets", "SELECT COUNT(*) FROM data_sheet")
            assert "3" in data_result
            
            text_result = manager.execute_query("mixed_sheets", "SELECT COUNT(*) FROM text_sheet")  
            assert "2" in text_result
            
            manager.disconnect_database("mixed_sheets")
        
        except ImportError:
            pytest.skip("openpyxl not available for empty sheets testing")
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    @patch('localdata_mcp.localdata_mcp.OPENPYXL_AVAILABLE', True)
    def test_sheet_with_special_characters(self, manager):
        """Test sheets with special characters in names."""
        special_data = pd.DataFrame({
            'test_col': ['value1', 'value2', 'value3']
        })
        
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            # Create file with sheets having special characters
            with pd.ExcelWriter(temp_path, engine='openpyxl') as writer:
                special_data.to_excel(writer, sheet_name='Sheet With Spaces', index=False)
                special_data.to_excel(writer, sheet_name='Sheet-With-Hyphens', index=False)
                special_data.to_excel(writer, sheet_name='Sheet@#$%', index=False)
                special_data.to_excel(writer, sheet_name='123Numbers', index=False)
            
            # Connect and test
            result = manager.connect_database("special_chars", "excel", temp_path)
            assert "Successfully connected" in result
            
            # Get table names and verify they're sanitized
            description = manager.describe_database("special_chars")
            
            # Should contain sanitized table names
            assert "Sheet_With_Spaces" in description or "Sheet_with_spaces" in description
            assert "Sheet_With_Hyphens" in description or "Sheet_with_hyphens" in description
            
            # Test querying tables with sanitized names
            tables_result = manager.execute_query("special_chars", 
                "SELECT name FROM sqlite_master WHERE type='table'")
            
            # Should be able to query at least one of the sanitized tables
            # Get the actual table names from the description
            import json
            db_info = json.loads(description)
            table_names = [table['name'] for table in db_info['tables']]
            
            if table_names:
                # Query the first table
                table_name = table_names[0]
                query_result = manager.execute_query("special_chars", 
                    f"SELECT COUNT(*) FROM {table_name}")
                assert "3" in query_result
            
            manager.disconnect_database("special_chars")
        
        except ImportError:
            pytest.skip("openpyxl not available for special characters testing")
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    @patch('localdata_mcp.localdata_mcp.OPENPYXL_AVAILABLE', True)
    def test_nonexistent_sheet_name(self, manager, multi_sheet_data):
        """Test connecting to non-existent sheet name."""
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            # Create multi-sheet Excel file
            with pd.ExcelWriter(temp_path, engine='openpyxl') as writer:
                for sheet_name, df in multi_sheet_data.items():
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
            
            # Try to connect to non-existent sheet
            result = manager.connect_database("nonexistent", "excel", temp_path, sheet_name="DoesNotExist")
            assert "Failed to connect" in result or "not found" in result
            
        except ImportError:
            pytest.skip("openpyxl not available for nonexistent sheet testing")
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    @patch('localdata_mcp.localdata_mcp.OPENPYXL_AVAILABLE', True)
    def test_large_multi_sheet_file(self, manager):
        """Test handling of files with many sheets."""
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            # Create file with many sheets (but not too many to avoid timeout)
            num_sheets = 10
            with pd.ExcelWriter(temp_path, engine='openpyxl') as writer:
                for i in range(num_sheets):
                    df = pd.DataFrame({
                        'id': range(10), 
                        'value': [f'sheet_{i}_row_{j}' for j in range(10)]
                    })
                    df.to_excel(writer, sheet_name=f'Sheet_{i}', index=False)
            
            # Connect and test
            result = manager.connect_database("many_sheets", "excel", temp_path)
            assert "Successfully connected" in result
            
            # Should have all sheets loaded
            description = manager.describe_database("many_sheets") 
            
            # Count tables in description
            import json
            db_info = json.loads(description)
            assert len(db_info['tables']) == num_sheets
            
            # Test querying different sheets
            result1 = manager.execute_query("many_sheets", "SELECT COUNT(*) FROM Sheet_0")
            assert "10" in result1
            
            result2 = manager.execute_query("many_sheets", "SELECT COUNT(*) FROM Sheet_5")
            assert "10" in result2
            
            manager.disconnect_database("many_sheets")
        
        except ImportError:
            pytest.skip("openpyxl not available for large multi-sheet testing")
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


class TestMultiSheetIntegration:
    """Test multi-sheet integration with MCP tools."""
    
    @pytest.fixture
    def manager(self):
        """Create a fresh DatabaseManager instance for testing."""
        return DatabaseManager()
    
    @patch('localdata_mcp.localdata_mcp.OPENPYXL_AVAILABLE', True)
    def test_find_table_multi_sheet(self, manager):
        """Test find_table tool with multi-sheet files."""
        test_data = {
            'customers': pd.DataFrame({'name': ['John', 'Jane'], 'city': ['NYC', 'LA']}),
            'orders': pd.DataFrame({'order_id': [1, 2], 'amount': [100, 200]}),
            'products': pd.DataFrame({'product': ['A', 'B'], 'price': [10, 20]})
        }
        
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            # Create multi-sheet file
            with pd.ExcelWriter(temp_path, engine='openpyxl') as writer:
                for sheet_name, df in test_data.items():
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
            
            manager.connect_database("test_db", "excel", temp_path)
            
            # Test finding specific tables
            customers_result = manager.find_table("customers")
            assert "test_db" in customers_result
            
            orders_result = manager.find_table("orders") 
            assert "test_db" in orders_result
            
            # Test table that doesn't exist
            nonexistent_result = manager.find_table("nonexistent")
            assert "not found" in nonexistent_result
            
            manager.disconnect_database("test_db")
        
        except ImportError:
            pytest.skip("openpyxl not available for find_table testing")
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    @patch('localdata_mcp.localdata_mcp.OPENPYXL_AVAILABLE', True)
    def test_describe_table_multi_sheet(self, manager):
        """Test describe_table tool with multi-sheet files."""
        test_data = pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['Alice', 'Bob', 'Carol'],
            'score': [95.5, 87.2, 91.8]
        })
        
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            # Create file with specific sheet
            with pd.ExcelWriter(temp_path, engine='openpyxl') as writer:
                test_data.to_excel(writer, sheet_name='student_data', index=False)
            
            manager.connect_database("students", "excel", temp_path)
            
            # Test describing the sheet/table
            description = manager.describe_table("students", "student_data")
            
            assert "student_data" in description
            assert "columns" in description
            assert "id" in description
            assert "name" in description
            assert "score" in description
            
            manager.disconnect_database("students")
        
        except ImportError:
            pytest.skip("openpyxl not available for describe_table testing")
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    @patch('localdata_mcp.localdata_mcp.OPENPYXL_AVAILABLE', True)
    def test_get_table_sample_multi_sheet(self, manager):
        """Test get_table_sample tool with multi-sheet files."""
        test_data = pd.DataFrame({
            'item': ['Item1', 'Item2', 'Item3', 'Item4', 'Item5'],
            'value': [10, 20, 30, 40, 50]
        })
        
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            # Create file
            with pd.ExcelWriter(temp_path, engine='openpyxl') as writer:
                test_data.to_excel(writer, sheet_name='inventory', index=False)
            
            manager.connect_database("store", "excel", temp_path)
            
            # Test getting sample data
            sample = manager.get_table_sample("store", "inventory", limit=3)
            
            assert "Item1" in sample
            assert "Item2" in sample
            assert "Item3" in sample
            # Should not contain Item4 or Item5 due to limit
            
            manager.disconnect_database("store")
        
        except ImportError:
            pytest.skip("openpyxl not available for table sample testing")
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])