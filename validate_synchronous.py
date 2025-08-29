#!/usr/bin/env python3
"""
Synchronous validation script for the LocalData MCP Server 9-tool architecture.

This script directly calls the DatabaseManager methods to validate functionality.
"""

import json
import os
import tempfile
import pandas as pd
from pathlib import Path

# Import the DatabaseManager directly
from localdata_mcp.localdata_mcp import DatabaseManager


class SynchronousValidator:
    """Validates the 9-tool streamlined architecture using direct method calls."""
    
    def __init__(self):
        self.manager = DatabaseManager()
        self.test_files = {}
        self.results = {
            "tool_count": 0,
            "tools_working": [],
            "tools_failed": [],
            "json_responses": 0,
            "non_json_responses": 0,
            "sql_flavors_detected": [],
            "memory_management_working": False,
            "chunking_working": False,
            "pagination_working": False,
            "errors": []
        }
        
        # Map tool names to actual method functions (extract from FastMCP wrappers)
        self.tool_methods = {
            'connect_database': self.manager.connect_database.fn,
            'disconnect_database': self.manager.disconnect_database.fn,
            'execute_query': self.manager.execute_query.fn,
            'next_chunk': self.manager.next_chunk.fn,
            'list_databases': self.manager.list_databases.fn,
            'describe_database': self.manager.describe_database.fn,
            'find_table': self.manager.find_table.fn,
            'describe_table': self.manager.describe_table.fn,
            'get_query_history': self.manager.get_query_history.fn
        }
    
    def setup_test_data(self):
        """Create test data files."""
        print("🔧 Setting up test data files...")
        
        # Create CSV test file (150 rows to test chunking)
        csv_data = pd.DataFrame({
            'id': range(1, 151),
            'name': [f'user_{i}' for i in range(1, 151)],
            'email': [f'user{i}@example.com' for i in range(1, 151)],
            'score': [i * 2.5 for i in range(1, 151)]
        })
        
        csv_path = './test_data.csv'
        csv_data.to_csv(csv_path, index=False)
        self.test_files['csv'] = csv_path
        
        # Create JSON test file
        json_data = [
            {'category': 'A', 'count': 10, 'active': True},
            {'category': 'B', 'count': 20, 'active': False},
            {'category': 'C', 'count': 15, 'active': True},
            {'category': 'D', 'count': 8, 'active': True}
        ]
        
        json_path = './test_data.json'
        with open(json_path, 'w') as f:
            json.dump(json_data, f)
        self.test_files['json'] = json_path
        
        print(f"   ✅ Created CSV file: {csv_path} (150 rows)")
        print(f"   ✅ Created JSON file: {json_path} (4 rows)")
    
    def cleanup_test_data(self):
        """Clean up test files."""
        for file_path in self.test_files.values():
            if os.path.exists(file_path):
                os.unlink(file_path)
        print("🧹 Cleaned up test data files")
    
    def is_json_response(self, response):
        """Check if response is valid JSON."""
        if isinstance(response, (dict, list)):
            return True
        try:
            json.loads(response)
            return True
        except (json.JSONDecodeError, TypeError):
            return False
    
    def count_tools(self):
        """Count available tools."""
        self.results["tool_count"] = len(self.tool_methods)
        return list(self.tool_methods.keys())
    
    def run_full_validation(self):
        """Run complete validation of the 9-tool architecture."""
        print("🚀 Starting LocalData MCP Server 9-Tool Architecture Validation")
        print("=" * 65)
        
        try:
            # Setup
            self.setup_test_data()
            available_tools = self.count_tools()
            print(f"\n🔧 Found {len(available_tools)}/9 expected tools")
            for tool in available_tools:
                print(f"   • {tool}")
            
            # Test Tool 1: connect_database
            print("\n🔌 Testing Tool 1: connect_database")
            self.test_connect_database()
            
            # Test Tool 2: list_databases
            print("\n📋 Testing Tool 2: list_databases")
            self.test_list_databases()
            
            # Test Tool 3: execute_query  
            print("\n🔍 Testing Tool 3: execute_query")
            self.test_execute_query()
            
            # Test Tool 4: next_chunk
            print("\n📄 Testing Tool 4: next_chunk")
            self.test_next_chunk()
            
            # Test Tool 5: describe_database
            print("\n🔍 Testing Tool 5: describe_database")
            self.test_describe_database()
            
            # Test Tool 6: find_table
            print("\n🔎 Testing Tool 6: find_table")
            self.test_find_table()
            
            # Test Tool 7: describe_table
            print("\n📊 Testing Tool 7: describe_table")
            self.test_describe_table()
            
            # Test Tool 8: get_query_history
            print("\n📜 Testing Tool 8: get_query_history")
            self.test_get_query_history()
            
            # Test Tool 9: disconnect_database
            print("\n🔌 Testing Tool 9: disconnect_database")
            self.test_disconnect_database()
            
            # Generate report
            success = self.generate_report()
            return success
            
        except Exception as e:
            print(f"\n💥 Critical error during validation: {e}")
            return False
        
        finally:
            try:
                # Clean up any remaining connections
                for name in list(self.manager.connections.keys()):
                    try:
                        self.manager.disconnect_database.fn(self.manager, name)
                    except:
                        pass
                self.cleanup_test_data()
            except Exception as e:
                print(f"⚠️  Cleanup error: {e}")
    
    def test_connect_database(self):
        """Test connect_database with SQL flavor detection."""
        try:
            # Test CSV connection
            result = self.manager.connect_database.fn(self.manager, "csv_db", "csv", self.test_files['csv'])
            
            if self.is_json_response(result):
                self.results["json_responses"] += 1
                response = json.loads(result) if isinstance(result, str) else result
                
                if response.get("success") and "sql_flavor" in response.get("connection_info", {}):
                    sql_flavor = response["connection_info"]["sql_flavor"]
                    self.results["sql_flavors_detected"].append(sql_flavor)
                    print(f"   ✅ CSV connection successful, SQL flavor: {sql_flavor}")
                    
                    # Test JSON connection
                    result2 = self.manager.connect_database.fn(self.manager, "json_db", "json", self.test_files['json'])
                    if self.is_json_response(result2):
                        self.results["json_responses"] += 1
                        response2 = json.loads(result2) if isinstance(result2, str) else result2
                        if response2.get("success"):
                            print(f"   ✅ JSON connection successful")
                    
                    self.results["tools_working"].append("connect_database")
                else:
                    print(f"   ❌ Unexpected response structure")
                    self.results["tools_failed"].append("connect_database")
            else:
                self.results["non_json_responses"] += 1
                print(f"   ❌ Non-JSON response: {result}")
                self.results["tools_failed"].append("connect_database")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            self.results["tools_failed"].append("connect_database")
            self.results["errors"].append(f"connect_database: {e}")
    
    def test_list_databases(self):
        """Test list_databases with SQL flavor info."""
        try:
            result = self.manager.list_databases.fn(self.manager)
            
            if self.is_json_response(result):
                self.results["json_responses"] += 1
                response = json.loads(result) if isinstance(result, str) else result
                
                if "databases" in response:
                    for db in response["databases"]:
                        if "sql_flavor" in db:
                            self.results["sql_flavors_detected"].append(db["sql_flavor"])
                    print(f"   ✅ Listed {response.get('total_connections', 0)} databases with SQL flavor info")
                    self.results["tools_working"].append("list_databases")
                else:
                    print(f"   ❌ Missing databases key in response")
                    self.results["tools_failed"].append("list_databases")
            else:
                self.results["non_json_responses"] += 1
                print(f"   ❌ Non-JSON response")
                self.results["tools_failed"].append("list_databases")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
            self.results["tools_failed"].append("list_databases")
            self.results["errors"].append(f"list_databases: {e}")
    
    def test_execute_query(self):
        """Test execute_query with chunking and memory management."""
        try:
            # Test small query (no chunking)
            small_result = self.manager.execute_query.fn(self.manager, "csv_db", "SELECT * FROM data_table LIMIT 5")
            
            if self.is_json_response(small_result):
                self.results["json_responses"] += 1
                small_response = json.loads(small_result) if isinstance(small_result, str) else small_result
                
                if "metadata" in small_response and small_response["metadata"].get("chunked") is False:
                    print(f"   ✅ Small query returned {len(small_response.get('data', []))} rows without chunking")
                    
                    # Check memory info
                    if "memory_info" in small_response["metadata"]:
                        self.results["memory_management_working"] = True
                        print(f"   ✅ Memory management working - memory info included")
            
            # Test large query (should trigger chunking)
            large_result = self.manager.execute_query.fn(self.manager, "csv_db", "SELECT * FROM data_table")
            
            if self.is_json_response(large_result):
                self.results["json_responses"] += 1
                large_response = json.loads(large_result) if isinstance(large_result, str) else large_result
                
                if "metadata" in large_response and large_response["metadata"].get("chunked") is True:
                    self.results["chunking_working"] = True
                    query_id = large_response["metadata"].get("query_id")
                    print(f"   ✅ Large query triggered chunking (query_id: {query_id})")
                    self.results["tools_working"].append("execute_query")
                    
                    # Store query_id for next_chunk test
                    self.query_id = query_id
                else:
                    print(f"   ❌ Large query should have triggered chunking")
                    self.results["tools_failed"].append("execute_query")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            self.results["tools_failed"].append("execute_query")
            self.results["errors"].append(f"execute_query: {e}")
    
    def test_next_chunk(self):
        """Test next_chunk (pagination)."""
        if not hasattr(self, 'query_id'):
            print("   ⏭️  Skipping next_chunk test (no query_id from execute_query)")
            return
        
        try:
            result = self.manager.next_chunk.fn(self.manager, self.query_id, start_row=11, chunk_size="20")
            
            if self.is_json_response(result):
                self.results["json_responses"] += 1
                response = json.loads(result) if isinstance(result, str) else result
                
                if "data" in response and len(response["data"]) == 20:
                    self.results["pagination_working"] = True
                    print(f"   ✅ Pagination working - retrieved 20 rows starting from row 11")
                    self.results["tools_working"].append("next_chunk")
                else:
                    print(f"   ❌ Expected 20 rows, got {len(response.get('data', []))}")
                    self.results["tools_failed"].append("next_chunk")
            else:
                self.results["non_json_responses"] += 1
                print(f"   ❌ Non-JSON response")
                self.results["tools_failed"].append("next_chunk")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
            self.results["tools_failed"].append("next_chunk")
            self.results["errors"].append(f"next_chunk: {e}")
    
    def test_describe_database(self):
        """Test describe_database."""
        try:
            result = self.manager.describe_database.fn(self.manager, "csv_db")
            
            if self.is_json_response(result):
                self.results["json_responses"] += 1
                response = json.loads(result) if isinstance(result, str) else result
                
                if "tables" in response and len(response["tables"]) > 0:
                    table_count = len(response["tables"])
                    print(f"   ✅ Database described successfully ({table_count} tables found)")
                    self.results["tools_working"].append("describe_database")
                else:
                    print(f"   ❌ No tables found in database description")
                    self.results["tools_failed"].append("describe_database")
            else:
                self.results["non_json_responses"] += 1
                print(f"   ❌ Non-JSON response")
                self.results["tools_failed"].append("describe_database")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
            self.results["tools_failed"].append("describe_database")
            self.results["errors"].append(f"describe_database: {e}")
    
    def test_find_table(self):
        """Test find_table."""
        try:
            result = self.manager.find_table.fn(self.manager, "data_table")
            
            if self.is_json_response(result):
                self.results["json_responses"] += 1
                databases = json.loads(result) if isinstance(result, str) else result
                
                if isinstance(databases, list) and "csv_db" in databases:
                    print(f"   ✅ Table found in {len(databases)} database(s)")
                    self.results["tools_working"].append("find_table")
                else:
                    print(f"   ❌ Table not found or unexpected response format")
                    self.results["tools_failed"].append("find_table")
            else:
                self.results["non_json_responses"] += 1
                if "was not found" in str(result):
                    print(f"   ❌ Table search returned error message instead of JSON")
                self.results["tools_failed"].append("find_table")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
            self.results["tools_failed"].append("find_table")
            self.results["errors"].append(f"find_table: {e}")
    
    def test_describe_table(self):
        """Test describe_table."""
        try:
            result = self.manager.describe_table.fn(self.manager, "csv_db", "data_table")
            
            if self.is_json_response(result):
                self.results["json_responses"] += 1
                response = json.loads(result) if isinstance(result, str) else result
                
                if "columns" in response and len(response["columns"]) > 0:
                    col_count = len(response["columns"])
                    row_count = response.get("size", 0)
                    print(f"   ✅ Table described successfully ({col_count} columns, {row_count} rows)")
                    self.results["tools_working"].append("describe_table")
                else:
                    print(f"   ❌ No columns found in table description")
                    self.results["tools_failed"].append("describe_table")
            else:
                self.results["non_json_responses"] += 1
                print(f"   ❌ Non-JSON response")
                self.results["tools_failed"].append("describe_table")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
            self.results["tools_failed"].append("describe_table")
            self.results["errors"].append(f"describe_table: {e}")
    
    def test_get_query_history(self):
        """Test get_query_history."""
        try:
            result = self.manager.get_query_history.fn(self.manager, "csv_db")
            
            # Note: get_query_history returns plain text, not JSON
            if isinstance(result, str) and ("SELECT" in result or "No query history" in result):
                print(f"   ✅ Query history retrieved successfully")
                self.results["tools_working"].append("get_query_history")
            else:
                print(f"   ❌ Unexpected query history response: {result}")
                self.results["tools_failed"].append("get_query_history")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
            self.results["tools_failed"].append("get_query_history")
            self.results["errors"].append(f"get_query_history: {e}")
    
    def test_disconnect_database(self):
        """Test disconnect_database."""
        try:
            result = self.manager.disconnect_database.fn(self.manager, "json_db")
            
            if "Successfully disconnected" in str(result):
                print(f"   ✅ Database disconnected successfully")
                self.results["tools_working"].append("disconnect_database")
            else:
                print(f"   ❌ Unexpected disconnect response: {result}")
                self.results["tools_failed"].append("disconnect_database")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
            self.results["tools_failed"].append("disconnect_database")
            self.results["errors"].append(f"disconnect_database: {e}")
    
    def generate_report(self):
        """Generate validation report."""
        print("\n" + "="*60)
        print("📊 VALIDATION REPORT")
        print("="*60)
        
        print(f"🔧 Architecture:")
        print(f"   • Tool count: {self.results['tool_count']}/9")
        print(f"   • Tools working: {len(self.results['tools_working'])}/9")
        print(f"   • Tools failed: {len(self.results['tools_failed'])}")
        
        if self.results['tools_failed']:
            print(f"   • Failed tools: {', '.join(self.results['tools_failed'])}")
        
        print(f"\n📡 Response Format:")
        print(f"   • JSON responses: {self.results['json_responses']}")
        print(f"   • Non-JSON responses: {self.results['non_json_responses']}")
        
        print(f"\n🗄️  SQL Flavors Detected:")
        unique_flavors = list(set(self.results['sql_flavors_detected']))
        for flavor in unique_flavors:
            count = self.results['sql_flavors_detected'].count(flavor)
            print(f"   • {flavor}: {count} instances")
        
        print(f"\n🧠 Enhanced Features:")
        print(f"   • Memory management: {'✅' if self.results['memory_management_working'] else '❌'}")
        print(f"   • Chunking: {'✅' if self.results['chunking_working'] else '❌'}")  
        print(f"   • Pagination: {'✅' if self.results['pagination_working'] else '❌'}")
        
        if self.results['errors']:
            print(f"\n❌ Errors Encountered:")
            for error in self.results['errors']:
                print(f"   • {error}")
        
        # Overall assessment
        working_tools = len(self.results['tools_working'])
        total_expected = 9
        success_rate = (working_tools / total_expected) * 100
        
        print(f"\n🎯 Overall Assessment:")
        print(f"   • Success rate: {success_rate:.1f}% ({working_tools}/{total_expected})")
        
        if success_rate >= 90:
            print("   • Status: ✅ EXCELLENT - Architecture fully validated")
        elif success_rate >= 75:
            print("   • Status: ✅ GOOD - Architecture mostly working")
        elif success_rate >= 50:
            print("   • Status: ⚠️  PARTIAL - Some issues need attention")
        else:
            print("   • Status: ❌ CRITICAL - Major issues found")
            
        return success_rate >= 75


def main():
    """Main validation entry point."""
    validator = SynchronousValidator()
    success = validator.run_full_validation()
    
    if success:
        print("\n🎉 Validation completed successfully!")
        return 0
    else:
        print("\n❌ Validation found critical issues!")
        return 1


if __name__ == "__main__":
    exit(main())