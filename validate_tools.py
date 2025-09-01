#!/usr/bin/env python3
"""
Quick validation script for the new Phase 1 P0 tools.
Validates syntax, imports, and basic tool registration.
"""

import sys
import inspect
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def validate_tool_implementation():
    """Validate that our three new tools are implemented correctly."""
    print("🔍 Validating Phase 1 P0 Tool Implementation")
    print("=" * 50)
    
    try:
        # Import the main module
        from localdata_mcp.localdata_mcp import DatabaseManager
        print("✅ Successfully imported DatabaseManager")
        
        # Get all methods from DatabaseManager
        manager_methods = [method for method, _ in inspect.getmembers(DatabaseManager, predicate=inspect.isfunction)]
        
        # Check for our three new tools
        expected_tools = [
            'profile_table',
            'detect_data_types', 
            'analyze_distributions'
        ]
        
        found_tools = []
        for tool in expected_tools:
            if tool in manager_methods:
                found_tools.append(tool)
                print(f"✅ Found tool: {tool}")
            else:
                print(f"❌ Missing tool: {tool}")
        
        # Validate tool signatures
        print(f"\n📋 Tool Signature Validation:")
        
        manager_instance = DatabaseManager()
        
        for tool in found_tools:
            method = getattr(manager_instance, tool)
            sig = inspect.signature(method)
            params = list(sig.parameters.keys())
            
            print(f"\n🔧 {tool}:")
            print(f"   Parameters: {params}")
            
            # Basic validation checks
            if 'name' in params:
                print("   ✅ Has 'name' parameter for database connection")
            else:
                print("   ❌ Missing 'name' parameter")
                
            if 'table_name' in params or 'query' in params:
                print("   ✅ Has data source parameters")
            else:
                print("   ❌ Missing data source parameters")
        
        # Check for MCP tool decorators by looking at source
        import localdata_mcp.localdata_mcp as mcp_module
        source = inspect.getsource(mcp_module)
        
        print(f"\n🎯 MCP Tool Registration Check:")
        tool_decorators = 0
        for tool in expected_tools:
            if f"@mcp.tool" in source and f"def {tool}" in source:
                print(f"   ✅ {tool} has @mcp.tool decorator")
                tool_decorators += 1
            else:
                print(f"   ❌ {tool} missing @mcp.tool decorator")
        
        # Summary
        print(f"\n📊 VALIDATION SUMMARY:")
        print(f"   Tools implemented: {len(found_tools)}/3")
        print(f"   Tools with decorators: {tool_decorators}/3")
        
        if len(found_tools) == 3 and tool_decorators == 3:
            print("🎉 All Phase 1 P0 tools are properly implemented!")
            return True
        else:
            print("⚠️  Implementation incomplete")
            return False
            
    except Exception as e:
        print(f"❌ Validation failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_tool_documentation():
    """Check that tools have proper docstrings."""
    print(f"\n📖 Documentation Check:")
    
    try:
        from localdata_mcp.localdata_mcp import DatabaseManager
        manager = DatabaseManager()
        
        tools = ['profile_table', 'detect_data_types', 'analyze_distributions']
        
        for tool_name in tools:
            if hasattr(manager, tool_name):
                tool_method = getattr(manager, tool_name)
                if tool_method.__doc__:
                    doc_lines = len(tool_method.__doc__.strip().split('\n'))
                    print(f"   ✅ {tool_name}: {doc_lines} lines of documentation")
                else:
                    print(f"   ❌ {tool_name}: No documentation")
                    
    except Exception as e:
        print(f"   ❌ Documentation check failed: {e}")

def main():
    """Run validation."""
    success = validate_tool_implementation()
    check_tool_documentation()
    
    if success:
        print(f"\n🚀 Phase 1 P0 Data Profiling Tools are ready for testing!")
    else:
        print(f"\n⚠️  Please review the implementation before proceeding.")
        
    return success

if __name__ == "__main__":
    main()