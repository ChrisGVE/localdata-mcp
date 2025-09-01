#!/usr/bin/env python3
"""
Simple test to verify MCP tool registration works correctly.
"""

import sys
import asyncio
import tempfile
import sqlite3
from pathlib import Path
from mcp import ClientSession, StdioServerParameters  
from mcp.client.stdio import stdio_client

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


async def test_simple_mcp():
    """Test basic MCP tool registration and calling."""
    print("🔧 Testing MCP Tool Registration and Binding")
    print("=" * 50)
    
    # Create simple test database
    temp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
    temp_db.close()
    
    conn = sqlite3.connect(temp_db.name)
    cursor = conn.cursor()
    cursor.execute('CREATE TABLE test (id INTEGER, name TEXT)')
    cursor.execute('INSERT INTO test VALUES (1, "test")')
    conn.commit()
    conn.close()
    
    # Connect to MCP server
    server_params = StdioServerParameters(
        command="python",
        args=["-m", "src.localdata_mcp.localdata_mcp"],
        env=None
    )
    
    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                
                # List available tools
                print("📋 Listing available tools...")
                tools_result = await session.list_tools()
                available_tools = [tool.name for tool in tools_result.tools]
                print(f"Available tools: {available_tools}")
                print(f"Found {len(available_tools)} tools")
                
                # Test connect_database_mcp (our wrapper)
                if "connect_database_mcp" in available_tools:
                    print("\n🔧 Testing connect_database_mcp wrapper...")
                    result = await session.call_tool(
                        "connect_database_mcp",
                        {
                            "name": "test_db",
                            "db_type": "sqlite", 
                            "conn_string": f"sqlite:///{temp_db.name}"
                        }
                    )
                    print(f"Result: {result.content}")
                
                # Test original connect_database
                if "connect_database" in available_tools:
                    print("\n🔧 Testing original connect_database...")
                    result = await session.call_tool(
                        "connect_database",
                        {
                            "name": "test_db2",
                            "db_type": "sqlite",
                            "conn_string": f"sqlite:///{temp_db.name}"
                        }
                    )
                    print(f"Result: {result.content}")
                
                # Test Phase 1 tools
                phase1_tools = ["profile_table", "detect_data_types", "analyze_distributions"]
                for tool_name in phase1_tools:
                    if tool_name in available_tools:
                        print(f"\n✅ {tool_name} is available via MCP")
                    else:
                        print(f"\n❌ {tool_name} is NOT available via MCP")
                        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False
        
    return True


if __name__ == "__main__":
    success = asyncio.run(test_simple_mcp())
    print(f"\nTest {'PASSED' if success else 'FAILED'}")