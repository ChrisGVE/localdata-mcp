#!/usr/bin/env python3
"""
Verification script for PyPI publication of localdata-mcp
Run this after successful PyPI upload to verify everything works correctly.
"""

import subprocess
import sys
import tempfile
import os
import time

def run_command(cmd, description):
    """Run a command and return success status"""
    print(f"\n🔍 {description}")
    print(f"Command: {cmd}")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            print(f"✅ SUCCESS: {description}")
            if result.stdout:
                print(f"Output: {result.stdout.strip()}")
            return True
        else:
            print(f"❌ FAILED: {description}")
            print(f"Error: {result.stderr.strip()}")
            return False
    except subprocess.TimeoutExpired:
        print(f"❌ TIMEOUT: {description}")
        return False
    except Exception as e:
        print(f"❌ ERROR: {description} - {e}")
        return False

def main():
    """Main verification function"""
    print("🚀 LocalData-MCP PyPI Publication Verification")
    print("=" * 50)
    
    # Create temporary directory for clean testing
    with tempfile.TemporaryDirectory() as temp_dir:
        os.chdir(temp_dir)
        print(f"Working in temporary directory: {temp_dir}")
        
        results = []
        
        # 1. Test package availability on PyPI
        results.append(run_command(
            "pip index versions localdata-mcp",
            "Check package availability on PyPI"
        ))
        
        # 2. Create virtual environment
        results.append(run_command(
            "python3 -m venv test_env",
            "Create test virtual environment"
        ))
        
        # 3. Install from PyPI
        results.append(run_command(
            "test_env/bin/pip install localdata-mcp",
            "Install localdata-mcp from PyPI"
        ))
        
        # 4. Test import
        results.append(run_command(
            "test_env/bin/python -c 'import localdata_mcp; print(f\"Version: {localdata_mcp.__version__}\")'",
            "Test Python import and version"
        ))
        
        # 5. Test command-line tool
        results.append(run_command(
            "test_env/bin/localdata-mcp --help | head -10",
            "Test command-line tool"
        ))
        
        # 6. Test MCP functionality
        results.append(run_command(
            "test_env/bin/python -c 'from localdata_mcp import DatabaseManager; dm = DatabaseManager(); print(\"DatabaseManager created successfully\")'",
            "Test DatabaseManager instantiation"
        ))
        
        # 7. Check all dependencies installed correctly
        results.append(run_command(
            "test_env/bin/pip check",
            "Check dependency conflicts"
        ))
        
        # Summary
        print("\n" + "=" * 50)
        print("📊 VERIFICATION SUMMARY")
        print("=" * 50)
        
        passed = sum(results)
        total = len(results)
        
        print(f"Tests passed: {passed}/{total}")
        
        if passed == total:
            print("🎉 ALL TESTS PASSED - PyPI publication successful!")
            return 0
        else:
            print("❌ Some tests failed - please investigate")
            return 1

if __name__ == "__main__":
    sys.exit(main())