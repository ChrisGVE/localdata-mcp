"""
Syntax validation for Phase 1 migration.

This script validates that the migration code is syntactically correct and
can be imported without runtime dependencies.
"""

import ast
import os


def validate_python_syntax(file_path):
    """Validate Python syntax of a file."""
    try:
        with open(file_path, 'r') as f:
            source = f.read()
        
        # Parse the AST to check syntax
        ast.parse(source, filename=file_path)
        return True, None
    except SyntaxError as e:
        return False, f"Syntax error: {e}"
    except Exception as e:
        return False, f"Error: {e}"


def validate_transformer_structure():
    """Validate transformer class structure."""
    transformer_file = "src/localdata_mcp/pipeline/phase1_transformers.py"
    
    if not os.path.exists(transformer_file):
        return False, "Transformer file not found"
    
    # Check syntax
    valid, error = validate_python_syntax(transformer_file)
    if not valid:
        return False, f"Syntax validation failed: {error}"
    
    # Parse and analyze structure
    try:
        with open(transformer_file, 'r') as f:
            source = f.read()
        
        tree = ast.parse(source)
        
        # Find classes
        classes = [node for node in tree.body if isinstance(node, ast.ClassDef)]
        class_names = [cls.name for cls in classes]
        
        expected_classes = [
            'ProfileTableTransformer',
            'DataTypeDetectorTransformer', 
            'DistributionAnalyzerTransformer'
        ]
        
        for expected_class in expected_classes:
            if expected_class not in class_names:
                return False, f"Missing class: {expected_class}"
        
        # Check that classes have required methods
        for cls in classes:
            if cls.name in expected_classes:
                methods = [node.name for node in cls.body if isinstance(node, ast.FunctionDef)]
                
                required_methods = ['fit', 'transform', 'get_feature_names_out', 'get_composition_metadata']
                for method in required_methods:
                    if method not in methods:
                        return False, f"Class {cls.name} missing method: {method}"
        
        return True, "All transformer classes and methods found"
        
    except Exception as e:
        return False, f"Structure analysis failed: {e}"


def validate_main_file_changes():
    """Validate changes to main localdata_mcp.py file."""
    main_file = "src/localdata_mcp/localdata_mcp.py"
    
    if not os.path.exists(main_file):
        return False, "Main file not found"
    
    # Check syntax
    valid, error = validate_python_syntax(main_file)
    if not valid:
        return False, f"Syntax validation failed: {error}"
    
    # Check for transformer imports
    try:
        with open(main_file, 'r') as f:
            content = f.read()
        
        # Check for transformer imports in the methods
        required_imports = [
            'from .pipeline.phase1_transformers import ProfileTableTransformer',
            'from .pipeline.phase1_transformers import DataTypeDetectorTransformer',
            'from .pipeline.phase1_transformers import DistributionAnalyzerTransformer'
        ]
        
        import_found = False
        for import_stmt in required_imports:
            if import_stmt in content:
                import_found = True
                break
        
        if not import_found:
            return False, "No transformer imports found in main file"
        
        return True, "Main file changes validated"
        
    except Exception as e:
        return False, f"Main file validation failed: {e}"


def run_syntax_validation():
    """Run all syntax validation tests."""
    print("🔍 Phase 1 Migration Syntax Validation")
    print("=" * 50)
    
    tests = [
        ("Transformer Structure", validate_transformer_structure),
        ("Main File Changes", validate_main_file_changes),
    ]
    
    all_passed = True
    
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}...")
        try:
            passed, message = test_func()
            if passed:
                print(f"  ✅ PASSED: {message}")
            else:
                print(f"  ❌ FAILED: {message}")
                all_passed = False
        except Exception as e:
            print(f"  ❌ ERROR: {e}")
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 ALL SYNTAX VALIDATION TESTS PASSED!")
        print("✅ Migration code is syntactically correct")
        print("✅ Required classes and methods are present")
        print("✅ Main file integration appears correct")
    else:
        print("❌ Some validation tests failed")
    
    return all_passed


if __name__ == '__main__':
    import sys
    success = run_syntax_validation()
    sys.exit(0 if success else 1)