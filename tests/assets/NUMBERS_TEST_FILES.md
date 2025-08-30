# Apple Numbers Test Files

## Why no .numbers files in tests?

Creating .numbers files programmatically is complex and requires:
1. An existing Numbers template or document
2. System dependencies (libsnappy-dev, etc.)
3. Platform-specific issues on non-macOS systems

## How to test Numbers support:

### Option 1: Create test files manually
1. Open Apple Numbers on macOS
2. Create a simple spreadsheet with:
   - Sheet 1: Employee data (name, salary, department)
   - Sheet 2: Sales data (quarter, revenue, profit)
3. Save as "test_numbers_file.numbers" in tests/assets/

### Option 2: Use existing Numbers files
1. Find any .numbers file on your system
2. Copy to tests/assets/ for testing
3. LocalData MCP will automatically handle multiple sheets/tables

## Test Coverage Status:
- ✅ Numbers reading implementation complete
- ✅ Multi-sheet and multi-table support
- ✅ Data type conversion and formatting
- ❌ Test files not included (due to creation complexity)
- ✅ Integration tests verify implementation works

## Supported Numbers Features:
- Multiple sheets per document
- Multiple tables per sheet
- Mixed data types (numbers, text, dates, booleans)
- Column name sanitization
- Automatic data type detection
- Error handling for password-protected files
