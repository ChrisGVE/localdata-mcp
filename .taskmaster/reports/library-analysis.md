# Spreadsheet Library Analysis for LocalData MCP Server Extension

**Analysis Date:** August 29, 2025  
**Python Version Requirement:** 3.10+  
**Focus:** Read operations for spreadsheet formats with pandas compatibility  

## Executive Summary

This analysis recommends specific Python libraries for extending the LocalData MCP Server with comprehensive spreadsheet format support. The selected libraries prioritize pandas compatibility, security, active maintenance, and performance while maintaining the existing architecture patterns.

## Current Implementation Context

The LocalData MCP Server currently supports:
- CSV via `pandas.read_csv()`
- JSON via `pandas.read_json()`
- YAML via `yaml.safe_load()` + `pandas.json_normalize()`
- TOML via `toml.load()` + `pandas.json_normalize()`

All formats are converted to pandas DataFrames and loaded into SQLite engines for SQL querying.

## Recommended Libraries by Format

### 1. Excel Support (.xlsx, .xls)

#### **Primary Recommendation: openpyxl**

**Rationale:**
- Industry standard for modern Excel files (.xlsx, .xlsm)
- Direct pandas integration via `pd.read_excel(engine='openpyxl')`
- Active maintenance and security updates
- Memory-efficient streaming capabilities
- Comprehensive feature support

**Technical Specifications:**
```python
# Dependencies
pip install openpyxl

# Integration
df = pd.read_excel(file_path, engine='openpyxl')
```

**Security Considerations:**
- **CRITICAL:** Install `defusedxml` to prevent XML billion laughs attacks
- No built-in protection against quadratic blowup attacks
- Regular security updates available

**Performance Characteristics:**
- Moderate read performance (slower than xlrd for pure reading)
- Constant memory usage in read-only mode
- Streaming support for large files
- Good for files up to ~100MB

#### **Legacy Support: xlrd**

**Use Case:** Only for legacy .xls files (Excel 97-2003)
```python
pip install xlrd
df = pd.read_excel(file_path, engine='xlrd')  # .xls only
```

**Limitations:**
- No longer supports .xlsx files (as of xlrd 2.0+)
- Password-protected files not supported
- Legacy format only

#### **Integration Complexity:** Low
- Drop-in replacement in existing `_create_engine_from_file()` method
- Pandas handles engine selection automatically
- Minimal code changes required

### 2. Apple Numbers Support (.numbers)

#### **Primary Recommendation: numbers-parser**

**Library Details:**
```python
pip install numbers-parser
```

**Features:**
- Read Apple Numbers files (version 10.3+, tested up to 14.1)
- Python 3.9+ compatibility
- Both reading and limited writing capabilities

**Integration Pattern:**
```python
from numbers_parser import Document

def _load_numbers_file(file_path: str) -> pd.DataFrame:
    doc = Document(file_path)
    # Convert to pandas DataFrame
    # Implementation would extract table data from doc
    return df
```

**Limitations:**
- Password-encrypted files not supported
- UTF-8 image filenames require Python 3.11+
- Windows x86 compatibility issues
- Requires python-snappy binary dependencies

**Security Profile:**
- Active maintenance (version 4.15.1 as of 2024)
- No known critical security vulnerabilities
- Limited attack surface (read-only operations)

**Performance:**
- Good for small to medium files
- Memory usage proportional to file size
- Not optimized for very large datasets

#### **Integration Complexity:** Medium
- Custom conversion logic required
- Additional error handling for format-specific limitations
- Binary dependency management (python-snappy)

### 3. LibreOffice Calc Support (.ods)

#### **Primary Recommendation: odfpy + pandas native support**

**Implementation:**
```python
pip install odfpy
df = pd.read_excel(file_path)  # pandas auto-detects .ods files
```

**Advantages:**
- Native pandas support since version 0.25
- Built-in validation and error checking
- Active maintenance through pandas ecosystem
- Consistent API with other pandas operations

**Alternative: pandas-ods-reader**
```python
pip install pandas-ods-reader
from pandas_ods_reader import read_ods
df = read_ods(file_path)
```

**Comparison:**
- **odfpy:** Better integration, consistent with pandas patterns
- **pandas-ods-reader:** Additional features (auto-cleanup of empty rows/columns)
- **ezodf:** Deprecated, not recommended (last update 2017)

**Security Considerations:**
- Built on XML parsing with standard protections
- Regular updates through pandas releases
- Lower risk profile than direct XML manipulation

**Performance:**
- Similar to Excel reading performance
- Good memory efficiency
- Handles moderate file sizes well

#### **Integration Complexity:** Low
- Seamless pandas integration
- No custom parsing logic required
- Standard pandas error handling applies

### 4. Additional Format Support

#### **TSV (Tab-Separated Values)**
**Implementation:** Native pandas support
```python
df = pd.read_csv(file_path, sep='\t')
```
**Integration Complexity:** Trivial (already supported via CSV engine)

#### **XML Files**
**Implementation:** Native pandas support (since 1.3.0)
```python
df = pd.read_xml(file_path)
```
**Limitations:** Complex nested structures may require custom parsing
**Integration Complexity:** Low

#### **INI Configuration Files**
**Implementation:** Python standard library + pandas
```python
import configparser
config = configparser.ConfigParser()
config.read(file_path)
# Convert to DataFrame structure
```
**Integration Complexity:** Medium (custom conversion logic required)

#### **Parquet Format**
**Primary Recommendation:** PyArrow
```python
pip install pyarrow
df = pd.read_parquet(file_path, engine='pyarrow')
```

**Advantages:**
- Excellent performance and compression
- Native pandas support
- Cross-language compatibility
- Mature ecosystem

#### **Feather/Arrow Format**
```python
# Already included with pyarrow
df = pd.read_feather(file_path)
```

**Performance Profile:** Superior to Parquet for loading speed, excellent compression

## Implementation Roadmap

### Phase 1: Core Spreadsheet Formats
1. **Excel Support (.xlsx/.xls)**
   - Install openpyxl + defusedxml
   - Add engine selection logic for .xls vs .xlsx
   - Update `_get_engine()` method with Excel support

2. **ODS Support**
   - Install odfpy
   - Add .ods detection to file type logic
   - Leverage existing pandas integration

### Phase 2: Extended Support
3. **Numbers Support (.numbers)**
   - Install numbers-parser + python-snappy
   - Implement custom conversion logic
   - Add comprehensive error handling

4. **High-Performance Formats**
   - Install pyarrow
   - Add Parquet/Feather support
   - Optimize for large file handling

### Phase 3: Additional Formats
5. **Structured Text Formats**
   - XML: Leverage pandas native support
   - INI: Custom parser with configparser
   - TSV: Extend CSV engine

## Security Assessment

### High Risk
- **openpyxl**: Requires defusedxml installation
- **XML parsing**: Potential XXE vulnerabilities

### Medium Risk  
- **numbers-parser**: Binary dependencies, less mature ecosystem
- **Custom parsers**: INI file parsing logic

### Low Risk
- **pandas native engines**: Regular security updates
- **pyarrow**: Mature, well-audited codebase

## Performance Characteristics

| Format | Library | Read Speed | Memory Usage | File Size Limit |
|--------|---------|------------|--------------|------------------|
| .xlsx | openpyxl | Medium | Constant* | ~100MB |
| .xls | xlrd | Fast | Full file | ~50MB |
| .numbers | numbers-parser | Medium | Proportional | ~50MB |
| .ods | odfpy | Medium | Medium | ~100MB |
| .parquet | pyarrow | Very Fast | Efficient | >1GB |
| .feather | pyarrow | Fastest | Very Efficient | >1GB |

*With read_only mode and streaming

## Maintenance and Community Support

### Excellent (Active Development)
- **openpyxl**: Regular updates, large community
- **pyarrow**: Apache Foundation backing
- **pandas**: Core ecosystem library

### Good (Stable, Maintained)
- **odfpy**: Stable releases, adequate documentation
- **numbers-parser**: Active maintainer, regular updates

### Fair (Limited Scope)
- **xlrd**: Maintenance mode, legacy support only

## Installation Dependencies Summary

```toml
# Core spreadsheet support
dependencies = [
    "openpyxl>=3.0.0",
    "defusedxml>=0.7.0",  # Security for openpyxl
    "odfpy>=1.4.0",       # ODS support
    "xlrd>=2.0.0",        # Legacy .xls support
]

# Extended format support
optional-dependencies.extended = [
    "numbers-parser>=4.15.0",  # Apple Numbers
    "pyarrow>=10.0.0",         # Parquet/Feather
    "python-snappy>=0.6.0",    # Required by numbers-parser
]
```

## Integration Code Changes

### Minimal Required Changes

1. **Update `_get_engine()` method:**
   - Add format detection for new file types
   - Map formats to appropriate libraries

2. **Extend `_create_engine_from_file()` method:**
   - Add cases for .xlsx, .xls, .numbers, .ods, .parquet, .feather
   - Implement error handling for format-specific limitations

3. **Update `connect_database()` documentation:**
   - Document new supported file types
   - Include security recommendations

### Error Handling Considerations

- **Format detection failures**: Graceful fallback to pandas auto-detection
- **Missing dependencies**: Clear error messages with installation instructions  
- **Corrupted files**: Format-specific error handling
- **Security violations**: Proper exception handling for XML attacks

## Conclusion

The recommended library stack provides comprehensive spreadsheet support while maintaining security, performance, and maintainability standards. The phased implementation approach allows for gradual rollout with immediate value from Excel and ODS support, followed by extended format capabilities.

**Priority Implementation Order:**
1. Excel (.xlsx/.xls) via openpyxl + xlrd
2. LibreOffice Calc (.ods) via odfpy  
3. High-performance formats (Parquet/Feather) via pyarrow
4. Apple Numbers (.numbers) via numbers-parser
5. Additional structured formats (XML, INI, TSV)

This approach balances immediate user value with comprehensive long-term functionality while maintaining the existing architecture's reliability and security posture.