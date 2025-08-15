# Contributing to LocalData MCP Server

Thank you for your interest in contributing to LocalData MCP Server! This guide will help you get started with contributing to our project.

## 🌟 Ways to Contribute

- **🐛 Bug Reports**: Help us identify and fix issues
- **✨ Feature Requests**: Suggest new features and improvements
- **🔧 Code Contributions**: Submit bug fixes and new features
- **📚 Documentation**: Improve documentation and examples
- **🧪 Testing**: Help expand test coverage and find edge cases
- **🔒 Security**: Report security vulnerabilities responsibly

## 📋 Before You Start

### Prerequisites

- **Python 3.8+** installed on your system
- **Git** for version control
- **Basic understanding** of MCP (Model Context Protocol)
- **Familiarity** with database concepts (SQL, NoSQL)

### Understanding the Codebase

LocalData MCP Server is structured as follows:

```
localdata-mcp/
├── src/localdata_mcp/           # Main package
│   ├── localdata_mcp.py         # Core MCP server implementation
│   └── __init__.py              # Package initialization
├── tests/                       # Test suite
│   └── test_basic.py           # Basic functionality tests  
├── .github/                     # GitHub templates and workflows
│   └── ISSUE_TEMPLATE/         # Issue templates
├── README.md                   # Project documentation
├── CONTRIBUTING.md             # This file
├── pyproject.toml              # Project configuration
└── LICENSE                     # MIT License
```

## 🚀 Getting Started

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/localdata-mcp.git
cd localdata-mcp

# Add the original repository as upstream
git remote add upstream https://github.com/ChrisGVE/localdata-mcp.git
```

### 2. Set Up Development Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev]"
```

### 3. Verify Setup

```bash
# Run basic tests
python -m pytest tests/

# Run the MCP server locally
python -m localdata_mcp.localdata_mcp
```

## 🔧 Development Workflow

### 1. Create a Feature Branch

```bash
# Sync with upstream
git fetch upstream
git checkout main
git merge upstream/main

# Create feature branch
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

### 2. Make Your Changes

- **Write clear, readable code** following existing patterns
- **Add comprehensive tests** for new functionality
- **Update documentation** as needed
- **Follow security best practices**

### 3. Testing Your Changes

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_basic.py -v

# Test with different Python versions (if available)
python3.8 -m pytest tests/
python3.9 -m pytest tests/
python3.12 -m pytest tests/
```

### 4. Security Testing

For security-related changes:

```bash
# Run comprehensive security tests
python test_security_validation.py

# Test with malicious inputs
python test_comprehensive.py
```

## 📝 Code Standards

### Code Style

- **Follow PEP 8** Python style guidelines
- **Use type hints** where possible
- **Write descriptive variable and function names**
- **Keep functions focused and modular**
- **Add docstrings** to all public functions

### Example Code Style

```python
def connect_database(name: str, db_type: str, conn_string: str) -> Dict[str, Any]:
    """
    Connect to a database or structured file.
    
    Args:
        name: Unique identifier for the connection
        db_type: Database type (sqlite, postgresql, mysql, csv, json, yaml, toml)  
        conn_string: Connection string or file path
        
    Returns:
        Dict containing connection status and metadata
        
    Raises:
        ValueError: If parameters are invalid
        ConnectionError: If connection fails
        SecurityError: If path is not allowed
    """
    # Implementation here
    pass
```

### Security Guidelines

- **Validate all inputs** thoroughly
- **Use parameterized queries** to prevent SQL injection
- **Restrict file access** to current working directory and subdirectories
- **Limit resource usage** (connections, memory, file sizes)
- **Handle errors gracefully** without exposing sensitive information

### Testing Standards

- **Write tests for all new functionality**
- **Include edge cases and error conditions**
- **Test security boundaries**
- **Maintain high test coverage**

```python
def test_connect_database_security():
    """Test that path traversal attempts are blocked."""
    with pytest.raises(SecurityError):
        connect_database("test", "sqlite", "../../../etc/passwd")
```

## 🧪 Testing Guidelines

### Test Categories

1. **Unit Tests**: Test individual functions and methods
2. **Integration Tests**: Test database connections and operations  
3. **Security Tests**: Test security boundaries and validation
4. **Performance Tests**: Test large dataset handling and buffering

### Running Tests

```bash
# Run all tests with coverage
python -m pytest tests/ --cov=localdata_mcp --cov-report=html

# Run specific test categories
python -m pytest tests/ -k "security"
python -m pytest tests/ -k "integration"

# Run tests for specific database types
python -m pytest tests/ -k "sqlite"
python -m pytest tests/ -k "postgresql"
```

### Test Database Setup

For database integration tests:

```bash
# SQLite (no setup required)

# PostgreSQL (requires local installation)
createdb test_localdata_mcp

# MySQL (requires local installation)  
mysql -u root -e "CREATE DATABASE test_localdata_mcp;"
```

## 📋 Pull Request Process

### 1. Prepare Your PR

- **Ensure all tests pass**
- **Update documentation** if needed
- **Write clear commit messages**
- **Rebase on latest main** if needed

```bash
# Rebase on latest main
git fetch upstream
git rebase upstream/main

# Run final tests
python -m pytest tests/
```

### 2. Submit Pull Request

- **Use descriptive PR title** following conventional commits
- **Fill out the PR template** completely  
- **Reference related issues** using `#issue_number`
- **Request review** from maintainers

### PR Title Examples

```
feat: Add support for TOML file connections
fix: Resolve SQL injection vulnerability in table queries
docs: Update README with new security features  
test: Add comprehensive CSV file handling tests
chore: Update dependencies to latest versions
```

### 3. PR Review Process

- **Maintainers will review** your code and provide feedback
- **Address requested changes** promptly
- **Keep discussions respectful** and constructive
- **Be patient** - reviews take time to ensure quality

## 🔒 Security Contributions

### Reporting Vulnerabilities

- **Critical vulnerabilities**: Email privately to `christian@berclaz.org`
- **Non-critical security issues**: Use the Security Report issue template
- **Include detailed reproduction steps** and impact assessment
- **We follow responsible disclosure** practices

### Security Development

- **Understand the threat model**: LocalData MCP is designed for trusted local environments
- **Test security boundaries**: Path traversal, SQL injection, resource exhaustion  
- **Document security implications** of new features
- **Follow principle of least privilege**

## 📚 Documentation

### What to Document

- **New features and tools** with usage examples
- **Configuration changes** and their implications
- **Security considerations** for new functionality
- **Breaking changes** and migration guides
- **Performance characteristics** of new features

### Documentation Style

- **Use clear, concise language**
- **Provide practical examples**
- **Include security warnings** where appropriate
- **Test all code examples** to ensure they work

## 🤝 Community Guidelines

### Code of Conduct

- **Be respectful** and inclusive in all interactions
- **Provide constructive feedback** and accept criticism gracefully
- **Help newcomers** learn and contribute
- **Focus on technical merit** in discussions
- **Report inappropriate behavior** to maintainers

### Communication

- **Use GitHub issues** for bug reports and feature requests
- **Use GitHub discussions** for questions and ideas
- **Be patient** and understanding with response times
- **Search existing issues** before creating new ones

## 📊 Release Process

### Versioning

We follow **Semantic Versioning** (semver):

- **Major (X.0.0)**: Breaking changes to API
- **Minor (0.X.0)**: New features, backward compatible  
- **Patch (0.0.X)**: Bug fixes, backward compatible

### Release Criteria

- **All tests passing** on supported Python versions
- **Security review** completed for security-related changes
- **Documentation updated** for user-facing changes
- **Changelog updated** with release notes

## 🏆 Recognition

### Contributors

All contributors are recognized in:

- **GitHub Contributors** section
- **Release notes** for significant contributions
- **Documentation credits** for documentation improvements
- **Security advisories** for security researchers (with permission)

### Types of Recognition

- **Code contributors**: Features, bug fixes, improvements
- **Documentation contributors**: README, examples, guides
- **Security researchers**: Responsible vulnerability disclosure
- **Community contributors**: Issue triage, user support

## ❓ Getting Help

### Where to Get Help

1. **📚 Read the README**: Comprehensive documentation and examples
2. **🔍 Search existing issues**: Your question might already be answered
3. **💬 GitHub Discussions**: Ask questions and discuss ideas
4. **📧 Email maintainers**: For private matters or security issues

### What Information to Include

When asking for help:

- **LocalData MCP version** you're using
- **Python version** and operating system
- **Database type** and configuration (sanitized)
- **Exact error messages** or unexpected behavior
- **Minimal reproduction steps**
- **What you've already tried**

## 📜 License

By contributing to LocalData MCP Server, you agree that your contributions will be licensed under the [MIT License](LICENSE).

---

**Thank you for contributing to LocalData MCP Server!** 

Your contributions help make secure database access available to the entire MCP community. Whether you're fixing bugs, adding features, improving documentation, or reporting issues, every contribution matters.

**Questions?** Don't hesitate to ask - we're here to help you succeed! 🚀