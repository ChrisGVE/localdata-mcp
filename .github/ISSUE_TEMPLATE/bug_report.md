---
name: Bug Report
about: Report a bug or issue with LocalData MCP Server
title: '[BUG] '
labels: ['bug']
assignees: ''

---

## 🐛 Bug Description

A clear and concise description of what the bug is.

## 🔍 Steps to Reproduce

Steps to reproduce the behavior:

1. Connect to database: `connect_database(...)`
2. Execute query: `execute_query(...)`
3. See error

## 🎯 Expected Behavior

A clear and concise description of what you expected to happen.

## 📊 Actual Behavior

A clear and concise description of what actually happened.

## 🖼️ Screenshots/Output

If applicable, add screenshots or error output to help explain your problem.

```
Paste error messages or output here
```

## 🌍 Environment

**LocalData MCP Version**: [e.g., v1.0.0]  
**Python Version**: [e.g., 3.9.7]  
**Operating System**: [e.g., macOS 12.0, Ubuntu 20.04, Windows 11]  
**MCP Client**: [e.g., Claude Code, Custom Client]  
**Database Type**: [e.g., PostgreSQL, SQLite, CSV]  

## 📝 Database Configuration

**Database Type**: [e.g., postgresql, sqlite, csv]  
**Connection String**: [redact sensitive information]  
**File Size** (if applicable): [e.g., 50MB CSV file]  

## 🔒 Security Context

- [ ] Using local files only (recommended)
- [ ] Using remote database connections
- [ ] Files are in current working directory
- [ ] Files are in subdirectories of working directory

## 📋 Additional Context

Add any other context about the problem here.

- Is this reproducible consistently?
- Does it happen with specific data types?
- Any recent changes to your setup?

## 🏥 Diagnostic Information

If comfortable sharing, please provide:

```bash
# Version info
python --version
pip show localdata-mcp

# Connection test (remove sensitive data)
connect_database("test", "your_db_type", "your_connection_string")
list_databases()
```

## 🚨 Workarounds

If you've found any workarounds, please describe them here.

---

**Thank you for helping improve LocalData MCP Server!**