---
name: Security Report
about: Report a security vulnerability (use private email for critical issues)
title: '[SECURITY] '
labels: ['security']
assignees: ''

---

## 🚨 Security Report

⚠️ **IMPORTANT**: For critical security vulnerabilities, please email privately to **christian@berclaz.org** instead of creating a public issue.

## 🔒 Vulnerability Type

Select the type of security issue:

- [ ] Path traversal vulnerability
- [ ] SQL injection vulnerability  
- [ ] Authentication bypass
- [ ] Authorization bypass
- [ ] Information disclosure
- [ ] Remote code execution
- [ ] Denial of service
- [ ] Input validation bypass
- [ ] Connection/resource exhaustion
- [ ] Other: ___________

## 📊 Severity Assessment

**Self-assessed severity level**:

- [ ] **Critical** - Remote code execution, data breach, complete system compromise
- [ ] **High** - Privilege escalation, significant data access, bypass security controls
- [ ] **Medium** - Information disclosure, limited access, moderate impact
- [ ] **Low** - Minor information disclosure, low impact

## 🔍 Vulnerability Details

**Affected Component(s)**:
- [ ] Path security (`_sanitize_path()` method)
- [ ] SQL query execution (`execute_query`, `execute_query_json`)
- [ ] Database connection management
- [ ] File handling (CSV, JSON, YAML, TOML)
- [ ] Query buffering system
- [ ] Connection limiting
- [ ] Input validation
- [ ] Other: ___________

**Detailed Description**:
A clear description of the security vulnerability. Include technical details but avoid full exploitation code if the issue is critical.

## 🎯 Steps to Reproduce

Provide minimal steps to demonstrate the vulnerability:

1. Setup: `connect_database(...)`
2. Action: `execute_query(...)`
3. Observe: [security issue manifests]

**Test Environment**:
- LocalData MCP Version: [e.g., v1.0.0]
- Python Version: [e.g., 3.9.7]
- Operating System: [e.g., macOS, Linux, Windows]
- Database Type: [e.g., PostgreSQL, SQLite]

## 📋 Evidence

If applicable, provide:

- Log outputs (sanitize sensitive information)
- Screenshots demonstrating the issue
- Proof-of-concept code (if low/medium severity)

```
Sanitized evidence here
```

## 💥 Impact Assessment

**What can an attacker achieve?**
- [ ] Read arbitrary files from the filesystem
- [ ] Write files outside allowed directories
- [ ] Execute arbitrary SQL commands
- [ ] Access unauthorized databases
- [ ] Cause denial of service
- [ ] Bypass connection limits
- [ ] Other: ___________

**Prerequisites for exploitation**:
- [ ] No special access required
- [ ] Local file system access required
- [ ] Database connection required
- [ ] Specific database configuration required
- [ ] Other: ___________

## 🛡️ Affected Versions

Which versions are affected?
- [ ] Current main branch
- [ ] v1.0.0
- [ ] All versions
- [ ] Specific version: ___________

## 🔧 Potential Mitigations

**Immediate Workarounds** (if known):
- Avoid using specific features
- Additional input validation
- Configuration changes
- Other: ___________

**Suggested Fix Approach**:
Brief, high-level suggestions for addressing the vulnerability.

## 📚 References

Any related security research, CVE numbers, or documentation:

- Similar vulnerabilities: 
- Security best practices:
- Related issues: #___

## 🤝 Responsible Disclosure

- [ ] I understand this may be a security issue
- [ ] I have not publicly disclosed details elsewhere
- [ ] I am willing to work with maintainers on responsible disclosure
- [ ] I would like to be credited in security advisories (optional)

## 📞 Contact Information

**Preferred contact method** (for follow-up):
- [ ] GitHub username: @___________
- [ ] Email: ___________@___________
- [ ] I prefer to remain anonymous

---

**Thank you for helping keep LocalData MCP Server secure!**

*We take security seriously and will respond promptly to verified reports.*