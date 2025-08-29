# GitHub Workflows Documentation

This document describes the automated workflows set up for the localdata-mcp project.

## Overview

The repository includes several automated workflows to ensure code quality, security, and streamlined releases:

1. **Continuous Integration (CI)** - Testing and code quality checks
2. **Release and PyPI Publishing** - Automated package publishing
3. **Security Scanning** - Vulnerability detection
4. **CodeQL Analysis** - Security code analysis
5. **Dependency Management** - Automated dependency updates

## Workflows

### 1. CI Workflow (`.github/workflows/ci.yml`)

**Triggers:** Push to `main`/`develop`, Pull requests to `main`

**Jobs:**
- **Test Matrix**: Tests against Python 3.8-3.12
- **Linting**: flake8, black, isort, mypy
- **Testing**: pytest with coverage reporting
- **Build Validation**: Package building and validation

**Required Status Checks:**
- All tests pass across Python versions
- Code formatting checks pass
- Type checking passes
- Package builds successfully

### 2. Release Workflow (`.github/workflows/release.yml`)

**Triggers:** GitHub release published

**Features:**
- Automatic version extraction from git tags (format: `v1.2.3`)
- Updates `pyproject.toml` with release version
- Builds and validates package
- Publishes to PyPI using trusted publishing
- Attaches built distributions to GitHub release

**Setup Required:**
1. Enable trusted publishing in PyPI project settings
2. Add GitHub Actions as trusted publisher
3. Create GitHub environment named `release`

### 3. Security Workflow (`.github/workflows/security.yml`)

**Triggers:** 
- Push to `main`
- Pull requests to `main`
- Weekly schedule (Mondays 8:00 AM UTC)

**Security Tools:**
- **safety**: Known vulnerability scanning
- **bandit**: Security linting for Python code
- **pip-audit**: Dependency vulnerability audit

### 4. CodeQL Workflow (`.github/workflows/codeql.yml`)

**Triggers:**
- Push to `main`
- Pull requests to `main`
- Weekly schedule (Tuesdays 2:43 PM UTC)

**Features:**
- GitHub's semantic code analysis
- Automated security vulnerability detection
- Results visible in Security tab

### 5. Dependabot (`.github/dependabot.yml`)

**Features:**
- Weekly dependency updates (Mondays 9:00 AM)
- Separate PRs for Python dependencies and GitHub Actions
- Auto-assignment to maintainer
- Proper labeling and commit message formatting

## Issue Templates

### Bug Report (`.github/ISSUE_TEMPLATE/bug_report.yml`)
Structured template for bug reports including:
- Problem description
- Reproduction steps
- Environment details (Python version, OS, database type)
- Error logs

### Feature Request (`.github/ISSUE_TEMPLATE/feature_request.yml`)
Template for feature requests including:
- Problem statement
- Proposed solution
- Use case description
- Priority level

## Pull Request Template

Comprehensive PR template (`.github/pull_request_template.md`) covering:
- Change description and type
- Testing requirements
- Database compatibility
- Documentation updates
- Code quality checklist

## Branch Protection Setup

Recommended branch protection rules for `main`:

```yaml
# GitHub repository settings
branches:
  main:
    protection:
      required_status_checks:
        strict: true
        contexts:
          - "test (3.8)"
          - "test (3.9)"
          - "test (3.10)"
          - "test (3.11)"
          - "test (3.12)"
          - "build"
      enforce_admins: true
      required_pull_request_reviews:
        required_approving_review_count: 1
        dismiss_stale_reviews: true
        require_code_owner_reviews: true
      restrictions: null
```

## PyPI Trusted Publishing Setup

To enable automated PyPI publishing:

1. Go to https://pypi.org/manage/account/publishing/
2. Add trusted publisher with these details:
   - Owner: `ChristianBerclaz`
   - Repository: `localdata-mcp`
   - Workflow: `release.yml`
   - Environment: `release`

3. Create GitHub environment:
   - Go to repository Settings > Environments
   - Create environment named `release`
   - Optionally add protection rules

## Release Process

1. **Prepare Release:**
   ```bash
   # Update version in pyproject.toml if needed
   # Commit all changes
   git tag v1.0.1
   git push origin v1.0.1
   ```

2. **Create GitHub Release:**
   - Go to repository > Releases > Create new release
   - Choose the tag (v1.0.1)
   - Generate release notes
   - Publish release

3. **Automatic Process:**
   - Release workflow triggers
   - Package builds and publishes to PyPI
   - Release assets attached to GitHub release

## Monitoring

**GitHub Actions:** Monitor workflow runs in the Actions tab
**Security:** Check Security tab for CodeQL and dependency alerts
**PyPI:** Verify package publication at https://pypi.org/project/localdata-mcp/

## Troubleshooting

### Failed PyPI Publication
1. Check PyPI trusted publisher configuration
2. Verify GitHub environment `release` exists
3. Ensure tag format is `vX.Y.Z`

### Failed Tests
1. Check Python version compatibility
2. Review linting errors (formatting, imports, types)
3. Verify all dependencies are properly declared

### Security Alerts
1. Review Security tab for detailed vulnerability reports
2. Update dependencies via Dependabot PRs
3. Address any CodeQL findings

## Maintenance

**Weekly Tasks:**
- Review and merge Dependabot PRs
- Check security scan results
- Monitor test success rates

**Monthly Tasks:**
- Review workflow efficiency
- Update Python version matrix as new versions release
- Audit security tool configurations