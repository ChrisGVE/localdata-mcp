# GitHub Repository Setup Guide

This guide will help you complete the setup for automated PyPI publishing and repository protection for localdata-mcp.

## ✅ **Already Completed**

- ✅ GitHub workflows created and pushed
- ✅ CI/CD pipeline with multi-Python testing
- ✅ Security scanning workflows
- ✅ Issue and PR templates
- ✅ Dependabot configuration

## 🚀 **Required Setup Steps**

### 1. Enable PyPI Trusted Publishing

**Why:** Allows automatic PyPI publishing without storing API tokens

**Steps:**
1. Go to https://pypi.org/manage/account/publishing/
2. Click "Add a new pending publisher"
3. Fill in:
   - **Owner:** `ChristianBerclaz` (or your GitHub username)
   - **Repository name:** `localdata-mcp`
   - **Workflow filename:** `release.yml`
   - **Environment name:** `release`
4. Click "Add"

### 2. Create GitHub Release Environment

**Why:** Required for secure PyPI publishing workflow

**Steps:**
1. Go to your repository: https://github.com/ChristianBerclaz/localdata-mcp
2. Go to Settings → Environments
3. Click "New environment"
4. Name: `release`
5. Click "Configure environment"
6. **Optional:** Add protection rules like requiring reviewers

### 3. Configure Branch Protection (Recommended)

**Why:** Ensures code quality before merging to main

**Steps:**
1. Go to Settings → Branches
2. Click "Add rule" for `main` branch
3. Configure:
   ```
   ✅ Require a pull request before merging
   ✅ Require status checks to pass before merging
   ✅ Require branches to be up to date before merging
   
   Required status checks:
   ✅ test (3.10)
   ✅ test (3.11)
   ✅ test (3.12)
   ✅ build
   ```
4. Save changes

### 4. Verify Workflows Are Running

**Check:**
1. Go to Actions tab in your repository
2. You should see workflows running after the recent push
3. All workflows should complete successfully

## 🎯 **Test the Complete Pipeline**

### Option A: Patch Release (Recommended for testing)
```bash
# Make a small change (like updating README)
git add README.md
git commit -m "docs: Update documentation"

# Create a patch release
git tag v1.0.1
git push origin v1.0.1

# Go to GitHub → Releases → Create new release
# Select tag v1.0.1, generate release notes, publish
```

### Option B: Minor Release
```bash
# For more significant changes
git tag v1.1.0
git push origin v1.1.0
# Then create GitHub release
```

## 📋 **Verification Checklist**

After completing setup and testing a release:

- [ ] PyPI trusted publisher configured
- [ ] GitHub `release` environment created
- [ ] Branch protection rules applied
- [ ] All CI workflows passing
- [ ] Test release published successfully
- [ ] Package available on PyPI: https://pypi.org/project/localdata-mcp/
- [ ] Package installable via `pip install localdata-mcp`

## 🔧 **Workflow Features**

### Automated CI/CD
- **Multi-Python testing** (3.10-3.12)
- **Code quality** (black, isort, flake8, mypy)
- **Security scanning** (safety, bandit, pip-audit, CodeQL)
- **Coverage reporting**

### Automated Publishing
- **Triggers:** GitHub release creation
- **Process:** Build → Test → Publish to PyPI
- **Security:** Uses trusted publishing (no tokens)

### Dependency Management
- **Dependabot:** Weekly dependency updates
- **Security:** Automated vulnerability alerts

## 🚨 **Troubleshooting**

### PyPI Publishing Fails
1. **Check trusted publisher:** Verify configuration at PyPI
2. **Check environment:** Ensure `release` environment exists
3. **Check tag format:** Must be `vX.Y.Z` (e.g., `v1.0.1`)
4. **Check logs:** View workflow run details in Actions tab

### CI Tests Fail
1. **Code style:** Run `black src/ tests/` and `isort src/ tests/`
2. **Linting:** Run `flake8 src/`
3. **Type checking:** Run `mypy src/`
4. **Tests:** Run `pytest tests/`

### Security Alerts
1. **Dependencies:** Update via Dependabot PRs
2. **Code issues:** Check Security tab for CodeQL results
3. **Vulnerabilities:** Review and update affected packages

## 📞 **Support**

- **GitHub Issues:** Use structured templates in your repository
- **Actions Documentation:** https://docs.github.com/en/actions
- **PyPI Trusted Publishing:** https://docs.pypi.org/trusted-publishers/

---

**Next Step:** Complete the PyPI trusted publisher setup, then test with a release!