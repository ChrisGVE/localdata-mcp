# PyPI Automation Setup Guide

## Overview
Automated PyPI publishing is now configured! The workflow will trigger whenever you push a git tag starting with `v` (like `v1.0.1`).

## Required Setup Steps

### 1. Create PyPI API Token
1. Go to https://pypi.org/account/login/ (create account if needed)
2. Go to https://pypi.org/manage/account/token/
3. Click "Add API token"
4. Name: `localdata-mcp-github-actions`
5. Scope: Select "Project: localdata-mcp" (if available) or "Entire account"
6. Click "Add token"
7. **IMPORTANT**: Copy the token immediately (it starts with `pypi-`)

### 2. Add Token to GitHub Secrets
1. Go to your GitHub repository: https://github.com/ChrisGVE/localdata-mcp
2. Click "Settings" tab
3. Click "Secrets and variables" → "Actions"
4. Click "New repository secret"
5. Name: `PYPI_API_TOKEN`
6. Value: Paste your PyPI token (including the `pypi-` prefix)
7. Click "Add secret"

## How It Works

### Automatic Triggers
The workflow automatically runs when you:
- Push a git tag starting with `v` (e.g., `v1.0.1`, `v2.0.0`)
- Create a GitHub release

### What It Does
1. Extracts version from git tag (removes `v` prefix)
2. Updates `pyproject.toml` with the correct version
3. Builds the package (wheel + source distribution)
4. Validates the package
5. Publishes to PyPI
6. Verifies publication

### Current Status
- ✅ GitHub Actions workflow created
- ✅ Version updated to 1.0.1 in pyproject.toml
- ✅ Test workflow added for CI/CD
- ⏳ Requires PyPI API token setup
- ⏳ Ready to test with v1.0.1 release

## Testing the Automation

Once you've set up the PyPI API token:

### Option 1: Push the existing v1.0.1 tag
```bash
git add .
git commit -m "feat: Add automated PyPI publishing workflow"
git push origin main

# The v1.0.1 tag already exists, so just push it
git push origin v1.0.1
```

### Option 2: Create a new tag for testing
```bash
git add .
git commit -m "feat: Add automated PyPI publishing workflow"
git push origin main

# Delete and recreate the tag to trigger the workflow
git tag -d v1.0.1
git tag v1.0.1
git push origin --force --tags
```

### Option 3: Create a GitHub Release
1. Go to https://github.com/ChrisGVE/localdata-mcp/releases
2. Click "Create a new release"
3. Tag: `v1.0.1`
4. Title: `Release v1.0.1`
5. Description: Add release notes
6. Click "Publish release"

## Monitoring
- Watch the Actions tab: https://github.com/ChrisGVE/localdata-mcp/actions
- Check PyPI after successful run: https://pypi.org/project/localdata-mcp/

## Future Releases
For future versions, the process is now automated:

1. Update your code
2. Commit changes
3. Create and push a new tag:
   ```bash
   git tag v1.0.2
   git push origin v1.0.2
   ```
4. The package will automatically be built and published to PyPI!

## Troubleshooting
- If the workflow fails, check the Actions log
- Verify the PyPI API token is correctly set in GitHub Secrets
- Ensure the token has the right permissions for your project