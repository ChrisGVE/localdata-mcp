# Contributing to LocalData MCP

Thank you for your interest in contributing. This guide covers how to set up a development environment, run tests, and submit changes.

## Ways to contribute

- **Bug reports**: Open an issue with reproduction steps and error output
- **Feature requests**: Describe the use case and expected behavior
- **Code**: Bug fixes, new features, performance improvements
- **Documentation**: Fix typos, improve examples, expand guides
- **Testing**: Expand test coverage, report edge cases
- **Security**: Report vulnerabilities responsibly (see below)

## Prerequisites

- **Python 3.10+**
- **uv** (recommended) or pip
- **Git**
- Basic familiarity with MCP (Model Context Protocol)

## Getting started

### Fork and clone

```bash
git clone https://github.com/YOUR_USERNAME/localdata-mcp.git
cd localdata-mcp
git remote add upstream https://github.com/ChrisGVE/localdata-mcp.git
```

### Set up the development environment

```bash
# Using uv (recommended)
uv sync --dev

# Or using pip
python -m venv venv
source venv/bin/activate   # macOS/Linux
pip install -e ".[dev]"
```

### Verify the setup

```bash
# Run unit tests
pytest tests/ -v --ignore=tests/integration

# Check that the server starts
localdata-mcp --version
```

## Project structure

```
localdata-mcp/
├── src/localdata_mcp/            # Main package
│   ├── localdata_mcp.py          # Entry point and MCP tool registration
│   ├── server/                   # CLI, database manager, query execution
│   ├── config_manager/           # Configuration loading and validation
│   ├── connection_manager/       # Database connections and pooling
│   ├── streaming/                # Memory-bounded query streaming
│   ├── security/                 # Path restrictions, SQL validation
│   ├── error_handler/            # Circuit breaker, retry, recovery
│   ├── file_processor/           # Tabular, Excel, JSON file processors
│   ├── tree_storage/             # JSON/YAML/TOML tree operations
│   ├── domains/                  # 8 data science domain modules
│   ├── pipeline/                 # Data preprocessing and transforms
│   └── ...                       # Additional modules
├── tests/                        # Test suite
│   ├── domains/                  # Domain-specific tests
│   ├── integration/              # Integration tests (databases, formats)
│   ├── pipeline/                 # Pipeline tests
│   └── assets/                   # Test data files
├── docs/                         # Sphinx documentation (Markdown)
├── .github/                      # CI workflows and issue templates
├── pyproject.toml                # Project metadata and dependencies
├── Dockerfile                    # Container build
├── docker-compose.yml            # Dev stack with databases
└── LICENSE                       # MIT License
```

## Development workflow

### Create a branch

```bash
git fetch upstream
git checkout main
git merge upstream/main
git checkout -b feature/your-feature-name
```

### Make changes

- Follow existing code patterns and module structure
- Add tests for new functionality
- Keep functions and files within the project's size limits (see `CLAUDE.md`)
- Update documentation when adding user-facing features

### Run tests

```bash
# Unit tests only (fast)
pytest tests/ -v --ignore=tests/integration

# Include integration tests (requires database services)
pytest tests/ -v

# Run a specific test file
pytest tests/test_config_manager.py -v

# Run tests matching a keyword
pytest tests/ -v -k "security"

# With coverage
pytest tests/ --cov=localdata_mcp --cov-report=html --ignore=tests/integration
```

### Integration test setup

Integration tests require running database services. The simplest approach is Docker Compose:

```bash
# Start database services
docker-compose up -d postgres mysql mongodb redis elasticsearch

# Run integration tests
pytest tests/integration/ -v

# Stop services when done
docker-compose down
```

## Code standards

### Style

- Follow PEP 8
- Use type hints on all public function signatures
- Write clear docstrings for public APIs
- Keep functions focused and within size limits

### Security

- Validate all inputs at system boundaries
- Use parameterized queries for SQL
- Restrict file access to allowed directories
- Handle errors without exposing sensitive information

### Testing

- Every new function or method needs at least one test
- Cover edge cases and error conditions
- Use mocks for filesystem scenarios (permissions, missing files)
- Test security boundaries (path traversal, injection)

## Pull request process

### Before submitting

- Ensure all tests pass
- Update documentation for user-facing changes
- Rebase on the latest `main` if needed:
  ```bash
  git fetch upstream
  git rebase upstream/main
  ```

### PR guidelines

- Use a descriptive title following conventional commits (`feat:`, `fix:`, `docs:`, `test:`, `chore:`, `refactor:`, `perf:`)
- Fill out the PR template
- Reference related issues with `#issue_number`
- One feature, one bug fix, or one improvement per PR

### Review process

- Maintainers review code and provide feedback
- Address requested changes promptly
- Keep discussions constructive

## Security vulnerabilities

- **Critical vulnerabilities**: Email `christian@berclaz.org` directly
- **Non-critical security issues**: Use the Security Report issue template
- Include reproduction steps and impact assessment
- We follow responsible disclosure practices

## Documentation

The main documentation lives under `docs/` and is built with Sphinx using Markdown (MyST). When adding features:

- Update the relevant page under `docs/`
- Add connection examples to `docs/data-sources/` for new data sources
- Add domain documentation to `docs/domains/` for new analytical tools
- Test code examples to ensure they work

## Versioning

We follow semantic versioning:

- **Major**: Breaking changes to the MCP tool API
- **Minor**: New features, backward compatible
- **Patch**: Bug fixes, backward compatible

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).
