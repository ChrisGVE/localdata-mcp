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
# Using uv (recommended). The dev tools are an extra, not a uv dependency group,
# so `uv sync --dev` will not install them — use --extra dev or --all-extras.
uv sync --extra dev

# Or using pip
python -m venv venv
source venv/bin/activate   # macOS/Linux
pip install -e ".[dev]"
```

`uv sync --all-extras` additionally installs the `modern-databases` and `enterprise` drivers, which the integration tests need.

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
├── src/localdata_mcp/            # The whole package — nine modules, no sub-packages
│   ├── server.py                 # The seven MCP tools, and nothing else
│   ├── slots.py                  # The registry: nicknames, lifecycle, eviction, spill
│   ├── loader.py                 # Reading a datasource in, and describing it
│   ├── dialects.py               # What differs per backend, and only that
│   ├── binding.py                # Type adapters — see CONSTRAINTS §1
│   ├── config.py                 # Configuration discovery and validation
│   ├── paths.py                  # Path containment at the trust boundary
│   └── export.py                 # Writing a result out
├── tests/                        # One test module per source module
│   └── assets/                   # Deliberately hostile test files
├── docs/
│   ├── architecture/LEVEL0.md    # The specification: premise, three arcs, seven verbs
│   └── CONSTRAINTS.md            # Measured behaviour, with the evidence
├── non_factual/                  # Quarantined prose — see its README before reading
├── skills/data/local-data/       # The skill that ships with the server
├── .claude-plugin/plugin.json    # Claude Code plugin manifest
├── server.json                   # MCP registry entry
├── scripts/                      # Test-data generation
├── .github/                      # CI workflows and issue templates
├── pyproject.toml                # Project metadata and dependencies
├── Dockerfile                    # Container build
├── docker-compose.yml            # Dev stack with databases
├── LICENSE                       # Apache License 2.0
└── NOTICE                        # Attribution notice required by Apache 2.0
```

Each skill is a directory holding a single `SKILL.md`; the directory name is the skill name and must match the `name` field in the file's frontmatter. Place a new skill in the domain directory it belongs to rather than at the top of `skills/`. Version bumps must stay in step across `pyproject.toml`, `.claude-plugin/plugin.json`, and `server.json`.

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
- Keep modules focused enough to read in one sitting. There is no enforced line
  limit, but `server/database_manager.py` is the cautionary example at 4,000-plus
  lines: new domains go in their own adapter and mixin rather than into it
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
- Keep functions focused enough to read without scrolling

### Security

- Validate all inputs at system boundaries
- Route every agent-supplied query through `SQLQueryParser`
  (`src/localdata_mcp/query_parser.py`). The agent writes the SQL, so there is
  nothing to parameterize; what protects the database is a whitelist that admits
  `SELECT` and `WITH` and rejects the other 25 named operations, plus a check
  that refuses multiple statements in one call. Never open a query path that
  bypasses it
- Parameterize any SQL the server itself composes from a value it did not write —
  a table name, a filter, a limit. That is a different case from the agent's
  query, and binding is the right tool for it
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

There are four documents and each has one job. Update the one that owns what you
changed, in the same commit as the change — documentation that lags is a defect,
not a chore:

- **`README.md`** — what the server is and how to use it. Any change to the tool
  surface lands here.
- **`docs/architecture/LEVEL0.md`** — the specification. Change it when the
  design changes, not when the code does.
- **`docs/CONSTRAINTS.md`** — behaviour established by measurement, with the
  numbers. Add to it when you measure something that shaped a decision; a
  constraint nobody recorded is one the next person re-derives.
- **`skills/data/local-data/SKILL.md`** — how an agent should talk to a user
  about their data. It ships with the server and is versioned with it, because
  the two are only correct against each other.

Test the code examples you write. A worked example that does not run is the
failure mode this project has hit most often — the README taught an addressing
its own code refused for a full session before a test caught it.

Everything under `non_factual/` is quarantined and unverified by construction.
Do not cite it, and do not restore anything from it without checking it against
the code first.

## Versioning

We follow semantic versioning:

- **Major**: Breaking changes to the MCP tool API
- **Minor**: New features, backward compatible
- **Patch**: Bug fixes, backward compatible

## License

By contributing, you agree that your contributions will be licensed under the [Apache License 2.0](LICENSE).
