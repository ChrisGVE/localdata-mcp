# Contributing to LocalData MCP

How to set up a development environment, run the tests, and submit a change.

Two things are worth knowing before anything else. **Branch off `new-v3`, not
`main`** — `main` still carries 2.x, a different product, and a patch against it
is a patch against code that is being deleted. And **there is no CI gating this
branch**, so the test run you describe in your pull request is the only evidence
there is; `.github/WORKFLOWS.md` says why.

Security vulnerabilities do not go through an issue — see
[Security vulnerabilities](#security-vulnerabilities).

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
uv sync --all-extras

# Or using pip — name it .venv, which is what every command below invokes
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
pip install -e ".[all,dev]"
```

**The development instructions assume macOS or Linux**, and are written that way
rather than claiming a portability nobody here has tested: every command below
spells `.venv/bin/python`, which on Windows is `.venv\Scripts\python.exe`. The
*user*-facing install routes are platform-neutral, and the configuration cascade
documents `%APPDATA%`. Contributing from Windows should work with that one
substitution; nobody has run it, so this says so instead of promising it.

**`--all-extras`, not `--extra dev`, and never `--dev`.** Test tooling here is a
project *extra*, not a uv dependency group, so `uv sync --dev` uninstalls pytest
rather than installing it. And the `dev` extra is pytest and nothing else: every
format library and every database driver is a separate extra, and the loader,
export and endpoint suites need them. Without them those tests do not fail —
they skip, which is worse, because the run still reports green.

### Verify the setup

```bash
# The fast suite: everything but the volume tests
.venv/bin/python -m pytest -q -m 'not slow'
```

At the time of writing that is `768 passed, 500 skipped, 5 deselected`, measured
at 64 s on this machine. **Every one of the 500 skips is an endpoint test with no
container listening**, and each names the command that would start one — see
"Endpoint tests" below.

Coverage is not in the `dev` extra either. To measure it, install the plugin
yourself and ask for it:

```bash
uv pip install pytest-cov           # into .venv, not as a standalone tool
.venv/bin/python -m pytest -q -m 'not slow' \
    --cov=localdata_mcp --cov-report=html
```

There is no `--version` flag on the server: the entry point starts it, so
anything after `localdata-mcp` on the command line is ignored. Check what you
have installed with `uv tool list` or `pip show localdata-mcp`.

## Project structure

```
localdata-mcp/
├── src/localdata_mcp/            # The whole package — thirteen modules, no sub-packages,
│                                 #   plus the package `__init__` and the typing marker
│   ├── __init__.py               # `__version__` — one of the six version sites below
│   ├── server.py                 # The nine MCP tools, and nothing else
│   ├── slots.py                  # The registry: nicknames, lifecycle, eviction, spill
│   ├── loader.py                 # Reading a datasource in, and describing it
│   ├── dialects.py               # What differs per backend, and only that
│   ├── formats.py                # The one table of what a file suffix means
│   ├── readers.py                # One function per input format
│   ├── writers.py                # One function per output format
│   ├── errors.py                 # LoadError and ExportError, below both sides
│   ├── temporal.py               # Recognising and canonicalising date columns
│   ├── binding.py                # Type adapters — see CONSTRAINTS §1
│   ├── config.py                 # Configuration discovery and validation
│   ├── paths.py                  # Path containment at the trust boundary
│   ├── export.py                 # Writing a result out
│   └── py.typed                  # PEP 561 marker — what `Typing :: Typed` rests on
├── tests/                        # 15 test modules — `test_<module>.py` for nine of the
│                                 #   thirteen, plus test_concurrency, test_streaming,
│                                 #   test_volume, test_endpoints and test_answer_shape
│   ├── assets/                   # Deliberately hostile test files
│   ├── conftest.py               # Shared fixtures
│   ├── foreign.py                # Cross-backend helpers
│   └── endpoints.py              # The endpoint catalogue and its auth-mode axis
├── docs/
│   ├── architecture/LEVEL0.md    # The specification: premise, three arcs, nine verbs
│   └── CONSTRAINTS.md            # Measured behaviour, with the evidence
├── assets/                       # The logo, referenced by absolute URL from the README
├── non_factual/                  # Quarantined prose — see its README before reading
├── skills/data/local-data/       # The skill that ships with the server
├── .claude-plugin/plugin.json    # Claude Code plugin manifest
├── server.json                   # MCP registry entry
├── scripts/                      # Fixture and test-data generation, the endpoint
│                                 #   batch runner, and the logo build
├── .github/                      # Workflows and issue templates — read .github/WORKFLOWS.md
├── pyproject.toml                # Project metadata and dependencies
├── uv.lock                       # The resolved environment — carries the project
│                                 #   version too, so a bump needs a relock
├── MANIFEST.in                   # What the sdist carries beyond the package
├── README.md                     # The product documentation
├── CHANGELOG.md                  # Keep a Changelog format
├── CONTRIBUTING.md               # This file
├── docker-compose.test.yml       # 22 services for the endpoint suite: one per endpoint,
│                                 #   plus the auth-axis variants and two helpers
├── Dockerfile                    # 2.x's, and broken — see .github/WORKFLOWS.md
├── docker-compose.yml            # 2.x's development stack, likewise
├── .gitignore                    # Ignore rules
├── .dockerignore                 # Build-context exclusions
├── LICENSE                       # Apache License 2.0
└── NOTICE                        # Attribution notice required by Apache 2.0
```

**`export.py`, `readers.py`, `writers.py` and `errors.py` have no test module of
their own.** They are exercised from `test_loader.py`, `test_server.py` and
`test_volume.py`, which is where a reader or a writer is actually reached from —
through a verb, against the hostile corpus, rather than called directly. A test
for a new reader or writer goes in whichever of those matches how it is reached;
do not add a `test_writers.py` for one writer alone. `formats.py` does have one,
because what it asserts is about the table itself rather than about reading any
particular file.

There are no sub-packages and no plugin registry: a new format is one entry in
`formats.FORMATS` — the reader, the writer, and whether a delimiter or chunked
reading mean anything for it, all on the one `Format` — and a new backend is a
`dialects.Backend` subclass **only if** the generic SQLAlchemy answer means
something different for it — several backends needed no subclass at all, which is
the result rather than an omission.

One skill ships, at `skills/data/local-data/`. A skill is a directory holding a
single `SKILL.md`, the directory name must match the `name` field in its
frontmatter, and a new one goes under its domain directory rather than at the top
of `skills/`. Version numbers live in more than one file and must move together —
see [Versioning](#versioning).

## Development workflow

### Create a branch

```bash
git fetch upstream
git checkout new-v3
git merge upstream/new-v3
git checkout -b feature/your-feature-name
```

**Branch off `new-v3`, not `main`.** `main` still carries 2.x — a different
product with 71 tools, none of which survive — and will until the 3.0.0 release
lands. A patch against `main` is a patch against code that is being deleted.

### Make changes

- Follow existing code patterns and module structure. Nine modules, no
  sub-packages: a change usually belongs in one of them rather than in a new one.
- Add tests for new functionality, and write the test first. The suite has caught
  wrong answers that no error surfaced — a number arriving as text, a `CREATE
  TABLE` that was permanent despite being refused, a rollback that returned
  normally over a write that stood.
- **Assert on query results, never on binding.** A value that binds without error
  and comes back wrong is the failure mode this project keeps meeting
  (`docs/CONSTRAINTS.md` §5.1), and a test that only checks the insert did not
  raise is blind to all of it.
- Keep modules focused enough to read in one sitting. There is no enforced line
  limit, and `dialects.py` at 126 KB is the one to watch: a new entry there
  should be a `Backend` subclass carrying only what the generic answer gets
  wrong for that engine, not a place for logic every backend shares.
- Update documentation in the same commit as the change, not afterwards.

### Run tests

```bash
# Everything but the volume suite — what to run before every commit
.venv/bin/python -m pytest -q -m 'not slow'

# One module
.venv/bin/python -m pytest tests/test_loader.py -v

# The volume and timing tests, which take minutes and gigabytes
.venv/bin/python -m pytest -m slow

# Only the dialects
.venv/bin/python -m pytest -m endpoint
```

There are two markers, both declared in `pyproject.toml`: `slow` for the volume
and timing tests, and `endpoint` for the dialects. `endpoint` exists to make the
*reverse* selection possible — every endpoint test already skips itself when its
container is not answering, so the marker is not needed to keep a Docker-free run
green.

### Endpoint tests

`tests/test_endpoints.py` holds twenty test functions, and each runs against
every entry in `tests/endpoints.py`'s `TARGETS` — **twenty-five**, not sixteen:
the sixteen containers plus the nine authentication-mode variants that four of
them carry. That is where the 500 skips above come from, 20 × 25.
`docker-compose.test.yml` defines the containers:

```bash
docker compose -f docker-compose.test.yml up -d localdata-test-postgres
.venv/bin/python -m pytest -m endpoint -k postgres
docker compose -f docker-compose.test.yml down
```

**This machine will not run the whole catalogue at once.** Six containers run;
at around seven the Docker VM starves them and the suite reports code failures
that are not. That one measurement is what every "six at a time" and "five
batches" in the other documents refers to. So the catalogue takes five batches
once the authentication variants are counted, and each batch reports a green
suite while the dialects it never reached stay silent
([#46](https://github.com/ChrisGVE/localdata-mcp/issues/46)).
`scripts/endpoint-batch.sh` holds the five batches — `a` through `e` — so they
live in a script rather than in prose that has already gone stale twice:

```bash
./scripts/endpoint-batch.sh a          # one batch, then remove its images
./scripts/endpoint-batch.sh all        # all five in sequence
./scripts/endpoint-batch.sh a --keep   # leave the containers and images up
```

The sixteen images are 33 GB, Exasol alone 12 GB, so the script removes them
afterwards by default: re-download every time, and leave nothing behind. **Say in
your PR which batches you actually ran** — a green run says nothing about a
dialect it never reached.

## Code standards

### Style

- Follow PEP 8. `pyproject.toml` configures black at line length 88, isort on
  the black profile, and mypy targeting 3.10 with `no_implicit_optional` and
  `strict_optional`. **None of the three is in the `dev` extra**, so install them
  yourself if you want to run them locally — `uv tool install black`, and the
  same for `isort` and `mypy` (one package per invocation).
- Use type hints on all public function signatures.
- **Write the docstring for a reader who has to justify the code, not describe
  it.** This codebase says why a thing is the way it is — which measurement
  forced it, which alternative was tried, what it costs — and that is the house
  style, not decoration. `dialects.py` and `loader.py` are the examples to match.
- Keep functions focused enough to read without scrolling.

### Security

- **Do not add a SQL parser.** There isn't one, deliberately. `query` runs on a
  connection that is read-only from the moment it opens, so there is no statement
  text to parse and mis-parse and no window in which the posture is briefly
  something else. The previous design matched SQL patterns and three issues found
  ways around it ([#33](https://github.com/ChrisGVE/localdata-mcp/issues/33),
  [#36](https://github.com/ChrisGVE/localdata-mcp/issues/36),
  [#38](https://github.com/ChrisGVE/localdata-mcp/issues/38)). A new backend
  makes its read connection refuse as far as that engine can, and where it cannot
  — Oracle commits DDL as it runs — the refusal says so rather than claiming
  otherwise.
- **Underneath every dialect is one dialect-free rule**: a statement that returns
  no rows, or rows of no columns, is not a read and is refused on that ground.
  Without it, a rolled-back write came back as a statement that succeeded and
  returned nothing, which is indistinguishable from success.
- **Parameterize any SQL the server itself composes** from a value it did not
  write — a table name, a filter, a limit. That is a different case from the
  agent's query, and binding is the right tool for it.
- **Every path crosses `paths.py`**, which resolves symlinks and `..` *before*
  the containment check, not after. A network URL is refused unless
  `network.enabled` is set.
- **Nothing may reach stdout but JSON-RPC.** There is no logging configuration
  and no `print` in the package, and that is not an accident: a stray write to
  stdout corrupts the protocol channel, which is what
  [#35](https://github.com/ChrisGVE/localdata-mcp/issues/35) and
  [#39](https://github.com/ChrisGVE/localdata-mcp/issues/39) were.
- **Never evaluate a caller's string as code.** The v2 tool that did was a host
  RCE ([#42](https://github.com/ChrisGVE/localdata-mcp/issues/42)).
- Handle errors without exposing sensitive information — a datasource URL is
  reported with its password masked, in the refusal and in `directory` alike.

### Testing

The rules that are specific to this project are under "Make changes" above —
write the test first, and assert on query results rather than on binding. Two
more that are not obvious from the code:

- **Mock the filesystem** for permission and missing-file scenarios rather than
  touching the real one. The endpoint suite is the exception: it runs against
  real containers, because the failures it exists to catch are the engine's.
- **A path-containment or refusal test must send a statement the parser would
  otherwise accept.** A test whose input is rejected for an unrelated reason
  stays green after the guard it is testing is deleted.

## Pull request process

### Before submitting

- Ensure all tests pass
- Update documentation for user-facing changes, in the same commit
- Rebase on the latest `new-v3` if needed — **not `main`**, for the reason at
  the top of this file:
  ```bash
  git fetch upstream
  git rebase upstream/new-v3
  ```
- **Say which tests you ran.** For anything touching `dialects.py` or a reader,
  name the endpoint batches — a Docker-free run reports green against every
  dialect it never reached.

### PR guidelines

- Use a descriptive title following conventional commits (`feat:`, `fix:`, `docs:`, `test:`, `chore:`, `refactor:`, `perf:`); a `!` marks a breaking change
- Fill out the PR template. Its "Database Support" checklist names SQLite,
  PostgreSQL and MySQL, which this version still supports; tick "Other databases
  tested" for any of the other fifteen
- Reference related issues with `#issue_number`
- One feature, one bug fix, or one improvement per PR
- **There is no CI gating this branch.** `.github/WORKFLOWS.md` explains why, so
  the test run in your PR description is the only evidence there is

## Security vulnerabilities

- **Critical vulnerabilities**: Email `christian@berclaz.org` directly
- **Non-critical security issues**: Use the Security Report issue template
- Include reproduction steps and impact assessment
- We follow responsible disclosure practices

## Documentation

**Seven documents are maintained, and this is the list.** Each has one job.
Update the one that owns what you changed, in the same commit as the change —
documentation that lags is a defect, not a chore. (`non_factual/README.md` is
tracked too, and is deliberately not on this list: it is a quarantine notice for the
abandoned v2 documents, not a document kept current.)

- **`README.md`** — what the server is and how to use it. Any change to the tool
  surface, the format registries or the backend catalogue lands here.
- **`CHANGELOG.md`** — every user-facing change, in Keep a Changelog form.
  Anything that would make an existing caller do something different is a
  breaking change and is called out as one.
- **`docs/architecture/LEVEL0.md`** — the specification. Change it when the
  design changes, not when the code does. An amendment gets a dated note saying
  what it supersedes rather than a silent edit, because the reasoning is what the
  document is for.
- **`docs/CONSTRAINTS.md`** — behaviour established by measurement, with the
  numbers. Add to it when you measure something that shaped a decision; a
  constraint nobody recorded is one the next person re-derives. It is append-only
  in spirit: a later measurement that changes the answer is a new section, not an
  edit to an old one.
- **`skills/data/local-data/SKILL.md`** — how an agent should talk to a user
  about their data. It ships with the server and is versioned with it, because
  the two are only correct against each other.
- **`.github/WORKFLOWS.md`** — what CI does and does not do.
- **`CONTRIBUTING.md`** — this file: how to build, test and submit. It owns the
  document inventory above, so a document added or retired is edited here first.

**Run the code examples you write.** A worked example that does not run is the
failure mode this project has hit most often — the README taught an addressing
its own code refused for a full session before a test caught it, and the shipped
skill told agents that dates were stored as integer ticks for weeks after the
code had settled on canonical UTC text. Reading the source and finding the claim
plausible is not verification.

Everything under `non_factual/` is quarantined and unverified by construction.
Do not cite it, and do not restore anything from it without checking it against
the code first.

## Versioning

We follow semantic versioning:

- **Major**: Breaking changes to the MCP tool API
- **Minor**: New features, backward compatible
- **Patch**: Bug fixes, backward compatible

A version lives in **six** hand-edited places across five files, and they must
move together — then `uv lock` has to be re-run, because `uv.lock` carries the
project's own version too. **`--locked` is what catches a bump without a relock,
and it is a general flag rather than a `sync` one.** Measured on a copy of this
tree with the version bumped and `uv.lock` untouched: `uv lock --check` and
`uv sync --locked` exit 1, and `uv run --locked`, `uv export --locked` and
`uv tree --locked` exit 2 — five commands, all of them refusing. Without the
flag nothing refuses. `uv sync --frozen` uses the lockfile without validating
it, so it exits 0 and installs the bumped version against the stale lock, and a
bare `uv run` is worse than silent: it exits 0 having **rewritten `uv.lock`**,
which is the relock you forgot, performed where nobody is looking at it. No
workflow in this repository runs `uv` at all — the two that did were deleted on
2026-08-11 (see `.github/WORKFLOWS.md`) — and nothing here runs `uv lock
--check`, so forgetting the relock is silent rather than self-announcing. The
one bare `uv run` that still executes is `.claude-plugin/plugin.json`'s launch
command, on a user's machine rather than in CI:

| File | Field | Today |
|---|---|---|
| `pyproject.toml` | `project.version` | `3.0.0.dev0` |
| `src/localdata_mcp/__init__.py` | `__version__` — **nothing derives this from `pyproject.toml`**; there is no `dynamic` key, so the two agree only because someone kept them in step, and no test checks that they do | `3.0.0.dev0` |
| `.claude-plugin/plugin.json` | `version` | `3.0.0-dev` |
| `server.json` | `version` — the registry entry's own | `2.1.0` |
| `server.json` | `packages[0].version` — **the PyPI release a client is told to fetch**, which is not the same thing and must name a release that exists | `2.1.0`, which PyPI 404s |
| `Dockerfile` | `LABEL version` | `2.0.0` |

They disagree, and which number 3.0.0 carries is a release decision rather than
something to fix in passing. `server.json`'s `packages[0].version` is the one
that is a defect either way: it names an artifact nobody can fetch, so the
registry submission is rejected whatever version is chosen.

## License

By contributing, you agree that your contributions will be licensed under the [Apache License 2.0](LICENSE).
