# Changelog

All notable changes to LocalData MCP Server will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [3.0.0] - unreleased

A ground-up rewrite. **Every one of the 71 tools 2.1.0 registered is gone**, and
nine verbs stand where they were. Nothing a 2.x client calls by name resolves,
and no alias, shim or compatibility layer is provided — the names are not
deprecated, they are absent.

### Upgrading from 2.x

**Your entire tool vocabulary is gone.** `connect_database`, `execute_query`,
`list_databases`, `describe_table`, `analyze_regression`, `forecast_time_series`,
`analyze_spatial_autocorrelation`, `get_node`, `export_graph` — all 71. A client
that names a tool explicitly will find nothing there; an agent that discovers
tools at runtime will simply see nine.

The seventy-one divide three ways: **sixteen map onto four of the new verbs**,
**forty-six have no successor at all**, and **nine managed machinery that no
longer exists**. The other five verbs are new in 3.0.0 and replace nothing.

| 3.0.0 verb | What it replaces from 2.x |
|---|---|
| `attach` | `connect_database` |
| `detach` | `disconnect_database` |
| `query` | `execute_query`, `analyze_query_preview`, `next_chunk`, `request_data_chunk`, `request_multiple_chunks`, `get_query_metadata`, `export_structured`, `search_data` |
| `directory` | `list_databases`, `describe_database`, `describe_table`, `find_table`, `export_schema`, `get_data_quality_report` |
| `create` | **new in 3.0.0** — reading a second datasource *into* an open one had no 2.x counterpart |
| `update` | **new in 3.0.0** |
| `drop` | **new in 3.0.0** |
| `save` | **new in 3.0.0** — 2.x had staging databases, which were not the user's to keep |
| `stats` | **new in 3.0.0** — 2.x reported no missing value under any tool |

That table accounts for sixteen 2.x tools: 1 + 1 + 8 + 6.

**Forty-six of the seventy-one have no replacement at all** — twenty analytical,
ten geospatial, seven graph and nine tree. Statistics, regression, clustering,
anomaly detection, time series, RFM, A/B testing, optimization, spatial analysis,
graph traversal and key-value trees are not in this product and are not planned.
The remaining nine — `manage_memory_bounds`, `get_streaming_status`,
`clear_streaming_buffer`, `cancel_query_operation`, `get_query_log`,
`get_error_log`, `get_metrics`, `check_compatibility` and `transform_data` —
managed machinery that no longer exists. Sixteen mapped, forty-six gone, nine
retired: seventy-one, with nothing unaccounted for. That platform
was abandoned, not deferred; its documentation is quarantined in `non_factual/`.
What this server does is SQL, and an agent that can write SQL can compute a
group-by, a correlation or an anti-join in the statement it was going to send
anyway.

**Every write path through query is refused.** In 2.x, `execute_query` ran what
it was given and a `security.readonly` setting tried to catch writes by matching
SQL patterns — a check three issues found ways around
([#33](https://github.com/ChrisGVE/localdata-mcp/issues/33),
[#36](https://github.com/ChrisGVE/localdata-mcp/issues/36),
[#38](https://github.com/ChrisGVE/localdata-mcp/issues/38)). There is no such
setting now, because there is no such check: `query` runs on a connection that is
read-only from the moment it opens, and composition has its own verbs. If your
2.x workflow wrote through a query, it must now call `create`, `update` or
`drop`, on a datasource attached with `writable=true`.

**Configuration is TOML, and thirty environment variables became one.** 2.x read
YAML from `~/.localdata.yaml` and elsewhere, and accepted roughly thirty
`LOCALDATA_*` environment overrides. 3.0.0 reads `config.toml` through a
first-found-wins cascade, and the only environment variable that carries any
part of the configuration is `LOCALDATA_CONFIG_PATH`, which *locates* the file
and never holds a setting. The cascade also consults `XDG_CONFIG_HOME` and, on
Windows, `APPDATA`, but those belong to the operating system and name a
directory rather than a setting.
There are three sections and five settings in total — `workspace.slots`,
`workspace.memory_budget_mb`, `paths.roots`, `paths.path_limited` and
`network.enabled`. **An unknown section or key is refused rather than ignored**,
because a mistyped `path_limitted = false` that silently kept the safe default is
a security switch you believe you have thrown. The file is read at startup, so a
configuration the server will not run under stops it starting rather than
surfacing inside a tool call — the refusal names the bad key and the keys the
section does know, and it arrives in the client's server log. No 2.x
configuration file is readable, and none is migrated.

**Nothing survives the session unless you `save` it.** 2.x persisted staging
databases and buffers across calls with their own eviction policies and disk
budgets. A 3.0.0 slot lives until `detach` or until the server stops.

**Ten datasources, and the oldest is evicted.** Check the `evicted` field on
every `attach` response: it describes the slot that was dropped well enough to
rebuild it.

### Added

- Nine MCP tools — **verbs**, the word used for them throughout: `attach`,
  `detach`, `query`, `directory`, `create`, `update`, `drop`, `save`, `stats`. Few and
  multi-faceted rather than many and narrow — `directory` alone absorbs six 2.x tools
  by varying on its two optional arguments.
- **`stats(nickname, table, columns?)` — the profile, and the only call that
  reports a missing value.** Every column answers with `nulls` and `non_nulls`, a
  numeric one adds `min`, `max` and `avg`, and a date column normalised to ISO
  8601 adds `min` and `max`. `median` and `stddev` are reported only where this
  server declares them for that backend — never emulated, never approximated,
  never a second pass in Python. That declaration defaults to none, and every
  backend has been driven against known values to fill it in (#95): a `median`
  on DuckDB, PostgreSQL, YugabyteDB, openGauss, CockroachDB, MonetDB,
  ClickHouse, Oracle and Exasol, and a sample `stddev` on every backend but
  SQLite. Where an engine had more than one median, the one that
  averages the two middle values and is exact was chosen — ClickHouse's
  `median` estimates, MonetDB's returns the lower middle value — and where
  its function refused a decimal or integer column the column is read as a
  double, the conversion PostgreSQL makes for itself. MySQL, MariaDB,
  Firebird, CrateDB, Trino, SQL Server and YDB report no median, each for a
  reason in `docs/CONSTRAINTS.md` §37. A **mixed**
  column and a column of **dates in no recognised standard** are profiled to
  their null count alone, each saying why in `withheld`, because an aggregate
  over either returns a real number that is not the number asked for. It is a
  verb rather than a flag on `directory` because a directory that reports statistics
  stops being one.
- **A text file is read and written at its standard separator unless you say
  otherwise.** `.csv`, `.tsv` and `.txt` are **one format** — character-separated
  text — and `delimiter` names its separator, on `attach`, `create` and
  `query(path=…)`. Left out, the separator a published standard gives the
  suffix is used: a comma for `.csv` (RFC 4180) and a tab for `.tsv` (IANA
  `text/tab-separated-values`). A `delimiter` you pass always wins, on every
  suffix, so a `;`-separated `.csv` reads with `delimiter=";"`. **`.txt` has no
  standard, so it now needs a `delimiter`**; without one the call is refused and
  says how to find the separator. Both directions follow the same rule, so a
  file this server writes with nothing declared is one it reads back with
  nothing declared. Handed to a format that has no separator it is refused on
  the way in and ignored on the way out. Nothing sniffs: a semicolon-separated
  file read at a comma — the `.csv` default or a wrong declaration — loads as
  one column, and the warning quotes the header and names the `;` in it rather
  than re-reading at a guessed separator
  ([#153](https://github.com/ChrisGVE/localdata-mcp/issues/153)).
- **Seventeen file formats read** (`.csv` `.tsv` `.txt` `.json` `.jsonl`
  `.ndjson` `.xml` `.yaml` `.yml` `.xlsx` `.xlsm` `.xls` `.ods` `.numbers`
  `.parquet` `.feather` `.orc`) and **fifteen written** (the same, less
  `.xlsm`, `.xls` and `.numbers`, plus `.md`). `.xlsx` and `.ods` are written as
  well as read. Seven of the readers and seven of the writers are guaranteed by
  the base install; `.yaml`/`.yml` also work without their extra today, because
  `fastmcp` requires `PyYAML` unconditionally — a fact about the dependency graph
  rather than a promise, so `yaml` stays the declared extra.
- **A file may hold more than one table, and all of them land.** A workbook
  becomes a database with a table per sheet, and a `.numbers` document one per
  **table** — a Numbers sheet is a canvas that may carry several — each under its
  own name put through the same snake_case rule as a nickname, so `Sheet1`
  becomes `sheet1` and `update` renames it. Two `.numbers` tables that would
  make the same table name are qualified by their sheet; names that still clean
  up alike — `Q2 Prices` and `Q2-Prices` — get a numeric suffix in file order,
  as repeated column headers do (`q2_prices`, `q2_prices_1`), and `attach` says
  which sheet became which table. 2.x read the
  first sheet and said nothing about the rest. A JSON, YAML or XML document
  holding several tables lands every one of them the same way — see the next
  entry.
- **A JSON, YAML or XML document holding several tables lands all of them**,
  where it used to be refused, naming both, with *"which one you want is not
  something this server should decide"*. Each non-empty array of objects in a
  JSON or YAML document is a table, found at any depth through objects (never
  inside an array, whose objects are rows), and is named from its **key path** —
  the keys from the root down, joined by `.` and cleaned like a sheet name, so
  `meta.orders` lands as `meta_orders`. In XML each element repeated under the
  root is a table, named after its tag. Names that clean up alike get the
  workbook's numeric suffix (`a_b`, `a_b_1`) and the workbook's clash warning;
  each `loaded` entry names its key path or tag as `sheet`, and `warnings` says
  once which path became which table and which keys were not part of any. A
  document holding one table loads exactly as before — under the file's name,
  with the same note — and that now includes one nested deeper than the top
  level, which used to be refused as holding no table. `create` still makes one
  table and refuses a document of several, naming what each would arrive as.
- **Seventeen database backends**: SQLite, DuckDB, PostgreSQL, MySQL, MariaDB,
  SQL Server, Oracle, ClickHouse, CockroachDB, YugabyteDB, Trino, MonetDB,
  CrateDB, Firebird, openGauss, YDB and Exasol. Fifteen are exercised
  against a container of their own in `docker-compose.test.yml`. The nine verbs
  are the whole surface on all of them; three verbs are refused on some engines,
  and those refusals are listed under *Known limitations* below.
- **Nine further ways of authenticating an endpoint are exercised by the test
  suite**, beyond the plain URL that was the only route 2.x ever took: a
  server configured to trust, a password from the environment, a password from a
  file, verified TLS, a client certificate, a Kerberos ticket, a MySQL option
  file, an empty password, and an ODBC data-source name in place of a host and
  port — ten routes in all, counting the plain URL that all fifteen endpoints
  still use, which carries a password on ten of them and a bare username on
  the other five (CockroachDB, YugabyteDB, Trino, CrateDB, YDB). All five of
  those are databases with authentication available and none of it configured,
  which is the same shape as `trust` — CockroachDB's `--insecure` accepts any
  password for `root`, and YDB's image configures none of the authentication it
  supports. `trust` is counted among the nine added modes because it is a named
  PostgreSQL setting this suite sets deliberately, not because those five differ
  in kind from it. **None of the ten is a setting of this server, and
  there is no parameter for any of them**: `attach` takes a datasource string,
  and each mode is expressed in the URL or in the driver's own environment,
  which this server passes through untouched. What 3.0.0 adds is the evidence
  that they work — `tests/endpoints.py` runs the whole endpoint suite as a
  second axis over them. `docs/CONSTRAINTS.md` §25 has the measurements, and
  `docs/architecture/LEVEL0.md` records which endpoint carries which mode.
- **`create(nickname, type="table", source=…)`** reads a second datasource in
  beside the tables already in a slot, which is what makes a cross-file lookup
  keepable: `save` writes one database, not a join.
- **`create(nickname, type="index", …)`** and `drop`, so a join that drags can be
  indexed. Nothing is indexed unless asked for.
- **`save(nickname, path, force?)`** relocates an in-memory or spilled database to
  a path the user chose. An occupied path is refused until `force` carries their
  consent, and a path a live slot sits on is refused regardless.
- **`query(…, path=…)`** writes an oversized result straight to a file instead of
  into the answer, and the suffix chooses the format.
- **Transparent spill to disk.** The working budget is 100 MB by default. A load
  that crosses it is allowed to finish; the *next* operation moves the largest
  in-memory database to a temp file and reconnects it in the same slot. No tool
  announces it. Residency is measured as
  `(page_count − freelist_count) × page_size` — measured, not estimated from file
  size, because a pre-flight estimate is the fail-open pattern that bit this
  project once.
- **Date canonicalisation.** Across the spellings of five instants
  spanning three years, every day-first and every month-name form ordered wrongly
  under `ORDER BY` and returned the earliest instant from `max()` — silently,
  with no error and no warning. (`docs/CONSTRAINTS.md` §8.1 asks that the classes
  be quoted rather than the counts: how many spellings fall in each class depends
  on which spellings the fixture happened to include.)
  ISO 8601 extended and Unix time are recognised; a recognised column is rewritten
  into one canonical UTC spelling and reported as
  `{"temporal": "iso8601_utc", "normalized": "UTC"}`. Everything else is left
  alone and reported with the offending values named. Ambiguous spellings such as
  `01/03/2025` are deliberately not guessed.
- **A mixed-type column is flagged on load**, with the non-numeric values named.
  An `avg()` over such a column counts text as zero and keeps it in the
  denominator, and nothing else says so.
- **Nickname collision is reported rather than resolved silently.** A second
  source deriving the same nickname gets a numeric suffix, and `collided_with`
  names the slot and source that forced it. The same source attached twice is
  refused outright, naming where it already lives.
- **Path confinement** to the working directory and configured roots, with
  symlinks and `..` resolved before the check. Network URLs are refused until
  `network.enabled = true`, and the refusal masks the password.
- **Installable as a plugin in Claude Code and in Codex**, from the marketplace
  file this repository carries (`.claude-plugin/marketplace.json`, which Codex
  also reads), with the server and the `local-data` skill in one step. Both launch the server with `uvx localdata-mcp`. The skill moved
  to `skills/local-data/`, one level up, because both loaders look exactly one
  directory below `skills/`. The README gives the plugin route and a manual one
  for each harness. The cross-vendor Agent Plugins form (`plugin.json` and
  `mcp.json` at the root) is deliberately not used: it starts the server in the
  plugin's own directory, and with paths confined to the working directory
  and the configured roots, it refused every file the user pointed at.
- `docs/architecture/LEVEL0.md` — the specification — and `docs/CONSTRAINTS.md`,
  which records every behaviour measured to arrive at these decisions, with the
  numbers.

### Changed

- **Breaking: Python 3.10 is no longer supported; the floor is 3.11.** Python
  3.10 reached end of life in October 2026 (PEP 619), and the project's
  dependencies are moving past it, so `requires-python` is now `>=3.11` and the
  3.10 classifier is gone. The `tomli` backport that 3.10 needed is no longer a
  dependency, because `tomllib` is in the standard library from 3.11. No release
  of this line had reached users on 3.10: 3.0.0 has not been published.
- **Built and tested on fastmcp 4.1.6 (MCP SDK 2.3.0), and still runs on
  fastmcp 3.** The requirement stays `fastmcp>=3.0.0`, uncapped. On fastmcp 4 the
  server answers both the classic `initialize` handshake and the sessionless
  2026-07-28 protocol, and reports `localdata` and this project's version in
  both. One thing a client will see differently: the per-parameter prose in each
  tool's docstring, the file-format lists included, now arrives in that
  parameter's schema `description` rather than in the tool description. No
  source change was needed. Measured in CONSTRAINTS §14.5.
- **Every datasource is a database**, whatever it came from, and the same nine
  verbs address all of them. In 2.x a CSV and a PostgreSQL connection were
  different kinds of thing: `connect_database` took both, and which of the
  seventy-one tools then worked depended on which you had handed it, with nothing
  in the type system to notice. Here the surface does not vary — where an engine
  cannot carry out a verb, the verb is still there and the refusal names the way
  round (see *Known limitations*).
- **A statement reaches one datasource.** Slots do not share a connection, so
  there is no join across nicknames; `create` copies one into the other and the
  join is then ordinary SQL.
- **The whole result comes back, and there is no row cap.** 2.x buffered results
  and handed them back in chunks. A row cap measures the wrong dimension — a
  hundred rows of a two-hundred-column table is the flood it was meant to
  prevent — so `LIMIT`, named columns and `path=` are the controls instead.
- **A delimited file is read twice rather than held once.** `.csv`, `.tsv` and
  `.txt` go through a measuring pass and an inserting pass, dropping
  the load's peak from 4,286 MB to 803 MB on a 1.22 GB CSV, at 1.4–1.75× wall
  clock. Below about 150 MB it costs slightly more than it saves. Formats that
  cannot be chunked still peak with the file, and that is stated rather than
  worked around.
- **A declared column type is named by the backend.** The portable spellings 2.x
  used could not create a table on Oracle at all, and quietly put float64 data
  into PostgreSQL's four-byte `REAL`.
- **A value leaving `query` is spelled for JSON.** A `Decimal` left alone reached
  the client as the string `"155000"`, and an agent then compares and adds text.
- **The dependency set is minimal**: `fastmcp`, `pandas`, `sqlalchemy` and a
  `tomli` backport on Python 3.10. 2.x carried scipy, scikit-learn, statsmodels,
  numpy, networkx, pydot, rdflib and SPARQLWrapper as core requirements. Every
  format library and every database driver is now an extra.
- **DuckDB 2.0.0 is the version the suite runs against** on Python 3.11 and later
  (it requires 3.11); Python 3.10 resolves to 1.5.6. The `duckdb` extra still
  accepts `duckdb>=1.0`, so a fresh install already got 2.0.0. 2.0.0 opens and
  writes files made by 1.5.5 and leaves them readable by 1.5.5, but a file
  *created* by 2.0.0 carries storage version 999, and 1.5.5 refuses to open it.
  This server never creates a DuckDB file, but a user's other tools may
  (`docs/CONSTRAINTS.md` §36).
- The licence is Apache 2.0, unchanged from 2.1.0.
- **Every endpoint test container is built from a Dockerfile in this
  repository** ([#139](https://github.com/ChrisGVE/localdata-mcp/issues/139)).
  `docker/test/<service>/Dockerfile` exists for each of the twenty-one services
  in `docker-compose.test.yml`, FROM the upstream image the service used to
  name, and adds nothing to it; the compose file still holds every service's
  environment, ports and healthcheck. The three services that tracked `latest`
  are pinned to the version last measured live, each tag resolving to the same
  digest the machine already held: CockroachDB `v26.3.4`, CrateDB `6.5.0`,
  MonetDB `Dec2025-SP3` (11.55.7). The CA and KDC helpers' inline scripts moved
  into files beside their Dockerfiles, and their packages are installed when the
  image is built rather than on every start.
- **The endpoint tests run in a container too** (`docker/test/runner/Dockerfile`,
  [#139](https://github.com/ChrisGVE/localdata-mcp/issues/139)): Python 3.11 on
  Debian bookworm, unixODBC with Microsoft's `msodbcsql18` from Microsoft's own
  repository, MIT Kerberos, and every extra's locked dependencies in a
  virtualenv outside the repository. The source is mounted at run time and never
  copied, so the image cannot test stale code, and the host's OpenSSL and ODBC
  no longer decide whether SQL Server can be tested.
- **`tests/endpoints.py` can address an endpoint from inside its compose
  network** ([#139](https://github.com/ChrisGVE/localdata-mcp/issues/139)).
  `LOCALDATA_TEST_IN_NETWORK=1` switches every address from the loopback and
  the published port to the service name and the container's own port: the
  URLs, the `.pgpass` host field, the ODBC DSN's `Server` line and the KDC the
  Kerberos mode asks. Any other value than `1`, `0` or unset is refused. Host
  mode is the default and unchanged. For the TLS and Kerberos modes to verify
  under a service name, the server certificate now also names
  `localdata-test-postgres-tls` and the keytab also holds
  `postgres/localdata-test-postgres-krb`. `docker/test/in-network.compose.yml`
  gives each run its own network and makes YDB advertise its service name.
- **ruff is the declared formatter and linter**
  ([#131](https://github.com/ChrisGVE/localdata-mcp/issues/131)).
  `pyproject.toml` declared black and an isort profile while the tree was
  formatted with ruff, so a tool honouring the declaration rewrote lines nobody
  had touched. `[tool.black]` and `[tool.isort]` are gone; `[tool.ruff]` states
  line length 88, the classic `E4`/`E7`/`E9`/`F` lint set, and keeps the
  formatter off Markdown. ruff is pinned at 0.17.3 in the `dev` extra, and
  `ruff format --check .` and `ruff check .` are clean on the tree at that
  version.
- **`scripts/endpoint-run.sh` replaces `scripts/endpoint-batch.sh`**
  ([#139](https://github.com/ChrisGVE/localdata-mcp/issues/139)). It runs one
  database at a time under the machine-wide lock `/tmp/localdata-docker.lock`.
  Each database is built, brought up in a compose project and network of its
  own, tested by the runner on that network, and then removed together with its
  image, any base image the run pulled, and, at the end, the runner and the build
  cache the run created. The five batches and the macOS `DYLD_LIBRARY_PATH`
  workaround for SQL Server are gone with the old script. `--keep` leaves one
  database up and `--dispose` removes it; `--seed` loads the standard assets
  into it through the new `scripts/seed_endpoint.py`, which uses the URL
  `tests/endpoints.py` builds and the server's own `attach` and `create`.
- **An endpoint run names the targets it did not reach**
  ([#46](https://github.com/ChrisGVE/localdata-mcp/issues/46)). A skipped
  endpoint test reads like a passing one, so a run against one container used to
  end on a green line that said nothing about the rest of the catalogue. Any
  pytest session that selects an endpoint test now ends with an
  `endpoint coverage` section: how many of the 24 targets in `TARGETS` it
  reached, and for each of the others whether there was no container, no driver,
  another missing precondition, a failed probe, or no test that asked for it.
  `scripts/endpoint-run.sh` collects what each service's session reached and
  ends with the targets the whole invocation did not reach, each with the
  command that would run it. The old closing line, "every one not listed
  above", is gone.

### Removed

- **All 71 MCP tools of 2.1.0.** See Upgrading above.
- **The data science platform** — statistical analysis, regression, pattern
  recognition, time series, business intelligence, optimization, sampling and
  geospatial domains, along with the pipeline framework and the sklearn-compatible
  transformers underneath them.
- **Graph, RDF and tree storage.** DOT, GML, GraphML, Mermaid, Turtle,
  N-Triples, SPARQL endpoints, and the key-value tree tools.
- **The non-SQL databases**: MongoDB, Redis, Elasticsearch, InfluxDB, Neo4j and
  CouchDB. This server speaks SQL through SQLAlchemy, and a database is in scope
  if and only if an open-source SQLAlchemy adapter exists.
- **`.html` and `.htm`**, from both registries, on 2026-07-28. Reading them was
  defensible; writing them was not — the writer emitted a bare `<table>` fragment
  rather than a document, and it was the one suffix that broke the round-trip
  property the overlap between the two registries is supposed to mean, since it
  wrote a table of any size and could not read back past lxml's 10,000,000-node
  XPath ceiling (about 417,000 rows of eleven columns). The `html` extra went with
  it, lxml being its only dependency.
- **`.hdf5`, `.ini`, `.toml` and `.arrow`** as datasource formats.
- **Databend**, suspended on 2026-10-29 before it shipped. It was landed during
  3.0.0's development and is taken out because its SQLAlchemy dialect,
  `databend-sqlalchemy`, has three defects that give a wrong outcome without
  saying so, and none has a released fix: `do_rollback()` swallows
  `NotSupportedError`, so a rollback reports success over a write that stands
  ([databendlabs/databend-sqlalchemy#77](https://github.com/databendlabs/databend-sqlalchemy/issues/77));
  `Index.create()` compiles to an empty string and creates nothing
  ([#78](https://github.com/databendlabs/databend-sqlalchemy/issues/78)); and
  Core `Time` renders as `DATETIME`, so a time of day is stored as a 1970
  timestamp ([#79](https://github.com/databendlabs/databend-sqlalchemy/issues/79)).
  Two more were worked around rather than fixed: `bytes` cannot be bound unless
  they happen to be valid UTF-8
  ([#80](https://github.com/databendlabs/databend-sqlalchemy/issues/80), fixed
  upstream and not yet released), and `databend-driver` does not expose the
  write progress the server reports
  ([databendlabs/bendsql#790](https://github.com/databendlabs/bendsql/issues/790)).
  The `databend` extra, its container and its endpoint tests went with it. This
  is a suspension, not a verdict on the engine: the measurements stay in
  `docs/CONSTRAINTS.md` §23, and the backend can return from git history once
  the dialect is fixed.
- **Eighteen skills and eleven agents.** One skill ships now,
  `skills/local-data/SKILL.md`, and it carries the idiom the tools
  deliberately do not: the naming conversation, and how to phrase an incomplete
  join in a user's own words rather than as an anti-join.
- **The `security.readonly` setting, `max_query_length` and `blocked_keywords`.**
  All three were pattern checks over SQL text; `query` is now read-only by the
  connection's posture, which cannot be reached around.
- **Every `LOCALDATA_*` environment variable except `LOCALDATA_CONFIG_PATH`.**
- **The staging-database and disk-budget subsystem**, the query audit log
  (`get_query_log`, `get_error_log`), the structured error taxonomy, the metrics
  endpoint and the compatibility manager.

### Fixed

Defects of 2.x that this rewrite removes by construction rather than by patch —
each was reported against 2.x and none of the code carrying it survives:

- **Console logging wrote to stdout, corrupting the MCP JSON-RPC channel**
  ([#35](https://github.com/ChrisGVE/localdata-mcp/issues/35),
  [#39](https://github.com/ChrisGVE/localdata-mcp/issues/39),
  [#41](https://github.com/ChrisGVE/localdata-mcp/issues/41)). This server writes
  nothing to stdout but JSON-RPC: there is no logging configuration, no log
  destination to point at stdout, and no `print` in the package.
- **`optimize_constrained` evaluated LLM-supplied strings through `eval()` behind
  a defeatable sandbox — a host RCE**
  ([#42](https://github.com/ChrisGVE/localdata-mcp/issues/42)). The tool is gone,
  and nothing in this server evaluates a caller's string as code.
- **The SELECT-only gate was SQLite-shaped and other dialects walked around it**
  ([#36](https://github.com/ChrisGVE/localdata-mcp/issues/36)). Enforcement is now
  the connection's, per backend, as far as each backend can enforce it — with
  Oracle's inability to refuse DDL stated in the refusal rather than papered over.
- **Ten of twelve analytical tools crashed on their documented happy path**
  ([#40](https://github.com/ChrisGVE/localdata-mcp/issues/40)). Those tools no
  longer exist.

And forty-five found in this release's own code before it shipped:

- **Two sheets whose names cleaned up to the same table name lost one of them
  without a word** ([#91](https://github.com/ChrisGVE/localdata-mcp/issues/91)).
  `Q2 Prices` and `Q2-Prices` are both `q2_prices` once made safe for SQL, so the
  second sheet overwrote the first, and `attach` reported two loads both showing
  the survivor's columns. Every sheet now lands as its own table — the later one
  as `q2_prices_1` — by the same rule that already kept repeated column headers
  apart, and the `.numbers` sheet qualifier is decided on the cleaned-up name
  rather than the raw one. Each `loaded` entry names the `sheet` it came from,
  and a warning names the sheets and the tables they became. That rule had a
  hole of its own, closed with it: a header row of `a, A, a_1` made `a_1` twice
  and the file could not be attached; it now arrives as `a, a_1, a_1_1`.
- **`query` on a DuckDB datasource read and wrote files outside the allowed
  paths** ([#105](https://github.com/ChrisGVE/localdata-mcp/issues/105)), even on a
  file attached read-only. `read_text('/etc/hosts')` and `read_csv` returned
  files from anywhere on disk, and `COPY (…) TO '/tmp/x.csv'` wrote one.
  `access_mode=read_only` covers only DuckDB's own database, and DuckDB opens
  those files itself without asking the path check. Every DuckDB connection now
  has `enable_external_access` switched off and its configuration locked, so a
  statement reaches only the database it was attached as. A second file still
  arrives through `create(type="table")`, and a result is still written with
  `query(path=…)`.
- **A short password corrupted the refusal it was masked out of**
  ([#82](https://github.com/ChrisGVE/localdata-mcp/issues/82)). The password was
  removed by replacing every occurrence of its characters, so password `e` turned
  a driver's "connection refused" into "conn***ction r***fus***d", and the shape
  of the damage gave away the password's length and letters. A password is now
  masked where it sits as a credential — in a URL's userinfo, or under the
  `password=`/`PWD=` key a driver echoes when a connection fails — and nowhere
  else, in its percent-encoded and decoded spellings both. One masker does it,
  and every refusal passes through it on the way out, so a URL that a verb read
  as a path (`query(path=…)`, `save(path=…)`) no longer comes back with its
  password in the refusal either — which since #108 names it as a URL.
- **`create` read a database URL as a path, and echoed its password**
  ([#88](https://github.com/ChrisGVE/localdata-mcp/issues/88)).
  `create(source="postgresql://u:hunter2@localhost:5432/db")` answered "No such
  file: <cwd>/postgresql:/u:hunter2@localhost:5432/db" — the password in clear,
  about a file nobody named. `create` reads a file, so a URL is now refused as a
  URL, shown the way `attach` shows it (`u:***`), with the route that does work:
  `attach` the database, write the rows wanted to a file with `query(path=…)`,
  and `create` that file.
- **`query(path=…)` and `save` read a database URL as a missing directory**
  ([#108](https://github.com/ChrisGVE/localdata-mcp/issues/108)). The write side
  of #88: `save(path="postgresql://u:hunter2@localhost:5432/db.sqlite")`
  answered "Directory does not exist: <cwd>/postgresql:/u:***@localhost:5432",
  about a directory nobody named. A destination is a file, so a URL is now
  refused as a URL before it is taken as a path — on `query` before the suffix
  is even looked at — shown the way `attach` shows it, with the verb that does
  put rows into a database: `create`, on that database attached writable.
- **Every failed statement on Exasol answered with an empty error**
  ([#117](https://github.com/ChrisGVE/localdata-mcp/issues/117)). A syntax error,
  `COMMIT; INSERT …` and a missing table all came back `{"ok": false, "error":
  ""}`, and a wrong password on `attach` stopped at the driver's class name. The
  Exasol DBAPI driver raises its `Error` with no words and keeps the server's
  reason on `__cause__`. A driver error with nothing of its own to say now takes
  its words from what it chained — the reason, not `pyexasol`'s block of DSN,
  user and query around it — in `query`'s answer, in every `Could not …`
  refusal of `create`, `update`, `drop` and `save`, and in a failed `attach`.
  Every other backend's wording is unchanged.
- **A date from a typed format compared wrongly against a date literal, with no
  warning** ([#87](https://github.com/ChrisGVE/localdata-mcp/issues/87)). A
  date or timestamp column from `.parquet`, `.feather`, `.orc`, a workbook or
  YAML loaded as integer nanoseconds since the epoch, so
  `WHERE seen > '2024-03-02'` returned zero rows with `ok: true` while the same
  frame written to a CSV returned the right row. On PostgreSQL and MySQL the
  ticks did not fit the column and `create` was refused outright. Such a column
  is now written in the CSV path's canonical UTC spelling and reported as
  `{"type": "TEXT", "temporal": "iso8601_utc", "normalized": "UTC"}`. A tz-aware
  value is converted to UTC rather than shifted, and a missing one is `NULL`.
  The description now also survives `query(path=…)` and a second `attach`.
  Durations and times of day are not instants and were fixed separately
  ([#109](https://github.com/ChrisGVE/localdata-mcp/issues/109)).
- **`detach` named the tables a slot arrived with, not the ones it held**
  ([#90](https://github.com/ChrisGVE/localdata-mcp/issues/90)). After a
  `create` and a `drop`, `detach` left out the table just added and listed the
  one just dropped, one call after `drop` had reported it gone. The tables it
  left out were the ones its own description warns are lost on detach. It now
  reads the tables from the database as it closes, the way `directory` does.
- **`attach` could not tell a datasource `save` takes from one it refuses**
  ([#81](https://github.com/ChrisGVE/localdata-mcp/issues/81)). A SQLite file
  and a DuckDB file both came back `"kind": "database"`, nothing in the answer
  named the engine, and only calling `save` showed which one would work.
  `attach` and `directory` now report `backend`, naming the engine (`sqlite`,
  `duckdb`, `postgresql` …), and `saveable`, which is `true` exactly where `save`
  writes the datasource out. Both are new keys; `kind` is unchanged.
- **A DuckDB datasource failed every verb but `query` when its two connections
  disagreed** ([#79](https://github.com/ChrisGVE/localdata-mcp/issues/79)). A
  slot opens a read and a write connection, and DuckDB refuses a second
  connection to one file under a different configuration. A DuckDB file attached
  `writable=true` therefore failed `directory`, `stats`, `create`, `update` and
  `drop`, and a `duckdb:///` URL attached read-only failed `directory` and
  `stats`. Both connections now share one configuration: `access_mode=read_only`
  on both for a read-only datasource, on neither for a writable one. `query` on
  a writable DuckDB datasource still only reads — every transaction it runs is
  `READ ONLY`, so DuckDB refuses an `INSERT … RETURNING` instead of answering
  with a row it then discards, and a statement that would begin, commit or roll
  back a transaction of its own is refused, since that is the one way out of a
  read-only transaction.
- **A time of day from a typed format compared wrongly against a time literal,
  and a duration claimed to count from the epoch**
  ([#109](https://github.com/ChrisGVE/localdata-mcp/issues/109)). A time-of-day
  column from `.parquet`, `.feather`, `.xlsx` or `.ods` loaded as integer
  nanoseconds since midnight, described as
  `"temporal": "timestamp", "unit": "nanoseconds_since_epoch"`, so
  `WHERE clock > '10:00:00'` returned zero rows with `ok: true` while a CSV
  holding `14:30:00` returned the right row. It is now written as the text the
  CSV holds — `HH:MM:SS`, with `.ffffff` only where the value has a fraction —
  and described `{"type": "TEXT", "temporal": "iso8601_time"}`, the description
  any text column in that spelling now gets, a CSV's included, so it survives
  `query(path=…)` and a second `attach`. A duration stays integer nanoseconds,
  which sum, average and order correctly, and is described
  `"temporal": "duration", "unit": "nanoseconds"` instead of counting from an
  epoch it has nothing to do with.
- **`query` on a writable DuckDB datasource refused a write in DuckDB's words**
  ([#116](https://github.com/ChrisGVE/localdata-mcp/issues/116)). An `INSERT`,
  `UPDATE`, `CREATE` or any other write was refused, and the file kept its rows,
  but the answer was *"TransactionContext Error: Cannot write to database … -
  transaction is launched in read-only mode"*, which speaks of a transaction the
  caller never opened and names no verb to use instead. It now reads *"query
  reads; it does not write. This statement asks to INSERT …"* and points to
  `create`, `update` and `drop`, as the refusal of `COMMIT; INSERT …` already
  did. A DuckDB datasource attached read-only still refuses a write in DuckDB's
  own words.
- **A typed date or time sharing a column with other values was stored as its
  tick count, written as text**
  ([#114](https://github.com/ChrisGVE/localdata-mcp/issues/114)). A YAML
  `seen: 2024-03-01` beside `seen: unknown`, or a workbook column holding a time
  cell and a datetime cell, is not one temporal kind, so the column is text, and
  the date in it was stored as `'1709251200000000000'` with `ok: true`. Each date
  or time in such a column is now written in its kind's spelling: `2024-03-01`,
  `2024-03-01T13:30:00Z` for an instant with a time of day, and `14:30:00` for a
  time. A filter such as `WHERE seen >= '2024-03-01'` then finds it, as it does
  in a CSV holding the same text.
- **A column of integers too wide for 64 bits was refused, though it is text**
  ([#72](https://github.com/ChrisGVE/localdata-mcp/issues/72)). A CSV column
  holding `99999999999999999999` beside ordinary integers was declared `TEXT`
  and then refused with *"integer wider than the 64-bit range"*, while the same
  value beside one decimal number loaded fine. It now loads as the digits the
  file held, and `attach` and `create` warn that the column is text: `sum()` and
  `avg()` over it convert each value to a float and lose its last digits, and
  `ORDER BY`, `min()` and `max()` compare the digits as text. An unsigned 64-bit
  value above the signed range, in a column declared `INTEGER`, is still
  refused.
- **On Firebird a text column grouped by storage handle, and then one accented
  letter failed the load** ([#56](https://github.com/ChrisGVE/localdata-mcp/issues/56)).
  Core's `Text` is a BLOB there, and `GROUP BY department` over five rows
  answered five groups with no error. A text column is now a `VARCHAR` sized
  from the widest value — declared `CHARACTER SET UTF8`, because without it the
  width counted bytes on a charset-`NONE` database and `été`, three characters
  and five bytes, was refused by the column sized for it. A column wider than
  8,191 characters stays a BLOB, keeping every value whole, and `create` now
  warns for that column that `GROUP BY`, `DISTINCT` and `ORDER BY` on it answer
  wrongly and that it cannot be indexed.
- **On SQL Server a CJK character or an emoji was stored as `?`, and the load said
  nothing** ([#128](https://github.com/ChrisGVE/localdata-mcp/issues/128)). A text
  column landed as `VARCHAR(n)`, which holds only the collation's code page:
  under the default `SQL_Latin1_General_CP1_CI_AS`, a CSV holding `été`, `日本`
  and `😀` loaded without an error and read back `été`, `??` and `??`. Under a
  UTF-8 collation the same column refused `été` instead, because there the width
  counts bytes. A text column is now `NVARCHAR(n)`, which holds any character
  under any collation, with `n` counted in UTF-16 code units, so an emoji counts
  two; past 4,000 it is `NVARCHAR(max)`, where it had been `VARCHAR(max)` past
  8,000. What it costs: a nonclustered index key holds 1,700 bytes, which is 850
  code units of `NVARCHAR` where it was 1,700 code-page characters of `VARCHAR`,
  and SQL Server refuses an index over a longer value with that limit in the
  error; and under a `SQL_` collation `ORDER BY` follows the Unicode rules,
  which pass over a hyphen or an apostrophe, where the code-page rules sorted
  punctuation first. A statement puts punctuation first again with
  `ORDER BY v COLLATE Latin1_General_BIN2`, which loses nothing but is
  code-point order and so tells `Ab` from `ab`, or with
  `ORDER BY CAST(v AS VARCHAR(n))`, which is the old order exactly and turns any
  character the code page lacks into `?`.
- **An instant cut to the microsecond was cut silently**
  ([#111](https://github.com/ChrisGVE/localdata-mcp/issues/111)). A date column
  is stored with six fractional digits, so two instants a nanosecond apart — from
  a nine-digit CSV value or a Parquet, Feather or ORC `timestamp[ns]` column —
  became one value, and `count(DISTINCT …)`, an equality join and `ORDER BY`
  answered as if they were, with nothing to say so. The cut stays, since
  PostgreSQL `timestamp`, MySQL `DATETIME(6)` and DuckDB `TIMESTAMP` hold six
  digits too, but `attach` and `create` now warn, naming each column it cut. A
  column whose only fraction was below the microsecond is written to the whole
  second, as the same instants from a file holding them to the second are.
- **The MCP registry manifest named a release that never existed**
  ([#78](https://github.com/ChrisGVE/localdata-mcp/issues/78)). `server.json`
  gave `2.1.0` as the PyPI version a registry client should install, and PyPI has
  no such release. It now carries the version `pyproject.toml` declares, and a
  test holds the two together and validates the manifest against the registry
  schema it names, which caps the description at 100 characters.
- **On PostgreSQL, `query` ran functions whose effects a rollback does not undo**
  ([#124](https://github.com/ChrisGVE/localdata-mcp/issues/124)). On a read-only
  attach, `lo_export` wrote a file on the server, `pg_reload_conf` reloaded its
  configuration, and `pg_advisory_lock` left a lock on the pooled connection;
  CockroachDB ran `setval` and the sequence stayed moved, and YugabyteDB and
  openGauss had the same hole. The reader now opens a read-only session and asks
  the server's own catalog what a read may not call: a server function that can
  act past the transaction, any function nobody installed as an extension, and a
  view, row-security policy or operator that calls one. A statement naming one,
  holding more than one statement, or spelling a name with `U&"…"` escapes is
  refused before it runs. YugabyteDB and openGauss get the gate under their own
  names.
- **On openGauss, a read reaching a user-defined cast moved a sequence**
  ([#125](https://github.com/ChrisGVE/localdata-mcp/issues/125)). A cast runs
  its function without the statement naming it — `1::money_c`, or `c + 1` over
  a column of the type — and openGauss's read-only session runs `nextval`, which
  PostgreSQL's refuses. The reader now also refuses a type that a cast converts
  through such a function, every type built from it (its arrays, domains and
  composites), and every table or view with a column of one, by name.
- **On openGauss, a read reaching a domain's `CHECK` moved a sequence**
  ([#126](https://github.com/ChrisGVE/localdata-mcp/issues/126)). `SELECT 1::d`
  runs the domain's check, and a check calling a function that calls `nextval`
  moved the sequence with neither name in the statement. The reader now also
  reads each domain's checks and refuses, by name, a domain whose check reaches
  something it refuses, with every type built from it and every table or view
  with a column of one.
- **On openGauss, a cast between two of the server's own types moved a
  sequence** ([#129](https://github.com/ChrisGVE/localdata-mcp/issues/129)).
  After an administrator's `CREATE CAST (INT AS BYTEA) WITH FUNCTION f(INT)`,
  `SELECT 1::bytea` names only `bytea`, and an implicit cast like it names
  nothing, so no word in a statement can be refused for it. The reader now fails
  closed: on a PostgreSQL, YugabyteDB, openGauss or CockroachDB datasource whose
  server holds such a cast through a function the gate refuses, **every**
  statement is refused (`attach` included), and the refusal names each cast and
  the function, and gives the `DROP CAST` an administrator must run. Dropping it
  restores reads. On openGauss 7.0.0-RC3 none of the server's own casts is like
  this, so a fresh server reads as before. The whole family is covered, not
  openGauss alone, because PostgreSQL's read-only session stops `nextval` but
  not `lo_export` or `pg_advisory_lock`.
- **The openGauss run of the domain-`CHECK` read-gate test left its domain and
  function behind** ([#130](https://github.com/ChrisGVE/localdata-mcp/issues/130)).
  openGauss refuses `DROP DOMAIN`, so the cleanup now falls back to `DROP TYPE`.
- **On SQL Server, a write sent to `query` after a `COMMIT` was kept**
  ([#122](https://github.com/ChrisGVE/localdata-mcp/issues/122)).
  `COMMIT; INSERT …` was refused, but the row stayed: pyodbc sends the string as
  one batch, SQL Server runs every statement in it, and the `COMMIT` ends the
  transaction the reader never commits, so the `INSERT` committed on its own.
  T-SQL needs no `;` between statements either, so `SELECT 1 SELECT 2` answered
  `ok`. SQL Server has no read-only session, so the reader now compiles each
  statement with `SHOWPLAN_XML` on, which plans without running anything, and
  runs it only if the plan holds exactly one `SELECT` and no remote operator. The
  cost is a read through a linked server, which is refused.
- **On MonetDB, `query` answered `ok` with rows from a write it then rolled
  back** ([#119](https://github.com/ChrisGVE/localdata-mcp/issues/119)).
  `INSERT … RETURNING`, the same under `WITH`, and `SELECT 1; INSERT …` all came
  back with rows, though nothing was kept. MonetDB has no read-only transaction,
  so the reader now asks MonetDB's own `EXPLAIN` what a statement is before
  running it, the way SQL Server's reader asks its plan. It runs only a single
  statement whose plan holds read operators alone and which calls no function
  that is not a side-effect-free system function. The last part matters because
  MonetDB records a SQL function whose body is an `INSERT` as having no side
  effects. The cost is a read with a `;` inside it, or one calling a
  user-defined function: both are refused. The `;` refusal says it fires inside
  a quoted string too, and offers `code(59)` for a `;` in the data:
  `SELECT 'a' || code(59) || 'b'` returns `a;b` (MonetDB has no `chr`, and
  `char(59)` does not parse there).
- **On PostgreSQL, CockroachDB, YugabyteDB, openGauss, MySQL and MariaDB, a `;`
  in a literal was refused as "more than one statement"**
  ([#127](https://github.com/ChrisGVE/localdata-mcp/issues/127)). These readers
  refuse any `;` but a trailing one without parsing SQL, so `SELECT 'a;b'`, one
  statement, was refused with words that were false for it and left no way
  forward. The rule is unchanged. The refusal now names the `;`, says one inside
  a quoted string or a comment is refused too and why, and offers a rephrase run
  through `query` on every engine it is offered on: `chr(59)` on the
  PostgreSQL family (`SELECT 'a' || chr(59) || 'b'` returns `a;b`), and
  `CHAR(59 USING utf8mb4)` on MySQL and MariaDB
  (`SELECT CONCAT('a', CHAR(59 USING utf8mb4), 'b')`). Without `USING`,
  `CHAR(59)` is a binary string there and comes back as `0x3b`.
- **On MySQL and MariaDB, a read-only `query` could write a file and change
  server settings, then report the statement as refused**
  ([#123](https://github.com/ChrisGVE/localdata-mcp/issues/123)). A read-only
  transaction refuses row changes and DDL, nothing else: `SELECT … INTO OUTFILE`
  and `INTO DUMPFILE`, `SET GLOBAL`, `SET PERSIST`, an executable comment
  (`/*! … */`, MariaDB's `/*M! … */`), and a stored function called from a
  `SELECT` all ran first. MySQL's `EXPLAIN` cannot say what such a statement does,
  so every statement on the reader is now checked before it is sent. It must be
  one statement, hold no executable comment and neither word `OUTFILE` nor
  `DUMPFILE`, and begin with `SELECT`, `WITH`, `TABLE`, `VALUES`, `SHOW`,
  `DESCRIBE` or `EXPLAIN`. It must also name no stored or loadable function, and
  no view that calls one. The cost is a read whose text contains one of those
  words or names, even inside a quoted string; the refusal says which. Such a
  view is still listed by `directory`, with `rows: null` and a warning, because
  counting it would run the function.
- **On DuckDB, a read-only `query` could overwrite a pending WAL and lose
  committed rows** ([#137](https://github.com/ChrisGVE/localdata-mcp/issues/137)).
  DuckDB exempts the database's own `.wal` from the switched-off file access, so
  `COPY (SELECT 'g') TO '<file>.wal' (USE_TMP_FILE false)` went through, on a file
  attached read-only too. That was accepted in
  [#106](https://github.com/ChrisGVE/localdata-mcp/issues/106) because the write
  stayed inside the database's own files. It is not harmless when a writer died
  before its checkpoint: the WAL still held a committed row, the `COPY` answered
  `ok` and replaced it, and the next open of the file had lost the row without an
  error. `EXPLAIN ANALYZE` of the same `COPY` wrote the WAL as well. The DuckDB
  reader now asks DuckDB's own parser (`extract_statements`) what a statement is
  before running it, and runs it only if it is exactly one query, or an `EXPLAIN`
  of one. Every spelling of a read that answered before still answers: `WITH`,
  `FROM t`, `SUMMARIZE`, `DESCRIBE`, `SHOW`, `CALL`, `EXPLAIN [ANALYZE] SELECT`.
  A `PRAGMA` that only reads, such as `PRAGMA show_tables` or
  `PRAGMA table_info('t')`, answers on DuckDB 1.5 and is refused on 2.0, whose
  parser calls it a `PRAGMA` where 1.5's called it a `SELECT`; the refusal names
  the same read asked as a query: `SHOW TABLES`, `SHOW ALL TABLES`,
  `DESCRIBE t`, `SELECT * FROM pragma_table_info('t')`, `pragma_version()`,
  `pragma_database_size()`. `LOAD`, `SET VARIABLE`, `PREPARE`, `EXECUTE`,
  `VACUUM` and `ANALYZE` answered `ok` before and are refused now, as is more than
  one statement in a call. The refusal names what DuckDB read the statement as
  (`docs/CONSTRAINTS.md` §29.4, amendment of 2026-11-03).

- **A database name containing `?` or `@` now survives a URL.** SQLAlchemy 2.0
  rendered the database component of a URL raw, so `a?b` was read back as `a`,
  and `a@b` without a password lost its database and took `b` as the host. The
  project now pins SQLAlchemy 2.1.6, which quotes the component on the way out
  and unquotes it on the way back
  ([#70](https://github.com/ChrisGVE/localdata-mcp/issues/70)).
- **DuckDB datasources reflect their columns under SQLAlchemy 2.1.** SQLAlchemy
  2.1 changed two PostgreSQL internals that `duckdb-engine` 0.17.0 inherits, so
  `describe` and every column read failed on DuckDB. Those two are adapted on
  DuckDB's dialect instance (see `dialects.py`); the rest of the reflection is
  SQLAlchemy's own.
- **A SQLite file whose name contains `?` opens under SQLAlchemy 2.1.** The
  file's URI was rebuilt from a string, and 2.1 unquoted it, so `why? not.db`
  opened as `why`. A file is now handed to the engine as a `URL` object.

- **`stats` reports the range and average of every floating-point column.**
  SQLAlchemy 2.1 stopped making `Float` a subclass of `Numeric`, so `REAL`,
  `FLOAT` and `DOUBLE PRECISION` columns lost their `min`, `max` and `avg`
  silently. The classifier now lists `Float` beside `Numeric` and `Integer`
  ([#70](https://github.com/ChrisGVE/localdata-mcp/issues/70)).
- **On SQLAlchemy 2.0, a URL whose database name holds `?` or `@` is refused
  instead of opening a different database.** The 2.1 pin lives in
  `[tool.uv] override-dependencies`, which never reaches the published package,
  so `pip install localdata-mcp[cockroachdb]` resolves to SQLAlchemy 2.0.55.
  There `postgresql://…/a%3Fb` reached the driver as the database `a%3Fb`, and
  `sqlite:///…/a%3Fb.db` opened a file literally called `a%3Fb.db` rather than
  `a?b.db`, with nothing said. `attach` now refuses such a URL on any
  SQLAlchemy older than 2.1, before the network gate, naming the database, the
  installed version and the remedy: SQLAlchemy 2.1 or later, which the
  `cockroachdb` extra holds back. A file reached by its path rather than a URL
  is unaffected and still attaches on 2.0. On 2.1 nothing changes
  ([#140](https://github.com/ChrisGVE/localdata-mcp/issues/140)).
- **On openGauss under Linux x86_64, a write refused through `query` is explained
  as a refused write again**, naming `create`, `update` and `drop`, instead of
  arriving as the server's `syntax error at or near "INSERT"`. The write was
  always refused; what depended on the machine was the explanation. The check
  looked for `DECLARE` in the server's error excerpt, which openGauss cuts from
  the left, and psycopg2's cursor name carries `id()`: 12 hex digits on Linux
  x86_64 against 9 on macOS, enough to push `DECLARE` out of the window. It now
  reads the end of the declaration, `CURSOR … FOR <verb>`, which sits next to
  the error and so stays in the excerpt wherever it is cut. A malformed `SELECT`
  is still reported in the server's own words
  ([#141](https://github.com/ChrisGVE/localdata-mcp/issues/141)).
- **On PostgreSQL, CockroachDB and YugabyteDB, `SELECT SELECT 1` is reported as
  the syntax error it is**, not as a refused write. The server objects to the
  second `SELECT`, but its message names `"SELECT"`, the statement's first word,
  beside the cursor's `FOR SELECT`, which is how a refused write looks. A
  statement that opens with `SELECT`, `VALUES`, `TABLE` or `WITH` is now never
  explained as a write, since a cursor is declared over exactly those. A
  misspelled first word such as `SELEC 1` was still explained as a refused write;
  that is the next entry.
- **On the PostgreSQL family, a misspelled first word is reported as the syntax
  error it is** ([#143](https://github.com/ChrisGVE/localdata-mcp/issues/143)).
  `SELEC 1` arrived as `syntax error at or near "SELEC"` beside the cursor's
  `CURSOR FOR SELEC`, the same words a refused `INSERT` arrives in, and was
  answered with `create`, `update` and `drop`. Before calling a statement a
  refused kind, the reader now asks the server to parse it: on PostgreSQL,
  CockroachDB and YugabyteDB with the protocol's Parse step, which runs
  nothing, so a statement that does not parse is answered with the server's own
  syntax error, without the cursor declaration. openGauss is reached through
  psycopg 2, which has no Parse step of its own; there a first word that is no
  keyword of the server's grammar is answered with the server's message. A
  keyword that cannot begin a statement (`FROM t`) was still read as a refused
  kind there; that is the next entry.
- **On openGauss, a keyword that cannot begin a statement is reported as the
  syntax error it is** ([#143](https://github.com/ChrisGVE/localdata-mcp/issues/143)).
  `FROM t` and `WHERE a = 1` were answered "it is not a read" with `create`,
  `update` and `drop`, because asking whether the first word was a keyword
  could not tell `FROM` from `CREATE`. openGauss is now asked the same Parse
  step as the other three engines, through the libpq its driver loaded, so
  every statement that does not parse there is answered with the server's own
  syntax error, without the cursor declaration, as on PostgreSQL, CockroachDB
  and YugabyteDB.
- **A refused locking read was explained as a statement that returns no rows**
  ([#144](https://github.com/ChrisGVE/localdata-mcp/issues/144)).
  `SELECT * FROM t FOR UPDATE` on the PostgreSQL family was answered "That
  statement returns no rows, so it is not a read", with advice to use `create`,
  `update` or `drop`. The read-only session refuses the clause with the code it
  refuses a write with, but the statement reads, and the way forward is to drop
  the clause. The refusal now names the clause, says it asks for row locks this
  reader does not take, and ends with the same statement without it, which reads
  the same rows. Measured on every engine with a locking clause: the PostgreSQL
  family refuses all four strengths, MySQL and MariaDB refuse `FOR UPDATE`, and
  Firebird refuses `WITH LOCK`; all are now explained this way. MySQL's
  `FOR SHARE` and `LOCK IN SHARE MODE`, Oracle's `FOR UPDATE` and SQL Server's
  lock hints are not refused, and run.
- **A write the database refused for a collision with another session said
  nothing about what had happened** ([#47](https://github.com/ChrisGVE/localdata-mcp/issues/47)).
  On YugabyteDB, a second session changing the same table made `drop` and
  `create(type='index')` fail with SQLSTATE 40001, and the answer was the
  driver's text alone: an exception class, a hybrid timestamp and a key dump.
  The refusal from `create(type='index')`, `update` and `drop` now says that
  another session's transaction collided with this one, that nothing was
  changed (naming what is still as it was), and that the same call goes
  through when sent again once the other session is done. The driver's text
  follows. Measured: nothing had been applied on any refusal, and every
  re-send succeeded. The server still sends a write once and never resends
  it by itself. Reading a table's columns before an index is built is a
  catalog read and is now run again once on 40001, like every catalog read;
  a 40001 that persists is reported as a failed read, where it used to be
  reported as "No such table" for a table that exists.
- **On MySQL and MariaDB, `query` read files on the database server through
  `LOAD_FILE()`** ([#36](https://github.com/ChrisGVE/localdata-mcp/issues/36)).
  It is a read, so the read-only session let it run, and as a built-in it is in
  no catalog the #123 gate asks. Whatever the account behind the URL could read
  (its `FILE` privilege, under the server's `secure_file_priv`) a `query` caller
  could read too. The reader now refuses a call to it before sending, in any
  case, in a subquery or an expression, backquoted, and with a comment between
  the name and its parenthesis; a view that calls it is refused by its name,
  like a view calling a stored function. A column, an alias or a literal that
  only holds the word is still answered. To read a file of your own, `attach` it.
- **A read refused for how it is written was explained as a refused write**
  ([#135](https://github.com/ChrisGVE/localdata-mcp/issues/135)). `SELECT 'a;b'`,
  a `U&"…"` name on the PostgreSQL family, and a read holding a refused word only
  inside a quoted string (`SELECT 'outfile'`, `SELECT 'set_config'`) came back as
  "query reads; it does not write", with advice to use `create`, `update` or
  `drop`. They are reads, and the advice sent the caller to the wrong verb. The
  rules are unchanged and still fail closed. Their refusal now opens `query did
  not run this statement`, says it is a rule about how a statement is written that
  a read meets too, names what it met, and gives a rephrase the gate admits: the
  literal in two halves (`'set_c' || 'onfig'` on the PostgreSQL family,
  `CONCAT('out', 'file')` on MySQL and MariaDB), the plain spelling of a `U&"…"`
  name, and the `;` forms from #127. A refused word or symbol outside a quoted
  string may be a call, and keeps the refused-write wording. MonetDB's `;` is
  framed the same way.
- **On DuckDB, `query` answered `CALL force_checkpoint()` in DuckDB's transaction
  words** ([#138](https://github.com/ChrisGVE/localdata-mcp/issues/138)).
  DuckDB refuses it inside the reader's read-only transaction with "Cannot FORCE
  CHECKPOINT: the current transaction has been started", which talks about a
  transaction to an agent that never opened one. Nothing was written. The reader
  now finds the call with DuckDB's own tokenizer, quoted, qualified or behind a
  comment, and refuses it before DuckDB is asked, saying that DuckDB runs it only
  outside a transaction and that `CHECKPOINT` and `FORCE CHECKPOINT` run here,
  both measured through `query`. A literal or a column holding the word is still
  a read.
- **`stats` answered a reader-gate refusal in the driver's raw words**
  ([#133](https://github.com/ChrisGVE/localdata-mcp/issues/133)). Where the gate
  refuses every read of a table, a view calling an unsafe function, or every
  statement on an openGauss server holding a cast between two of its own types,
  `stats` came back as `ProgrammingError: (psycopg2.ProgrammingError) query may
  not …`, with the SQL it had built and a SQLAlchemy link: the wrong verb's name,
  a statement the caller never wrote, and nothing to act on. It now says `stats
  did not profile <table>: reading it would …`, with the gate's own reason, as
  `directory` does for a count, and any other failure of its aggregate in the
  driver's words alone.
- **On Oracle a text column of fewer than 4,000 characters but more than 4,000
  bytes was refused on insert** ([#132](https://github.com/ChrisGVE/localdata-mcp/issues/132)).
  A text column lands as `VARCHAR2(n CHAR)`, sized in characters, and became a
  `CLOB` only past 4,000 characters; but the 4,000 ceiling of a database without
  extended string sizes counts bytes whatever the length semantics, so 1,400 CJK
  ideographs (4,200 bytes in UTF-8) were refused by the `VARCHAR2(1400 CHAR)`
  they were given. The widest value is now measured in UTF-8 bytes as well, and
  a column past 4,000 bytes is a `CLOB`, which holds it whole. Measured on
  Oracle Free: `été`, `日本` and `😀` at exactly their column's width read back
  byte for byte under the session's `BYTE` semantics, before and after, because
  SQLAlchemy writes `CHAR` into the declaration — the defect #132 predicted
  there was not there.
- **A formula column in a workbook a program wrote loaded as all NULL, silently**
  (task 20). pandas reads an `.xlsx` through openpyxl with `data_only=True`, which
  returns the result a spreadsheet application stored beside each formula; a
  workbook written by openpyxl, this server's own writer or most export
  pipelines stores the formula and no result, so every such cell read as NULL
  and nothing said why. `attach` now warns, per sheet, naming each column under
  the name it landed as and its spreadsheet letter, how many cells have no
  result, and the first formula, so the column can be rebuilt in SQL from the
  columns it is computed from. Formulas are still not evaluated. `.xlsx`,
  `.xlsm` and `.ods`: an OpenDocument sheet written by a program, odfpy for
  one, loses the value the same way and gets the same warning, with repeated
  rows and columns counted at the positions they cover. `.xls` is not flagged
  (see *Known limitations*).
- **A second `scripts/endpoint-run.sh` no longer loses its runner to the first**
  ([#142](https://github.com/ChrisGVE/localdata-mcp/issues/142)). Both
  invocations tagged the runner `localdata-test/runner:local` and built it
  before taking the lock, so the run that finished first removed the image of
  the run still waiting, which then failed at `docker run` with exit 125. The
  end-of-run pass also pruned every build cache record new since the run
  started, the other run's included. Each run now builds its runner under the
  lock with a tag of its own, records the cache its own builds created, and
  removes only that, before it releases the lock.
- **A `create(type="table")` that fails while inserting and leaves its table
  behind now says so** ([#147](https://github.com/ChrisGVE/localdata-mcp/issues/147)).
  The table is made and filled in one transaction, on the premise that a
  failed insert takes the `CREATE TABLE` with it. Measured, it does not on
  SQLite, where the driver commits the `CREATE` before the first insert, nor on
  YugabyteDB 2.25, where it survives the abort, whether the insert failed on a
  value no 64-bit column holds or on a collision with another session (SQLSTATE
  40001), nor on CockroachDB 26.3. The refusal said only why the insert failed, so
  the same call sent again was refused with "already exists, holding 0 rows".
  After a failed create the server now asks the database, on a fresh
  connection, whether the table is there; if it is, the refusal names it with
  the rows it holds and ends with the `drop` call that clears it. The table is
  not dropped for you: another session can write into it while it is being
  filled (measured on YugabyteDB), so a drop could destroy rows that are not
  this server's.
- **`rows` in a successful `create(type="table")` answer is the number of rows
  the call sent.** It used to be a `COUNT(*)` read after the commit, so it
  counted other sessions' rows too: on YugabyteDB the new table is visible to
  other sessions while it fills, one inserted 462 rows, and a 50,000-row create
  answered `rows: 50462`. The table is still counted afterwards, because that is
  the only check that every row sent landed. Where the two differ, the answer
  carries a warning giving both numbers and the `query` that counts what the
  table holds now. `directory` still reports the table's own count.
- **`directory` no longer lists the placeholder CockroachDB keeps beside an
  index it is still building** ([#148](https://github.com/ChrisGVE/localdata-mcp/issues/148)).
  While an index is being built, CockroachDB 26.3 builds it through a temporary
  index over the same columns, named `crdb_internal_index_<n>_name_placeholder`,
  and reflection reported both. `directory` listed an index nobody made and
  nobody can drop, and `create(type="index")` sent again for the same columns
  answered "already indexed on salary, by crdb_internal_index_3_name_placeholder"
  rather than naming the index being built. The placeholder is now left out;
  the index being built is still listed.
- **On CockroachDB a write answers within 30 seconds, and says what it left**
  ([#148](https://github.com/ChrisGVE/localdata-mcp/issues/148)). A schema
  change there waits, inside the statement, until every other session's open
  transaction on the table has ended; a plain `SELECT` left open is enough.
  `create(type="index")`, `update` and `drop` used to answer only once that
  transaction ended, and since every call runs under the server's one lock, no
  other call on any datasource answered in the meantime. Statements on the
  write engine now carry a 30 second `statement_timeout`, and a write cut off
  by it says what it waited on, what is in place, and what to do next.
  Measured on CockroachDB 26.3: the rename and either drop are already made
  when the bound fires, so the answer says so and not to send it again. An
  index is not usable yet: CockroachDB keeps building it after the call, a
  second `create` is refused while it exists, and it is usable once the other
  transaction has ended. `directory` shows each of these.
- **A `.json`, `.jsonl` or `.ndjson` file starting with a UTF-8 byte-order mark
  attaches** ([#154](https://github.com/ChrisGVE/localdata-mcp/issues/154)).
  Windows editors, Excel and PowerShell write one, and the JSON readers refused
  the file with Python's codec message ("Unexpected UTF-8 BOM (decode using
  utf-8-sig)"), which says nothing an agent can act on. The mark is now skipped,
  as RFC 8259 §8.1 allows and as `.csv`, `.tsv`, `.yaml` and `.xml` already did,
  and the first column's name carries no trace of it. Only a mark at the very
  start of the file is skipped: one at the start of a later JSON Lines record is
  stray data and is still refused by line number.

### Known limitations

- **On CockroachDB, a write held up by another session's open transaction
  holds every other call for up to 30 seconds**
  ([#148](https://github.com/ChrisGVE/localdata-mcp/issues/148)). The bound
  cuts the wait off, but the server's one lock is held while it runs; letting
  other datasources answer meanwhile is
  [#150](https://github.com/ChrisGVE/localdata-mcp/issues/150). A `query` on a
  table another session has written to and not committed is not bounded, and
  waits for that transaction as before.
- **On CockroachDB, `create(type="index")` over a large table answers at 30
  seconds even with no other session connected**
  ([#148](https://github.com/ChrisGVE/localdata-mcp/issues/148)). The bound
  covers the whole statement, and an index over 2,000,000 rows took 53 seconds
  to build. The answer is the same as for a wait: the index is still being
  built, `directory` lists it meanwhile, and it is usable once the build
  completes (measured: 132 seconds after the call, by itself).

- **On Firebird, `query` refuses a `SELECT` carrying `FOR UPDATE`**
  ([#145](https://github.com/ChrisGVE/localdata-mcp/issues/145)). Measured on
  Firebird 5.0.4 with `firebirdsql` 1.5.0, on a read-only and a read-write
  connection alike: the driver fetches rows only for a plain select statement,
  and the server reports a `FOR UPDATE` select as a different kind, so the
  answer was `ok` with no rows whatever the table held. It is now refused
  before it runs, with or without `OF` or `WITH LOCK`. The refusal names the
  clause, says why the empty answer would be false, and ends with the same
  `SELECT` without it, which reads the rows. A `FOR UPDATE` inside an
  `EXECUTE BLOCK` is not refused, because its rows are fetched. `WITH LOCK`
  alone is refused by the read-only transaction (#144). The driver defect
  itself is upstream's.

- **A formula an `.xls` writer stored with an empty result loads as NULL
  without a warning.** Measured with xlwt 1.3.0: every formula it writes carries
  the result "empty string", which reads as NULL, the case `.xlsx`, `.xlsm` and
  `.ods` warn about. A workbook Excel saved stores the real result and reads
  correctly. xlrd hands back a formula's result and not the formula, so an
  empty-string result is indistinguishable from a formula that really computed
  `""`, and no warning is given (`docs/CONSTRAINTS.md` §38).

- **Known-unsupported pin: `sqlalchemy-cockroachdb` 2.0.4 is run on SQLAlchemy
  2.1.6 by an override.** The dialect declares `SQLAlchemy<2.1`, so the resolver
  cannot combine them. `[tool.uv] override-dependencies` removes that bound and
  nothing else, because 2.1 is what fixes #70. The CockroachDB dialect is tested
  live on 2.1, but a combination its maintainer has not declared is not one
  this project can promise. The override is removed when `sqlalchemy-cockroachdb`
  ships 2.1 support.
- **The override reaches only this repository's lock, not an installed
  package.** Installed from PyPI with the `cockroachdb` extra, SQLAlchemy
  resolves to 2.0.55. On that version a database URL whose database name holds
  `?` or `@` is refused
  ([#140](https://github.com/ChrisGVE/localdata-mcp/issues/140)), and a
  percent-escape of any other character in the database name (`%20`, `%2F`) is
  passed to the driver undecoded, as 2.0 always did.
- **`save` is refused on every backend but SQLite** — sixteen of the seventeen,
  a local DuckDB file included. `save` writes out a database this server is
  holding, and a slot reached over its own connection has none: the rows live in
  the engine, not here. Every file-derived slot *is* SQLite, so `save` works on
  all of them. For the rest it takes three calls: send the rows to a file with
  `query(path=…)`, `attach` that file as a datasource of your own — file-derived,
  so writable — and save that. The refusal says so, and `attach` reports
  `saveable: false` before anyone has to try.
- **`create(type="index")` is refused on DuckDB, ClickHouse, Trino, CrateDB
  and Exasol**, each for its own reason — an index this server could not find
  again to list or drop, indexes of another kind that cannot be reflected
  either, no storage to index, every column indexed already, or an engine that
  maintains its own. `directory` lists no indexes on those five, so an index a
  DuckDB file already held is not shown.
- **`update(type="table")` is refused on Firebird**, which has no rename-table
  statement and never has.
- **On Firebird a text column wider than 8,191 characters is a BLOB**, the
  widest `VARCHAR` being 8,191 characters. Its values are whole, `=` and `LIKE`
  work, but `GROUP BY`, `DISTINCT` and `ORDER BY` on it compare storage handles
  and answer wrongly with no error, and it cannot be indexed. `create` warns for
  that column. An index on a narrower text column is refused past 2,047
  characters on Firebird's default 8 KB page.
- **On Oracle a statement `query` refuses can still have happened.** The
  read-only guarantee is enforced on the connection, and Oracle commits DDL as
  it runs it, so there is no open transaction left to withhold when the refusal
  is composed: a refused `CREATE` or `DROP` stands, and the refusal says so
  ([#84](https://github.com/ChrisGVE/localdata-mcp/issues/84)). DML still rolls
  back, so the guarantee holds everywhere it can.

  This is the limitation where following the documentation can still leave
  your database changed. Weigh it before pointing this at a production Oracle.
  CrateDB used to share it for rows and schema alike, having no transactions;
  it now refuses anything but a query before the statement is sent
  ([#118](https://github.com/ChrisGVE/localdata-mcp/issues/118)).
- **`.xlsx` and `.ods` are refused above 65,535 rows.** That is the older
  worksheet's own limit and it is what bounds the writer's memory: uncapped,
  `.xlsx` held 12.9 GB while writing a million rows. `.ods` is 13.5× slower than
  `.xlsx` at the cap.
- **Formats that are not delimited are parsed whole**, so their load peak still
  tracks the file size.
- **A normalised temporal column loses its original offset.** A file that needs
  it must keep it in a column of its own. Its fractional seconds stop at the
  microsecond, so two instants a nanosecond apart become one value; the load
  warns when it cut one
  ([#111](https://github.com/ChrisGVE/localdata-mcp/issues/111)).
- **The backend catalogue is tested one database at a time.**
  `scripts/endpoint-run.sh all` covers it in one invocation, but serially. A run
  of fewer services says so: its summary names every target it did not reach
  ([#46](https://github.com/ChrisGVE/localdata-mcp/issues/46),
  [#139](https://github.com/ChrisGVE/localdata-mcp/issues/139)).
- **Db2 and OceanBase are eligible and unreachable** from a macOS host —
  measured, in `docs/CONSTRAINTS.md` §24 and §26.
- **A DuckDB file with a `?` in its name opens and reads, but attached writable it
  cannot commit**, because DuckDB names its WAL differently (`docs/CONSTRAINTS.md`
  §29.3).
- **On DuckDB, `query` runs one statement per call, and an `EXPLAIN` only over a
  query** ([#137](https://github.com/ChrisGVE/localdata-mcp/issues/137)). So
  `SELECT 1; SELECT 2` is refused, and so is `PIVOT … USING …` without an `IN`
  list, which DuckDB 1.5 reads as a `CREATE` and a `SELECT` and 2.0 as one
  statement of kind `MULTI`; with `IN ('a', 'b')` it is one query and answers,
  and the refusal says so, adding that `SELECT DISTINCT k FROM t` lists the
  values to name. On DuckDB 2.0 a `PRAGMA` is
  refused even when it only reads; its refusal names the query that reads the
  same.

## [2.1.0] - unreleased

### Upgrading from 2.0.0

Read this section even if you use only the MCP server. Two components were
renamed, three groups of tools that never worked now do, and the geospatial
statistics in 2.0.0 returned wrong numbers.

**Renames and moves.** Both renames break callers by name, and nothing warns
you — an old name simply resolves to nothing.

1. **Rename `graph-explore` to `graph-data-explore`** wherever you invoke it:
   `/graph-explore` becomes `/graph-data-explore`.
2. **Rename `graph-analyst` to `graph-data-analyst`** in any request or
   configuration that names the agent.
3. **Update any pinned path** of the form `skills/<name>/SKILL.md`. All eighteen
   skills now sit one level deeper, under a group directory:
   `skills/<group>/<name>/SKILL.md`. The eight that existed in 2.0.0 moved to
   `skills/exploration/explore-data/`, `skills/statistical/ab-test/`,
   `skills/statistical/analyze-correlations/`,
   `skills/modeling/cluster-analysis/`, `skills/modeling/forecast/`,
   `skills/modeling/regression/`, `skills/graph-data/graph-data-explore/`, and
   `skills/workflow/data-pipeline/`.
4. **Reload the plugin** so the moved directories are rescanned. Confirm with
   `claude --plugin-dir <path> plugin details localdata-mcp`, which should
   report 18 skills and 11 agents.

**Re-run any geospatial statistic computed on 2.0.0.** Moran's I, Geary's C and
Getis-Ord Gi* all returned plausible, wrong numbers — inflated z-scores, a
Geary's C sixty times too small, and a `p_value` of `None` on every Geary call
ever made. The corrected formulas are described under Fixed below. Any
conclusion drawn from those statistics needs recomputing, not adjusting.

**Recheck any YAML configuration you wrote for 2.0.0.** Settings the server
parsed and then ignored now take effect: `query.chunk_size`,
`query.buffer_timeout`, `connections.max_concurrent` and nineteen of the
twenty-five logging fields, `enable_metrics` among them. Defaults are unchanged
when nothing is configured, so an installation that configured nothing behaves
as before — but a file that set one of those keys will now be obeyed for the
first time.

**Nothing was removed from the tool surface.** It grew from 53 tools to 71; the
53 that existed in 2.0.0 keep their names, parameters and return shapes. A
client that names tools explicitly needs no edit for the new ones to appear.

### Added

- Eighteen MCP tools, taking the surface from 53 to 71. Seventy are registered
  by `DatabaseManager`; the seventy-first, `get_metrics`, is registered at import
  when metrics collection is enabled, which is the default.
  - **Sampling and estimation (4):** `generate_sample`, `bootstrap_statistic`,
    `monte_carlo_simulate`, `bayesian_estimate`.
  - **Optimization (4):** `solve_linear_program`, `optimize_constrained`,
    `analyze_network`, `solve_assignment_problem`. These four read a whole table
    rather than a query, since a solver needs the full constraint set.
  - **Geospatial (10):** `check_geospatial_capabilities`,
    `analyze_spatial_autocorrelation`, `find_spatial_hotspots`,
    `calculate_spatial_distances`, `optimize_route`, `analyze_accessibility`,
    `generate_service_isochrones`, `perform_spatial_join`,
    `perform_spatial_overlay`, `aggregate_points_in_polygons`. geopandas,
    shapely and pyproj are required; call `check_geospatial_capabilities` to see
    which optional backends (scikit-gstat, rasterio) are present.
- `LOCALDATA_LOGGING_ENABLE_METRICS`, `LOCALDATA_LOGGING_METRICS_PORT` and
  `LOCALDATA_LOGGING_METRICS_ENDPOINT`. `enable_metrics` had no environment
  variable and no working YAML path, so the `get_metrics` tool could not be
  turned off.
- Ten skills: data-quality, find-reference-data, anomaly-detection,
  dimensionality-reduction, geospatial, optimization, hypothesis-test,
  sampling-estimation, process-control, research-pipeline. The plugin now ships 18.
- Five agents: data-researcher, data-scientist, geospatial-analyst,
  operations-analyst, research-analyst. The plugin now ships 11.
- `server.json` describing the package for the official MCP registry, plus the
  `mcp-name` marker in README.
- Docker Hub publish workflow and MCP metadata label on the image.
- Kaggle demo: an RFM replication notebook over the UCI Online Retail dataset,
  with a Google Colab variant.
- `uv.lock`, committed so development environments are reproducible.
- `docs/plugin.md`, documenting every shipped skill and agent.

### Changed

- Configuration that was parsed and discarded now reaches the code that uses it.
  `query.chunk_size`, `query.buffer_timeout` and `connections.max_concurrent`
  had no call sites, so the live paths used hardcoded literals — a
  `Semaphore(10)`, 600s and 3600s buffer expiries, a 1000-row chunk fallback.
  `get_logging_config()` built `LoggingConfig` from a hand-written list of six
  keys and dropped the other nineteen. Where a setting has two homes, the newer
  `query`/`connections` section wins over the older `performance` keys, which
  stay honoured. Every existing default is unchanged when nothing is configured.
- Licence migrated from MIT to Apache 2.0. `LICENSE` carries the Apache 2.0 text,
  `NOTICE` carries the attribution notice Apache 2.0 requires, and `pyproject.toml`
  declares `license = "Apache-2.0"` as a PEP 639 expression.
- **Breaking:** skills are grouped into five domain directories (exploration,
  statistical, modeling, graph-data, workflow). All eight 2.0.0 skills moved, and
  `graph-explore` is now `graph-data-explore`. See Upgrading above.
- **Breaking:** `graph-analyst` is now `graph-data-analyst`. See Upgrading above.
- `statistical-analyst` covers sampling design and estimation alongside hypothesis testing.
- PyPI metadata: added AI and data-science classifiers and cross-platform keywords.
- Version raised to 2.1.0 in `pyproject.toml`, `.claude-plugin/plugin.json`, and `server.json`.

### Removed

- The skill name `graph-explore` and the agent name `graph-analyst`. Both are
  renames, not deletions of function — see Upgrading above for the replacements.
- The flat `skills/<name>/SKILL.md` layout, replaced by
  `skills/<group>/<name>/SKILL.md`.
- Four root documents, each describing a system that no longer matched the code:
  `API_REFERENCE.md`, `ADVANCED_EXAMPLES.md`, `ARCHITECTURE.md` and
  `CONFIGURATION.md`. They are replaced by `docs/tools-reference.md`,
  `docs/advanced-examples.md`, `docs/architecture/index.md` and
  `docs/configuration.md`.
- A duplicated tool section in the configuration documentation, which documented
  two parameters that do not exist: `style` on `export_graph` and `include_path`
  on `export_structured`.

### Fixed

- **`analyze_regression`'s `regularization` parameter did nothing at all.** It
  was forwarded into the modelling pipeline, which selects its estimator from
  `model_type` alone and never reads a `regularization` key, so `l1`, `l2`,
  `elastic_net` and even a misspelled value all returned the identical
  unpenalised fit. Anyone who reached for it to control overfitting on a wide
  feature set received ordinary least squares and no indication of it. The
  penalty now resolves to the estimator it names -- `l1` fits lasso, `l2` fits
  ridge, `elastic_net` fits elastic net -- and refines the expanded basis when
  combined with `model_type="polynomial"`. An unrecognised value, or one that
  contradicts an explicitly chosen `model_type`, now raises with the accepted
  values named.
- **`analyze_rfm` aborted on an ordinary order log.** When every customer places
  the same number of orders, the frequency quartiles collapse onto a single
  value and `pandas.cut` rejects the duplicate bin edges with `ValueError: Bin
  edges must be unique` -- an entirely normal shape for a subscription or
  replenishment business, failing the whole analysis. Duplicate edges are now
  collapsed and the surviving buckets spread back across the same 1-4 scale, and
  a dimension with no spread at all scores every customer the neutral 2 rather
  than branding them all with an extreme. Data with three distinct quartile
  boundaries scores exactly as it did before.
- **`analyze_network` never returned shortest paths.** NetworkX's
  `floyd_warshall_predecessor_and_distance` returns a `(predecessors,
  distances)` tuple, which was passed to `dict()`; that raised, the exception
  was swallowed, and the result carried `{"algorithm": "failed"}` instead of any
  paths. The Floyd-Warshall branch handles every graph of 100 nodes or fewer, so
  the feature was dead for graphs of the size the tool is used on. Distances and
  predecessors are now returned for every pair.
- **`analyze_hypothesis_test` never compared the groups it was given.** With the
  default `test_type="auto"` and a `group_column`, it returned only Shapiro-Wilk
  and Kolmogorov-Smirnov normality tests -- no group comparison at any point.
  This is the call shown as the flagship analytical example in the README, so an
  agent following the documentation received a well-formed result that answered a
  different question and reported no difference where one existed. The automatic
  path now selects from the data's assumptions: Welch's or Student's t-test for
  two normal groups depending on Levene, Mann-Whitney U with a rank-biserial
  effect size when either group is non-normal, one-way ANOVA with eta squared for
  three or more normal homoscedastic groups, Kruskal-Wallis H with epsilon
  squared otherwise. The comparison leads `test_results` and the normality
  profile follows it. Behaviour without a `group_column` is unchanged.
- **`analyze_anova` returned no post-hoc comparisons, ever.** `post_hoc_results`
  was `{}` on every call, including a three-group fixture with F = 27.9 and
  p = 1.2e-10, because the extraction indexed a raw observation array as though it
  were a labelled table and the resulting error was swallowed. Pairwise
  comparisons are now returned with their mean differences, p-values and
  confidence intervals, and the exception handling around them is narrowed so a
  future failure of this kind surfaces instead of returning an empty result.
- **`export_structured(name, "markdown")` returned an empty document.** The call
  produced `{"format": "markdown", "content": "## "}` for every tree, flat or
  nested, while `json`, `yaml` and `toml` returned the full structure. Markdown
  export now renders the tree.
- **Spatial statistics returned wrong numbers.** Anyone who ran them on 2.0.0
  should re-run rather than reinterpret. Four defects, all in the variance terms:
  the S2 term shared by Moran's I and Geary's C summed squared row sums instead
  of squared row-plus-column sums (60 instead of 256.9 for a row-standardised
  8-nearest-neighbour matrix over 60 points); Moran's variance ran roughly ten
  times small, inflating a z-score of 18.9 to 59.9, and now uses the closed-form
  normality variance of Cliff & Ord (1981); Geary's C divided by S0 twice,
  deflating C by a factor of sixty, and its variance came out negative, so its
  `p_value` was `None` on every call ever made; Getis-Ord Gi* used a variance
  with no textbook counterpart and added a self-weight to an already
  row-standardised matrix, and now uses the standard form of Getis & Ord (1992).
  On three well-separated blobs it now labels every high-value point a hot spot
  and every low-value point a cold spot, where before it found neither.
- **All twelve data science tools raised before returning.** The adapter between
  the MCP tools and the domain modules called almost every domain function with
  the wrong signature — column selectors forwarded as keywords the transformers
  do not accept, a DataFrame passed where a positional ndarray was required, and
  renamed parameters (`amount_column`, not `value_column`; `algorithm`, not
  `method`). The MCP signatures themselves were correct throughout.
- **The time series tools could not run at all.** Every analyzer subclassed an
  abstract pipeline base whose five hooks none of them implement, so
  instantiation failed with "Can't instantiate abstract class"; and
  `TimeSeriesResult.to_dict()` listed the base fields by hand, silently dropping
  every field a subclass added.
- **`request_data_chunk`, `request_multiple_chunks` and `get_query_metadata`
  were non-functional from the day they were registered.** The chunk loader was a
  placeholder returning `None`, so every chunk of every query came back as "Chunk
  N not available". Chunk availability was sized from the buffered head rather
  than the whole result, so a 50,000-row result advertised ten chunks covering
  2,170 rows and refused every id past that. `get_query_metadata` serialized its
  response with the stdlib encoder while the quality report carries numpy
  scalars, so it returned "Object of type bool is not JSON serializable" and
  nothing else, always.
- Clustering results keyed per-cluster statistics by numpy integer labels, which
  `json.dumps` rejects as mapping keys regardless of the `default` handler, so the
  tool failed at serialization after the analysis had already succeeded.
- Four geospatial entry points read result attributes their transformers do not
  set, returning `{"error": "Analysis failed"}` while the analysis underneath had
  succeeded.
- The advertised tool count. Six files stated 53; the server registers 71.
- Performance benchmarks workflow was invalid YAML and had never executed. Embedded
  Python scripts written at column 0 terminated the enclosing `run:` block scalar,
  so GitHub could not parse the file and every run completed with zero jobs.
- Performance benchmarks workflow installed dependencies from `requirements.txt` and
  `requirements-dev.txt`, which no longer exist; it now installs from `pyproject.toml`.
  Deprecated actions were updated and the Python matrix aligned with CI.
- Benchmark harness called token estimation with the wrong parameter name.
- Retry-logic test counted sleeps raised on other threads and failed intermittently.
- Plugin manifest and contributing guide still referenced the MIT licence after the
  migration to Apache 2.0. The `License :: OSI Approved :: MIT License` classifier was
  removed, since a PEP 639 `license` expression supersedes it.
- README documented seven graph tools that do not exist (`get_node_graph`,
  `set_node_graph`, `delete_node_graph`, `get_value_graph`, `set_value_graph`,
  `delete_key_graph`, `list_keys_graph`). Graph node operations are served by the tree
  tools, which read `path` as a node ID for graph connections.
- README named four environment variables the server never reads
  (`LOCALDATA_MAX_CONNECTIONS`, `LOCALDATA_CHUNK_SIZE`, `LOCALDATA_BUFFER_TTL`,
  `LOCALDATA_WORKING_DIR`) and gave a chunk-size default of 500 where the code uses 100.
- README linked `DATABASE_CONNECTIONS.md` and `FAQ.md`, both deleted before 2.0.0.
- `CONTRIBUTING.md`, `TROUBLESHOOTING.md`, and `docs/getting-started.md` told
  contributors to run `uv sync --dev`. The dev tools are a project extra, not a uv
  dependency group, so that command uninstalls pytest, black, and mypy instead of
  installing them. Corrected to `uv sync --extra dev`.
- Four domain guides documented internal Python adapters whose first parameter is
  a SQLAlchemy engine as if they were MCP tools. `docs/domains/` now names the
  tools an MCP client can actually call.

## [2.0.0] - 2026-04-06

### Added
- Data science domain modules with sklearn-compatible transformers:
  - Statistical analysis: hypothesis testing, ANOVA, effect sizes
  - Regression modeling: linear, regularized, polynomial with diagnostics
  - Pattern recognition: clustering, anomaly detection, dimensionality reduction
  - Time series: decomposition, ARIMA/ETS forecasting, change detection
  - Business intelligence: RFM segmentation, cohort analysis, A/B testing
  - Optimization: linear programming, constrained optimization, network analysis
  - Sampling & estimation: bootstrap, Bayesian estimation, Monte Carlo simulation
  - Geospatial analysis: spatial joins, routing, clustering (optional geopandas)
- Pipeline framework: sklearn-compatible data science pipeline with streaming support
- 12 new MCP tools for data science operations (52 total)
- Integration shim layer for domain interoperability
- Graph format support: DOT, GML, GraphML, Mermaid
- RDF/SPARQL support: Turtle, N-Triples files and remote SPARQL endpoints
- Memory-aware query execution with 3-path decision flow (in-memory, staging, refinement)
- Graceful abort handler for streaming with partial result return
- Claude Code plugin structure: 8 skills, 6 agents, plugin manifest
- Comprehensive documentation: tools reference, data sources, 8 domain guides

### Changed
- Added scipy, scikit-learn, statsmodels, numpy as core dependencies
- Added networkx, pydot, rdflib, SPARQLWrapper as core dependencies
- Geospatial imports guarded with try/except for optional geopandas dependency
- README rewritten for v2.0 (663 lines to 261 lines)
- Replaced fixed 100-row query threshold with dynamic memory-based threshold
- Refactored oversized modules into sub-packages for maintainability

## [1.7.1] - 2026-04-02

### Added
- Enterprise-scale integration testing with NYC Taxi dataset (100K rows per database)
- 62 enterprise tests across 7 databases: PostgreSQL, MySQL, MSSQL, Oracle, SQLite, MongoDB, Elasticsearch
- Oracle integration tests via oracledb thin mode (no Oracle Client required)
- Automated integration test runner (`scripts/run_integration_tests.py`) with Docker lifecycle management
- Public dataset download script (`scripts/download_test_datasets.py`) for NYC Taxi, Ames Housing, World Bank
- Test fixture file generator (`scripts/generate_test_files.py`) for all supported formats
- InfluxDB, Neo4j, CouchDB integration test suites and Docker services

### Fixed
- Pydantic v1 to v2 migration for configuration validators
- Oracle SQLAlchemy connection string: use `?service_name=` format for thin mode
- MSSQL bulk loader: switch to executemany (2100 parameter limit prevents multi-row INSERT)
- Oracle bulk loader: switch to executemany (multi-row INSERT VALUES not supported)
- Enterprise parquet reader: stream via `iter_batches` to avoid loading 9.5M rows into memory
- Enterprise fixture: read dataset once and pass to all database loaders
- Various test assertion fixes for MCP response shapes, timeout manager mocks, security event fields

### Changed
- README updated: added Oracle, MSSQL, RDF/SPARQL to supported databases; documented 18 missing tools; removed 6 dead tool references; updated test stats and examples
- Version references updated from v1.3.1 to v1.7.0 across API_REFERENCE.md, ARCHITECTURE.md, CONFIGURATION.md, ADVANCED_EXAMPLES.md
- Test count: 1,621 unit tests + 234 integration tests + 62 enterprise tests

## [1.7.0] - 2026-03-28

### Added

#### Regex Search & Transform Tools
- `search_data` MCP tool: search query results for regex patterns with column filtering, case sensitivity, and match position tracking
- `transform_data` MCP tool: apply regex find/replace to query result columns with capture group support and change sampling
- ReDoS (Regular Expression Denial of Service) prevention via `sre_parse` AST analysis detecting nested quantifiers and dangerous patterns
- Pattern complexity scoring with configurable limits (200 char max, 10 groups max)
- Timeout-protected execution (5 second default) via ThreadPoolExecutor

#### Enhanced Graph & Tree Export
- Graph hierarchy export (`style="hierarchy"`): renders DAGs as indented trees with multi-parent annotations (`also child of: X`)
- Graph detailed export (`style="detailed"`): full node property sections with incoming/outgoing edge tables
- Graph adjacency list export (`style="adjacency"`): compact `A -> B [label]` format for token efficiency
- Tree export with path breadcrumbs (`include_path=True`): shows `root > parent > child` navigation context
- Cycle detection with automatic fallback to adjacency list for non-DAG graphs

#### Schema Export
- `export_schema` MCP tool with four output formats:
  - JSON Schema with type mapping, required fields, maxLength, x-primary-key, x-foreign-keys, x-indexes
  - Python dataclass definitions with Optional fields for nullable columns
  - TypeScript interfaces with optional property markers
  - SQL DDL with CREATE TABLE, PRIMARY KEY, FOREIGN KEY, CREATE INDEX statements
- `SchemaIntrospector` class using SQLAlchemy Inspector for cross-database schema extraction

#### Query Audit Log
- `get_query_log` MCP tool: retrieve recent query execution history with database, status, and time filters
- `get_error_log` MCP tool: filtered view of errors and timeouts with structured classification
- In-memory ring buffer (configurable max 1000 entries) with thread-safe deque
- Automatic audit recording integrated into `execute_query` with timing and error capture
- Query hash generation for deduplication statistics

#### Readonly Mode
- `security.readonly: true` configuration option that hard-blocks write-disguised-as-read SQL
- Detects SELECT INTO, CREATE TABLE AS SELECT, COPY TO, INSERT ALL, MERGE INTO, OUTPUT INTO
- Database-specific pattern coverage for Oracle, MS SQL, PostgreSQL, MySQL

## [1.6.0] - 2026-03-28

### Added

#### Enterprise Database Support
- Oracle Database support via `oracledb` driver with password, Oracle Wallet, Kerberos, and certificate authentication
- MS SQL Server support via `pymssql`/`pyodbc` with SQL Auth, Windows Integrated Auth, Azure AD, Kerberos, and certificate authentication
- Enterprise authentication framework (`auth` parameter on `connect_database`) with support matrix per database type
- Oracle and MSSQL error mappers with ORA-XXXXX and Msg/Severity code classification

#### Configuration System Upgrade
- OS-aware configuration paths: XDG on Linux, `~/Library/Application Support` on macOS, `%APPDATA%` on Windows
- Legacy `~/.localdata.yaml` path deprecated with migration support
- New config sections: `staging`, `memory`, `query`, `connections`, `security`, `disk_budget`
- Environment variable overrides for all new config sections (`LOCALDATA_STAGING_*`, `LOCALDATA_MEMORY_*`, etc.)
- CLI flags: `--config`, `--version`, `--migrate-config`, `--force`, `--validate-config`, `--show-config`, `--init-config`
- Default config file creation with `create_default_config()`

#### Structured Error Classification
- `StructuredErrorResponse` with `error_type`, `is_retryable`, `message`, `suggestion` for LLM agents
- Database-specific error mappers for SQLite, PostgreSQL, MySQL, DuckDB, Oracle, MS SQL
- Error mapper registry with pluggable architecture for custom databases
- Structured errors integrated into `execute_query`, `connect_database`, and streaming executor
- Helper functions: `classify_error()`, `is_error_retryable()`, `get_error_suggestion()`

#### Query Execution Intelligence
- Size estimation engine combining column type metadata with EXPLAIN row estimates
- EXPLAIN parsers for SQLite, PostgreSQL, MySQL, Oracle (DBMS_XPLAN), MS SQL (SHOWPLAN_XML)
- Pre-flight query estimation via `preflight=True` parameter on `execute_query`
- Memory-aware execution with `MemoryBudget` auto-calculated from available RAM (10%, max 512MB)
- Aggressive mode when RAM drops below 1GB (5%, max 128MB)
- BLOB column detection with placeholder handling (`[BLOB: size, mime_type]`) and opt-in base64 encoding

#### Staging and Disk Management
- Staging database manager with LRU eviction for large query results
- Disk space monitoring during streaming with configurable thresholds
- Graceful abort with partial results when disk limits are reached
- `include_staging` parameter for `list_databases` to show staging databases
- Cascade cleanup of staging databases on `disconnect_database`

#### Markdown Export
- Markdown table export for query results with alignment and truncation
- Tree/structured data export as heading hierarchy
- Graph export with summary stats, node/edge tables, and Mermaid diagram embedding
- Registered as `markdown`/`md` format in `export_structured` and `export_graph`

#### Documentation and Testing
- Comprehensive configuration reference (`docs/configuration.md`)
- Error classification reference (`docs/error-classification.md`)
- Docker-based integration testing infrastructure (`docker-compose.test.yml`)
- 663 tests across 26 test files

## [1.5.2] - 2026-03-26

### Added
- Validation warnings on graph import and edit operations

## [1.5.1] - 2026-03-26

### Added
- Mermaid flowchart parser support for graph import
- Mermaid export support with round-trip fidelity
- Main README section documenting graph file, Mermaid, and SPARQL support

### Fixed
- Include node and edge metadata in graph exports
- Six graph bugs found during code review
- Mermaid parsing and export edge cases
- File and function size limit compliance for graph modules

## [1.5.0] - 2026-03-24

### Added
- Graph file format support: DOT, GML, GraphML
- RDF format support: Turtle and N-Triples parsing
- GraphStorageManager and RDFStorageManager for structured graph persistence
- SPARQL query endpoint for RDF data
- Graph-specific MCP tools for node, edge, path, and statistics operations
- RDF export and describe_database support for graph connections
- End-to-end integration tests for graph/RDF/SPARQL pipeline
- move_node tool to relocate subtrees in tree storage

### Fixed
- Return dicts from tree tools to prevent double JSON serialization
- Deserialize string values when explicit value_type is provided

## [1.4.2] - 2026-03-23

### Fixed
- Use StaticPool for in-memory SQLite to fix cross-thread access errors
- Clean up tree managers on disconnect to prevent resource leaks

## [1.4.0] - 2026-03-23

### Added
- Tree storage model, manager, and hierarchical data persistence
- Tree parsers for TOML, JSON, and YAML formats
- MCP tool functions for tree navigation and mutation (get_node, set_node, get_children, delete_node)
- Tree export to TOML, JSON, and YAML formats
- Integration of tree storage with connect_database and tool registration
- Data summary returned on connect_database
- Comprehensive tests for tree tools, parsers, and export

### Fixed
- Migrate to fastmcp v3 API for tool registration
- LogContext now accepts arbitrary keyword arguments
- Row-normalise deeply nested TOML/YAML instead of column explosion

## [1.3.1] - 2025-08-31

### Added
- SQL query parser and safety validator
- QueryAnalyzer class with pre-query analysis system
- ConfigManager class with pydantic-based configuration
- StreamingQueryExecutor for memory-bounded streaming pipeline
- QueryTimeoutManager for advanced timeout management
- Streaming file processors integrated into file engine creation
- TokenManager class with DataFrame-based token estimation
- EnhancedConnectionManager with connection pooling and health monitoring
- SecurityManager class with OWASP-aligned security patterns
- Enhanced response metadata and LLM communication protocol with progressive data loading
- Error handling system with custom exceptions, retry mechanisms, and circuit breaker pattern
- Enhanced database manager with integrated error handling and recovery strategies
- Structured logging system with structlog, Prometheus metrics, and JSON logging
- CompatibilityManager for backward compatibility with v1.3.0
- Performance benchmarking suite with CI integration
- Comprehensive documentation: architecture, configuration, API reference, migration guide

### Changed
- Clean tool surface for production release (breaking: removed deprecated tool aliases)
- Major cleanup and reorganization of project structure
- README restructured with table of contents

### Fixed
- Environment variable names in migration script generation
- Boolean column categorization and test data alignment
- TokenManager integration robustness
- DataFrame boolean ambiguity in return statements
- Function abuse attack pattern regex
- Leaked credential detection in documentation examples

## [1.3.0] - 2025-08-30

### Added
- Modern database support: Redis, Elasticsearch, MongoDB, InfluxDB, Neo4j, CouchDB
- DuckDB support for analytical SQL queries
- HDF5 format support for scientific data
- Apple Numbers (.numbers) document support
- Docker containerization with complete docker-compose setup
- Comprehensive test coverage improvements (68% coverage)

### Changed
- Updated documentation to reflect actual test coverage (68% vs previous claims)
- Enhanced README with modern database features and Docker support
- Improved error handling and graceful degradation for optional dependencies

### Fixed
- ODS import detection with proper library imports
- JSON/TOML serialization for complex nested data structures
- INI file parsing with special character handling
- InfluxDB service configuration for proper startup

## [1.2.0] - 2025-08-29

### Added
- Multi-sheet spreadsheet support (Excel .xlsx/.xls, LibreOffice .ods)
- Enhanced file format support: XML, INI, TSV
- Analytical formats: Parquet, Feather, Arrow
- Large file handling with automatic SQLite conversion (100MB+ files)
- Query buffering system for large result sets (100+ rows)
- Advanced security features and input validation

### Changed
- Improved performance for large datasets
- Enhanced error messages and validation
- Better memory management and resource cleanup

### Fixed
- Path traversal security issues
- SQL injection prevention
- Connection limit enforcement

## [1.1.0] - 2025-08-23

### Added
- Enhanced spreadsheet format support
- Improved file type detection
- Better error handling for malformed files

### Changed
- Performance optimizations for file processing
- Updated documentation and examples

## [1.0.3] - 2025-08-23

### Fixed
- Minor bug fixes and stability improvements
- Documentation updates

## [1.0.2] - 2025-08-23

### Fixed
- Package dependencies and compatibility issues
- Installation process improvements

## [1.0.1] - 2025-08-23

### Fixed
- Initial release bug fixes
- Documentation corrections

## [1.0.0] - 2025-08-23

### Added
- Initial release of LocalData MCP Server
- Basic database connectivity (PostgreSQL, MySQL, SQLite)
- File format support (CSV, JSON, YAML)
- Core MCP protocol implementation
- Security features and input validation