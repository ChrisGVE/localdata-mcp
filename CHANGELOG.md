# Changelog

All notable changes to LocalData MCP Server will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [3.0.0] - unreleased

A ground-up rewrite. **Every one of the 71 tools 2.1.0 registered is gone**, and
eight verbs stand where they were. Nothing a 2.x client calls by name resolves,
and no alias, shim or compatibility layer is provided — the names are not
deprecated, they are absent.

### Upgrading from 2.x

**Your entire tool vocabulary is gone.** `connect_database`, `execute_query`,
`list_databases`, `describe_table`, `analyze_regression`, `forecast_time_series`,
`analyze_spatial_autocorrelation`, `get_node`, `export_graph` — all 71. A client
that names a tool explicitly will find nothing there; an agent that discovers
tools at runtime will simply see eight.

The seventy-one divide three ways: **sixteen map onto four of the new verbs**,
**forty-six have no successor at all**, and **nine managed machinery that no
longer exists**. The other four verbs are new in 3.0.0 and replace nothing.

| 3.0.0 verb | What it replaces from 2.x |
|---|---|
| `attach` | `connect_database` |
| `detach` | `disconnect_database` |
| `query` | `execute_query`, `analyze_query_preview`, `next_chunk`, `request_data_chunk`, `request_multiple_chunks`, `get_query_metadata`, `export_structured`, `search_data` |
| `info` | `list_databases`, `describe_database`, `describe_table`, `find_table`, `export_schema`, `get_data_quality_report` |
| `create` | **new in 3.0.0** — reading a second datasource *into* an open one had no 2.x counterpart |
| `update` | **new in 3.0.0** |
| `drop` | **new in 3.0.0** |
| `save` | **new in 3.0.0** — 2.x had staging databases, which were not the user's to keep |

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
first-found-wins cascade, and the only environment variable is
`LOCALDATA_CONFIG_PATH`, which *locates* the file and never carries a setting.
There are three sections and five settings in total — `workspace.slots`,
`workspace.memory_budget_mb`, `paths.roots`, `paths.path_limited` and
`network.enabled`. **An unknown section or key is refused rather than ignored**,
because a mistyped `path_limitted = false` that silently kept the safe default is
a security switch you believe you have thrown. The file is read on the first tool
call rather than at startup, so the server launches and the handshake completes;
the refusal arrives on that first call, naming the bad key and the keys the
section does know. No 2.x configuration file is readable, and none is migrated.

**Nothing survives the session unless you `save` it.** 2.x persisted staging
databases and buffers across calls with their own eviction policies and disk
budgets. A 3.0.0 slot lives until `detach` or until the server stops.

**Ten datasources, and the oldest is evicted.** Check the `evicted` field on
every `attach` response: it describes the slot that was dropped well enough to
rebuild it.

### Added

- Eight MCP tools — **verbs**, the word used for them throughout: `attach`,
  `detach`, `query`, `info`, `create`, `update`, `drop`, `save`. Few and
  multi-faceted rather than many and narrow — `info` alone absorbs six 2.x tools
  by varying on its two optional arguments.
- **Eighteen file formats read** (`.csv` `.tsv` `.txt` `.fwf` `.json` `.jsonl`
  `.ndjson` `.xml` `.yaml` `.yml` `.xlsx` `.xlsm` `.xls` `.ods` `.numbers`
  `.parquet` `.feather` `.orc`) and **fifteen written** (the same, less `.fwf`,
  `.xlsm`, `.xls` and `.numbers`, plus `.md`). `.xlsx` and `.ods` are written as
  well as read. Eight of the readers and seven of the writers are guaranteed by
  the base install; `.yaml`/`.yml` also work without their extra today, because
  `fastmcp` requires `PyYAML` unconditionally — a fact about the dependency graph
  rather than a promise, so `yaml` stays the declared extra.
- **A file may hold more than one table, and all of them land.** A workbook
  becomes a database with a table per sheet, and a `.numbers` document one per
  **table** — a Numbers sheet is a canvas that may carry several — each under its
  own name put through the same snake_case rule as a nickname, so `Sheet1`
  becomes `sheet1` and `update` renames it. Where two tables in one document
  share a name, the sheet name is prefixed to break the collision. 2.x read the
  first sheet and said nothing about the rest. A JSON, YAML or XML document with
  **two** candidate tables is refused instead, naming both, rather than one being
  picked silently.
- **Eighteen database backends**: SQLite, DuckDB, PostgreSQL, MySQL, MariaDB,
  SQL Server, Oracle, ClickHouse, CockroachDB, YugabyteDB, Trino, MonetDB,
  CrateDB, Firebird, openGauss, YDB, Databend and Exasol. Sixteen are exercised
  against a container of their own in `docker-compose.test.yml`. The eight verbs
  are the whole surface on all of them; three verbs are refused on some engines,
  and those refusals are listed under *Known limitations* below.
- **Nine further ways of authenticating an endpoint are exercised by the test
  suite**, beyond the credentialed URL that was the only route 2.x ever took: a
  server configured to trust, a password from the environment, a password from a
  file, verified TLS, a client certificate, a Kerberos ticket, a MySQL option
  file, an empty password, and an ODBC data-source name in place of a host and
  port — ten routes in all, counting the credentialed URL that all sixteen
  endpoints still use. **These are not server settings and there is no parameter
  for them**: `attach` takes a datasource string, and each mode is expressed in
  the URL or in the driver's own environment, which this server passes through
  untouched. What 3.0.0 adds is the evidence that they work —
  `tests/endpoints.py` runs the whole endpoint suite as a second axis over them.
  `docs/CONSTRAINTS.md` §25 has the measurements, and
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
- **A `delimiter` parameter on `attach`, `create` and `query`.** Nothing sniffs:
  a semicolon-separated file read at `,` loads as one column, and the warning says
  so and names the parameter rather than re-reading at a guessed separator.
- **Path confinement** to the working directory and configured roots, with
  symlinks and `..` resolved before the check. Network URLs are refused until
  `network.enabled = true`, and the refusal masks the password.
- `docs/architecture/LEVEL0.md` — the specification — and `docs/CONSTRAINTS.md`,
  which records every behaviour measured to arrive at these decisions, with the
  numbers.

### Changed

- **Every datasource is a database**, whatever it came from, and the same eight
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
- **A delimited file is read twice rather than held once.** `.csv`, `.tsv`,
  `.txt` and `.fwf` go through a measuring pass and an inserting pass, dropping
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
- The licence is Apache 2.0, unchanged from 2.1.0.

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
- **Eighteen skills and eleven agents.** One skill ships now,
  `skills/data/local-data/SKILL.md`, and it carries the idiom the tools
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

### Known limitations

- **`save` is refused on every backend but SQLite** — seventeen of the eighteen,
  a local DuckDB file included. `save` writes out a database this server is
  holding, and a slot reached over its own connection has none: the rows live in
  the engine, not here. Every file-derived slot *is* SQLite, so `save` works on
  all of them. For the rest, `create` the rows you want into a slot of your own
  and save that, or send the result straight to a file with `query(path=…)`. The
  refusal says so.
- **`create(type="index")` is refused on ClickHouse, Trino, CrateDB, Databend and
  Exasol**, each for its own reason — indexes that cannot be reflected, no
  storage to index, every column indexed already, a statement that compiles to
  nothing, or an engine that maintains its own.
- **`update(type="table")` is refused on Firebird**, which has no rename-table
  statement and never has.
- **On two backends a statement `query` refuses can still have happened.** The
  read-only guarantee is enforced on the connection, and on these two engines
  there is no open transaction left to withhold when the refusal is composed:
  - **Oracle** commits DDL as it runs it, so a refused `CREATE` or `DROP`
    stands. DML still rolls back, so the guarantee holds everywhere it can.
  - **CrateDB** has no transactions at all, so **both** DDL *and* DML survive:
    a refused `INSERT` stands. It is the worse of the two, and the refusal text
    names only `CREATE`/`DROP`, which reads as *this was not DDL, therefore
    nothing happened*
    ([#84](https://github.com/ChrisGVE/localdata-mcp/issues/84)).

  These are the limitations where following the documentation can still leave
  your database changed. Weigh them before pointing this at a production Oracle
  or CrateDB.
- **`.xlsx` and `.ods` are refused above 65,535 rows.** That is the older
  worksheet's own limit and it is what bounds the writer's memory: uncapped,
  `.xlsx` held 12.9 GB while writing a million rows. `.ods` is 13.5× slower than
  `.xlsx` at the cap.
- **Formats that are not delimited are parsed whole**, so their load peak still
  tracks the file size.
- **A normalised temporal column loses its original offset.** A file that needs
  it must keep it in a column of its own.
- **No single test run covers the backend catalogue.** This machine runs six
  containers and starves them at around seven, so it takes five batches, and each
  reports a green suite while the dialects it never reached stay silent
  ([#46](https://github.com/ChrisGVE/localdata-mcp/issues/46)).
- **Db2 and OceanBase are eligible and unreachable** from a macOS host —
  measured, in `docs/CONSTRAINTS.md` §24 and §26.

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