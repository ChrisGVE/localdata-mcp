# Architecture

How LocalData MCP is put together: what runs, what a tool call touches on its way
through the process, and where the boundaries are. This page describes the server
as it ships. Where a claim here was checked by running the code rather than by
reading it, it says so.

If you are looking for the reasoning behind the interface rather than its
mechanics, the design documents at the end of this page cover that — with the
caveat noted there about which of them describe built machinery.

## The shape of the thing

LocalData MCP is a single Python process. An MCP client — Claude Desktop, Claude
Code, any other — launches `localdata-mcp` and talks to it over stdio. There is
no network listener, no daemon, no database of its own. Everything it knows about
lives in memory for the lifetime of the process, plus whatever temporary files it
stages to disk and cleans up on exit.

The process runs one [FastMCP](https://github.com/jlowin/fastmcp) server instance
and one `DatabaseManager`. The server holds the tool registry; the manager holds
the connections, the query buffers, and the implementation of nearly every tool.

```
MCP client
    | stdio (JSON-RPC)
    v
FastMCP server ......... the tool registry: 71 tools
    |
    v
DatabaseManager ........ connections, buffers, tool implementations
    |
    +-- SQLAlchemy engines ......... SQL databases
    +-- file processors ............ CSV, Excel, Parquet, ... loaded into SQLite
    +-- tree / graph / RDF managers  structured documents, graphs, triples
    +-- domain adapters ............ the analytical toolkit
```

## Startup

`localdata-mcp` runs `main()` in `server/cli.py`. Before that function executes, importing
`localdata_mcp.localdata_mcp` has already done most of the work of standing the
server up, and the ordering matters:

1. **Configuration resolves first.** `get_config_manager()` merges defaults, then
   YAML files in discovery order, then `LOCALDATA_*` environment variables — later
   sources winning. Nothing downstream can see a half-built configuration.
2. **Logging is built from that configuration**, which is why the metrics decision
   can be made at import time.
3. **The FastMCP instance is created**, and `get_metrics` is registered on it if
   `logging.enable_metrics` is on. It is the only tool whose presence depends on
   configuration, and it defaults to on.
4. **`DatabaseManager` is imported last**, because the server modules import the
   module above by absolute name to reach these singletons.

`main()` then handles the configuration subcommands (`--show-config`,
`--validate-config`, `--init-config`, `--migrate-config`), each of which exits
without starting a server, constructs the `DatabaseManager`, and calls
`mcp.run(transport="stdio")`.

Constructing the manager is what registers the other seventy tools:
`DatabaseManager.__init__` calls `self._register_tools(mcp)`, which binds each
tool method with `add_tool`. So the registry has two feeders — the import-time
registration above and this one — and a count that consults only one of them will
be wrong. `tests/test_tool_registration.py` reads the assembled server for exactly
that reason.

## Where a tool call goes

Tools come in two shapes, and the difference is worth knowing because it decides
which protections apply.

### Data tools

`connect_database`, `execute_query`, `describe_table` and their neighbours are
methods on `DatabaseManager` itself. `execute_query` is the fullest path:

1. **Dispatch by connection kind.** SPARQL endpoints and RDF connections are
   handled by separate methods before anything else runs.
2. **Security validation.** `parse_and_validate_sql` parses the statement and
   rejects anything that is not a `SELECT` or a `WITH` CTE. This is a parser, not
   a pattern match — it raises `SQLSecurityError`, and the tool returns the error
   rather than the data.
3. **Analysis and estimation.** `analyze_query` inspects the query and, with
   `EXPLAIN` where the backend supports it, estimates how large the result will be.
4. **One of three execution paths**, chosen from that estimate against the memory
   budget:
   - **In memory** when the result fits the RAM budget.
   - **Staging** when it exceeds RAM but fits the disk budget: rows stream through
     a temporary SQLite database under disk monitoring.
   - **Refinement** when it exceeds both: nothing executes, and the response asks
     for a narrower query and suggests how — a `LIMIT`, a `WHERE`, an aggregate.
5. **Buffering.** Large results are held in a `QueryBuffer` and handed out in
   chunks by `next_chunk` and the `request_*_chunk` tools, so a result larger than
   the client's context can still be read.
6. **Audit.** `_record_audit` records the query, its status and its timing, which
   is what `get_query_log` and `get_error_log` read.

The refinement path is the architectural decision worth naming: the server would
rather refuse a query and explain why than return a result that blows up the
agent's context or the machine's RAM.

### Analytical tools

The eight analytical domains follow a different, deliberately thin pattern. Each
domain contributes two files:

- `<domain>_tools.py` — pure adapters with no MCP knowledge, taking a SQLAlchemy
  engine and a query and returning a plain dictionary.
- `server/<domain>_mixin.py` — the MCP wrappers: resolve the named connection with
  `self._get_connection`, call the adapter, serialize with `safe_dumps`.

The mixin is added to `DatabaseManager`'s bases and its methods listed in
`_register_tools`. Sampling, optimization and geospatial each live in their own
mixin; the twelve older data science tools and the graph, tree and search tools
are still implemented directly in `server/database_manager.py`.

Nearly every analytical tool takes a connection name and a SQL query as its first
two parameters. The query selects the data; there is no separate load step and no
data-frame parameter. Column arguments name columns in that query's result set.
The four optimization tools are the exception: they take a `table_name` and read
the whole table, because a linear program is defined over a full constraint
matrix rather than a projection of one.

**These tools do not pass through the SELECT-only gate.** The gate guards
`execute_query` and `analyze_query_preview`; an analytical tool hands its query
to pandas directly. Running a `DELETE` through `execute_query` is refused, and
running the same statement through `analyze_clusters` reaches the driver instead
(both verified by execution). It did not persist in that test, but only because
the statement ran in an uncommitted transaction — that is transaction semantics,
not a control. This is tracked as issue #25 and is a known gap, not a design.

## Connections

`connect_database` maps a name to a live connection and remembers its type.
What it builds depends on the source:

- **SQL databases** get a SQLAlchemy engine — PostgreSQL, MySQL, SQLite, DuckDB,
  Oracle, SQL Server.
- **Files** are read by a streaming file processor and materialized into a SQLite
  database, which is why a CSV connection answers SQL and why its table is called
  `data_table`. Large files stream rather than loading whole.
- **Document, graph and RDF sources** get their own managers — `TreeStorageManager`
  for JSON/YAML/TOML trees, `GraphStorageManager` for DOT/GML/GraphML/Mermaid,
  `RDFStorageManager` for Turtle and N-Triples.
- **Non-SQL databases** — MongoDB, Redis, Elasticsearch, InfluxDB, Neo4j, CouchDB —
  are handled by their own clients.

Every optional backend is imported behind a `try`/`except ImportError` that sets an
availability flag, so a missing driver disables one source rather than breaking
startup. `check_compatibility` reports what is available.

Concurrency is bounded by a semaphore sized from
`connections.max_concurrent` (default 10), which the config manager reads in
preference to the older `performance.max_concurrent_connections`. Buffers expire
on a timer.

## Security boundaries

Three controls, and it is worth being precise about what each covers:

- **Path restriction.** File connections must resolve inside
  `security.allowed_paths`, which defaults to `["."]` — the working directory.
  This is why fixtures and data files must live inside the project rather than in
  `/tmp`. `security.restrict_paths: false` disables it.
- **SELECT-only SQL.** Enforced by parsing, on `execute_query` and
  `analyze_query_preview`. See the gap noted above for the analytical tools.
- **Connection limits.** A semaphore caps concurrent connections; buffers and
  temporary files are reclaimed on a timer and at exit.

`--show-config` redacts passwords, secrets, tokens, connection strings and
certificate paths before printing.

## Configuration

One resolution order, applied at startup and on reload:

1. Built-in defaults
2. YAML file, first found of: `LOCALDATA_CONFIG`, `./.localdata.yaml`, the
   per-user path (`$XDG_CONFIG_HOME/localdata/config.yaml`,
   `~/Library/Application Support/localdata/config.yaml`, or `%APPDATA%\localdata`),
   `/etc/localdata/config.yaml`, then legacy `~/.localdata.yaml` with a
   deprecation warning
3. `LOCALDATA_*` environment variables

Later sources win. The manager tracks which keys the user actually set, separately
from those defaults seeded — needed because several settings have both a current
and a legacy home, and "present in the merged data" cannot distinguish a default
from a choice. `docs/configuration.md` documents the settings themselves.

## Code that is present but not on the live path

The tree carries more than the server runs, and reading it as though every package
were load-bearing will mislead you. Verified by importing the server, exercising
tools, and inspecting what actually loaded and what actually held state:

- **`connection_manager/` and `enhanced_database_manager.py`** are imported as a
  side effect of the analytical domain packages, but they govern nothing. After
  `connect_database` succeeds, `DatabaseManager.connections` holds the connection
  and the enhanced manager's `list_databases()` returns `[]`. Configuration wired
  into them has no effect on a real connection.
- **`enhanced_database_tools.py`, `auth_manager.py`, `blob_handler.py` and
  `performance_benchmarks/`** are not imported at all during normal operation.
- **`pipeline/`** is loaded, because domain classes inherit from it, and it
  provides the scikit-learn-style transformer layer the domains are built on. No
  MCP tool exposes it: an agent cannot compose a pipeline through the protocol.
- **`domains/time_series_analysis/`** — 29 modules — is unreachable. The live
  time-series code is the single module `domains/time_series.py`, which is what
  `domains/__init__.py` and the adapters import. Tracked as issue #28.

## Design documents

Six documents live in `docs/architecture/` in the repository and none of them are
published: `FIRST_PRINCIPLES.md`, `DESIGN_DECISIONS.md`, `CORE_PATTERNS.md`,
`LIBRARY_STRATEGY.md`, `domain-integration-layer.md` and
`integration-shims-architecture.md`.

All six predate the implementation and describe a system that was designed but
not built. `IntentResolver`, `DomainRegistry`, `DomainBridge`, `SemanticWrapper`,
`AdaptiveProcessor`, `DataStream` and `AnalysisResult` appear in them and in no
source file. Their worked examples call `explore_relationships`, `profile_data`,
`model_relationships` and `generate_business_insights`, none of which is a tool
this server registers — and the shipped interface is the one those examples hold
up as the pattern to avoid: `analyze_clusters` does take `n_clusters`, and
`analyze_hypothesis_test` does take `alpha`.

The first two are the project's design constitution and still govern how the
interface evolves, which is why they are kept in the repository. That is a
different job from documenting the server, and it does not
earn them a place in the published docs: a reader who found a prescribed API here
would go looking for tools that do not exist. Publishing a proposal as a
description is the failure this page exists to correct, and the rule has to apply
to the documents this page likes as much as to the ones it does not.

## Where to look next

- [Tools reference](../tools-reference.md) — all 71 tools, by category
- [Configuration](../configuration.md) — every setting and where it goes
- [Data sources](../data-sources/index.md) — what each source type supports
- [Analytical domains](../domains/index.md) — what each domain offers
