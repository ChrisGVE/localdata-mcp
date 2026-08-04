<p align="center">
  <img src="assets/logo.png" alt="LocalData MCP Server" width="250">
</p>

# LocalData MCP Server

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-yellow.svg)](LICENSE)
[![GitHub Release](https://img.shields.io/github/v/release/ChrisGVE/localdata-mcp)](https://github.com/ChrisGVE/localdata-mcp/releases)
[![PyPI version](https://img.shields.io/pypi/v/localdata-mcp.svg)](https://pypi.org/project/localdata-mcp/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastMCP](https://img.shields.io/badge/FastMCP-Compatible-green.svg)](https://github.com/jlowin/fastmcp)

<!-- mcp-name: io.github.chrisgve/localdata-mcp -->

SQL over your local data files and databases, for LLM agents.

**Every datasource becomes a database.** A CSV, a workbook, a Parquet file, a
SQLite or DuckDB file, a PostgreSQL URL — each is attached under a nickname, and
each call names the nickname it is for. That one idea is why a spreadsheet and a
warehouse are the same kind of thing here, and why there are eight tools rather
than seventy: a database already has verbs, and they are the same verbs whatever
filled it.

Tables are addressed by their own names inside the datasource you named —
`query(nickname="shop", sql="SELECT * FROM sales")`. **The nickname is an
argument, never a prefix on the table**: qualifying the table with it names a
table in a database this statement was not pointed at, and the statement fails.
One statement reaches one datasource. To look one file up against another,
`create` copies the second into the first; the join is then an ordinary statement
over two tables in one database.

These are raw capabilities, not a workflow. Nothing here guesses a join key,
decides an index would help, or turns a mismatch into a sentence — those need to
know what was actually asked, and the caller is the one holding that.

> **This branch is 3.0.0 and it is not on PyPI yet.** The latest published
> release is **2.0.0**, a different product — a data-science platform with 71
> tools, none of which survive here. `uv tool install localdata-mcp` installs
> that one until 3.0.0 is tagged. To run what this document describes, clone the
> repository (see [Development](#development)). `CHANGELOG.md` has the full list
> of what changed, and it is long.

## Quick start

```bash
uv tool install localdata-mcp             # CSV, TSV, TXT, FWF, JSON, JSONL, NDJSON, XML
uv tool install 'localdata-mcp[all]'      # every format and every database driver
```

The base install carries no format libraries and no database drivers. Eight of
the eighteen readable formats need nothing beyond it; the rest arrive as extras
(`parquet`, `excel`, `ods`, `xls`, `numbers`, `yaml`, `markdown`) and each
database is one more (`postgres`, `duckdb`, `oracle`, …). A format is **known
whether or not its library is installed**, so a missing one is an instruction
rather than a mystery:

```
Reading .parquet needs pyarrow, which is not installed. Install it with:
pip install 'localdata-mcp[parquet]' (or [all] for every format).
```

Quote the brackets — `[` and `]` are glob characters in zsh.

Add the server to your MCP client configuration:

```json
{
  "mcpServers": {
    "localdata": {
      "command": "localdata-mcp"
    }
  }
}
```

Then point it at a file and ask:

```python
attach("./sales.csv")
# → {"ok": true, "nickname": "sales", "kind": "file", "writable": true,
#    "tables": ["sales"], "loaded": [{"table": "sales", "rows": 4,
#    "columns": [{"name": "sku", "type": "TEXT"},
#                {"name": "qty", "type": "INTEGER"}], "mixed_columns": []}],
#    "collided_with": null, "evicted": null}

query("sales", "SELECT sku, sum(qty) AS qty FROM sales GROUP BY sku")
# → {"ok": true, "columns": ["sku", "qty"], "rows": [["a", 5], ["b", 7], ["c", 1]],
#    "row_count": 3}
```

`attach` derives the nickname from the filename and **returns the one it actually
used** — if that name was taken by a different source, you get `sales_2` and
`collided_with` names the slot that forced it. Always read it back rather than
assuming. The answer for a file already carries its columns, types, row count and
any warning, so there is nothing for an `info` call straight afterwards to add.

## The eight verbs

| Verb | What it does |
| --- | --- |
| `attach(database, nickname?, writable?, delimiter?)` | Open a datasource as a database. Returns the nickname used, plus anything it collided with or evicted. A workbook or `.numbers` document becomes a database holding all its sheets. |
| `detach(nickname)` | Close it and free the slot. Deletes the temp file if the slot had spilled. |
| `query(nickname, sql, path?, force?, delimiter?)` | Run SQL. **Reads only.** Returns the whole result; with `path`, writes it to a file whose suffix chooses the format. |
| `info(nickname?, table?)` | Three altitudes: every slot, one datasource's tables, or one table's columns, row count and indexes. |
| `create(nickname, type, table?, source?, columns?, delimiter?)` | `type="table"` lands a datasource *inside* an open database; `type="index"` indexes columns of a table already there. |
| `update(nickname, type, name, to)` | Rename a table, keeping its rows, types and indexes. For when the file chose the name — a workbook's `Sheet1`, which arrives as `sheet1`. |
| `drop(nickname, type, name)` | Remove a table or an index. |
| `save(nickname, path, force?)` | Write the database out to a SQLite file you keep. |

## What it reads, writes and connects to

Reading and writing are two registries keyed on the file suffix, and the suffix
is what chooses the format. Writing CSV under a `.parquet` name was the defect
this replaced, so a suffix with no writer is **refused by name, before the
destination is touched**:

```
No writer for '.docx'. The suffix chooses the format. Supported: .csv, .feather,
.json, .jsonl, .md, .ndjson, .ods, .orc, .parquet, .tsv, .txt, .xlsx, .xml,
.yaml, .yml
```

**Eighteen formats read, fifteen write:**

| Group | Read | Write |
|---|---|---|
| Flat | `.csv` `.tsv` `.txt` `.fwf` | `.csv` `.tsv` `.txt` `.md` |
| Structured | `.json` `.jsonl` `.ndjson` `.yaml` `.yml` `.xml` | same |
| Spreadsheet | `.xlsx` `.xlsm` `.xls` `.ods` `.numbers` | `.xlsx` `.ods` |
| Columnar | `.parquet` `.feather` `.orc` | same |

**Eighteen database backends**, each of which runs all eight verbs: SQLite,
DuckDB, PostgreSQL, MySQL, MariaDB, SQL Server, Oracle, ClickHouse, CockroachDB,
YugabyteDB, Trino, MonetDB, CrateDB, Firebird, openGauss, YDB, Databend and
Exasol. SQLite and DuckDB arrive as files; the other sixteen as a URL, and each
is exercised against a container of its own in `docker-compose.test.yml`.

**A file may hold more than one table.** A workbook or a `.numbers` document
becomes a database with a table per sheet, so nothing in the file is unreachable.
Sheet names go through the same snake_case rule as nicknames, so `Sheet1` arrives
as `sheet1` and `Q2 Prices` as `q2_prices`; `update` renames one, keeping its rows
and indexes. A JSON, YAML or XML document is the other way round: one candidate
table loads with a note naming the key it came from, and **two are refused,
naming both**, rather than one being picked silently.

The backend catalogue is closed rather than open-ended, and a database is in
scope **iff an open-source SQLAlchemy adapter exists**. TiDB and HyperSQL fail that rule.
Db2 and OceanBase pass it and still cannot be reached from a macOS host — Db2's
native client links against a C++ runtime macOS no longer ships, and OceanBase's
server needs an `rdtscp` instruction the Docker VM does not expose. Both are
measured rather than assumed, in `docs/CONSTRAINTS.md` §24 and §26.

## Looking one file up against another

The common request — *"can you cross-reference this with that other file?"* —
uses `create`, not a second `attach`:

```python
attach("./sales.csv")                                              # → "sales"
create("sales", type="table", source="./prices.csv")
query("sales", "SELECT s.sku, s.qty * p.price AS total "
               "FROM sales s JOIN prices p ON s.sku = p.sku")
```

`query` reads and only reads. `INSERT`, `CREATE TABLE`, `CREATE VIEW` and
`PRAGMA` are refused there however writable the datasource is:

```
query reads; it does not write. This statement asks to INSERT, which is refused
here even on a writable datasource. To add a table or an index use create, to
remove one use drop; there is no verb for arbitrary DDL by design.
```

Changing a slot goes through `create`, `update` and `drop`, which is what
`writable=true` governs. The enforcement is the connection's rather than a check
that could be reached around, and each backend goes as far as it can: SQLite
refuses at statement preparation through an authorizer, DuckDB opens
`access_mode=read_only`, MySQL and MariaDB open a read-only session because their
DDL commits itself, ClickHouse carries `readonly=1` because it has no transaction
to withhold. Oracle is the honest exception — it commits DDL before anything can
object and has no session-level read-only posture, so a `CREATE` sent to `query`
there really does take effect, and the refusal says so rather than claiming
otherwise.

Landing the second file inside the first database is not just tidier. **`save`
writes one database, not a join** — so attaching the two files separately gives
you an answer now and nothing to come back to, while this gives you a lookup you
can keep.

Whether the match is actually complete is an anti-join you write, in whichever
direction you care about:

```python
query("sales", "SELECT sku FROM sales WHERE sku NOT IN (SELECT sku FROM prices)")
query("sales", "SELECT sku FROM prices WHERE sku NOT IN (SELECT sku FROM sales)")
```

If either drags, index the key first and ask again. Nothing is indexed unless you
say so, and `info` tells you what already is:

```python
create("sales", type="index", table="prices", columns=["sku"])
# → {"ok": true, "index": "ix_prices_sku", "table": "prices",
#    "columns": ["sku"], "unique": false}
drop("sales", type="index", name="ix_prices_sku")
```

## Nothing survives unless you save it

Attached data lives until `detach`, or until the server stops. `save` writes the
whole database — including tables you added — to a file:

```python
save("sales", "./analysis.db")
# → {"ok": true, "path": "…/analysis.db", "tables": ["prices", "sales"]}
```

Attaching that file again later is an ordinary attach, so it comes back
**read-only** unless you pass `writable=true`.

An existing file is refused. The destination is a name a person chose, so whether
to replace what is already there is their decision — ask, and pass `force=true`
once they have said yes. A file an attached datasource is sitting on is refused
either way, forced or not.

## What it will not do

These are deliberate, and each one is measured rather than assumed.

- **Ten datasources at once.** A chosen number, not a forced one: each slot holds
  two live connections and, until it is spilled, memory. The oldest is evicted
  when the limit is reached, and the eviction is *reported* in `evicted` with
  everything needed to rebuild it.
- **Write is not the default.** Anything attached from outside is read-only; the
  grant is per-attach and is carried by the connection itself. A database built
  from a flat file is yours, and is writable.
- **The same file twice is refused**, naming the datasource already holding it
  and the tables it holds.
- **Paths are confined** to the working directory and any configured roots.
  Symlinks and `..` are resolved before the check, not after, so `/etc/hosts`
  is refused as `/private/etc/hosts`.
- **A network URL is refused** until `network.enabled = true` is set in the
  configuration file. The refusal masks the password.
- **A mixed-type column is flagged on load.** An `avg()` over a column holding
  both numbers and text silently counts the text as zero and keeps it in the
  denominator — the answer is wrong and nothing says so, unless something says
  so. The warning names the offending values, because a column that came from a
  spreadsheet stores all of them as text and `typeof()` cannot tell them apart.
- **What SQL cannot hold is encoded, not dropped.** A nested JSON value becomes
  its JSON text and a nested XML element its XML text — lossless and reversible,
  reachable with `json_extract(column, '$.key')` — and the warning names the
  columns. `pandas.read_xml` drops the subtree and reports nothing, which is why
  that reader is written directly on ElementTree.
- **An inference is stated.** Fixed-width column boundaries are inferred from
  which character positions are blank on every line, because nothing in the file
  declares them, and the warning says so.
- **Nothing sniffs a delimiter.** A semicolon-separated file read at `,` loads as
  one column named `a_b_c`, and the warning says exactly that and names the
  `delimiter` parameter. It does not re-read at a guessed separator: a guess that
  is usually right is the worst kind.
- **`.xlsx` and `.ods` are refused above 65,535 rows** — the older worksheet's
  own limit, and what bounds the writer's memory. Uncapped, `.xlsx` held 12.9 GB
  while writing a million rows (`docs/CONSTRAINTS.md` §10.7). `.ods` is 13.5×
  slower than `.xlsx` at the cap; reach for `.xlsx` unless OpenDocument was
  specifically asked for.

## Dates

A file holds dates as text, and text compares as text — so `'30.11.2023'` sorts
*after* `'01.03.2025'`, `ORDER BY` runs backwards and `max()` returns the
earliest instant. Measured across twenty-four spellings of five instants spanning
three years, **seven ordered wrongly**, and the day-first and month-name forms
among them returned the earliest instant from `max()` — silently, with no error
and no warning (`docs/CONSTRAINTS.md` §8.1). The year span is what exposes them:
`03/01/2025` and `01.03.2025` both sort correctly by accident inside a single
year.

The answer is not a better parser, because most of those spellings are genuinely
ambiguous: `01/03/2025` is the first of March or the third of January depending
on who wrote the file, and the file does not say. So exactly two forms are
recognised — **ISO 8601 extended** calendar dates and datetimes, and **Unix
time**, which is left untouched because integer comparison already is
chronological comparison.

A recognised column is rewritten into one canonical UTC spelling and stays text.
`attach` and `info` report it:

```python
{"name": "order_date", "type": "TEXT", "temporal": "iso8601_utc", "normalized": "UTC"}
```

An offset is honoured and normalised, so the same instant written `+00:00` and
`-05:00` compares equal. The original offset is **not** recoverable afterwards; a
file that needs it must keep it in a column of its own. One spelling means one
spelling for the whole column: a single value carrying a time takes the column to
`…T00:00:00Z` throughout. Anything in no recognised standard is left alone and
reported, with the offending values named.

## Memory

The working budget is small by default — 100 MB — because this runs on a machine
doing other things. When a load crosses it, **the load finishes**: the overshoot
is tolerated once. The *next* operation then moves the largest in-memory database
out to a temp file and re-attaches it under the same nickname, and nothing in any
response mentions that it happened.

There is deliberately no pre-flight estimate. What a file *will* cost is guessed
from metadata and guessed wrong; what a database *holds* is read from the
database. Residency is measured as `(page_count − freelist_count) × page_size`,
freelist-corrected so a slot emptied by a `DROP` is not moved to disk for data it
no longer has.

**A delimited file is read twice rather than held once.** `.csv`, `.tsv`, `.txt`
and `.fwf` go through a measuring pass and then an inserting pass, which stops the
load's peak tracking the file: 4,286 MB → 803 MB against a 1.22 GB CSV
(`docs/CONSTRAINTS.md` §28). The second read costs 1.4–1.75× wall clock, and
below about 150 MB it costs slightly more than it saves. The formats that are not
delimited are parsed whole by the libraries that read them — there is no chunk to
ask for and no line that is a row — so their peak still tracks the file.

## Configuration

Optional. Discovery is a cascade — **first found wins**, not a merge:

1. `$LOCALDATA_CONFIG_PATH`
2. `$XDG_CONFIG_HOME/localdata/config.toml` (defaults to `~/.config`)
3. `./localdata.toml`
4. `~/Library/Application Support/localdata/config.toml` (macOS) or `%APPDATA%\localdata\config.toml` (Windows)

```toml
[workspace]
slots = 10                # 1-10; each slot costs two connections and, unspilled, memory
memory_budget_mb = 100    # before a database is moved to disk

[paths]
roots = ["~/data"]        # in addition to the working directory
path_limited = true       # false removes the confinement entirely

[network]
enabled = false           # true allows a datasource URL naming a service
```

`LOCALDATA_CONFIG_PATH` is the only environment variable this server reads, and
it *locates* the file rather than carrying a setting.

**An unknown section or key is refused rather than ignored.** A mistyped
`path_limitted = false` that silently kept the safe default would be a security
setting you believe you have changed.

## Claude Code plugin

The repository doubles as a Claude Code plugin, registering the server and
shipping the `local-data` skill — the mental model, the naming conversation, and
how to phrase an incomplete join in the user's own words rather than as an
anti-join. The tools stay mechanical precisely because the skill carries that.

## Documentation

- [Level 0 specification](docs/architecture/LEVEL0.md) — the premise, the three user journeys, the eight verbs
- [Measured constraints](docs/CONSTRAINTS.md) — the behaviour that shapes the design, with the numbers behind it
- [The shipped skill](skills/data/local-data/SKILL.md) — how an agent should talk to a user about their data
- [Changelog](CHANGELOG.md) — and read the 3.0.0 entry before upgrading from 2.x, because none of that tool surface survived
- [Contributing](CONTRIBUTING.md) — development setup, tests, and which document owns what

Everything else that used to live under `docs/` described a data-science platform
that was never built, and now sits in [`non_factual/`](non_factual/README.md) —
quarantined rather than deleted, and not to be cited or acted on until somebody
has checked it against the code.

## Development

```bash
git clone https://github.com/ChrisGVE/localdata-mcp.git
cd localdata-mcp
uv sync --all-extras
.venv/bin/python -m pytest -q -m 'not slow'
```

`-m 'not slow'` skips the volume suite. The endpoint tests skip themselves — with
the command to start the container — when nothing is listening, so a Docker-free
run is green with several hundred skips. `docker-compose.test.yml` has the
containers; this machine runs six at a time before the Docker VM starves them, so
the catalogue takes five batches
([#46](https://github.com/ChrisGVE/localdata-mcp/issues/46)).

## Contributing

Contributions are welcome. Please read [CONTRIBUTING.md](CONTRIBUTING.md) before
submitting a pull request.

## License

Apache License 2.0 — see [LICENSE](LICENSE) for details.
