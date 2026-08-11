<p align="center">
  <img src="https://raw.githubusercontent.com/ChrisGVE/localdata-mcp/main/assets/logo.png" alt="LocalData MCP Server" width="250">
</p>

# LocalData MCP Server

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-yellow.svg)](LICENSE)
[![GitHub Release](https://img.shields.io/github/v/release/ChrisGVE/localdata-mcp)](https://github.com/ChrisGVE/localdata-mcp/releases)
[![PyPI version](https://img.shields.io/pypi/v/localdata-mcp.svg)](https://pypi.org/project/localdata-mcp/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

The release and PyPI badges both resolve to 2.x, which is the last published
version and a different product — see the note below.

<!-- mcp-name: io.github.chrisgve/localdata-mcp -->

SQL over your local data files and databases, for LLM agents.

**Every datasource becomes a database.** A CSV, a workbook, a Parquet file, a
SQLite or DuckDB file, a PostgreSQL URL — each is attached under a nickname, and
each call names the nickname it is for. That one idea is why a spreadsheet and a
warehouse are the same kind of thing here, and why there are eight tools —
**verbs**, the word this repository uses for them throughout — rather than
seventy: a database already has verbs, and they are the same verbs whatever
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
> release is **2.0.0**, a different product — a data-science platform with
> fifty-three tools, none of which survive here. (`main` carries an unreleased
> 2.1.0 that grew that to seventy-one; the CHANGELOG counts from there, because
> that is the surface this rewrite replaced.) Every PyPI and `uvx` command that
> names `localdata-mcp` fetches 2.0.0 until 3.0.0 is tagged, so the Quick start
> below installs from a clone instead. `CHANGELOG.md` has the full list of what
> changed, and it is long.

## Quick start

**Install from a clone. Do not install from PyPI yet** — the published package
is 2.0.0, and none of what this document describes is in it:

```bash
git clone https://github.com/ChrisGVE/localdata-mcp.git
cd localdata-mcp
uv sync --all-extras                      # every format library and every database driver
```

Once 3.0.0 is tagged, and not before, the same thing installs from PyPI:

```bash
uv tool install localdata-mcp             # CSV, TSV, TXT, FWF, JSON, JSONL, NDJSON, XML
uv tool install 'localdata-mcp[all]'      # every format and every database driver
```

The base install declares no format libraries and no database drivers. **Eight of
the eighteen readable formats are guaranteed by it**; the remaining ten are
declared behind six extras (`parquet` covering all three columnar suffixes, `excel`,
`ods`, `xls`, `numbers`, `yaml`), and each database is one more (`postgres`,
`duckdb`, `oracle`, …). One extra is for the write side only: `markdown`
installs the table formatter `.md` output needs, and there is no `.md` reader.

In practice a base install reads **ten of the eighteen** and writes nine, because `fastmcp`
requires `PyYAML` unconditionally and `.yaml`/`.yml` therefore work without their
extra. That is a fact about today's dependency graph and not a promise this
project makes: `yaml` stays the declared extra, and code that needs YAML should
install it rather than rely on a transitive arriving.

A format is **known whether or not its library is installed**, so a missing one
is an instruction rather than a mystery:

```
Reading .parquet needs pyarrow, which is not installed. Install it with:
pip install 'localdata-mcp[parquet]' (or [all] for every format).
```

Quote the brackets — `[` and `]` are glob characters in zsh.

Add the server to your MCP client configuration. From a clone, name the clone —
`localdata-mcp` is not on `PATH` until the package is installed:

```json
{
  "mcpServers": {
    "localdata": {
      "command": "uv",
      "args": ["run", "--all-extras", "--directory", "/path/to/localdata-mcp", "localdata-mcp"]
    }
  }
}
```

After `uv tool install 'localdata-mcp[all]'` puts the entry point on `PATH`,
`"command": "localdata-mcp"` with no arguments is equivalent. It has to be the
`[all]` form: the bare `uv tool install localdata-mcp` carries no extras, so that
configuration would reach ten of the eighteen formats and one of the eighteen
backends — which is what `--all-extras` above exists to avoid.

Then point it at a file and ask:

```python
attach("./sales.csv")
# → {"ok": true, "nickname": "sales", "kind": "file",
#    "source": "…/sales.csv", "writable": true,
#    "tables": ["sales"], "loaded": [{"table": "sales", "rows": 4,
#    "source": "…/sales.csv",
#    "columns": [{"name": "sku", "type": "TEXT"},
#                {"name": "qty", "type": "INTEGER"}], "mixed_columns": []}],
#    "collided_with": null, "evicted": null}

query("sales", "SELECT sku, sum(qty) AS qty FROM sales GROUP BY sku")
# → {"ok": true, "columns": ["sku", "qty"], "rows": [["a", 5], ["b", 7], ["c", 1]],
#    "row_count": 3}
```

`info()` with no arguments answers for the session rather than for a datasource —
what is open, how many slots there are, and which paths the server will accept:

```python
info()
# → {"ok": true, "datasources": [], "slots_used": 0, "slots_available": 10,
#    "roots": ["/path/to/your/project"], "path_limited": true}
```

**`slots_available` is the slot limit, not the number of free slots.** It is
constant for the session, so a full session reports `{"slots_used": 10,
"slots_available": 10}` rather than zero. Free slots are the subtraction:
`slots_available - slots_used`. The example above is an empty session, which is
the one occupancy where the two readings give the same number.

`attach` derives the nickname from the filename and **returns the one it actually
used** — if that name was taken by a different source, you get `sales_2` and
`collided_with` names the slot that forced it. Always read it back rather than
assuming. The answer for a file already carries its columns, types, row count and
any warning, so there is nothing for an `info` call straight afterwards to add.

**A slot is one attached datasource** — its database, the two connections that
reach it, and the nickname it answers to. There are ten of them; the word is
used throughout this document and in `docs/architecture/LEVEL0.md`, which states
the same premise from the other end — *a slot **is** a database*, since that is
what every datasource becomes.

## The eight verbs

| Verb | What it does |
| --- | --- |
| `attach(database, nickname?, writable?, delimiter?)` | Open a datasource as a database. Returns the nickname used, plus anything it collided with or evicted. A workbook becomes a database holding a table per sheet, and a `.numbers` document one per table. |
| `detach(nickname)` | Close it and free the slot. Deletes the temp file if the slot had been spilled to disk (see [Memory](#memory)). |
| `query(nickname, sql, path?, force?, delimiter?)` | Run SQL. **Reads only.** Returns the whole result; with `path`, writes it to a file whose suffix chooses the format and answers `rows_written` and the column names instead of the rows. |
| `info(nickname?, table?)` | Three levels of detail: bare → the session (see above); nickname → its tables; nickname and table → columns, row count and indexes. |
| `create(nickname, type, table?, source?, columns?, delimiter?)` | `type="table"` lands a **file** *inside* an open database — one holding a single table, since `create` makes one; `type="index"` indexes columns of a table already there. |
| `update(nickname, type, name, to)` | Rename a table, keeping its rows, types and indexes. For when the file chose the name — a workbook's `Sheet1`, which arrives as `sheet1`. |
| `drop(nickname, type, name)` | Remove a table or an index. |
| `save(nickname, path, force?)` | Write the database out to a SQLite file you keep — see [Not every verb reaches every backend](#not-every-verb-reaches-every-backend). |

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
| Flat | `.csv` `.tsv` `.txt` `.fwf` | `.csv` `.tsv` `.txt` |
| Structured | `.json` `.jsonl` `.ndjson` `.yaml` `.yml` `.xml` | `.json` `.jsonl` `.ndjson` `.yaml` `.yml` `.xml` |
| Spreadsheet | `.xlsx` `.xlsm` `.xls` `.ods` `.numbers` | `.xlsx` `.ods` |
| Columnar | `.parquet` `.feather` `.orc` | `.parquet` `.feather` `.orc` |
| Markdown | — | `.md` |

**Eighteen database backends**: SQLite, DuckDB, PostgreSQL, MySQL, MariaDB, SQL
Server, Oracle, ClickHouse, CockroachDB, YugabyteDB, Trino, MonetDB, CrateDB,
Firebird, openGauss, YDB, Databend and Exasol. SQLite and DuckDB are normally
files, and can also be reached by URL; the other sixteen are URLs only, and each
of the sixteen is exercised against a container of its own in
`docker-compose.test.yml`.

**Each backend needs its driver extra installed, and the refusal will not tell
you which.** A missing *format* library produces the instruction quoted above. A
missing *driver* produces one of two raw exceptions, neither naming the extra:
eight backends give `ModuleNotFoundError` naming the Python module
(`No module named 'psycopg'`), and nine give `NoSuchModuleError. Can't load
plugin: sqlalchemy.dialects:<name>` naming a SQLAlchemy dialect entry point —
which for CockroachDB, openGauss and YDB is a composite name (`yql.ydb`) that is
not importable as a module path at all.

Which of those two exceptions you get is mostly whether SQLAlchemy ships the
dialect itself, but not entirely: CrateDB and Exasol are third-party dialects
that would raise `NoSuchModuleError` too, and give `ModuleNotFoundError` only
because this server imports their driver eagerly to build a type converter,
before the engine exists. So six of the eight `ModuleNotFoundError` backends are
the plain case and two are ours.

Both shapes are what the *URL* route returns. A datasource attached as a
**file** — the common DuckDB route — is wrapped by the attach path instead, and
reads `Could not attach <path>: Can't load plugin: sqlalchemy.dialects:duckdb`:
`slots.py:482` renders the exception's message, so the class name is not part of
it.

Either way the refusal never names the extra, which is why the map is here. Two
rows point at another backend's extra:

| Backend | Extra |
|---|---|
| SQLite | none — stdlib |
| DuckDB | `duckdb` |
| PostgreSQL | `postgres` |
| MySQL | `mysql` |
| MariaDB | **`mysql`** — addressed `mariadb+pymysql` |
| SQL Server | `mssql` |
| Oracle | `oracle` |
| ClickHouse | `clickhouse` |
| CockroachDB | `cockroachdb` |
| YugabyteDB | **`postgres`** |
| Trino | `trino` |
| MonetDB | `monetdb` |
| CrateDB | `cratedb` |
| Firebird | `firebird` |
| openGauss | `opengauss` |
| YDB | `ydb` |
| Databend | `databend` |
| Exasol | `exasol` |

`databases` installs all fifteen at once — fifteen and not eighteen because
SQLite needs no extra and MariaDB and YugabyteDB share another backend's.
`formats` installs the seven format extras (the six for reading, plus `markdown`
for `.md` output), and `all` installs both.

### Not every verb reaches every backend

The eight verbs are the whole surface everywhere. Three of them cannot be
carried out on some engines, and the refusal says so and names the way round:

- **`save` writes out a database this server is holding.** A file-derived slot
  — every CSV, workbook, Parquet file and the like — is an in-memory SQLite
  database, and `save` writes it out. **A slot reached over its own connection
  has no such database, so `save` is refused on all seventeen non-SQLite
  backends, a local DuckDB file included.** The way to keep the result is three
  calls: `query(path=…)` writes the rows to a file, `attach` on that file makes
  a database of your own — file-derived, so writable — and `save` writes it
  out. The middle verb is `attach`, not `create`: `create` lands a file inside a
  slot that is already open. The refusal says exactly this.
- **`create(type="index")` is refused on ClickHouse, Trino, CrateDB, Databend
  and Exasol** — each for its own reason: indexes that cannot be reflected, no
  storage to index, every column indexed already, a statement that compiles to
  nothing, or an engine that maintains its own.
- **`update(type="table")` is refused on Firebird**, which has no rename-table
  statement.

Per-backend detail, and what each engine needed, is in
[`docs/architecture/LEVEL0.md`](docs/architecture/LEVEL0.md#the-backends-and-what-each-one-needed).

**A file may hold more than one table.** A workbook becomes a database with a
table per sheet and a `.numbers` document one **per table** — a Numbers sheet is
a canvas that may carry several — so nothing in the file is unreachable. Each
table arrives under its own name put through the same snake_case rule as a
nickname, so `Sheet1` arrives as `sheet1` and `Q2 Prices` as `q2_prices`; where
two tables in one document share a name, the sheet name is prefixed to break the
collision. `update` renames one, keeping its rows and indexes. A JSON, YAML or
XML document is the other way round: one candidate table loads with a note naming
the key it came from, and **two are refused, naming both**, rather than one being
picked silently.

`create(type="table")` is stricter than `attach` about the same file, and for a
reason worth stating: **it makes one table, so a source holding more than one is
refused**, naming them. A two-sheet workbook that attaches happily as a database
of two tables cannot be read into an open database as a table, and `table=` does
not select a sheet — there is no way to take one sheet out of a workbook this
way. The refusal says to attach the file as its own datasource, which gives you
the workbook as a database of its own; it does not put those sheets beside the
tables you already have.

There is a route to one sheet beside them, and it takes three calls: attach the
workbook, write the sheet you want out to a flat file with `query(path=…)`, then
`create` a table in the open database from that file.

```
attach("/path/book.xlsx")                                   # → "book"
query("book", "SELECT * FROM prices", path="/path/prices.csv")
create(nickname="shop", type="table", source="/path/prices.csv")
```

The backend catalogue is closed rather than open-ended, and a database is in
scope **if and only if an open-source SQLAlchemy adapter exists**. TiDB and
HyperSQL fail that rule.
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
remove one use drop, to rename one use update; there is no verb for arbitrary
DDL by design.
```

Changing a slot goes through `create`, `update` and `drop`, which is what
`writable=true` governs. The enforcement is the connection's rather than a check
that could be reached around, and each backend goes as far as it can: SQLite
refuses at statement preparation through an authorizer, DuckDB opens
`access_mode=read_only`, MySQL and MariaDB open a read-only session because their
DDL commits itself, ClickHouse carries `readonly=1` because it has no transaction
to withhold. **Two are exceptions.** Oracle commits DDL before anything
can object and has no session-level read-only posture, so a `CREATE` sent to
`query` there really does take effect; its DML still rolls back. CrateDB has no
transactions at all, so **both** a refused `CREATE` and a refused `INSERT` stand
— the worse of the two, and the refusal names only `CREATE`/`DROP`, so on
CrateDB it must not be read as meaning nothing happened
([#84](https://github.com/ChrisGVE/localdata-mcp/issues/84)).

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
# → {"ok": true, "nickname": "sales", "index": "ix_prices_sku",
#    "table": "prices", "columns": ["sku"], "unique": false}
drop("sales", type="index", name="ix_prices_sku")
# → {"ok": true, "nickname": "sales", "dropped": "ix_prices_sku",
#    "table": "prices"}
```

## Nothing survives unless you save it

Attached data lives until `detach`, or until the server stops. `save` writes the
whole database — including tables you added — to a file:

```python
save("sales", "./analysis.db")
# → {"ok": true, "nickname": "sales", "path": "…/analysis.db",
#    "tables": ["prices", "sales"]}
```

Attaching that file again later is an ordinary attach, so it comes back
**read-only** unless you pass `writable=true`.

**`save` only works on a slot this server built** — anything that came from a
file, and SQLite. A slot reached over its own connection (PostgreSQL, a DuckDB
file, any of the other fifteen backends) has no local database to write out, and
`save` is refused there; send the result to a file with `query(path=…)`, `attach`
that file as a datasource of your own, and save that. See [Not every verb
reaches every backend](#not-every-verb-reaches-every-backend).

An existing file is refused. The destination is a name a person chose, so whether
to replace what is already there is their decision — ask, and pass `force=true`
once they have said yes. A file an attached datasource is sitting on is refused
either way, forced or not.

## What it refuses

These are deliberate. Some are forced by a measurement recorded in
`docs/CONSTRAINTS.md`; the rest are decisions, and each bullet says which it is.

- **Ten datasources at once.** A chosen number, not a forced one: each slot holds
  two live connections and, until it is spilled to disk, memory. The oldest is
  evicted when the limit is reached, and the eviction is *reported* in `evicted`
  with everything needed to rebuild it.
- **Write is not the default.** Anything attached from outside is read-only; the
  grant is per-attach and is carried by the connection itself, so changing it
  means `detach` and then attach again with `writable=true` — re-attaching a
  source that is still open is refused by the rule below. A database built from
  a flat file is yours, and is writable.
- **The same file twice is refused**, naming the datasource already holding it
  and the tables it holds.
- **Paths are confined** to the working directory and any configured roots.
  Symlinks and `..` are resolved before the check, not after, so `/etc/hosts`
  is refused as `/private/etc/hosts`.
- **A network URL is refused** until `network.enabled = true` is set in the
  configuration file. The refusal masks the password.
- **`.xlsx` and `.ods` are refused above 65,535 rows** — the older worksheet's
  own limit, and what bounds the writer's memory. Uncapped, `.xlsx` held 12.9 GB
  while writing a million rows (`docs/CONSTRAINTS.md` §10.7). `.ods` is 13.5×
  slower than `.xlsx` at the cap; reach for `.xlsx` unless OpenDocument was
  specifically asked for.

## What it tells you rather than fixing

The other half of the same policy: where the server can see a problem but the
answer belongs to whoever wrote the file, it reports and carries on.

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
- **A timestamp out of a typed format is reported only on the column itself**,
  and this is the one entry here with **no warning attached**. It loads as an
  integer count of nanoseconds since the epoch, which orders correctly and
  compares wrongly against a date string; the `temporal` and `unit` fields on the
  column say so and nothing else does. *Dates*, below, has the detail.

  The two directions differ deliberately. On the way in, `attach` and `create`
  **refuse** a `delimiter` for any suffix that has no separator. On the way out,
  `query(path=…)` **ignores** it for such a suffix, so one default can be carried
  across a mixed batch of destinations without the caller stripping it per file.

## Dates

A flat file holds dates as text, and text compares as text — so `'30.11.2023'`
sorts *after* `'01.03.2025'`, `ORDER BY` runs backwards and `max()` returns the
earliest instant. Measured across the spellings of five instants spanning
three years, **every day-first and every month-name form ordered wrongly, and
returned the earliest instant from `max()`** — silently, with no error and no
warning (`docs/CONSTRAINTS.md` §8.1, which asks that the classes be quoted rather
than the counts: how many spellings land in each class depends on which spellings
the fixture happened to include). The year span is what exposes them:
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
`…T00:00:00Z` throughout. Text in no recognised standard is left alone and
reported, with the offending values named.

All of that is the flat-file case, where the only thing a date can be is text. A
**typed** format — `.parquet`, `.feather`, `.orc`, and the spreadsheet formats —
carries a timestamp type of its own, and such a column arrives as an integer
count of **nanoseconds since the Unix epoch**:

```python
{"name": "order_date", "type": "INTEGER", "temporal": "timestamp", "unit": "nanoseconds_since_epoch"}
```

Ordering and `max()` are correct on it, for the same reason Unix time is left
untouched above. What is not correct is comparing it against a date string:
`WHERE order_date > '2024-03-02'` compares an integer to text and returns **zero
rows with no error** — the same silent wrong answer this section opens with,
arrived at from the other direction. Compare against a tick value, or read `unit`
and convert. Two things make this the sharpest edge in the tool:

- **Nothing warns about it.** Every entry under *What it tells you rather than
  fixing* pairs its problem with a warning; this one has no warning channel at
  all, so the `temporal` and `unit` pair on the column is the only signal there
  is.
- **That signal does not survive a round trip.** A column written out with
  `query(path=…)` or `save()` and attached again comes back as a bare `INTEGER`
  with no temporal key, so the second reader cannot know what the integers mean.

Whether the split is the right design is open —
[#87](https://github.com/ChrisGVE/localdata-mcp/issues/87).

## Memory

The working budget is small by default — 100 MB — because this runs on a machine
doing other things. When a load crosses it, **the load finishes**: the overshoot
is tolerated once. The *next* operation then **spills** the largest in-memory
database — moves it out to a temp file and re-attaches it under the same
nickname — and nothing in any response mentions that it happened. A slot that has
been spilled is on disk rather than in memory; `detach` deletes the temp file,
and so does eviction and termination.

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

Optional. **`$LOCALDATA_CONFIG_PATH` is not the head of the search — it replaces it.** Set it and
the three locations below are not consulted at all, and a value naming something that is not a
file is refused rather than skipped, on the first tool call. Unset it to use the search.

With it unset, discovery is a cascade — **first found wins**, not a merge:

1. `$XDG_CONFIG_HOME/localdata/config.toml` (defaults to `~/.config`)
2. `./localdata.toml`
3. `~/Library/Application Support/localdata/config.toml` (macOS) or `%APPDATA%\localdata\config.toml` (Windows)

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

`LOCALDATA_CONFIG_PATH` is the only environment variable that carries any part of
the configuration, and it *locates* the file rather than holding a setting. (The
cascade also consults `XDG_CONFIG_HOME` and, on Windows, `APPDATA`, but those
belong to the operating system and name a directory, not a setting.)

**An unknown section or key is refused rather than ignored, and the refusal is
not silent.** A mistyped `path_limitted = false` that silently kept the safe
default would be a security setting you believe you have changed. What the
refusal looks like in practice: the configuration is read on the **first tool
call**, not at startup, so the server launches and the MCP handshake completes
normally, and then the first `attach` fails with a message naming the bad key and
the keys that section does know. If a client shows the tools but every call
fails, read the error text — it names the typo.

## Claude Code plugin

The repository doubles as a Claude Code plugin, registering the server and
shipping the `local-data` skill — the mental model, the naming conversation, and
how to phrase an incomplete join in the user's own words rather than as an
anti-join. The tools stay mechanical precisely because the skill carries that.

**There is no install route for it yet.** `claude plugin install` resolves a
plugin from a marketplace, and this repository publishes no marketplace entry —
there is no `.claude-plugin/marketplace.json` and nothing lists the plugin
elsewhere. Until one exists, use the ordinary client configuration above; it
reaches the same server, and the skill is the only thing the plugin adds on top.

When that route does open, `.claude-plugin/plugin.json` launches the server with
`uv run --all-extras`, so the first start resolves every format library and every
database driver — 139 distributions, slow on a cold `uv` cache and a few seconds
after that. That is deliberate: the plugin builds its own environment, and
without the flag it would reach ten of the eighteen formats and one of the
eighteen backends while that file's own description advertises all of them. The
client configuration above carries the same flag for the same reason, so neither
path depends on `uv sync --all-extras` having been run in the clone first.

## Documentation

- [Level 0 specification](docs/architecture/LEVEL0.md) — the premise, the three user journeys, the eight verbs
- [Measured constraints](docs/CONSTRAINTS.md) — the behaviour that shapes the design, with the numbers behind it
- [The shipped skill](skills/data/local-data/SKILL.md) — how an agent should talk to a user about their data
- [Changelog](CHANGELOG.md) — and read the 3.0.0 entry before upgrading from 2.x, because none of that tool surface survived
- [Contributing](CONTRIBUTING.md) — development setup, tests, and which document owns what
- [What CI does and does not do](.github/WORKFLOWS.md) — read it before trusting a green check

Those six and this file are the documents kept current;
[`CONTRIBUTING.md`](CONTRIBUTING.md#documentation) is the single inventory and
says which one owns what, so a change lands in exactly one of them.

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
containers; this machine runs six at a time and starves them at around seven, so
the catalogue takes five batches
([#46](https://github.com/ChrisGVE/localdata-mcp/issues/46)).

**No workflow gates this branch**, so that local run is the only evidence there
is. `.github/WORKFLOWS.md` says which workflows exist, which of them work, and
what has to be fixed before a release tag is pushed.

The level-0 surface is built and its scope is closed; what is left before it
ships is documentation currency and a pass driving the live server through a real
client. `docs/architecture/LEVEL0.md` ends with what remains.

## Contributing

Contributions are welcome. Please read [CONTRIBUTING.md](CONTRIBUTING.md) before
submitting a pull request.

## License

Apache License 2.0 — see [LICENSE](LICENSE) for details.
