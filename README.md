<p align="center">
  <img src="assets/logo.png" alt="LocalData MCP Server" width="250">
</p>

# LocalData MCP Server

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-yellow.svg)](LICENSE)
[![GitHub Release](https://img.shields.io/github/v/release/ChrisGVE/localdata-mcp)](https://github.com/ChrisGVE/localdata-mcp/releases)
[![CI](https://img.shields.io/github/actions/workflow/status/ChrisGVE/localdata-mcp/ci.yml?branch=main&label=CI)](https://github.com/ChrisGVE/localdata-mcp/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/localdata-mcp.svg)](https://pypi.org/project/localdata-mcp/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastMCP](https://img.shields.io/badge/FastMCP-Compatible-green.svg)](https://github.com/jlowin/fastmcp)

<!-- mcp-name: io.github.chrisgve/localdata-mcp -->

SQL over your local data files, for LLM agents.

**Every datasource becomes a database.** A CSV, a TSV, a SQLite file — each is
attached under a nickname, and each call names the nickname it is for. That one
idea is why a spreadsheet and a SQLite file are the same kind of thing here, and
why there are seven tools rather than seventy: a database already has verbs, and
they are the same verbs whatever filled it.

Tables are addressed by their own names inside the datasource you named —
`query(nickname="shop", sql="SELECT * FROM sales")`. To look one file up against
another, `create` copies the second into the first; the join is then an ordinary
statement over two tables in one database.

These are raw capabilities, not a workflow. Nothing here guesses a join key,
decides an index would help, or turns a mismatch into a sentence — those need to
know what was actually asked, and the caller is the one holding that.

> **Rebuild in progress.** This branch is a ground-up rewrite. It currently
> supports flat files (`.csv`, `.tsv`, `.txt`) and SQLite, done properly — see
> [Where this is going](#where-this-is-going). Earlier releases claimed far more
> surface than they held; this one claims what it has.

## Quick start

```bash
uv tool install localdata-mcp     # or: uvx localdata-mcp
```

Add it to your MCP client configuration:

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
attach("./sales.csv")                     # → {"nickname": "sales", "tables": ["sales"], ...}
query("sales", "SELECT sku, sum(qty) FROM sales GROUP BY sku")
```

`attach` derives the nickname from the filename and **returns the one it
actually used** — if that name was taken, you get `sales_2` and are told what it
collided with. Always read it back rather than assuming.

## The eight verbs

| Verb | What it does |
| --- | --- |
| `attach(database, nickname?, writable?, delimiter?)` | Open a datasource as a database. Returns the nickname used, plus anything it collided with or evicted. A file holding several tables (a workbook's sheets, a page's tables) becomes a database holding all of them. |
| `detach(nickname)` | Close it and free the slot. |
| `query(nickname, sql, path?, force?, delimiter?)` | Run SQL. **Reads only.** Returns the whole result; with `path`, writes it to a file whose suffix chooses the format. |
| `info(nickname?, table?)` | Three altitudes: the whole session, one datasource, or one table's columns, row count and indexes. |
| `create(nickname, type, table?, source?, columns?, delimiter?)` | `type="table"` lands a file *inside* an open database; `type="index"` indexes columns of a table already there. |
| `update(nickname, type, name, to)` | Rename a table, keeping its rows, types and indexes. For when the file chose the name — a workbook's `Sheet1`. |
| `drop(nickname, type, name)` | Remove a table or an index. |
| `save(nickname, path, force?)` | Write the database out to a file you keep. |

### Looking one file up against another

The common request — *"can you cross-reference this with that other file?"* —
uses `create`, not a second `attach`:

```python
attach("./sales.csv")                                              # → "sales"
create("sales", type="table", source="./prices.csv")
query("sales", "SELECT s.sku, s.qty * p.price AS total "
               "FROM sales s JOIN prices p ON s.sku = p.sku")
```

`query` reads and only reads — `INSERT`, `CREATE TABLE`, `CREATE VIEW` and the
rest are refused there however writable the datasource is. Changing a slot goes
through `create` and `drop`, which is what `writable=true` governs.

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

If either drags, index the key first and ask again. Nothing is indexed unless
you say so, and `info` tells you what already is:

```python
create("sales", type="index", table="prices", columns=["sku"])     # → "ix_prices_sku"
drop("sales", type="index", name="ix_prices_sku")
```

### Nothing survives unless you save it

Attached data lives until `detach`, or until the server stops. `save` writes the
whole database — including tables you added — to a file:

```python
save("sales", "./analysis.db")
```

Attaching that file again later is an ordinary attach, so it comes back
**read-only** unless you pass `writable=true`.

An existing file is refused. The destination is a name a person chose, so
whether to replace what is already there is their decision — ask, and pass
`force=true` once they have said yes. A file an attached datasource is sitting
on is refused either way, forced or not.

## What it will not do

These are deliberate, and each one is measured rather than assumed.

- **Ten datasources at once.** A chosen number, not a forced one: each slot
  holds live connections and, until it is spilled, memory. The oldest is evicted
  when the limit is reached, and the eviction is *reported* with everything
  needed to rebuild it.
- **Write is not the default.** Anything attached from outside is read-only; the
  grant is per-attach and is carried by the connection's own URI, so SQLite
  enforces it rather than a check that could be reached around. A database built
  from a flat file is yours, and is writable.
- **The same file twice is refused**, naming the datasource already holding it.
- **Paths are confined** to the working directory and any configured roots.
  Symlinks and `..` are resolved before the check, not after.
- **A mixed-type column is flagged on load.** An `avg()` over a column holding
  both numbers and text silently counts the text as zero and keeps it in the
  denominator — the answer is wrong and nothing says so, unless something says
  so.

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

## Configuration

Optional. Discovery is a cascade — **first found wins**, not a merge:

1. `$LOCALDATA_CONFIG_PATH`
2. `$XDG_CONFIG_HOME/localdata/config.toml` (defaults to `~/.config`)
3. `./localdata.toml`
4. `~/Library/Application Support/localdata/config.toml` (macOS) or `%APPDATA%\localdata\config.toml` (Windows)

```toml
[workspace]
slots = 10                # 1-10; SQLite refuses the eleventh attachment
memory_budget_mb = 100    # before a database is moved to disk

[paths]
roots = ["~/data"]        # in addition to the working directory
path_limited = true       # false removes the confinement entirely

[network]
enabled = false           # true allows a datasource URL naming a service
```

**An unknown section or key is refused rather than ignored.** A mistyped
`path_limitted = false` that silently kept the safe default would be a security
setting you believe you have changed.

## Claude Code plugin

The repository doubles as a Claude Code plugin, registering the server and
shipping the `local-data` skill — the mental model, the naming conversation, and
how to phrase an incomplete join in the user's own words rather than as an
anti-join. The tools stay mechanical precisely because the skill carries that.

## Where this is going

Level 0 is a gate: flat files and SQLite, done well. Only then more input and
output formats, then more backends — file-based and endpoint-based, anything
SQLAlchemy speaks. Building blocks first.

The full specification is in [docs/architecture/LEVEL0.md](docs/architecture/LEVEL0.md).

## Documentation

- [Level 0 specification](docs/architecture/LEVEL0.md) — the premise, the three user journeys, the eight verbs
- [Measured constraints](docs/CONSTRAINTS.md) — the behaviour that shapes the design, with the numbers behind it

Those two are the whole of it. Everything else that used to live under `docs/` described
a data-science platform that was never built, and now sits in
[`non_factual/`](non_factual/README.md) — quarantined rather than deleted, and not to be
cited or acted on until somebody has checked it against the code.

## Development

```bash
git clone https://github.com/ChrisGVE/localdata-mcp.git
cd localdata-mcp
uv sync --all-extras
uv run --extra dev pytest
```

`-m 'not slow'` skips the volume suite.

## Contributing

Contributions are welcome. Please read [CONTRIBUTING.md](CONTRIBUTING.md) before
submitting a pull request.

## License

Apache License 2.0 — see [LICENSE](LICENSE) for details.
