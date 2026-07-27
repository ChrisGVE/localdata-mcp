# Level 0 — files and databases, done well

Level 0 is the gate. Until the surface below is built and behaving, nothing else is
started. The reason is that everything above level 0 is the *same* building blocks
pointed at more kinds of source, so a block that is wrong here is wrong everywhere later.

The surface is built, and the gate has since opened onto its own breadth: more formats
and more backends are **still level 0**, because they are nothing new — the same verbs
pointed at more kinds of source. Formats are done, and so are the backends: SQLite,
DuckDB, PostgreSQL, MySQL, MariaDB, SQL Server and Oracle each run every verb, each
against a container of its own.

## The premise

**A slot is a database.** Not a table, not a file — a database, addressed by a
nickname. A CSV becomes a fresh in-memory database holding one table named after the
file; a workbook becomes one holding a table per sheet; a SQLite or DuckDB file arrives
with the tables it already has; a service URL becomes its own engine. Because all of
them are databases, the same eight verbs work on any of them. Each call names one
datasource and the SQL addresses tables inside it by their own names; putting two
datasources together is `create`, which copies one into the other so the join is an
ordinary statement.

**A file may hold more than one table, and all of them land.** Sheets in a workbook and
tables on an HTML page are the cases that force it. Reading the first and ignoring the
rest would leave data that is present in the file unreachable through the server — the
same silent loss as dropping a value — so the datasource, being a database, holds every
table the file had, under the names the file gave them.

That premise was already true of the registry. What level 0 changes is the *verbs*:
they were file verbs wearing database names. Attach made a database, and after that you
could only read it. A thing you can only read is a view over a file, which is the model
this design leaves behind.

So the surface grows the verbs a database actually needs — lifecycle, composition, and
introspection — and it grows them as **few and multi-faceted** rather than many and
narrow. The LLM has less to choose between, and each choice is obvious.

## The three arcs

Every part of the surface exists to serve one of three user journeys.

### Arc 1 — Simple

*"Analyze this file."* Attach it, ask questions, detach. Ephemeral: the data dies on
detach or when the harness quits, and nothing is written anywhere.

The thing the caller must stay cognizant of is that **even one flat file is a table
inside a database**. Questions about the data become SQL that names the table. The slot
is named after the file unless the caller names it at attach time.

### Arc 2 — Lookup

*"Can you do a lookup with this other file that shares field X?"* Many users talk
exactly like that, and it means three things:

1. **Add a table to the database already open** — not a second slot.
2. Join the two tables in ordinary SQL through `query`, which reads and only reads.
   Index the key first with `create` if the join drags; nothing is indexed unless asked.
3. **Check whether the join is complete**, and say so in the user's terms: *"X, Y and Z
   have no match in that file."*

> **Amended (2026-07-26).** Step 3 used to be the server's work: `add_table` took a
> `join_on` parameter and returned unmatched counts and sample values in both
> directions. It is now the caller's, and step 2's indexing note is the other half of
> the same decision.
>
> Two things were wrong with it. The parameter was **an authority grant for a judgement
> the caller was already making** — which column is the key is something only the party
> that read the user's question knows, so asking for it in a parameter and then also
> deciding what to do with it split one act across two. And having been handed the key,
> the code used it for five correlated `NOT EXISTS` subqueries against an unindexed
> column and *not* for the one thing that would have justified asking: an index.
>
> An anti-join is ordinary SQL over two tables in one database. A caller that can write
> the join can write the check, and it can ask for the index that makes both fast. What
> it gets from the server is the primitives and the facts; what it owes the user is the
> sentence.

Adding to the existing database rather than attaching a second one is not only the
friendlier mental model — it is **what makes the result keepable**. `save` writes one
database, not a join, so a lookup meant to outlive the session needs both sides inside
the slot being saved.

So the two halves behave differently and are easy to conflate. Joining across slots
works and is worth doing for a one-off answer. Keeping that relationship — coming back
to it next session — requires both tables in one database, which is what `create` is
for.

> **Superseded (2026-07-26).** An earlier draft made step 2 *"build a view over the
> join"* and justified `create` by SQLite's refusal of a cross-database `CREATE VIEW`
> (CONSTRAINTS §2.3). Creating views was never part of this product; that justification
> rested on a feature that does not exist, and the reason above holds without it.

### Arc 3 — Spill

*"Will we have enough memory?"* The working budget is small — order **100 MB**, a config
knob rather than a constant. If a new file takes the session to 1.2 GB we are probably
not dead yet, so **the overshoot is tolerated once**. But the **next operation**,
whatever it is, first unloads the largest in-memory database to a temp file and
**reconnects it in the same slot**, now on disk.

This is **transparent, including to the LLM**. No tool announces it and no caller asks
for it.

> **No pre-flight estimate.** Deciding admission upfront from file size or metadata is
> the fail-open pattern that already bit this project once. This design measures what is
> *actually* resident and reacts after the fact. That is the sound version; keep it.

Residency is `(page_count − freelist_count) × page_size` per attached schema — measured,
not inferred, and freelist-corrected because `page_count` does not shrink after a
`DROP TABLE` (CONSTRAINTS §6).

Temp-file lifecycle: deleted on **eviction**, on **detach**, and on **termination with
connections live**. Nothing more. Richer heuristics are possible and not worth the time.

## The eight verbs

| Verb | Shape | Notes |
|---|---|---|
| `attach` | `database`, `nickname?`, `writable?`, `delimiter?` | Multipurpose — flat file, database file (SQLite or DuckDB, told apart by header), or a URL. Returns the nickname **actually used**. A file holding several tables becomes a database holding all of them. `delimiter` applies to character-separated text only. |
| `detach` | `nickname` | Drop a slot deliberately instead of waiting for FIFO to guess. Deletes the temp file if spilled. |
| `query` | `nickname`, `sql`, `path?`, `force?` | The `path` suffix chooses the output format and one with no writer is refused by name. **Reads only** — every write is refused whatever the slot allows, by the connection rather than by a check, as far as each backend can refuse (see *Write is not the default*). Returns the whole result; the optional path is where an oversized one is written instead, which **absorbs `export_query`**. `force` is the same overwrite consent `save` takes, for the same reason. |
| `info` | — \| `nickname` \| `nickname`+`table` | Polymorphic: bare → every slot; nickname → its tables; nickname+table → schema, row count and indexes. **Absorbs `list_tables` + `describe_table`.** |
| `create` | `nickname`, `type`, `table?`, `source?`, `columns?`, `delimiter?` | `type="table"` reads a datasource in beside the tables already there, which is what makes arc 2 possible. `type="index"` indexes columns of a table already there — asked for, never inferred. |
| `update` | `nickname`, `type`, `name`, `to` | Rename a table, keeping its rows, types and indexes. The third of create/update/drop, and the answer to a file that names its own tables — a workbook's sheets arrive as the spreadsheet named them. Renaming onto a taken name is refused, not allowed to replace. |
| `drop` | `nickname`, `type`, `name` | Composition needs both directions, for both types. The index name is the one `create` returned and `info` lists. |
| `save` | `nickname`, `path`, `force?` | Relocate an in-memory or spilled database to a path the user chose — the "actually, keep this" escape from ephemerality. An occupied path is refused until `force` carries the user's consent, and a path a live slot sits on is refused regardless. |

### Write is not the default

Only databases **the MCP created** are read/write. A flat file becomes our own in-memory
database, so it is writable by construction. Everything attached from outside is
**read-only**. The caller can grant write on an external database at attach time
(`writable=true`), and that grant is per-attach.

The grant governs `create` and `drop` — and only those, because **`query` never
writes to anything**. A query reads: `INSERT`, `CREATE TABLE`, `CREATE VIEW`, `PRAGMA`
and the rest are refused there even on a database the caller owns outright. So there is
exactly one way to change a slot, and it is a named verb rather than a clause buried in
a statement.

That is deliberately not a permission model. The enforcement is the connection's, not a
check that could be reached around — and **each backend enforces it as far as it can**,
which is not equally far:

- **SQLite** refuses at statement preparation, through an authorizer that whitelists four
  read actions; a read-only attach also carries `mode=ro` in its connection URI. **DuckDB**
  opens `access_mode=read_only`. Neither runs the statement at all.
- **MySQL and MariaDB** open a read-only session, and they have to: DDL commits itself
  there, so a `CREATE TABLE` on a connection that never commits would otherwise be
  permanent.
- **PostgreSQL and SQL Server** keep the transactional floor — the read connection never
  commits, so a write is rolled back — which is sufficient because their DDL is
  transactional too.
- **Oracle** keeps the same floor, and it holds for DML. It does *not* hold for DDL:
  Oracle commits DDL as it runs it, before anything can object, and it has no session-level
  read-only posture to reach for. So a `CREATE` sent to `query` there really does take
  effect, and the refusal says so rather than claiming otherwise.

Underneath all of them is one dialect-free rule: **a statement that returns no rows is not
a read**, and is refused on that ground. No SQL is parsed to decide it — a `SELECT` returns
rows even when it matches none — and without it a rolled-back write came back as a
statement that succeeded and returned nothing, which is indistinguishable from success.

A database that was `save`d and is attached again later is, like any other external
database, **read-only by default** until the caller says otherwise.

## Sources and targets

Reading and writing are two registries keyed on the file suffix — `loader.READERS` and
`export.WRITERS` — and a format is one entry in each. Adding one touches nothing else,
because everything downstream of a reader works from a DataFrame and everything upstream
of a writer works from columns and rows.

| Group | Read | Write |
|---|---|---|
| Flat | `.csv` `.tsv` `.txt` `.fwf` | `.csv` `.tsv` `.txt` `.md` |
| Structured | `.json` `.jsonl` `.ndjson` `.yaml` `.yml` `.xml` | same, less `.fwf` |
| Spreadsheet | `.xlsx` `.xlsm` `.xls` `.ods` `.numbers` | `.xlsx` `.ods` |
| Columnar | `.parquet` `.feather` `.orc` | same |
| Web | `.html` `.htm` | same |

**The suffix chooses the format, and one with no writer is refused by name.** Writing CSV
under a `.parquet` name was the defect this replaced: the file's name lied about its
contents and nothing reported it. Refusal happens before the destination is touched, so a
request that could never succeed does not cost the user a file on its way to failing.

**Every format is known whether or not its library is installed**, and the refusal names
the extra to install (`pip install 'localdata-mcp[parquet]'`). Listing only what happens
to be present would make the tool's own description vary by environment, so an agent could
not learn what this server does without discovering what it has.

Four decisions recur across the readers, and they are the same decision each time — *take
what is unambiguous, say what was assumed, refuse an actual choice*:

- **One candidate is not a choice.** A JSON or YAML object with exactly one array of
  objects under it loads from that key, with a note naming it. Two candidates are two
  tables and are refused, naming both.
- **What SQL cannot hold is encoded, not dropped.** A nested JSON value becomes its JSON
  text and a nested XML element its XML text — lossless and reversible — and a note names
  the columns. `pandas.read_xml` drops the subtree and reports nothing, which is why that
  reader is written directly on ElementTree.
- **An inference is stated.** Fixed-width column boundaries are inferred from alignment,
  because nothing in the file declares them, and the note says so.
- **Nothing sniffs.** A file separated by something other than what its extension implies
  loads as one fat column; `delimiter` is how the caller says otherwise. The parameter
  alone would not have been enough — a caller who does not know would never reach for it —
  so a single column whose *name* still contains a common delimiter says exactly that and
  names the parameter. It reports what it sees; it does not re-read at a guessed
  separator, because a guess that is usually right is the worst kind.

`delimiter` earns its place on the same test `join_on` failed: it declares a **fact about
the source** the server cannot know and the caller often does, rather than a judgement the
caller was already making. It applies to character-separated text only, and is refused —
not ignored — anywhere else.

### The backends, and what each one needed

A database is reached through a SQLAlchemy engine, and `create_engine` is generic — so
`Backend` is **not** an interface a database must implement to be reachable. A dialect
nobody has subclassed still opens, still queries, still composes. A subclass exists only
where the generic answer means something different here, or nothing at all:

| Backend | Reached as | What it could not be asked portably |
|---|---|---|
| SQLite | file, URL | `query_only` and an authorizer; `VACUUM INTO` for `save`; residency from the page count; `typeof()` for mixed columns; the declared type *is* the affinity |
| DuckDB | file, URL | nothing — the generic answers are the whole answer |
| PostgreSQL | URL | nothing |
| MySQL / MariaDB | URL | a read-only session, since DDL commits itself; an index over a *prefix*, since `TEXT` cannot be a key |
| Oracle | URL | `VARCHAR2` sized from the data, since `CLOB` cannot be a comparison key; and the admission that DDL survives refusal |
| SQL Server | URL | `sp_rename`; `VARCHAR` sized from the data, since `TEXT` is deprecated and unindexable |

Two things generalised out of that table and became generic rather than per-dialect. **A
declared type is named by the backend**, because the portable spellings are what make one
column `TEXT` on PostgreSQL, `CLOB` on Oracle and `INTEGER` on all of them — the literal
uppercase forms used before could not create a table on Oracle at all, and quietly put
float64 data into PostgreSQL's four-byte `REAL`. And **a value leaving `query` is spelled
for JSON**, because a server-side database returns far more than SQLite's three storage
classes: a `Decimal`, a `date`, a `timedelta`, a `UUID`, raw `bytes`. A `Decimal` left
alone reached the client as the *string* `"155000"`, and an agent then compares and adds
text.

Every dialect is exercised against a live container (`docker-compose.test.yml`), and the
tests skip — with the command to start one — rather than fail where none is running.

## Dates, and the two spellings that carry their own meaning

A file holds dates as text, and text compares as text — so `'30.11.2023'` sorts
*after* `'01.03.2025'`, `ORDER BY` runs backwards and `max()` returns the
earliest instant. Measured across twenty-four spellings, seven ordered wrongly
and four reported the earliest as the maximum, silently (CONSTRAINTS §8.1).

The answer is not a better parser, because most of those spellings are
**genuinely ambiguous**: `01/03/2025` is the first of March or the third of
January depending on who wrote the file, and the file does not say. A server
that guesses is wrong silently, which is the failure it was meant to prevent.
So exactly two forms are recognised, both of which mean one thing everywhere:

- **ISO 8601 calendar dates and datetimes, extended format** — `2024-03-01`,
  `2024-03-01T14:30:00`, `2024-03-01T14:30:00Z`, and `2024-03-01 14:30:00`
  (the space separator is RFC 3339's relaxation, and is what pandas and every
  SQL engine emit). ISO 8601 *basic* format (`20240301`) is **not** accepted —
  it is indistinguishable from an order number.
- **Unix time** (IEEE Std 1003.1), an integer count from
  1970-01-01T00:00:00Z — which is left exactly as it is. Integer comparison
  *is* chronological comparison, so it already orders, ranges and joins
  correctly; and it could not be detected anyway, since `1766664000` is equally
  a timestamp, an identifier or a count. The numeric form is supported by not
  touching it.

A recognised column is rewritten into **one canonical UTC spelling** and stays
text. An offset is honoured and normalised, so the same instant written
`+00:00` and `-05:00` compares equal — the join that used to return zero rows.
The original offset is **not** recoverable afterwards; a file that needs it must
keep it in a column of its own. A column of plain dates stays `YYYY-MM-DD`.

> **Why not integer ticks**, which CONSTRAINTS §1.4 otherwise calls for. Ticks
> move the silent wrong answer rather than removing it: against a tick column
> `WHERE order_date > '2025-01-01'` compares an integer to text and returns
> **zero rows with no error**, and that is a likelier query than a cross-file
> instant join. §1.4's evidence is about *offset-preserving* text and about
> durations; canonical UTC text has neither problem, and SQLite's own date
> functions all take ISO 8601 text.

**Anything else is left alone and reported.** A column that reads as dates in
no recognised standard comes back with the offending values named, saying that
its comparisons are alphabetical rather than chronological. That is the same
contract as the mixed-column signal: the server states the limitation, and the
caller decides what to do about it.

## Nicknames

**snake_case, always** — derived nicknames and table names alike. `2024 Sales Report.csv`
is not a legal SQL identifier: lowercase it, map non-alphanumerics to `_`, and prefix it
when it would otherwise lead with a digit.

Collision is judged on the *derived* nickname, and **the URI decides what it means**:

- **Same source, already attached → refuse**, naming the slot it already lives in. A
  second copy of identical data burns one of ten slots for nothing. The answer is the
  same when a different nickname is requested for a source that is already attached:
  tell the caller where it is.
- **Different source, colliding nickname → accept and disambiguate.** `~/q1/sales.csv`
  and `~/q2/sales.csv` are two real datasources and both deserve a slot. A numeric
  suffix does it: `sales`, `sales_2`.

Not the parent directory folded into the name — `q1_sales` is longer without being
self-explanatory, and the thing that actually distinguishes the two is the source, which
`attach` returns and `info` lists.

**`attach` always returns the nickname it actually used**, along with what it collided
with and that source's URI. The caller must never assume the name derived from the
filename is the name it got. Silent renaming is the same class of defect as a config
that keeps its default after a typo.

## The division of labour with the skill

The deliverable is **tools plus skill**. The contract between them is one sentence:

> **The server never blocks on a question, but surfaces enough that the skill can ask a
> good one.**

The server takes the deterministic default (`sales_2`) and proceeds. The skill decides
when to pull the user in — on a collision: *"you already have a `sales` from `~/q1` —
this one's `sales_2`, want to call it something that'll mean more later?"*

Concretely, the server returns structured facts and the skill carries the idiom. Join
completeness is the clearest case: the *server* offers only the primitives — a second
table in the same database, an index when one is asked for, and SQL that reads — while
the skill writes the anti-join, decides which direction the user cared about, and says
*"Acme, Globex and Initech have no match"* rather than the phrase "anti-join".

The skill ships in this repo and is versioned with the server, because the two are only
correct against each other.

## What comes after

Still level 0, and in this order: the format catalogue above (**done**), then more
backends (**done** — the seven in the table above, each against a container of its own).

The expectation going into the backends was that they would need a test harness rather
than a code path, and that was half right: nothing about *reaching* a dialect needed
writing, and the two that needed nothing at all — DuckDB and PostgreSQL — are the proof.
What the harness found instead was four defects the file-backed dialects could not have
shown, each of them a wrong answer rather than an error: a number arriving as text, a
write reported as a success, a `CREATE TABLE` that was permanent despite being refused,
and a type name that no other database has.

What is left: measuring `CONSTRAINTS.md` against the formats added since it was written,
and whether SQLAlchemy needs extending for anything after that.

Building blocks first: **simple, composable, multi-faceted, and where possible
transparent even to the LLM.**
