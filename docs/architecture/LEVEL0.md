# Level 0 — files and databases, done well

Level 0 is the gate. Until the surface below is built and behaving, nothing else is
started. The reason is that everything above level 0 is the *same* building blocks
pointed at more kinds of source, so a block that is wrong here is wrong everywhere later.

The surface is built, and the gate has since opened onto its own breadth: more formats
and more backends are **still level 0**, because they are nothing new — the same verbs
pointed at more kinds of source. Formats are done. The backends are a **closed** catalogue
of **eighteen**, worked through one at a time: SQLite, DuckDB, PostgreSQL, MySQL, MariaDB,
SQL Server, Oracle, ClickHouse, CockroachDB, YugabyteDB, Trino, MonetDB, CrateDB, Firebird,
openGauss, YDB, Databend and Exasol. Each is reached by the same nine verbs, and each is
exercised against a container of its own — except SQLite and DuckDB, which are files and
need none. **The catalogue is worked through**: every entry that can be reached on this
machine is landed.

Reached is not the same as *carried out*. Three verbs mean nothing on some engines and are
refused there, with the reason named: **`save` on every backend but SQLite** (a slot reached
over its own connection holds no database here to write out), **`create(type='index')` on
ClickHouse, Trino, CrateDB, Databend and Exasol**, and **`update(type='table')` on
Firebird**. The per-backend table below carries each one, and they are the reason the surface
is uniform while the outcomes are not.

Five candidates are **out**, and for three different reasons. **TiDB** and **HyperSQL** fail
the eligibility rule — a database is in scope if and only if an open-source SQLAlchemy adapter exists,
and TiDB has none while HyperSQL's reaches it only through a JVM and a JDBC jar that have
to be on the host. **Greenplum** was dropped as scope. **Db2** and **OceanBase** pass the
rule and still cannot be reached, each one blocked a layer lower than the last: Db2's
adapter installs and cannot be imported, because the native client beneath it links against
a C++ runtime macOS no longer ships (`CONSTRAINTS.md` §24), and OceanBase's server crashes
at startup on an instruction — `rdtscp` — that the virtual machine Docker runs here does not
expose (§26). Exasol, the last entry, landed (§27) — and the list ends there.

Sixteen of those are containers, and this machine will run six at a time before the Docker
VM starves them, so **no single test run covers the catalogue** — it takes five batches once
the authentication variants are counted, and each one reports a green suite while the
dialects it never reached stay silent (issue #46).

## The premise

**A slot is a database.** Not a table, not a file — a database, addressed by a
nickname. A CSV becomes a fresh in-memory database holding one table named after the
file; a workbook becomes one holding a table per sheet; a SQLite or DuckDB file arrives
with the tables it already has; a service URL becomes its own engine. Because all of
them are databases, the same nine verbs work on any of them. Each call names one
datasource and the SQL addresses tables inside it by their own names; putting two
datasources together is `create`, which copies one into the other so the join is an
ordinary statement.

**A file may hold more than one table, and all of them land.** Sheets in a workbook are
the case that forces it, and a `.numbers` document is the same shape — a canvas per sheet,
each carrying its own named tables. Reading the first and ignoring the rest would leave
data that is present in the file unreachable through the server — the same silent loss as
dropping a value — so the datasource, being a database, holds every table the file had,
under the names the file gave them **put through the same snake_case rule as a nickname**
(a sheet called `Sheet1` becomes the table `sheet1`).

> **Corrected (2026-08-04) — a multi-table JSON is not this shape, and the code says so.**
> An earlier draft of this paragraph named "a JSON file carrying several arrays" alongside
> the workbook. It is not the same case and never was built as one: a JSON, YAML or XML
> document with **two or more** candidate tables is **refused**, naming both, per *"One
> candidate is not a choice"* below. Measured through the shipped surface: a JSON object
> holding a `customers` array and an `orders` array comes back as
>
> ```
> Could not read multi.json: it holds more than one table ('customers', 'orders'), and
> which one you want is not something this server should decide. Split the file, or
> attach it as one table per file.
> ```
>
> The two rules genuinely point different ways and only one of them is built. A sheet is
> **declared** a table by the format — the file says so — whereas an array under a key is a
> table only by inference, and a document may equally be one record with a list inside it.
> That is a defensible line, and it is the line the code draws. But the refusal's own
> wording — *"which one you want"* — is the pick-one model this design left behind, and
> the premise above is the reason to revisit it. **Open, not decided.**

That premise was already true of the registry. What level 0 changes is the *verbs*:
they were file verbs wearing database names. Attach made a database, and after that you
could only read it. A thing you can only read is a view over a file, which is the model
this design leaves behind.

So the surface grows the verbs a database actually needs — lifecycle, composition,
introspection and, since 2026-08-12, the profile that says whether the values inside can be
trusted — and it grows them as **few and multi-faceted** rather than many and narrow. The
LLM has less to choose between, and each choice is obvious.

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

## The nine verbs

> **Amended (2026-08-12) — the surface gained a ninth verb, `stats`, and this document
> counted eight everywhere.** The count was never the principle. What this design states is
> *"few and multi-faceted rather than many and narrow"*, and nine is still few; eight was an
> observation about the surface rather than a constraint on it. Both numbers are updated
> throughout rather than left to disagree.
>
> **Why it is a verb and not a flag on `directory`.** The trigger was
> [#94](https://github.com/ChrisGVE/localdata-mcp/issues/94): a cold agent read `directory`'s
> `mixed_columns: []` as a clean bill of health and reported 3,000 rows as trustworthy,
> while 52 values in a `REAL` column were null. Nothing in `attach`, `directory` or the documented
> checklist would ever have said so. The framing the issue proposed — *should `attach` or
> `directory` report a null count* — is the wrong question, and both halves of it are refused:
> **`attach` reports load-time facts** and why a format could not be read, **`directory` is a
> directory** — sources, tables, schema — and a directory that reports statistics stops
> being one. A profile is a third role, so it is a third verb. An `directory(stats=true)` flag
> would re-import exactly the confusion this split exists to remove.
>
> **Why it is still level 0**, given this document's own test — *nothing new, the same
> building blocks pointed at more kinds of source*. Under the cost rule below, `stats` adds
> no statistics engine, no second pass, and no dependency: it is SQL aggregates the engine
> already computes, issued over the same slot machinery, and refused per-dialect through the
> same capability declaration `builds_indexes` and `renames_tables` already use. Anything
> needing emulation is level 1 — and that line is drawn *through* the verb rather than
> around it, which is what keeps the verb here and the statistics platform above.

The table gives the shape and the one-line purpose. What each verb costs and why it is
drawn where it is follows underneath.

Every verb answers with a payload carrying `ok`, and a refusal is one of those answers
rather than a protocol error: `{"ok": false, "error": "…"}`, the reason in plain words and,
where a name was wrong, the names that were right. That is the envelope the whole surface
shares, and it is why every refusal below is described as something the caller reads rather
than something they catch.

| Verb | Shape | What it is for |
|---|---|---|
| `attach` | `database`, `nickname?`, `writable?`, `delimiter?` | Open a datasource as a database — flat file, database file (SQLite or DuckDB, told apart by header), or a URL. Returns the nickname **actually used**. |
| `detach` | `nickname` | Close a slot deliberately instead of waiting for FIFO to guess. Deletes the temp file if it had spilled. |
| `query` | `nickname`, `sql`, `path?`, `force?`, `delimiter?` | Run SQL. **Reads only.** Returns the whole result, or writes it to `path` when it is too large to return. |
| `directory` | — \| `nickname` \| `nickname`+`table` | Three levels of detail: bare → every slot and the path posture; nickname → its tables; nickname and table → schema, row count and indexes. |
| `create` | `nickname`, `type`, `table?`, `source?`, `columns?`, `delimiter?` | `type="table"` reads a **file** in beside the tables already there — one holding a single table, since `create` makes one — which is what makes arc 2 possible. `type="index"` indexes columns of a table already there — asked for, never inferred. |
| `update` | `nickname`, `type`, `name`, `to` | Rename a table, keeping its rows, types and indexes — the answer to a file that named its own tables. |
| `drop` | `nickname`, `type`, `name` | Remove a table or an index. Composition needs both directions, for both types. |
| `save` | `nickname`, `path`, `force?` | Relocate an in-memory or spilled database to a path the user chose — the "actually, keep this" escape from ephemerality. |
| `stats` | `nickname`, `table`, `columns?` | Profile a table's columns — missing values, and the range of the ones that are there. The verb that answers *"can I trust this data?"*, which no other verb does. |

**`attach`.** A workbook becomes a database holding a table per sheet and a `.numbers`
document one per table, snake_cased. `delimiter` applies to character-separated text only
and is refused elsewhere.

**`query`.** Every write is refused whatever the slot allows, by the connection rather than
by a check, and as far as each backend can refuse — see *Write is not the default*. The
`path` suffix chooses the output format and one with no writer is refused by name; `force`
is the same overwrite consent `save` takes, for the same reason. `delimiter` separates
fields on the way *out*, for `.csv`/`.tsv`/`.txt` only, and is ignored rather than refused
elsewhere.

**`update` and `drop`.** `update`'s `name` is the snake_cased name `directory` lists rather than
the spelling in the spreadsheet, and case is the part of that the server closes: a name
differing from the stored one only in case resolves to it, so these verbs answer the
existence question the way the database does and the way `query` always did. A spelling the
snake_casing changed by more than case — `Q2 Prices` — still names no table and is refused
with the names that do. Renaming onto a taken name is refused rather than allowed to replace,
and so is renaming to the same name in another case, which is not a free name on an engine
that folds. The index name `drop` takes is the one `create` returned and `directory` lists.
`update(type='table')` is refused on Firebird, and `create(type='index')` on the five
engines named in the backend table.

**`save`.** An occupied path is refused until `force` carries the user's consent, and a path
a live slot sits on is refused regardless. It writes out a database this server holds, so it
is refused on every backend but SQLite; the refusal names the route round it, which is
three calls — `query(path=…)`, `attach` that file as a datasource of your own, `save` that.

**`stats`.** Governed by one rule — **cost: free we take, expensive we leave.** Every column
reports `nulls` and `non_nulls`, because `count(*) - count(col)` is stock SQL on every engine
here and that is the floor the whole design rests on. A numeric column adds `min`, `max` and
`avg`; a date column normalised to ISO 8601 adds `min` and `max`, its text order being
chronological order. **A function the engine lacks is simply not reported** — never emulated,
never approximated, never a second pass in Python — so `median` and `stddev` appear only where
the database computes them itself, and their absence is a fact about the datasource rather
than about the column. Strings get the null count and nothing more; length statistics are
level 1. Every aggregate for every column goes into one `SELECT`, so a profile costs one scan
whatever its width.

**Two kinds of column are deliberately profiled to their null count and no further**, each
saying why in `withheld`, and they are the reason this verb is not merely a convenience. A
**mixed** column's `avg()` coerces its text values to 0 and keeps them in the denominator, so
the average of 1..5 plus two text rows is 2.14 rather than 3.0. A column of **dates in no
recognised standard** compares alphabetically, so `min()` and `max()` answer with the
alphabetically first and last value rather than the earliest and latest instant. Both return
a real number from a real column, which is what makes them worse than silence under a verb
whose whole promise is that the numbers are true — the same judgement `snapshot` and
`builds_indexes` make about an answer that would be a lie the caller then builds on.

Which statistics a backend offers is declared per-dialect and **defaults to none**, so a
dialect nobody has measured under-reports rather than failing a statement mid-profile.
Measured so far: SQLite has neither `median` nor `stddev`; DuckDB has both. The other
sixteen are a declared coverage gap and are left at the floor until a container run measures
them.

### Write is not the default

Only databases **the MCP created** are read/write. A flat file becomes our own in-memory
database, so it is writable by construction. Everything attached from outside is
**read-only**. The caller can grant write on an external database at attach time
(`writable=true`), and that grant is per-attach — changing it on an open slot means
`detach` and attach again, since the same source attached twice is refused.

The grant governs the three verbs that change a slot — `create`, `update` and `drop`
— and only those, because **`query` never writes to anything**. A query reads: `INSERT`, `CREATE TABLE`, `CREATE VIEW`, `PRAGMA`
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
- **ClickHouse** has no transactions at all, so there is no floor to keep: a write sent to
  a connection that never commits is simply applied. Its read engine therefore carries
  `readonly=1` in the URL, and the database refuses DML and DDL alike before either runs —
  which makes it *stronger* here than the backends that rely on rollback, not weaker.
- **CrateDB** has no transactions either and no read-only posture to reach for, so it is
  the one backend where the floor is absent on **both** axes: `dml_survives_refusal()` and
  `ddl_survives_refusal()` are both `True`, and a refused `INSERT` is in the index before
  this server has anything to say about it. It is worse than Oracle, which loses only the
  DDL half. Both halves of the axis are read when the refusal is composed, so on CrateDB it
  names the rows as well as `CREATE`/`DROP`: a caveat that names the wrong half points a
  reader away from what happened, which is worse than none at all
  ([#84](https://github.com/ChrisGVE/localdata-mcp/issues/84)).

Underneath all of them is one dialect-free rule: **a statement that returns no rows is not
a read**, and is refused on that ground. No SQL is parsed to decide it — a `SELECT` returns
rows even when it matches none — and without it a rolled-back write came back as a
statement that succeeded and returned nothing, which is indistinguishable from success.

A database that was `save`d and is attached again later is, like any other external
database, **read-only by default** until the caller says otherwise.

## Sources and targets

Everything keyed on a file suffix is declared once, in `formats.FORMATS`: a format is one
`Format` naming its reader, its writer, whether a delimiter means anything for it and
whether it can be read in chunks. Adding one is that single entry, because everything
downstream of a reader works from a DataFrame and everything upstream of a writer works
from columns and rows. `READERS`, `WRITERS`, `DELIMITED` and `STREAMED` still exist as the
names the rest of the code uses, but they are **views** of that table now rather than four
tables that could disagree — which they had begun to (`localdata#98`, `#99`).

Read-only and write-only are a missing callable rather than membership of some other set.
`.md` has no reader, because a Markdown table has no types and no quoting; `.xls` has no
writer, because xlrd dropped writing.

**The table is closed-world, deliberately unlike `dialects.BACKENDS`.** An unregistered
dialect there gets a generic `Backend` that works, since an unknown database still speaks
SQL. An unregistered suffix has no such fallback — there is no generic way to read a file
whose format nobody declared, and guessing produces exactly the answer this server exists
to prevent: data that loaded, looks fine, and is wrong. Absence here is a refusal by name.

**A delimited file is read twice rather than held once.** `.csv`, `.tsv`, `.txt` and
`.fwf` — the formats declaring `streamed` — go through a measuring pass and then an inserting pass, so
the load's peak stops tracking the file: 4,286 MB → 803 MB against a 1.22 GB CSV
(CONSTRAINTS §28). The file is read twice and that costs 1.4–1.75x wall clock, which is
the whole of the trade.

This is not a chunk size. Everything that decides the *table* is a whole-column
measurement made before the first insert — the declared type, the width of the widest
text value, the numeric split of a mixed column, whether a text column is dates — and
pandas infers dtypes per chunk, so a naively chunked insert declares a column from chunk
one and meets a value it cannot hold in chunk five. Pass one accumulates those answers
across chunks and pass two applies them. Which canonical spelling a date column is
written in is the one that does not compose (both flags are whole-column aggregates), so
it is measured over the column and the writer is told it.

Every format keeps one insert path: a frame held in memory is a source of exactly one
chunk, so a workbook and a million-row CSV reach the same code and nothing downstream
knows which it got. The formats that are not delimited are parsed whole by the libraries
that read them — there is no chunk to ask for and no line that is a row — so their peak
still tracks the file, and that is stated rather than worked around.

| Group | Read | Write |
|---|---|---|
| Flat | `.csv` `.tsv` `.txt` `.fwf` | `.csv` `.tsv` `.txt` |
| Structured | `.json` `.jsonl` `.ndjson` `.yaml` `.yml` `.xml` | same |
| Spreadsheet | `.xlsx` `.xlsm` `.xls` `.ods` `.numbers` | `.xlsx` `.ods` |
| Columnar | `.parquet` `.feather` `.orc` | same |
| Markdown | — | `.md` — write only, there is no `.md` reader |

> **`.html` and `.htm` were removed from both registries on 2026-07-28.** They had been a
> "Web" group of their own. Reading them was defensible — a saved page is a real thing to
> be handed — but writing them was not: the writer emitted a bare `<table>` fragment
> rather than a document, so it served a person no better than `.md` and a program worse
> than `.csv`. And it was **the one suffix that broke the round-trip property the overlap
> between these two registries is supposed to mean** — it wrote a table of any size and
> could not read back past lxml's 10,000,000-node XPath ceiling, about 417,000 rows of
> eleven columns. Dropping both sides was chosen over capping the writer, because a format
> kept only for a reader that a document-shaped input rarely satisfies is a format earning
> its place by history. lxml was its only dependency and the `html` extra went with it.

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
caller was already making. It applies to character-separated text only, and **the two
directions treat a mismatch differently, deliberately**. Reading — `attach` and
`create` — **refuses** it: a `.parquet` handed a `delimiter` is a caller who has
misunderstood the file, and the refusal names the suffix and the three that qualify.
Writing — `query(path=…)` — **ignores** it, because one default can then be carried across
a mix of destinations without the call having to know which suffix it is about to hit.

### The backends, and what each one needed

A database is reached through a SQLAlchemy engine, and `create_engine` is generic — so
`Backend` is **not** an interface a database must implement to be reachable. A dialect
nobody has subclassed still opens, still queries, still composes. A subclass exists only
where the generic answer means something different here, or nothing at all.

**Seventeen rows for eighteen backends**: MySQL and MariaDB need the same overrides and
share one. **Every row but SQLite's refuses `save`** — that refusal is the generic answer,
not a per-dialect one, so it is stated once here rather than repeated in seventeen cells:
`Backend.snapshot` raises, and only `SQLiteBackend` overrides it, because only a database
this server built is a database it holds.

| Backend | Reached as | What it could not be asked portably |
|---|---|---|
| SQLite | file, URL | `query_only` and an authorizer; `VACUUM INTO`, which is what makes it the one backend `save` works on; residency from the page count; `typeof()` for mixed columns; the declared type *is* the affinity |
| DuckDB | file, URL | `access_mode=read_only` on the read connection — refused by DuckDB itself at open time, so a read connection cannot write however it is reached, which is stronger than the generic transactional floor. That is the whole of the subclass, and everything else is the generic answer |
| PostgreSQL | URL | nothing |
| MySQL / MariaDB | URL | a read-only session, since DDL commits itself; an index over a *prefix*, since `TEXT` cannot be a key |
| Oracle | URL | `VARCHAR2` sized from the data, since `CLOB` cannot be a comparison key; and the admission that DDL survives refusal |
| SQL Server | URL | `sp_rename`; `VARCHAR` sized from the data, since `TEXT` is deprecated and unindexable |
| ClickHouse | URL | `readonly=1`, since there is no transaction to withhold; `Nullable` columns, since a non-nullable one takes a missing value and stores `''`; `RENAME TABLE`; an engine clause on every `CREATE TABLE`; and the refusal of `create(type='index')`, since its indexes cannot be reflected |
| CockroachDB | URL | nothing — and on a different engine speaking PostgreSQL's wire, that is the result rather than an absence |
| YugabyteDB | URL | nothing — reached on PostgreSQL's dialect, which is why the seam had to learn to ask the *engine* what it is rather than trust the dialect's name |
| Trino | URL | an isolation level, since the driver connects in `AUTOCOMMIT` and there is no transaction left to withhold; `LargeBinary`; and `create(type='index')`, since it owns no storage to index |
| MonetDB | URL | nothing — a column store, and not the first here in any sense: DuckDB (a file) and ClickHouse (a URL, which creates its tables as `MergeTree`) both came earlier. What it shows is that it answers every axis the way a row store does |
| CrateDB | URL | that a refused write really happened, since it has neither transactions nor a read-only session; a `REFRESH TABLE` before a write can be read back; the driver's type converter, without which a date arrives as epoch milliseconds; `LargeBinary` and `Time`, which it does not have, and `Numeric`, which its dialect silently truncates; and the refusal of `create(type='index')`, since every column is indexed already |
| openGauss | URL | nothing — but it is the first PostgreSQL fork here that cannot be *addressed* as PostgreSQL, since its version banner does not parse and SQLAlchemy's own PGDialect raises while initialising the connection; it ships its own dialect, so there is no impostor to resolve either |
| Firebird | URL | that rows cannot be written in the transaction that created the table, since DDL is transactional *and* prepared against committed metadata; the refusal of `update(type='table', to=…)`, since no statement renames a table here; `DOUBLE PRECISION`, since the dialect renders `Double` as a keyword Firebird lacks; and `VARCHAR` sized from the data, since `Text` becomes a `BLOB` that groups by identity rather than by value |
| YDB | URL | a **primary key on every loaded table**, since it has no heap tables and a file has no key to offer — so one holding the row's position is added and reported; a **read-only isolation level**, since an uncommitted write here is not rolled back at all and the floor has to be a refusal rather than an undo; the two codes that refusal arrives under, one for rows and one for schema; and `Time`, which it does not have |
| Databend | URL | that a statement is **a read before it runs**, since it has no transaction, no read-only session, and a write that answers with a named result set no examination of the result can tell from a query — so the server itself is asked to plan the statement as a subquery first; `Time`, which it does not have and its dialect renders `DATETIME`; `LargeBinary`, which binds or not depending on whether the bytes are valid UTF-8; and the refusal of `create(type='index')`, since the statement for one compiles to nothing and succeeds |
| Exasol | URL | **autocommit turned off for the read engine**, since its driver commits every statement by default and the transactional floor is otherwise no floor at all; the driver's own type mapper, without which a number widened by `SUM` arrives as the *string* `'155000'` and a `TIMESTAMP` as text; `RENAME TABLE`; `LargeBinary`, which the database does not have and its dialect refuses at compile time; and the refusal of `create(type='index')`, since the engine maintains its own indexes and offers no statement for one |

Two things generalised out of that table and became generic rather than per-dialect. **A
declared type is named by the backend**, because the portable spellings are what make one
column `TEXT` on PostgreSQL, `CLOB` on Oracle and `INTEGER` on all of them — the literal
uppercase forms used before could not create a table on Oracle at all, and quietly put
float64 data into PostgreSQL's four-byte `REAL`. And **a value leaving `query` is spelled
for JSON**, because a server-side database returns far more than SQLite's three storage
classes: a `Decimal`, a `date`, a `timedelta`, a `UUID`, raw `bytes`. A `Decimal` left
alone reached the client as the *string* `"155000"`, and an agent then compares and adds
text.

> **Corrected (2026-08-04) — DuckDB's row said "nothing", and DuckDB is the reason the
> subclass exists.** The row above read *"nothing — the generic answers are the whole
> answer"*. That contradicted this document's own *Write is not the default* section, which
> already records that DuckDB opens `access_mode=read_only`, and it contradicted the class:
> `DuckDBBackend`'s docstring says the read-only posture carried by the URL is *"the one
> thing it adds, and the reason it is registered at all"*. The row now says so.
>
> **And a second thing is open, not decided.** A DuckDB file attached `writable=true` is
> broken today: the slot opens a read engine carrying `access_mode=read_only` and a write
> engine without it, and DuckDB refuses two connections to one file under different
> configurations. Measured through the shipped surface — `attach` and `query` succeed,
> `directory` and `create` raise, `update` and `drop` return a refusal naming the driver error,
> and `save` is refused for the unrelated reason above. The read-only arm is clean. The two
> ways out point different directions: open one engine and carry the posture per statement,
> or refuse `writable=true` on a DuckDB file outright and say why. Filed as
> [#79](https://github.com/ChrisGVE/localdata-mcp/issues/79). **Open, not decided.**

Every dialect that needs a server is exercised against a live container
(`docker-compose.test.yml`) — sixteen of the eighteen; SQLite and DuckDB are files and need
none. The tests skip — with the command to start one — rather than fail where none is
running.

## Dates, and the two spellings that carry their own meaning

A file holds dates as text, and text compares as text — so `'30.11.2023'` sorts
*after* `'01.03.2025'`, `ORDER BY` runs backwards and `max()` returns the
earliest instant. Measured across the spellings of five instants spanning
three years, every day-first and every month-name form ordered wrongly and
reported the earliest as the maximum, silently (CONSTRAINTS §8.1, which asks that
the class be quoted rather than the count — how many spellings land in each class
depends on which spellings the fixture happened to include).

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
keep it in a column of its own. A column of plain dates stays `YYYY-MM-DD` —
but only where *every* value is a plain date. **One spelling means one spelling
for the whole column**: a single value carrying a time takes the column to
`…T00:00:00Z` throughout, and one carrying fractional seconds takes it to
`…T00:00:00.000000Z`, because a column written two ways does not sort as one
(CONSTRAINTS §28.4, [#75](https://github.com/ChrisGVE/localdata-mcp/issues/75)).

> **Why not integer ticks**, which CONSTRAINTS §1.4 otherwise calls for. Ticks
> move the silent wrong answer rather than removing it: against a tick column
> `WHERE order_date > '2025-01-01'` compares an integer to text and returns
> **zero rows with no error**, and that is a likelier query than a cross-file
> instant join. §1.4's evidence is about *offset-preserving* text and about
> durations; canonical UTC text has neither problem, and SQLite's own date
> functions all take ISO 8601 text.
>
> **This governs the text path only, and the typed readers do not take it.** A
> `.parquet`, `.feather`, `.orc` or spreadsheet column that is a real timestamp
> in the source arrives as integer nanoseconds since the epoch — reported as
> `{"temporal": "timestamp", "unit": "nanoseconds_since_epoch"}` — and therefore
> carries the exact failure this paragraph rejects, with no warning raised. The
> reasoning above still holds for everything parsed out of text; it simply never
> reached the readers that get a type handed to them. Whether that split is
> intended is [#87](https://github.com/ChrisGVE/localdata-mcp/issues/87).

**Any other text is left alone and reported.** A column that reads as dates in
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
`attach` returns and `directory` lists.

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

## What is left, and what it took to get here

**What is left before level 0 closes is no longer code**: a pass driving the live server
through a real client, since every verb has changed since the last one, and a review of the
issues that were fixed forward. Three design questions are open and are recorded where they
arise, under an **Open, not decided** mark (rows 1 and 3 share one, since the second is moot
if the first resolves against the refusal):

| Question | Where |
|---|---|
| Whether a JSON, YAML or XML document with two candidate tables should still be refused, given the premise that a file may hold more than one table | *The premise*, above |
| What a DuckDB file attached `writable=true` should do, since two engines on one file is refused by DuckDB ([#79](https://github.com/ChrisGVE/localdata-mcp/issues/79)) | *The backends*, above |
| Whether the refusal's wording — *"which one you want"* — should survive, since it is the pick-one model this design left behind | *The premise*, above |

The rest of this section is how the two catalogues went, and is history rather than plan.
Both are **done**: the format catalogue first, then the backends — every entry that can be
reached on this machine is landed, and the two that cannot, Db2 and OceanBase, were measured
rather than assumed.

The expectation going into the backends was that they would need a test harness rather
than a code path, and that was half right: nothing about *reaching* a dialect needed
writing, and PostgreSQL — which needed nothing at all — is the proof. (DuckDB was named
here too until the correction above: it carries `access_mode=read_only`, which is the one
thing its subclass adds.)
What the harness found instead was defect after defect the file-backed dialects could not
have shown, most of them a wrong answer rather than an error: a number arriving as text, a
write reported as a success, a `CREATE TABLE` that was permanent despite being refused, a
rollback that returned normally over a write that stood, and a write no examination of its
result could tell from a read. They are measured one section at a time in
`docs/CONSTRAINTS.md`, and the client-library half of them is reported upstream.

The authentication matrix is **done**, and it was the item whose shape was unknown. A
database is reached one of several ways and only one of them was ever exercised: a plain URL,
carrying a password where the server wants one. Nine more are exercised now:

| | Modes | Where |
|---|---|---|
| The route 2.x took | 1 — a plain URL | all sixteen endpoints — **eleven with a password in it, five with none**, since CockroachDB, YugabyteDB, Trino, CrateDB and YDB are reached by a bare username |
| Added by this work | 9 — trust, a password from the environment, a password from a file, verified TLS, a client certificate, a Kerberos ticket, an option file, an empty password, a data-source name in place of an address | four endpoints |
| **Exercised in total** | **10** | |
| Real and out of reach here | 2 — a Unix socket, which does not cross the container boundary, and Windows integrated authentication, there being no Windows host | — |

They run as a **second axis** over the endpoint table rather than as tests of their own, so
each endpoint test runs against every mode its endpoint carries: `tests/test_endpoints.py`
holds **twenty test functions**, and each is parameterised over `TARGETS` — one entry per
endpoint *per authentication mode*, **twenty-five** of them against sixteen containers. That
is where the suite's 500 endpoint tests come from: 20 × 25. Nothing in
the server implements them: each is expressed in the URL or in the driver's own environment,
which this server passes through untouched, so what was added is the evidence rather than a
feature. `docs/CONSTRAINTS.md` §25 has the measurements.

The load half of the volume work is **done**. It was the last gate with code behind it, and
what it cost was a second read of the file rather than a smaller buffer: the peak used to
track the file because every measurement deciding the table is a whole-column one, and no
chunk size reaches that. Measured in §28, and the honest edges are recorded there too —
below about 150 MB streaming costs slightly *more*, the formats nobody can chunk are
unchanged, and the earlier sampled figures for this path were under-reports.

Building blocks first: **simple, composable, multi-faceted, and where possible
transparent even to the LLM.**
