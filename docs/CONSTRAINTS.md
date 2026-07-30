# CONSTRAINTS — measured behaviour that shapes the design

Every entry below was established by **direct execution**, not by reading documentation and not by
reasoning. Each one cost real measurement time, and each one is a trap that a from-scratch
implementation walks into by default.

**How to use this file.** Do not read it end to end before starting. It is organised by *when each
constraint bites*, so the relevant section is consulted at the moment the code reaches that
capability. A constraint you read six weeks early is a constraint you forget.

**Measurement environment.** Python 3.12.9 · pandas 3.0.2 · numpy 2.4.4 · sqlite3 3.47.1
(library ≥ 3.27 assumed) · SQLAlchemy 2.0.49 · fastmcp 3.2.0, macOS. Where behaviour depends on a
library version, that is called out. **If a version here moves, re-measure rather than assume the
finding carries** — several of these are library behaviours, not language or engine guarantees.

---

## §1 — Loading a file into SQLite

### 1.1 Values that bind without error and answer wrong (silent BLOB)

The most dangerous class in this document. These bind, insert, commit, and report success. Then
`sum()` returns `0.0`.

| Value produced by a reader | Stored as | `sum()` returns | Truth |
|---|---|---|---|
| `np.int64` | `blob` `X'0100000000000000'` | **0.0** | correct integer |
| `np.bool_` | `blob` `X'01'` | — | correct boolean |
| non-null `pd.array(dtype="Int64")` | `blob` | **0.0** | correct integer |
| non-null `pd.array(dtype="boolean")` | `blob` | — | correct boolean |
| `np.datetime64` inside an `object` column | `blob` | **0.0** | correct instant |

Two facts that make this worse than it looks:

- **A declared `INTEGER` affinity does not rescue it.** BLOB storage class outranks column affinity,
  so the column stays blob-typed. Measured: `avg` = `0.0` and `sum` = `0.0` against a truth of 2.0
  and 6, with `count` = 3 — on a *clean, single-dtype* column with no mixed data anywhere.
- **`np.float64` survives only by accident.** It subclasses Python `float`, so the builtin binder
  accepts it. `np.int64` does not subclass `int`, and `np.bool_` does not subclass `bool`, on this
  platform. The happy path is luck, not design.

**Do:** register an explicit adapter per producible scalar type, and assert storage class in tests.

### 1.2 Values that raise mid-insert

`datetime64[ns]`, `datetime64[us]`, tz-aware timestamps, `timedelta64`, `Decimal`, `Period`,
`complex`, `list` cells, `dict` cells, and any `pd.NA`-bearing nullable dtype all raise a raw DBAPI
error on a bare cursor bind.

Enumerated across the readers a data tool actually uses (csv incl. `parse_dates=`, excel via
openpyxl/xlrd/odf, parquet/feather via pyarrow, json, hdf5, and `convert_dtypes` output):
**25 cases — 10 raise, 2 store silent BLOBs, 13 are fine.** With a full adapter layer: **23 of 25
work**; `list` and `dict` cells are genuinely not scalar and need an explicit refusal or a stated
JSON-encoding rule.

Note `datetime.time` (Excel time cells, parquet `time32/64`, TOML local time) **raises today** — an
ordinary spreadsheet is refused without an adapter. And `datetime.date` binds *only* via a default
adapter **deprecated in Python 3.12**; it is scheduled to break and must not be relied on.

### 1.3 Values that cannot be represented at all

Not fixable by an adapter — these need a **value-domain** check, not a type-keyed one:

- **`np.uint64` above 2⁶³−1** — silently stores a BLOB, `sum` returns `0.0`. Adapters do not help
  (`np.uint64` is a distinct type from `np.int64`), and the value is genuinely unrepresentable.
- **Python `int` wider than 64 bits** — raises `OverflowError`. SQLite `INTEGER` is 64-bit signed,
  full stop.

Refusal is the only correct answer for both, and it must be keyed on the **value**, not the column
dtype.

### 1.4 Temporal columns must be stored as INTEGER ticks

Storing temporals as ISO-8601 text binds cleanly, passes a `typeof` check, and is **wrong on every
aggregate**. Truth for the row set below: 2 d + 10 d + 1 h.

| Query | ISO-8601 text | INTEGER ns ticks |
|---|---|---|
| `sum` | **0.0** | correct |
| `avg` | **0.0** | correct |
| `max` | **`'P2DT0H0M0S'`** (truth: 10 days) | correct |
| `ORDER BY` | **`1 h, 10 d, 2 d`** | `1 h, 2 d, 10 d` |

For timestamps, offset-preserving text is worse still: the same instant written as
`2024-11-03 06:30 UTC` and as its `America/New_York` conversion **joins 0 rows** where integer
epoch-ns joins 1. A UTC file joined against a local-offset file silently returns nothing.

**Store INTEGER ticks with a declared epoch and unit.** State the round-trip contract explicitly,
including what happens to a tz-aware column's original offset — it is **not** recoverable from
epoch-ns alone, so it needs its own column or the contract must say it is dropped.

(A lexical-ordering failure across a DST fall-back was *not* reproducible — `-04:00` and `-05:00`
happen to sort chronologically as text. Don't claim it.)

### 1.5 A column with mixed types cannot be fixed by choosing an affinity

Five numeric rows (1..5, true mean 3.0) followed by two text rows in the same column:

| Query | declared `INTEGER` | no affinity | rebuilt to `TEXT` | truth |
|---|---|---|---|---|
| `avg(col)` | 2.142857 | 2.142857 | 2.142857 | **3.0** |
| `count(*) WHERE col > 1` | 6 | 6 | 6 | **4** |
| `typeof` histogram | integer 5, text 2 | integer 5, text 2 | **text 7** | — |

**Two conclusions, the second counter-intuitive:**

1. **Affinity is irrelevant to the wrong answer.** SQLite coerces text to 0 *and keeps it in the
   denominator* — `(15+0+0)/7`. That is a property of the aggregate, not of the declared type. No
   affinity available to us makes a mixed column answer correctly.
2. **Rebuilding the column to `TEXT` is strictly worse.** It does not fix the aggregate, and it
   **destroys the only surviving signal**: the per-value `typeof` histogram. Before the rebuild,
   `integer 5 / text 2` distinguishes the numerics from the text; after it, everything reads `text 7`
   and recovering the numerics needs `GLOB` pattern-matching, which is a guess. It also costs a
   **2.20× residency transient**, paid exactly when memory is already under pressure.

**Do:** keep the declared affinity, detect the conflict, and **signal** it — record the column as
mixed and expose the per-storage-class histogram (`SELECT typeof(col), count(*) … GROUP BY 1` —
cheap, exact, measured rather than predicted). Silence is the defect; refusing the whole load is an
over-correction that fails the flagship path for a condition the caller can work around once told.
Make strictness a config knob defaulting to signal-and-continue.

> **The work-around is not one work-around (2026-07-26).** This section used to close by naming
> `WHERE typeof(col)='integer'` as the thing to tell the caller — and conclusion 2 above already
> said why that cannot be right in general: once everything reads `text 7`, `typeof` has no signal
> left to give. **A column read from a CSV is always that case.** pandas types it `object`, it is
> declared `TEXT`, and every value stores as text however numeric it looks, so the prescribed
> filter returns the whole column. The advice was copied from here into the server's warning and
> into the shipped skill, and live-agent validation caught it there (§7.2): four of six agents ran
> it, got everything back, and had to go and find the sentinel value themselves. The remedy has to
> branch on which signal fired — `typeof` where the storage classes really differ, and **the
> offending values, by name** where they do not.

**State plainly, wherever the SQL surface is documented, that aggregates over a mixed column
silently coerce text to 0.** Nobody infers it from the affinity discussion, and it is the entire
reason the signal is mandatory.

---

## §2 — Querying, and joining across files

### 2.1 `INTEGER` affinity silently collapses distinct keys in a JOIN

Left keys `['1','2','3']`; right keys `['1','1.0','01','1.00','2']` — numerically equal, textually
distinct:

| left declared | right declared | rows returned |
|---|---|---|
| `INTEGER` | `INTEGER` | **5** |
| `INTEGER` | `TEXT` | 5 |
| `TEXT` | `INTEGER` | 5 |
| `TEXT` | `TEXT` | **2** |

(Control, with keys distinct both numerically and textually: 3 rows in all four cells.)

The intuitive diagnosis — "the affinity *mismatch* causes the fan-out" — **is wrong**.
`INTEGER`/`INTEGER` fans out identically. The cause is `INTEGER` affinity on *either* side collapsing
`1`, `1.0`, `01`, `1.00` into a single value. Only all-`TEXT` preserves the truthful 2.

Consequence: comparing column profiles between two files **cannot** detect the `INTEGER`/`INTEGER`
case, because both sides look identical and correct. This is a real limitation to state, not a bug
to fix.

### 2.2 A connection holds ten attached databases, and the eleventh raises

> **No longer constrains this server (2026-07-26).** Every datasource now holds its own
> engine, so nothing `ATTACH`es and no ceiling is reached. The ten-slot limit survives as a
> *choice* — each slot costs live connections and, until it is spilled, memory — and both
> `config.py` and the README now say so rather than citing this number. The measurement
> below is still true of SQLite; it just no longer decides anything here.

`SQLITE_LIMIT_ATTACHED` reports **10** (SQLite 3.47.1), and the eleventh `ATTACH` fails hard rather
than degrading:

```
sqlite3.OperationalError: too many attached databases - max 10
```

Three facts follow from measurement rather than from the documentation:

- **`main` is not counted.** After ten successful attaches, `PRAGMA database_list` returns **11**
  rows. So ten is ten *besides* the host database, which stays free.
- **An in-memory attach costs a slot like any other.** `ATTACH DATABASE ':memory:' AS nick` yields a
  writable, private database — and consumes one of the ten. So a design where every datasource is an
  attached database (a flat file included) has a **hard ceiling of ten datasources**, and the number
  is not a policy choice.
- **`DETACH` genuinely frees a slot**, so eviction recovers capacity rather than merely forgetting.

Cross-database joins between two independently attached in-memory databases work in one ordinary
statement — verified, not assumed. That is what makes "one nickname per datasource" cheap: the join
is SQLite's, not ours.

Consequence for the configuration: `workspace.slots` is validated against this ceiling and refuses a
larger value, because the failure it would otherwise produce arrives mid-session, at attach time,
long after the mistake was made.

### 2.3 A join may cross attached databases; a **view** over that join may not

> **No longer reachable (2026-07-26).** A statement reaches one datasource, so there is no
> cross-database join to store as a view. Retained because it records why the cross-database
> `CREATE VIEW` refusal was once cited as a justification for `add_table` (now `create`) — a feature that
> did not exist justifying a design decision, which is the error this file exists to prevent.

Querying across two attached databases works in one ordinary statement (§2.2). Storing that same
statement as a view does not:

```
sqlite3.OperationalError: view named cannot reference objects in database wh
```

The refusal is at `CREATE VIEW` time, not at read time — so this is not a fragile view that later
goes stale, it is a view that never exists. The two halves are easy to conflate and behave
completely differently:

| Statement | Across attached databases |
|---|---|
| `SELECT … JOIN other.table` | works |
| `CREATE VIEW … AS SELECT … JOIN other.table` | refused outright |

**Deduced, not discovered.** A view is SQL text stored inside one file; a file has no way to name a
table in another. The behaviour follows from that in one step and needed no experiment. It is
recorded here because the exact error text is worth quoting, not because it was a surprise.

**Consequence for the design.** Little, as it turns out. Views are not part of this package, so
this refusal is not what justifies `create` — an earlier draft of this section claimed it did.
`create` exists for a simpler reason that holds whether or not views are in play: `save` writes
one database, not a join. A lookup meant to outlive the session needs both sides *inside* the slot
being saved. Guidance that says "attach both files and join them" offers an answer to a question,
not something the user can keep.

### 2.4 A view that names its own schema is not portable, and poisons the whole file

> **Sharper now, not gone (2026-07-26).** Such a view used to be usable under exactly one
> nickname — the one it was built under, because the file was `ATTACH`ed under that name.
> Each datasource is now opened as a database in its own right, so **no** name resolves it:
> the file is refused at attach, under every nickname. The save-time openability check this
> section used to justify became unreachable and was removed.

Following on from §2.3: a view *may* name its own database, and SQLite accepts it at `CREATE VIEW`
time.

**Mostly deduced too.** A nickname is a connection-scoped label that appears nowhere in the file,
and a view is SQL text that does live in the file — so a qualified name inside a view cannot
survive being attached under a different label. One deduction, no experiment. What *was* worth
measuring is the blast radius below: lazy failure of the single view is the reasonable expectation,
and it is not what happens.

```sql
CREATE VIEW shop.revenue AS SELECT … FROM shop.sales s JOIN shop.prices p ON …   -- accepted
```

The nickname is then stored inside the view's SQL. Copy that database to a file and attach it under
any other name and the schema no longer parses:

```
sqlite3.OperationalError: malformed database schema (revenue)
  - view revenue cannot reference objects in database shop
```

Three things make this worse than it first looks:

- **It takes the whole database down, not just the view.** `ATTACH` itself fails, so every table in
  the file becomes unreachable because of one view.
- **The wording points at corruption.** "Malformed database schema" reads as a damaged file; the
  data is perfectly intact and the problem is a name.
- **It is latent.** The view is created, read and saved without complaint. The failure only appears
  in some later session, under a different nickname — usually the one the user picked because it
  read better.

**The portable spelling is the short one, and it is verified:**

```sql
CREATE VIEW shop.revenue AS SELECT … FROM sales s JOIN prices p ON …   -- travels
```

Inside a view an unqualified table name already resolves to the view's own database, both in place
and after re-attaching under a new name. So dropping the qualifier costs nothing and fixes it.

**Consequence for the design — and both halves of it have moved (2026-07-27).** This section used
to end by describing a save-time check (`save` writes the file, re-attaches it under another name,
and deletes it if that fails) and by arguing that refusing `CREATE VIEW` outright "would need
statement classification — more code, and fragile". Neither describes the shipped code.

**Such a file is refused at attach, and the refusal explains itself.** There is no save-time
re-attach check; the header note above records why it became unreachable. What `Registry` does
instead is catch the failure where it actually surfaces — the first table listing after attaching —
and translate SQLite's "malformed database schema" into the recoverable thing that really happened,
including the fix (`FROM sales`, not `FROM shop.sales`). That is the §5.1 rule applied where it
belongs: the diagnosis comes from what the operation returned, never from scanning the view's SQL
for the nickname, which guesses wrong on a table *aliased* to the same word
(`SELECT shop.qty FROM sales shop`).

**And `query` can no longer make a view at all, at no cost in statement classification.** The
read-only engine installs a SQLite authorizer whose whitelist admits four action codes
(`SELECT`, `READ`, `FUNCTION`, `RECURSIVE`); `SQLITE_CREATE_VIEW` is not among them, so the
statement is refused at preparation and never runs. No SQL text is parsed anywhere, which is why
the refusal cannot be walked around by a comment, a leading CTE or stray whitespace — all three
verified live, 2026-07-27, along with `ATTACH`, `ANALYZE` and `CREATE TEMP TABLE`. The argument
against refusing views was reasoning about a mechanism the design did not end up using.

---

## §3 — Volume, performance, concurrency

### 3.1 The write primitive decides the memory profile — by ~35×

Loading 200,000 rows, peak measured by `tracemalloc` with the frame allocated before tracing starts:

| Path | Frame `memory_usage(deep=True)` | Peak during write | Ratio |
|---|---|---|---|
| `to_sql` via SQLAlchemy | 3.20 MB | **113.75 MB** | **35.6×** |
| `to_sql`, numeric + text | 6.09 MB | 118.37 MB | 19.4× |
| `to_sql` via raw sqlite3 | 3.20 MB | 28.70 MB | 9.0× |
| **`executemany` over a lazy iterator** | 6.10 MB | **0.008 MB** | **~0×** |

**Sub-batching does not bound it.** `to_sql(chunksize=K)` lowers the constant but the peak still
scales with *total rows*, not with `K` — 100k → 17.67 MB, 200k → 32.93 MB, 400k → 63.46 MB, 800k →
124.50 MB at a fixed `chunksize=5000`. Sweeping chunksize at 200k rows asymptotes at ~5× and never
approaches zero. Cause: pandas materialises the whole frame into insert-ready sequences **once,
before** it chunks, so chunking bounds the *statement* size and not the *allocation*.

**`executemany` over a lazy row iterator is flat.** `conn.executemany(INSERT…, df.itertuples(index=False, name=None))`
held a **0.008 MB** peak from 100,000 rows through 1,600,000 — a 16× growth in row count moved the
peak not at all. It is also **~4–5× faster** (200k rows: 0.92 s vs 4.57 s). There is no
speed-for-memory trade to weigh.

### 3.2 The invariant that makes 3.1 true, and how to break it

**The row sequence handed to `executemany` is never materialised.** That single sentence is the
whole property, and it is sharp and testable:

| Path | Peak at 200k rows |
|---|---|
| `exec_driver_sql(sql, list(itertuples))` | 40.19 MB — **materialising kills it** |
| `raw_connection()` → cursor → `executemany(iterator)` | **0.020 MB** |

Reaching the DBAPI cursor through `engine.raw_connection()` borrows the connection from the
SQLAlchemy pool, so the engine abstraction survives and pragmas remain reachable on the same engine
afterwards. **Test this as a growth property** — assert the traced peak stays flat as row count grows
by an order of magnitude, not that it is below some absolute number.

The adapter layer §1 requires does **not** break the invariant, but it does move the constant:

| Rows | Natively bindable | With adapters (2 of 4 columns adapted) |
|---|---|---|
| 400,000 | 0.0091 MB, 2.09 s | **2.5684 MB**, 6.29 s |
| 1,600,000 | 0.0075 MB, 9.04 s | **2.5677 MB**, 25.89 s |

Flat across a 4× row increase — a constant, not a per-row accumulation. Quote **~2.5 MB** for the
adapted path, not the bare path's 0.008 MB. Adaptation costs **~2.9× wall clock**.

### 3.2a SQLAlchemy Core cannot take the lazy iterator, and chunking it is what bounds the peak

Measured 2026-07-26, deciding the insert path when the workspace moved onto Core.

**Core refuses an iterator outright.** `connection.execute(table.insert(), rows)`,
`connection.execute(text(...), rows)` and `connection.exec_driver_sql(sql, rows)` all raise on a
generator — `ArgumentError: mapping or list expected for parameters` — so the property in §3.2
cannot simply be carried over. Materialising for it is worse than the path already rejected:

| Path | 100,000 rows | 800,000 rows |
|---|---|---|
| driver `executemany` over a lazy iterator | 0.0137 MB / 1.44 s | 0.0106 MB / 12.14 s |
| **Core `insert()`, list of dicts** | **63.78 MB** / 4.43 s | **511.12 MB** / 33.06 s |

**Chunking a lazy iterator bounds it, and pandas' failure does not carry over.** `to_sql(chunksize=K)`
does not bound its peak because pandas materialises the whole frame *before* it chunks (§3.1). Pulling
`K` rows at a time from a generator has no such upstream step, and the peak tracks the chunk size and
nothing else:

| Chunk | 100,000 rows | 800,000 rows |
|---|---|---|
| 1,000 | **0.8446 MB** | **0.8139 MB** |
| 5,000 | 3.5849 MB | 3.6000 MB |
| 20,000 | 13.7852 MB | 13.8244 MB |
| 100,000 | 63.7352 MB | 68.2710 MB |

800,000 rows at chunk 1,000 cost what 100,000 do. **The smallest chunk measured is also the fastest**,
so there is no memory-for-speed trade to weigh inside Core and nothing to tune. What Core does cost is
a flat **~2.6× wall clock** against handing the driver an iterator (33 s vs 12.8 s at 800,000),
independent of chunk size — it is per-row parameter processing, not batching. That is the price of the
abstraction, and it is paid once per load rather than per query.

**Residency has no portable form.** `inspect()` covers schema, not storage, so
`(page_count − freelist_count) × page_size` stays a per-dialect answer. It reports `None` — not `0` —
for a database whose data is not in this process: a tag that cannot be measured is not a tag holding
nothing, and zero would mean never spilling it.

### 3.3 `memory_usage(deep=True)` undercounts object columns by ~6×

A frame of 3 cells, each holding a list of 100,000 distinct strings:

```
memory_usage(deep=True)             = 2,403,108 bytes
actual payload (one cell's strings) = 4,688,890 bytes   → ~6× undercount
```

pandas applies `sys.getsizeof` **per element, non-recursively** on `object` dtype, so the list
objects are counted and the strings inside them are invisible. Any budget decision made from this
number is unsound for nested content.

Related: instrumenting the write path with `tracemalloc` costs **5.71×** wall clock (4.564 s vs
0.799 s). It is a diagnostic, not a production measurement.

### 3.4 The MCP server dispatches tools concurrently — on real OS threads

Verified end to end against a real stdio subprocess driven by a `ClientSession` issuing two
`call_tool`s under `asyncio.gather`. The lowlevel server does `tg.start_soon(handler, …)` per message
and never awaits the handler:

| Case | Result |
|---|---|
| Two blocking **sync** tools | Fully concurrent, 1.205 s overlap, **on two different OS threads** |
| Two **async** tools | Fully concurrent, 1.200 s overlap, same loop thread |
| A fast call during a slow sync call | Completed while the slow one was in flight |

**Sync tool bodies are dispatched to worker threads.** Any "only one thing happens at a time"
assumption is false from the first tool.

### 3.4a The GIL does not make this safe

Worth stating precisely, because "Python has a GIL, so there is no concurrency" is half right and
leads to the wrong conclusion. Measured on CPython 3.12.9, two threads against one:

| Work | 2 threads / 1 thread | Reading |
|---|---|---|
| Pure-Python CPU loop | ~2.0× | No overlap — the GIL is held, as expected |
| SQLite scan, **shared** connection | 2.14× | No overlap — but from SQLite's per-connection mutex, not the GIL |
| SQLite scan, **separate** connections | **1.02×** | Genuine parallelism — `sqlite3` releases the GIL around its C calls |

So parallelism is real for I/O and C-extension work. More importantly, **the data-loss hazard does
not require parallelism at all — only interleaving**, which threads provide whatever the GIL is
doing. A writer with an open transaction, and a reader that runs one statement and calls
`rollback()` on the same connection:

```
rows inserted by the writer: 1000
rows actually in the table:     0
```

No exception on either side. A connection has exactly one transaction, so any second user of that
connection can discard the first's uncommitted work. This is §3.5's 79,807-row loss reduced to
twenty lines, and it is why serialising access is not optional.

**Corollary worth having: serialising a shared connection is nearly free.** It already serialises
statements internally (2.14× above), so a lock costs throughput that was never available, and buys
transaction-level safety that nothing else provides. The 11–16× throughput figure in §3.6 is the
price of a lock in a *different* configuration — separate connections, where real parallelism exists
to lose.

### 3.5 A shared connection loses data silently, at scale

With `StaticPool` + `check_same_thread=False`, every thread shares **one** DBAPI connection. Three
measured consequences:

- **An unrelated concurrent read rolls back an in-flight load.** SQLAlchemy's pool default is
  `reset_on_return = rollback`. A reader checks out the (same) connection, runs one `SELECT`, closes
  its checkout — and the writer's open transaction is gone. At load scale this lost **79,807 of
  200,000 rows with no exception anywhere**.
- **A read-only guarantee implemented as a per-connection pragma leaks across threads.** Thread A
  opens a writable window; thread B, arriving as an ordinary read, **INSERTed and committed
  durably**.
- **Any concurrent reader holding a partially-fetched cursor blocks maintenance.** `VACUUM` fails
  with `cannot VACUUM - SQL statements in progress`.

**`in_transaction()` cannot be trusted to detect any of this.** It read `True` throughout a run in
which **0 rows landed** and `commit()` did **not** raise. (An earlier observation of it flipping
`True`→`False` is also real.) No design may detect loss via that signal.

### 3.6 Connection posture — measured, four candidates

200,000-row load on two OS threads against a concurrent cursor-scanning reader:

| Posture | Reader p50 | Scans done | Rows landed | `CREATE TABLE` under read | Maintenance w/ cursor open |
|---|---|---|---|---|---|
| One conn + process-wide lock | 693.4 ms | 1 | 200,000/200,000 | 40/40 | **BLOCKED** |
| One conn, no lock | 579.7 ms | 11 | **120,193/200,000** | — | **BLOCKED** |
| shared-cache `:memory:`, 2 conns | 573.4 ms | 9 | 200,000/200,000 | **0/40** | OK |
| **file-backed + WAL, 2 conns** | **465.7 ms** | **17** | 200,000/200,000 | **40/40** | OK |

**File-backed + WAL with separate reader/writer connections wins on every axis at once.**

Two eliminations worth keeping:

- **shared-cache `:memory:` is out.** 0/40 `CREATE TABLE` under a concurrent read, every one
  `database table is locked: sqlite_master`. Shared-cache uses **table-level** locks and signals
  `SQLITE_LOCKED`, which **`busy_timeout` cannot cure** — resolving it needs `sqlite3_unlock_notify`,
  which Python's `sqlite3` does not expose. Its `read_uncommitted=1` variant restores concurrency and
  buys it with correctness: a reader saw **100 rows that were rolled back and never existed**.
- **A process-wide lock starves readers.** Across four configurations the reader completed **exactly
  one** read per run. Its price is ~**1.2× latency and 11–16× throughput** — *not* the 5,200× a
  naive comparison suggests (see §5.2).

**Set a read-only connection's posture once, at connect, via `PRAGMA query_only = ON` in a `connect`
event listener** — never by toggling shared state around an operation.

---

## §4 — Writing files out

### 4.1 Created files are world-readable by default

A file created by SQLite's own `VACUUM INTO` lands at **`0o644`** — world-readable on a multi-user
host, containing the user's actual data. Any file this tool writes on the user's behalf must be
created `0o600` unless the user asked otherwise, and the mode must be set by pre-creating with a
restrictive umask or `chmod`-ing immediately, with the race acknowledged.

### 4.2 The overwrite guard is unconditional, and belongs at the path boundary

An earlier draft of this section conflated two separate questions — *whether* to refuse an existing
target, and *who* enforces the refusal — and got both answers from the same place. They are
unrelated.

**Whether: refuse by default, and let the user's answer come back as `force`.** A destination path
in this server arrives from an LLM relaying a name the *user* chose, so consent to destroy whatever
sits at that name is the user's to give. `force` is how that consent travels — the user was asked
and said yes, and the round trip is spared.

What makes this different from the `overwrite=True` it replaces is **the wording of the refusal.**
The old message ended `"Pass overwrite=true to replace it"`, which reads as an instruction to the
agent, and an agent will take it — the flag then quietly becomes the agent's own judgement. The
message now names the file and says *"Ask the user whether to replace it — if they say yes, call
again with force=true"*: a decision to put to somebody, not a retry to make. The parameter is the
same shape; who it is understood to speak for is not. There is a test asserting on that wording,
because the wording is the guard.

**And `force` has a hard limit: a file some live slot is sitting on is refused regardless.** That
covers every attached SQLite database, every flat file a slot was built from and would be rebuilt
from after eviction, and every spill file — `Registry.claimed_paths()`. The failure mode if it were
allowed is silent: on POSIX the unlink *succeeds*, the holding slot keeps answering from an inode
with no name, and nothing anywhere reports that the file the user believes they are looking at has
diverged from the datasource. `force` is authority over the user's spare files; it is not authority
over this server's open state.

**Who: the path boundary, never SQLite.** `VACUUM INTO` does refuse *some* existing targets, and
the earlier draft leaned on that. It does not hold. Measured across three target states:

| Existing target | Result |
|---|---|
| A valid SQLite database | refused — `output file already exists` |
| A zero-length file | **allowed**, written in place |
| A non-empty non-database | refused — `file is not a database`, an unrelated complaint |

Only the first row is the refusal the design wants, and a zero-length file is exactly what a
half-finished earlier write leaves behind. **The overwrite guard therefore belongs at the path
boundary**, where it is unconditional and reads the same for every writer, rather than being
delegated to SQLite.

Worth being plain about the weight of this one: delegating a filesystem policy to a SQL statement's
incidental behaviour was never sound, whatever that behaviour turned out to be. Measuring the three
rows corrected a *claim in this document*; it did not change a design, because nothing should have
been resting there in the first place.

### 4.3 `VACUUM INTO` cannot run inside a transaction

`cannot VACUUM from within a transaction`. Any flow that wants to copy or compact mid-write must
commit first — which means, at that moment, there is no transaction left to roll back and the undo
path has to be something else. Verified working from an in-memory source (311,296 bytes, all rows
present), and `page_count × page_size` on the source matched the resulting file size exactly.

---

## §5 — How to verify things (the method that caught the errors above)

These are not style preferences. Each was learned by getting it wrong first.

### 5.1 Assert on query results, never on binding

**The recurring failure in this project's history is a check that covers the question asked rather
than the question that matters.** Five separate instances.

The sharpest one: an adapter mapping was scored "working" on the strength of a `typeof` check after
insert. That established only that values *bind without raising*. They bound perfectly and returned
`sum` = `0.0`.

**Rule:** for any value-transforming mechanism, the probe asserts on what a query **returns** —
`sum`, `avg`, `min`/`max`, `ORDER BY`, and an equality join. Never on `typeof`, and never on the
absence of an exception.

### 5.2 An order-of-magnitude ratio is usually two different operations

A comparison reporting a 5,200× penalty turned out to be measuring a cursor-iterating read against a
single-step `SELECT count(*)`. Holding read shape fixed, the true cost was 1.2×.

**Rule:** when two legs of a comparison differ by orders of magnitude, suspect the operation before
believing the ratio.

### 5.3 Enumerate the case space; a sample presented as coverage is the bug

Two separate wrong conclusions came from probing the happy path (`int64`/`float64`/`str`/`bool`) and
presenting the result as settled. **Rule:** enumerate what the readers actually produce, and test all
of it.

### 5.4 A near-zero measurement has the same shape as a no-op

Every performance case must assert the rows actually landed and spot-check first and last values.
A write that does nothing is very fast.

### 5.5 A guarantee implemented as an interception point can be walked around

Demonstrated mechanically: a handle designed to route all access through a checked accessor was
bypassed by reaching the wrapped object directly (`getattr(handle, "engine")`), enumerating
everything with the checked path called **zero** times.

**Rule:** prefer guarantees carried by an object's own state (a connection that *is* read-only,
set once at open) over guarantees enforced by a wrapper that callers must go through. There is
nothing to bypass in the former.

### 5.6 Report a failed probe as failed

One measurement intended to isolate lock granularity failed to isolate its variable — the reader
starved under both arms, so every cell returned a single sample, and a ratio from one sample is not
a ratio. It is recorded as a failed probe and its output is not quoted anywhere. A probe that did
not isolate its variable produces no evidence, however plausible its numbers look.

---

## §6 — Deferred: only relevant once data outgrows memory

Not needed for a single-file, fits-in-RAM path. Recorded so it is not re-derived.

- **Freelist ratchet.** `page_count × page_size` does **not** shrink after `DROP TABLE` — 1058 pages
  before, 1058 after, with `freelist_count` = 1057. So N failed loads leave the database measuring at
  its high-water mark while holding no rows. Fix either by `PRAGMA auto_vacuum=FULL` **before the
  first table exists** (only settable then; reclaimed 1059 of 1060 pages), or by measuring
  `(page_count − freelist_count) × page_size`.
- **Neither fix returns memory to the OS.** RSS stays high after `FULL` reclaims pages. Both correct
  the *measurement* ratchet only. Under the default, a reload after a `DROP` reuses the freed pages
  and grows RSS by ~0.2 MB, which is the more useful property.
- **`PRAGMA cache_size` is signed, and the sign is the unit.** **Positive = pages. Negative = KiB.**
  Passing a byte budget of 512 MiB straight through as a positive number yields `cache_size =
  536,870,912` *pages* at a 4,096-byte page size — an implied bound of **2.00 TiB** instead of
  0.50 GiB. **Fail-open by 4,096×.** The correct spelling is negative KiB: `-524288` → 0.50 GiB,
  verified. Any code touching this pragma asserts the value reads back **negative**, not merely
  non-zero.
- **Page-cache occupancy is not observable from Python.** No pragma, no `sqlite3_status` binding. A
  file-backed database's memory use cannot be measured from inside the process; only the configured
  ceiling can be charged, and that is conservative *only if the unit is right* — which is why the
  point above matters.

---

## §7 — Driving the surface as an agent (2026-07-26)

A different instrument from the rest of this file. Everything above was established by executing
code; this section was established by giving the **finished tool surface** to agents that had never
seen it, with the source unreachable and every route to the data except the tools closed off, and
watching which verb they reached for. It answers the one question a test suite cannot: not *does
the surface work*, but *does it read the way it was meant to*.

**Method.** Six runs — the three arcs of `LEVEL0.md`, each driven once with the shipped skill
loaded and once without it, so the skill's contribution is visible rather than assumed. Each agent
got a request phrased the way a person phrases it, a command that spoke the real MCP protocol
in-process, and nothing else. Every call was logged by the harness independently of the agent, and
scored against answers computed outside the server; the agents' own accounts were read against that
log rather than taken. The fixtures were built to arm four specific traps: a mixed column, a
nickname collision, a join incomplete in **both** directions, and an occupied `save` path.

### 7.1 What the surface got right, measured rather than hoped

- **Every number matched ground truth**, in all six runs, across both the 300-row and the
  400,000-row file.
- **Nobody wrote `nickname.table`.** Not once, in any of six runs, with or without the skill. The
  addressing change of the previous session reached the model through the instructions and the
  docstrings.
- **Nobody tried to write through `query`.** No `INSERT`, no `CREATE TABLE`, in any run.
- **Nobody attached the second file as a second slot.** Every arc-2 and arc-3 agent reached for
  `create`, which is the single choice this design most wanted to make obvious — and the bare
  runs made it as reliably as the skill-loaded ones.
- **The collision protocol held.** The agent that hit `q1_sales_2` read the returned nickname and
  used it, and quoted `collided_with` back.
- **The mixed-column *signal* did its job.** Every agent that touched the column checked it rather
  than averaging blind. The naive average is 10.82 against a true 12.53 — a wrong answer nothing
  would have flagged, and none of them reported it.

### 7.2 What it got wrong

Each of these is fixed; they are recorded because the *class* recurs.

| Found | Where the fix belonged |
|---|---|
| `info` raised a driver error for **every** table inside an attached database — schema inspection ran on the read engine, whose authorizer refuses the `PRAGMA` the inspector speaks | the code: inspect over the write engine, as residency already does |
| The mixed-column warning prescribed `typeof(col)='integer'`, which cannot discriminate on the column that produced it (§1.5) | the code and the skill, and §1.5 above, which is where the advice came from |
| `query`'s docstring advertised writes it refuses, contradicting the server instructions | the docstring |
| The README taught `nickname.table` in both worked examples | the README, plus a test that greps for it |
| All three skill-less agents called `info` immediately after `attach`, for a payload they already held | the docstrings — the skill already said it, and its readers mostly skipped the call |

**The two that matter beyond themselves.** The `info` failure had been shipped and green for a
session: every table the test suite describes is one the loader *remembered loading*, so the branch
that asks the database was never entered. A suite can be exhaustive over the path its fixtures
build and blind to the path a user takes. And the mixed-column advice was correct where it was
written, copied to two other places, and wrong in the case those places actually meet — the same
shape as the four instances in this project's log of specifications written from side-effects.

### 7.3 Arc 3, where the agent is the wrong instrument

The spill is invisible by design, so an agent cannot confirm it and neither can this method. Both
arc-3 agents were asked "will memory be a problem" and both answered honestly that they could not
tell, offered behavioural evidence, and named the absence — which is the right behaviour and not a
defect. Verified separately from inside, at an 8 MB budget: residency measured **16.4 MB** after the
load, the operation that crossed the budget completed, the **next** operation found the database
moved to a temp file, and the same query returned identical rows either side of the move.

### 7.4 What this method cannot see

Stated so its coverage is not overread. The harness closed off `Read` and `Glob`, so an agent's
complaint that it could not list a directory before attaching is an artefact of the harness, not of
the product — a real client has those tools. Both arc-2 agents invented a `save` name rather than
the obvious one, so the occupied-path refusal was never reached by an agent; its wording is verified
only by direct execution. And a single run of one model is a sample, not a distribution: what six
runs establish is that a choice is *reachable*, never that it is *reliable*.

---

## §8 — Driving the live server through a real client (2026-07-27)

A second live pass, and a different instrument again from §7: the server connected to a real MCP
client rather than an in-process harness, driven by an agent with the filesystem available. §7 asked
whether the surface *reads* right. This pass asked what happens when the inputs stop being the ones
the fixtures build — every container format, and twenty-four spellings of the same five instants.

The seven verbs, the guards and the arcs came through it intact. What did not is recorded below.

### 8.1 A temporal from a flat file is never converted, and four spellings report the earliest
instant as the maximum

**The headline measurement of this pass.** `binding.py` converts temporals to INTEGER ticks exactly
as §1.4 requires, and its tests prove it. No caller can reach that code with a temporal. The reader
table is `.csv`, `.tsv`, `.txt` → `pd.read_csv` with no `parse_dates` and no `to_datetime` anywhere
downstream, so a date column arrives as `object`, is declared `TEXT`, and is compared as text.

Twenty-four spellings of five instants spanning three years, each scored by which label `ORDER BY`
put first and last (fixture and method: the run's own build script; the year span matters, because
`MM/DD/YYYY` and `DD.MM.YYYY` both sort correctly *by accident* inside a single year):

| Spelling | Orders correctly? | `max()` returns |
|---|---|---|
| `iso_date`, `iso_datetime`, `iso_t`, `iso_t_z`, `iso_micros` | yes | correct |
| `off_utc`, `off_local`, `ampm`, `year_quarter`, `iso_week` | yes | correct |
| `compact_ymd`, `epoch_s`, `epoch_ms`, `excel_serial` | yes (numeric) | correct |
| **`us_mdy`** (`03/01/2025`) | **no** — earliest wrong | correct by luck |
| **`eu_dmy_dot`** (`01.03.2025`) | **no — fully inverted** | **the earliest instant** |
| **`eu_dmy_slash`**, **`dmy_dash`** | **no — fully inverted** | **the earliest instant** |
| **`month_name`** (`Mar 01, 2025`), **`month_long`** | **no — fully inverted** | **the earliest instant** |
| **`rfc2822`** | **no** | **the earliest instant** |

Two further measurements on the same table, both silent:

- **The same five instants, one column offset-`+00:00` and one offset-`-05:00`, join 0 rows.**
  Joined on `epoch_s` they join 5. §1.4's tz trap, reproduced through the shipped surface.
- **`WHERE eu_dmy_dot > '01.01.2025'` returns all 5 rows; the true answer is 2.** A range filter on
  a European-format date column silently selects everything.

Nothing warned. The mixed-column detector fires correctly for a text-in-numeric column (§1.5) and had
no counterpart for this, which is the larger silent-wrong-answer class of the two: a date column is
*uniformly* text, so no storage-class signal exists to trip.

> **Resolved 2026-07-27**, and the shape of the fix is the finding. Only the two spellings that
> carry their own meaning are recognised — ISO 8601 extended calendar forms, and Unix time — and
> everything else is *reported* rather than parsed, because `01/03/2025` is March or January
> depending on who wrote the file and a server that guesses is silently wrong. The grammar is
> pandas' own `to_datetime(format="ISO8601")`, which rejected every ambiguous spelling tried
> against it; two holes were closed on top of it (it accepts basic-format `20240301`, which would
> turn order numbers into dates, and maps `''` to `NaT`). Re-measured against the same fixture: the
> seven correct spellings still correct, the cross-offset join **5 where it was 0**, the range
> filter **2 where it was 5**, and all seven ambiguous spellings now named in a warning. Full
> contract in `LEVEL0.md`.
>
> **Stored as canonical UTC text, not integer ticks** — a deliberate departure from §1.4, recorded
> because §1.4 is otherwise unambiguous. Ticks were built first and measured worse *here*: they
> move the silent wrong answer instead of removing it, since `WHERE order_date > '2025-01-01'`
> against a tick column compares integer to text and returns **zero rows with no error**. §1.4's
> evidence is about offset-*preserving* text and about durations; canonicalising to a single offset
> answers the first, and no reader here produces the second. §1.4 stands for typed readers, which is
> where its measurements came from.
>
> **The deciding measurement, because the intuition runs the other way** (SQLite 3.47.1). "Store
> epochs so the SQL date functions work" is backwards: ISO 8601 text is the *native* input to
> SQLite's date functions, and an integer is not. Against an epoch column every one of them returns
> **NULL — silently, not as an error** — unless each call carries the `'unixepoch'` modifier.
>
> | Task | From ISO text | From an epoch integer |
> |---|---|---|
> | `date(col)` | `2025-12-25` | **NULL** |
> | `strftime('%Y-%m', col)` | `2025-12` | **NULL** |
> | `col > '2025-01-01'` | true | **false** |
> | get the epoch | `unixepoch(col)` → `1766664000` | it is already that |
> | difference in days | `julianday(a)-julianday(b)` | `(a-b)/86400.0` |
>
> **Text is a strict superset**: it yields the epoch on demand in one call and everything else
> natively, while the epoch form yields nothing text cannot and costs three silent failure modes.
> The only operation epochs win is raw subtraction, and `julianday()` covers that.
>
> One consequence worth stating: **`binding.py`'s temporal conversion is still unreachable from a
> flat file**, and that is now by design rather than by oversight. It is correct code waiting for
> the typed readers (parquet, feather, Excel) that arrive above level 0, and it is what will convert
> a column that arrives already typed.

**The pattern, sixth instance and the sharpest yet.** `tests/test_binding.py` contains
`test_same_instant_in_two_offsets_joins`, asserting `matched == 1`, and it passes — because it hands
`insert_frame` two tz-aware `pd.Timestamp`s. Handed the same two instants *by a CSV*, the server
returns 0. The test asserts the exact property the shipped surface fails, on the one input shape the
shipped readers cannot produce. It is not a wrong test; it guards a real conversion. It is a test
whose fixture reaches past the layer where the defect lives — §5.1's rule pointed one level up.

### 8.2 A delimiter that is not a comma is misparsed into one column, silently

`.csv` and `.txt` both dispatch to `pd.read_csv` with the default separator. A semicolon-delimited
file — the Excel default across much of Europe — is accepted, and every row lands in a single TEXT
column whose *name* is the joined header:

| File | Result |
|---|---|
| `a;b;c` / `1;2;3` | one column `a_b_c`, values `"1;2;3"` |
| a 25-column `;`-separated `.txt` | one column, a 250-character name |

Loud failures by contrast, all correctly refused: Latin-1 and UTF-16 (raw codec error), a parquet
wearing a `.csv` extension, and every unsupported extension (`.parquet`, `.xlsx`, `.json`, …), whose
refusal names the three supported suffixes. UTF-8 BOM, CRLF and a headers-only file all load
correctly.

### 8.3 A slot's table list was a snapshot, and three messages reported it after it went stale

Found by composing a slot with `create` and then forcing eviction. `Slot.tables` is taken at attach
time; `Registry.tables()` exists precisely to avoid reading it, and says so. Three user-facing
messages read the snapshot anyway, so every table added by `create` — the whole point of arc 2 —
went unnamed:

- the eviction record (`evicted.tables`): reported `["victim"]` for a slot holding `victim` and
  `sidecar`;
- the evicted-nickname explanation, whose advice *"attach it again to use it"* is then **actively
  wrong**, because re-attaching the source restores only the source's own table;
- the duplicate-attach refusal, which sends the caller to a slot it under-describes.

`detach` and `info` were correct throughout — they call `Registry.tables()`. Fixed 2026-07-27 at all
three sites, with tests that compose before they evict.

**Why the suite missed it**, and it is the §7.2 shape exactly: the covering test is named
`test_the_eviction_record_carries_what_is_needed_to_rebuild_the_slot` and its docstring reads *"Once
a slot can hold several tables, its source alone is not enough"* — but its fixture never calls
`create`, so the slot held one table, the snapshot equalled the live truth, and the assertion passed
against code that was wrong for every composed slot.

### 8.4 Epochs vary, and that is an argument for touching numbers less, not more

Measured 2026-07-27, from the question "if we store dates as epochs, whose epoch is it?". The
answer turns out to settle a different question than the one asked.

**There is no single epoch.** A number meaning "a date" means nothing without the epoch and unit it
counts from, and the ones in circulation are far apart:

| Epoch | Unit | Used by |
|---|---|---|
| 1970-01-01 | seconds | Unix/POSIX (IEEE Std 1003.1), C, Python, Java, JavaScript, Arrow/Parquet |
| **2001-01-01** | seconds | **Apple Cocoa `NSDate` / Core Data** — 978,307,200 s ahead of Unix |
| 1904-01-01 | days | classic Mac OS, and Excel's alternate date system |
| 1899-12-30 | days | Microsoft Excel (default), Lotus 1-2-3 |
| 1601-01-01 | 100 ns | Windows `FILETIME`, NTFS |
| 0001-01-01 | 100 ns | .NET `DateTime.Ticks` |
| 1960-01-01 | days / ms | SAS, Stata |
| 4713 BC | days | Julian day (SQLite's `julianday()`) |
| 1980-01-06 | weeks | GPS |

A Core Data timestamp read as Unix time is **31 years early**; an Excel serial read as Unix seconds
lands in 1970. These are not near-misses that a sanity check would catch by range.

**But the readers already resolve it, and that is the finding.** A typed format *declares* its epoch,
so the library that understands the format converts on the way in. Round-tripped through pandas, a
date column comes back as `datetime64` from every one of them:

| Format | Reads back as |
|---|---|
| `.xlsx`, `.ods` | `datetime64` — **including a workbook set to the 1904 system**, verified |
| `.parquet`, `.feather`, `.orc` | `datetime64` |
| `.dta` (Stata) | `datetime64` |
| `.numbers` (Apple) | `datetime` — see below |
| **`.json`** | **`int64`** — pandas writes and reads epoch-milliseconds, and does not convert back |
| **`.csv` / `.tsv` / `.txt`** | **text, or a bare number** |

**Apple Numbers is the case that looked most likely to leak an epoch, and does not.** It genuinely
stores dates the Cocoa way — seconds from **2001-01-01** — confirmed in `numbers-parser`'s own
source (`constants.py`: `EPOCH = datetime(2001, 1, 1)`, applied at `cell.py:911` and
`model.py:2612` as `EPOCH + timedelta(seconds=…)`) rather than inferred from a round-trip. But the
epoch is *internal*: a `DateCell` yields a `datetime`, verified on a synthetic file across both
boundaries that would expose an epoch error — 2001-01-01, which is zero in Cocoa seconds, and
1970-01-01, which is negative.

Two things to know before that reader is added. `.numbers` is **not a flat file**: it is a package
holding Snappy-compressed protobuf (`.iwa`) archives, so it needs `numbers-parser` — pandas has no
reader for it and never will. And a Numbers *table* is not a frame: it carries merged cells, empty
trailing rows (a fresh document reads back with eight of them) and header rows that are structural
rather than data, so mapping one to a table is its own decision and not a `read_*` call.

So the epoch zoo is the reader's problem for every format that carries its own types, and the
readers solve it. It reaches us in exactly two places, and both are formats that declare nothing:
JSON, which needs `convert_dates` / `date_unit` decided when that reader is added; and the flat
files level 0 actually reads.

**Which is the argument for leaving numbers alone.** In a CSV, `978307200` is a Core Data instant,
a Unix instant, an order number and a population count, and the file says which. Nothing does.
Converting it would be a guess with a 31-year error mode, which is the same class as reading
`01/03/2025` as March — so the same answer applies, and the numeric variant stays untouched
(`LEVEL0.md`). What made this worth measuring is that it *looked* like an argument for storing
epochs and is in fact an argument against inferring them.

### 8.5 What held

Recorded because a pass with findings should not read as a failing report.

- **The read-only floor is not walkable.** `INSERT`, `UPDATE`, `REPLACE`, `CREATE TABLE`,
  `CREATE VIEW`, `CREATE TEMP TABLE`, `PRAGMA`, `ANALYZE` and `ATTACH` all refused, and so were a
  comment-prefixed write, a CTE-prefixed `INSERT` and a whitespace-prefixed `UPDATE`. Nothing landed:
  row count, sentinel values and a target row all unchanged afterwards. The authorizer whitelist
  parses no SQL, so there is no text to evade.
- **The path guards are uniform across `save` and `query(path=)`** — occupied path refused with the
  §4.2 wording, a zero-length file refused (the case `VACUUM INTO` alone would overwrite), `force`
  overriding a spare file but **not** a file a live slot sits on, both written `0600` (§4.1).
- **Arc 2 end to end.** `create` → join → anti-join in both directions (4 and 6 orphans, exact) →
  index → `save` → re-attach: 300/61 rows, aggregate to the cent, index survived, read-only again.
- **The §7.2 fixes hold.** `info` on a table inside an attached database works; the mixed-column
  warning now names the offending value, and the remedy it prescribes was run verbatim and returned
  the true mean (20.6407) against the naive 20.2967 it prevents.
- **Nicknames.** Leading digit prefixed, spaces snake_cased, duplicate source refused by name,
  colliding-but-distinct sources disambiguated to `sales`/`sales_2` with `collided_with` populated.
- **Errors teach.** `orders.customers` is met with the addressing fix, an unknown table lists the
  real ones, an unknown slot lists what is attached, and a path outside the roots names the config
  knob.

---

## §9 — Memory footprint, stress and performance (2026-07-27)

Measured in-process against the real tool functions, at Chris's direction: *"the next step will
have to be memory footprint (along processing), stress testing (including the memory), and
performance measurement."* Probes live in `tmp/perf/`.

**Nothing here is a pass/fail budget.** Every number is an observation. What counts as an
acceptable footprint or latency is not a measurement's to decide, and none is asserted below.

### 9.1 A returned cell costs ~53 bytes plus one byte per character

The result of `query` is fully resident by the time the method returns. Its cost is **per cell, not
per row**, and flat across both axes — 200,000 rows of 9 columns and 5,000 rows of 201 columns are
within 6% of each other per cell:

| Shape | Cells | Peak | Bytes/cell |
|---|---|---|---|
| 10,000 × 9 | 90,000 | 5.00 MB | 58.3 |
| 50,000 × 9 | 450,000 | 25.68 MB | 59.8 |
| 200,000 × 9 | 1,800,000 | 104.36 MB | 60.8 |
| 5,000 × 21 | 105,000 | 5.79 MB | 57.8 |
| 5,000 × 101 | 505,000 | 27.48 MB | 57.0 |
| 5,000 × 201 | 1,005,000 | 55.04 MB | 57.4 |
| 100 × 201 | 20,100 | 1.10 MB | 57.2 |

Sweeping value width at a fixed 200,000 cells separates the constant from the content, and the fit
is clean:

```
bytes/cell = 53.1 + 0.998 x characters        (str4 57.3 · str12 64.9 · str40 92.9 · str120 172.9)
integer cells                = 44.3
```

So a result's cost is predictable before it is asked for: **cells × (53 + average value length)**.

**This is the evidence for the design decision already taken.** The tool docstring argues a row cap
"measures the wrong thing — a hundred rows of a two-hundred-column table is the flood it would be
meant to prevent". That was reasoning; it is now measured. A 100 × 201 result costs 1.10 MB and a
5,000 × 21 result of *half* as many cells costs 5.79 MB — rows do not predict cost, cells do.

**`yield_per` buys nothing observable here.** It bounds what the driver hands back at a time, but
the rows accumulate into a list regardless, and the peak tracks the finished result exactly. Do not
read the streaming in `Workspace.query` as bounding the server's memory; it bounds the driver
buffer.

### 9.2 `path=` bounds the answer, not the peak

> **Fixed 2026-07-28, and the section is kept for the shape of the finding.** The materialisation
> this section measures is gone: `Workspace.query_stream` hands the writer the open cursor, so a
> result bound for a file is never assembled, and eleven of the seventeen suffixes now hold a flat
> peak whatever the row count. What survives is the *reasoning* — a bound that is real and measures
> the wrong thing — and the six suffixes that still materialise because their format requires it.
> The corrected numbers are the second table below; the tables above it are what was true before.
> Both tables still name `.html` and `.htm`, which were dropped from the catalogue later the same
> day (§10.7); nothing else about the measurement changes.

`query(path=…)` is the documented route for a result that "does not belong in an answer". It is
accurate about the *answer*, and an agent may not infer more than it says — because the rows are
materialised in full by `Workspace.query` before `export_rows` ever sees them:

| Shape | In the answer | With `path=` | Ratio |
|---|---|---|---|
| 200,000 × 9 | 104.36 MB | 104.40 MB | **1.0004** |
| 5,000 × 201 | 55.04 MB | 55.15 MB | 1.0020 |
| 50,000 × 9 | 25.68 MB | 25.84 MB | 1.0059 |

What `path=` does save is everything *downstream* of the materialisation: the tool body's
`[list(row) for row in rows]`, a second full copy costing a further **1.14–1.28×**, and the JSON
text (20.79 MB for the 200,000-row result). Real, and not the dominant term.

**The export half streams for nine suffixes of seventeen, and not for the other eight.** Measured
2026-07-27 by handing `export_rows` a generator that records the file's size as each row is consumed
— a writer that streams grows the file while the generator is still running, one that materialises
leaves it at zero until the end (50,000 rows, ~1.2 MB, far past any file buffer):

| | Suffixes | Bytes on disk as the last row was consumed |
|---|---|---|
| **Streams** — writes row by row, never holds the result | `.csv` `.tsv` `.txt` `.json` `.jsonl` `.ndjson` `.xml` `.html` `.htm` | 0.87–2.77 MB |
| **Materialises** — builds the whole result first | `.yaml` `.yml` `.md` `.parquet` `.feather` `.orc` `.xlsx` `.ods` | **0** |

For the columnar three the materialisation is the format's shape — a columnar file stores each
column contiguously, so nothing can be written until everything exists — and they are still the
*fastest* writers measured. For YAML, Markdown and the workbooks it is simply how they are written:
`_write_yaml` builds a list of dicts, `_write_markdown` a list of lists, and both workbook formats a
whole `DataFrame`. That is not free at volume — **`_write_yaml` cost 22.9 minutes and ~5.4 GB for
1,000,000 × 11 where `.csv` cost 30 s** (measured under a memory configuration whose absolute
numbers do not otherwise transfer; this one does, because the list of dicts is built whatever the
budget is).

So `path=` bounds the answer for every format, and bounds the *peak* for none of them — but for the
eight materialising suffixes it is doubly so: the caller pays `Workspace.query`'s full result list
**and** the writer's own copy of it. **A large result wanted on disk should be asked for in a
streaming format.**

What does not exist for any of them is a way for `Workspace.query` to hand the writer an
unmaterialised cursor. Closing that is a change to the read path, not a fix to the export — **open,
and tracked as task 21**.

#### The correction (2026-07-28) — the cursor reaches the writer

`Workspace.query_stream` yields the column names and a lazy iterator over the open cursor;
`query` is that method plus a `list`, so there is still one read path. `server.query` uses the
streaming form when — and only when — `path=` is given.

Peak Python allocation for the whole `query(path=…)` call, measured with `tracemalloc` through the
tool function itself, four columns of mixed int/text/float:

| Suffix | 50,000 rows | 200,000 rows | Growth for 4× the rows | 200,000 rows, seconds |
|---|---|---|---|---|
| `.csv` | 0.18 MB | 0.18 MB | 1.00× | 2.4 |
| `.jsonl` `.ndjson` | 0.05 MB | 0.05 MB | 1.01× | 9.2 |
| `.json` | 0.05 MB | 0.05 MB | 1.01× | 9.3 |
| `.xml` | 0.05 MB | 0.06 MB | 1.00× | 4.3 |
| `.yaml` `.yml` | 3.13 MB | 2.46 MB | 0.79× | 90.9 |
| `.md` | 36.42 MB | 144.19 MB | **3.96×** | 22.8 |
| `.parquet` | 13.12 MB | 52.89 MB | **4.03×** | 1.8 |
| `.feather` | 13.12 MB | 52.89 MB | **4.03×** | 1.7 |
| `.orc` | 13.12 MB | 52.82 MB | **4.03×** | 1.7 |

The same measurement on the CSV path before and after, so the size of what was removed is on the
record: **9.04 → 0.17 MB** at 50,000 rows, **35.80 → 0.17 MB** at 200,000, **143.68 → 0.17 MB** at
800,000. The old figure is the list of tuples, and it tracked the row count exactly.

**YAML moved from the materialising group to the streaming one** by being dumped a chunk at a time —
a top-level sequence dumped in pieces concatenates into the same sequence, byte for byte. It cost
540 MB at 200,000 rows before and 2.46 MB after. **It remains by far the slowest writer**: 90.9 s
against `.csv`'s 2.4 s on the same result, which is PyYAML serialising rather than anything about
the peak. Confirmed at the full 1M × 11 corpus, where it writes 1.39 GB in **237.6 s adding no
measurable RSS over its baseline** — the size at which the old writer held gigabytes (§10.7).

So the group boundary now falls at **nine streaming suffixes and six materialising ones**
(`.md` `.parquet` `.feather` `.orc` `.xlsx` `.ods`), and each of the six is deliberate.
(The measurements above were taken while `.html` and `.htm` were still in the catalogue; they
streamed, at 0.07 MB flat, and were removed later the same day for a reason unrelated to their
peak — see §10.7.)

* the **columnar three** must have every value before they write any of it, because a columnar file
  stores each column contiguously. They are also the fastest and the most compact writers here, and
  the right destination for a large result;
* the **two workbooks** are capped at 65,535 rows (§10.7), so their peak is bounded by the cap;
* **`.md`** builds a list of lists and one string, because a Markdown table's column widths are not
  known until the last row has been seen. It is the most expensive per row of anything measured —
  144 MB for 200,000 rows — and it is a format for putting a small result in a document. **No cap
  has been set on it**: nothing has failed there, and the number above is on the record so the
  question can be settled with evidence rather than by analogy to the spreadsheets.

**The read path still materialises**, and that is the other half of task 21: `read_file` builds the
whole pandas frame before a row is inserted (§10.6). A very large YAML this server writes is
therefore one it may not be able to read back — the cliff belongs to loading, not to YAML.

### 9.3 Two memory dimensions, and the budget is on only one of them

`relieve_memory` reads `resident_bytes`, which is SQLite's `(page_count − freelist_count) ×
page_size` — **page storage**. A result set is Python objects. Both are memory this process holds;
they differ by most of an order of magnitude on the *same table*:

| Table | Resident (budget sees) | Payload (budget does not) | Ratio |
|---|---|---|---|
| 200,000 × 3, no dates | 5.70 MB (10.0 B/cell) | 52.71 MB (92.1 B/cell) | **9.2×** |
| 200,000 × 4, ISO dates | 9.79 MB (12.8 B/cell) | 65.86 MB (86.3 B/cell) | **6.7×** |

§5.2 warns that an order-of-magnitude ratio is usually two different operations. Here it
demonstrably *is* two different operations — which is the finding rather than a confound, and why
both were measured on one table. **A `memory_budget_mb` of 100 does not cap the process at 100 MB**;
a single `SELECT *` against a database sitting comfortably inside the budget can allocate several
times it, and nothing spills in response because nothing on the budget's books moved.

### 9.4 Arc 3, end to end through the tool surface — deferred, cheap, invisible

§7.4 said an agent is the wrong instrument for the spill because the move is invisible by design.
This is the in-process harness it asked for: real tool functions, an 8 MB configured budget, 400,000
rows (120,000 was tried first and occupied only 3.13 MB of pages — the probe reported that it had
created no pressure rather than measuring nothing, per §5.6).

Every property held:

| Property | Result |
|---|---|
| Spilled *during* the load that crossed the budget | **No** — the overshoot is tolerated once, as designed |
| Spilled on the next operation | Yes |
| Time to spill 10.80 MB | **0.0027 s** |
| Python memory to spill it | **0.015 MB** — the work is inside SQLite |
| Temp file on disk | 10.80 MB, matching residency exactly |
| Residency afterwards | `None` — unknown, not zero |
| Same nickname, same answer | Yes (checksum identical either side) |
| Still writable | Yes — `create` succeeded afterwards |
| Query cost, spilled ÷ in memory | **0.95×** |

A spilled slot is not slower to query. **That figure is warm-cache**: the file was written moments
earlier and the OS is still holding it. A genuinely cold spilled slot is not measured here.

### 9.5 What an attach costs, and what the temporal path adds to it

200,000 rows through `attach`, timing taken untraced (tracing inflates wall clock ~5×, §3.3):

| Shape | File | Peak | Wall clock |
|---|---|---|---|
| No date column | 5.25 MB | 22.65 MB | 1.69 s |
| Date already ISO 8601 | 9.25 MB | 24.15 MB | 2.60 s |
| Date needing conversion | 10.21 MB | 32.38 MB | 3.82 s |

Session 34 measured the temporal work in isolation; this is it paid inside a full attach. A column
that genuinely needs normalising costs **+1.22 s and +8.2 MB** over one already ISO, and **+2.13 s**
over no date column at all — against a load that is otherwise 1.69 s. Residency is identical for
both date shapes (9.79 MB), which is the expected consequence of both landing in the same canonical
form and a useful check that the conversion is not storing something different.

### 9.6 A long session plateaus — and a short measurement of it reads as a leak

**The mistake is recorded because it is the instructive part.** A 120-cycle run showed RSS climbing
**0.82 MB per cycle, linearly**, while `tracemalloc` saw 1160× less (0.0007 MB/cycle) and the slot
count stayed pinned at its capacity of ten. Read alone, that is a C-side leak: 818 MB projected over
1,000 cycles.

It is not. 120 cycles was inside the ramp. Run from a cold start for **800** cycles, the per-block
slope collapses:

| From cycle | 0 | 100 | 200 | 300 | 400 | 500 | 600 | 700 |
|---|---|---|---|---|---|---|---|---|
| MB/cycle | 1.654 | 0.404 | 0.254 | 0.311 | 0.062 | 0.026 | 0.071 | 0.028 |

137.3 MB → 442.8 MB overall, with the **final 200 cycles adding 9.5 MB**. That is a working-set
high-water mark being reached, not a leak — the shape is asymptotic, and only ten slots are ever
live. A residual ~0.05 MB/cycle remains at cycle 800 and a much longer run would be needed to call
it zero.

**The general point, which outlives this measurement:** a growth rate sampled inside the ramp
extrapolates to a leak that is not there. Distinguish the two by *shape over blocks*, never by a
slope through one window — the same discipline §5.2 states for ratios.

**And `tracemalloc` is the wrong instrument for footprint.** At the end of that run it reported
1.71 MB against an RSS of 340 MB — **~200×** apart. It traces Python allocations only, and SQLite is
a C extension with its own arena. Any question about how much memory this server uses has to be
answered from RSS.

Steady state also holds structurally: 40 attaches against a capacity of 10 ended with exactly 10
slots and 1.48 → 1.58 MB traced (1.06×), and 150 attach/detach cycles grew 46 KB in total with the
per-cycle increment *halving* between thirds (0.70 → 0.36 KB/cycle).

### 9.7 The global lock: correct under load, and it converts concurrency into latency

`server._lock` serialises every tool call. Eight threads issuing 20 `query` calls each:

| Threads | Throughput | p50 | p95 | Wrong or failed answers |
|---|---|---|---|---|
| 1 | 318 calls/s | 2.9 ms | 4.4 ms | **0** |
| 2 | 323 calls/s | 6.0 ms | 9.6 ms | **0** |
| 4 | 322 calls/s | 11.7 ms | 24.1 ms | **0** |
| 8 | 323 calls/s | 23.4 ms | 45.7 ms | **0** |

**Throughput is flat and latency is linear in caller count** — the textbook signature of a global
lock, and exactly what §3.6 predicts. Nothing is gained by adding callers and nothing is lost but
time; no answer was ever wrong or partial, which is the property the lock is there to buy. §3.6's
file-backed-WAL-with-separate-connections posture is what to reach for *if* this latency ever
matters. At interactive scale it does not.

### 9.8 Still unmeasured

- **A cold spilled slot** — 9.4's 0.95× is warm-cache.
- **The plateau's residual** — ~0.05 MB/cycle at cycle 800, indistinguishable from zero without a
  much longer run.
- **Concurrency against residency** — every 9.7 call was a read against one slot; spilling while
  another thread queries is not covered.
- **Non-SQLite datasources** — every figure here is SQLite. A server-backed URL reports `None`
  residency by construction, so the budget never sees it at all.

---

## §10 — A million rows and ten million, end to end (2026-07-27)

Two generated files, driven through the real tool functions: a **wide** one (1,000,000 rows x 11
columns, 1.22 GB — a date, a datetime, an id, text at 10/100/1000 characters, small and very large
integers, ordinary and very large floats, and a float inside (-1, 1)) and a **tall** one
(10,000,000 rows x 5 columns, 0.76 GB, carrying an undeclared `foreign_id` into the wide file's id
space). Generator and harness in `tmp/perf/`. **19 steps, 0 failed.**

### 10.1 The catalogue is one extraction format, and `path=` ignores the suffix

Worth stating first, because it bounds everything else here. `loader.READERS` holds `.csv`, `.tsv`
and `.txt` — all three `pandas.read_csv` variants — and `export.py` exports `export_csv` and nothing
else. There is no JSON, YAML, TOML, XML, Excel, ODS or Parquet, in either direction.

**And `query(path=…)` calls `export_csv` unconditionally**, so the suffix selects nothing:

| Asked for | Reported | Written |
|---|---|---|
| `out.json` | `ok: true` | `a,b\r\n1,x\r\n2,y\r\n` |
| `out.parquet` | `ok: true` | `a,b\r\n1,x\r\n2,y\r\n` |
| `out.wibble` | `ok: true` | `a,b\r\n1,x\r\n2,y\r\n` |

An agent that asks for Parquet is told it succeeded and gets a file whose name lies about its
contents — the same silent-wrong-answer shape as §1.1 and §8.1, in the export path. A `WRITERS`
registry keyed on suffix, refusing an unknown one by name the way `read_file` already does, is the
fix and is also the seam every new format arrives through.

> **Resolved 2026-07-27**, and the fix is the registry this section asked for. Re-measured against
> the code on `new-v3` at `af95ce79`, through the tool functions rather than the writers directly:
>
> | Asked for | Reported | Written |
> |---|---|---|
> | `out.csv` | `ok: true` | `a,b\r\n1,x\r\n2,y\r\n` |
> | `out.json` | `ok: true` | `[\n  {"a": 1, "b": "x"…` |
> | `out.parquet` | `ok: true` | `PAR1…` |
> | `out.feather` | `ok: true` | `ARROW1…` |
> | `out.orc` | `ok: true` | `ORC…` |
> | `out.xlsx`, `out.ods` | `ok: true` | `PK\x03\x04…` (zip) |
> | `out.wibble` | **`ok: false`** | nothing — *"No writer for `'.wibble'`. The suffix chooses the format. Supported: …"* |
> | a name with no suffix | **`ok: false`** | nothing — refused the same way |
>
> **The suffix now selects the format, and an unknown one is refused by name before the path is
> resolved** — deliberately before, so a request the server was never going to satisfy cannot cost
> the caller an existing file on its way to failing.
>
> The catalogue is no longer one format in either direction. `loader.READERS` holds **20 suffixes
> across 10 readers**; `export.WRITERS` holds **17 suffixes across 9 writers**:
>
> | | Suffixes |
> |---|---|
> | Read and written (16) | `.csv` `.tsv` `.txt` `.json` `.jsonl` `.ndjson` `.xml` `.yaml` `.yml` `.parquet` `.feather` `.orc` `.xlsx` `.ods` `.html` `.htm` |
> | Read only (4) | `.fwf` `.numbers` `.xls` `.xlsm` |
> | Written only (1) | `.md` |
>
> **The sixteen-suffix overlap is the round-trip property, not a coincidence**: a file this server
> writes is one it can read back, and the two `DELIMITED` sets are held identical for the same
> reason. The five that do not overlap each name a real asymmetry — `.xls` lost its writer when xlrd
> dropped writing, `.numbers` and `.fwf` have no writer worth having, and Markdown is deliberately
> write-only because a Markdown table has no types and no quoting, so no reader could return what
> went in.
>
> What this section got right and is worth keeping: **the suffix is the whole of the format
> decision**, in both directions, so a new format is one registry entry and nothing upstream of it
> changes.
>
> **Corrected 2026-07-27 (session 40): the round-trip property held for fifteen of those sixteen,
> not sixteen.** The `out.xlsx, out.ods` row above is where the defect was hiding, and the row
> itself shows how: both were checked only as far as `PK\x03\x04`, and **both formats are Zip
> archives, so the magic bytes cannot tell them apart**. `.ods` had been writing XLSX. `file(1)`
> called the output *"Microsoft Excel 2007+"*, and pandas' own ODF engine refused it —
> `KeyError "There is no item named 'META-INF/manifest.xml' in the archive"`.
>
> The cause is one absent argument, and it is worth stating precisely because it is a trap any
> `Path`-passing caller can fall into: **`pandas.DataFrame.to_excel` infers its engine from the
> suffix of a `str` path but not of a `pathlib.Path`** — measured on pandas 3.0.2, where a `Path`
> falls back to openpyxl silently. `_write_workbook` is handed a `Path`. The engine is now named
> explicitly rather than inferred.
>
> **Why nothing caught it, which matters more than the bug.** The round-trip test *passed*: pandas
> reads a workbook by sniffing its contents rather than trusting its suffix, so the wrong file
> returned the right rows. A round trip through a reader that auto-detects cannot verify format
> identity — it verifies only that *something* readable was written. The check that does work is
> the archive membership (`mimetype` and `META-INF/manifest.xml` for ODS, `[Content_Types].xml` for
> XLSX), and that is what the two tests added at `813920f5` assert.
>
> **Real ODS costs about thirteen times what the workbook did**, which is the measure of how much
> was being skipped: 50,000 rows extracted in **156.2 s** against **11.8 s**, and read back in
> **66.3 s** against **13.4 s**. Any ODS timing recorded before `813920f5` is a timing of XLSX.
>
> **Corrected again 2026-07-28: `.html` and `.htm` were removed from both registries**, so the
> counts above now read **18 suffixes across 9 readers** and **15 across 8 writers**, and the
> overlap is **fourteen**. The round-trip property is stated the same way and is now true without
> the exception HTML had always been — it wrote a table of any size and could not read back past
> ~417,000 rows of eleven columns (§10.7). The reasoning is in `LEVEL0.md`.

> **The timings below carry a wide environmental variance, measured 2026-07-27.** They were
> taken in a single pass, and a later run of the same steps came out 2–3x faster, which was
> briefly read as a code improvement. It is mostly not one. Repeating the *identical* attach
> on *unchanged* code, in a fresh process, at the default memory budget:
>
> | Rep | attach | resident |
> |---|---|---|
> | 1 | 68.55 s | 1305.4 MB |
> | 2 | 53.64 s | 1305.4 MB |
> | 3 | 60.11 s | 1305.4 MB |
> | a fourth, through the full harness | 80.44 s | 1305.4 MB |
>
> **Within one condition the same operation spans 1.50x** (53.64–80.44 s), while the storage
> figure reproduces to the tenth of a megabyte every time — and matches §10.6's 1,305 MB
> exactly. Same corpus, same code path; only the machine's timing conditions differ. So the
> quantities this file states in *bytes* are reproducible and the ones in *seconds* are
> reproducible only to about ±25%.
>
> This is §5.2's rule applied to a ratio between runs rather than between operations: a
> difference between two conditions is a finding only if it exceeds the noise inside one
> condition, and the control that establishes that noise is the step most easily skipped.
> What it does **not** explain is the 149.6 s below, which sits outside the band — that gap
> is still open, and the likeliest reading is that the original pass ran under load, since it
> was measured while the full format sweep was in progress.

### 10.2 Loading dominates; everything else is comparatively cheap

> **Re-measured in §10.7 (2026-07-28).** The timings below are single draws taken before the
> harness could repeat a condition or record what the machine was doing. §10.7 supersedes them with
> medians of three and a per-step spread. The **conclusion** of this section — that loading
> dominates — survives; the absolute seconds do not, and the 149.6 s below is explained there.

| Step | Wide (1M x 11) | Tall (10M x 5) |
|---|---|---|
| `attach` | **149.6 s** — 6,683 rows/s, 8.1 MB/s | **472.5 s** — 21,162 rows/s, 1.6 MB/s |
| `SELECT count(*)` | 1.15 s | 1.54 s |
| extract `SELECT *` to CSV | 88.5 s — 11,300 rows/s | 160.8 s — 62,173 rows/s |
| `save` to a database file | 15.1 s — 90 MB/s | 6.7 s — 115 MB/s |
| re-attach that saved file | **3.2 s** | **0.6 s** |

Extraction runs **1.7-2.9x faster than loading** the same data. Of a ~35-minute run, ~22 minutes
was spent in the four load steps. If anything here is ever optimised, it is the load path.

Rows per second is the wrong headline for the wide file and the right one for the tall: the wide
file moves 8.1 MB/s at 6,683 rows/s because a 1000-character column dominates each row, while the
tall file moves 21,162 rows/s at only 1.6 MB/s. **Bytes and rows disagree by 13x across these two
files**, so neither alone predicts a load.

### 10.3 Re-opening a saved database skips the work rather than doing it faster

> **Re-measured in §10.7 (2026-07-28), and the conclusion held** — 67x for the wide file and 568x
> for the tall, against the 46x and 787x below.

3.2 s against 149.6 s (**46x**) for the wide file, 0.6 s against 472.5 s (**787x**) for the tall.

§5.2 says an order-of-magnitude ratio is usually two different operations, and here it plainly is —
which is the point rather than a confound. Attaching a CSV parses text and inserts every row;
attaching a saved database opens a file. Nothing got faster; the work stopped being done. That is
the concrete argument for `save` as a working habit rather than only an escape from ephemerality:
**one `save` converts an eight-minute reload into six-tenths of a second.**

`create(type="table")` reading the same 10M-row file cost 525.5 s against `attach`'s 472.5 s — 1.11x,
the same operation with the same shape, which is the expected result and a useful negative.

### 10.4 An index pays for itself on the first join, not the second

> **Re-measured in §10.7 (2026-07-28), and the conclusion held.** Every absolute number moved —
> the index wins by 1.93x counting the build rather than 2.16x, and 14.2x afterwards rather than
> 12.1x — but the ordering and the argument are unchanged.

The tall file's `foreign_id` points into the wide file's `id`, and nothing declares it. Composed the
way the surface intends — `create` lands the second table in the first's database, then ordinary SQL:

| | Time |
|---|---|
| Join, no index | 124.8 s |
| Build index on `tall.foreign_id` | 47.4 s |
| Join, indexed | **10.3 s** |

47.4 + 10.3 = 57.7 s against 124.8 s, so the index wins by **2.16x even counting the build**, on a
single join, and by **12.1x** on every join after. All 10,000,000 rows matched, which is the correct
answer and confirms the join is doing real work rather than failing to.

**Nothing in the server says any of this** — principle 3 holds, no index is inferred and no join key
is guessed. What changes is that the numbers now exist for an agent to reason with.

### 10.5 The round trip is value-exact and format-lossy

CSV in, CSV out, compared cell by cell over 300,003 rows:

```
formatting-only numeric differences : 92,783
real numeric value differences      : 0      (worst relative error 0.00e+00)
text differences                    : 0
```

Every difference is a spelling: `2893.90` → `2893.9`, `0.698008716370` → `0.69800871637`,
`5.605320e+17` → `5.60532e+17`. Trailing zeros and exponent normalisation, because the value goes
through SQLite `REAL` and comes back through Python's float repr. **No value changed**, including
`big_int` near 2^62 and floats spanning 1e12-1e18.

Two consequences worth carrying:

* **Both temporal columns round-tripped identically** — `2024-10-10` and `2024-10-10T03:01:26Z` came
  back character for character, on a 1,000,000-row file. The §8/`LEVEL0.md` date contract holds at
  volume.
* **Text lost its unnecessary quoting.** The generator wrote `"0_propatag"`; the export writes
  `0_propatag`, because `csv.writer` quotes only when it must. Semantically identical, not byte
  identical — so a byte-comparison of a round trip will report differences that are not losses, and
  any future round-trip test has to compare parsed values rather than lines.

### 10.6 What it costs while it runs

Peak RSS **~3.0 GB**, against a 1.22 GB source. The load is the peak, not the extract: `read_file`
builds the whole pandas frame before a single row is inserted, so the load peak tracks the *file*
and no chunk size bounds it (§3.2a bounds the *insert*, which is a later step). **This is the read
side of §9.2's gap and is tracked as task 21** — the memory budget bounds resting pages, not either
transient peak, so no configuration closes it. **The export side of that gap was closed on
2026-07-28** (§9.2); this side is what is left of the task, and it is the larger half: every reader
produces a whole frame, so it is the loading of a large file — not the writing of one — that now
sets the peak. Residency afterwards
is far smaller — 1,305 MB for the wide file, 735 MB for the tall — because that measures SQLite
pages, which is §9.3's point restated at volume.

Storage is close to parity throughout: wide 1.22 GB CSV → 1,305 MB resident → 1.37 GB saved
database; tall 0.76 GB → 735 MB → 0.77 GB. Nothing expands or compresses meaningfully.

### 10.7 The re-measurement — with repetitions, and with the conditions written down

**Measured 2026-07-27/28 (session 40).** Everything above in §10 is a **single draw**. This section
re-measures it as the **median of three repetitions of the whole condition**, at the server's own
**100 MB default budget** — never disabled, so the slot spills and the arm is `disk`, which is the
only arm comparable with an endpoint database (§5.2). Each step also records its own load average
and the kernel's `Pageouts`/`Swapouts` differenced across it, so **a timing that includes paging
says so instead of being explained afterwards.**

**Read the spread before the number.** A per-step spread is max/min over the repetitions, and it is
the bar any comparison has to clear: a format 1.3x faster in a step whose own spread is 1.4x has
not been shown to be faster. The spread belongs to the pair being compared — the widest spread
anywhere in a run is *not* a universal floor.

#### Load and lifecycle

| Step | Wide (1M x 11) | spread | Tall (10M x 5) | spread |
|---|---|---|---|---|
| `attach` | **107.3 s** | 1.23x | **261.0 s** | 1.24x |
| `SELECT count(*)` — *includes the spill, see below* | 13.4 s | 1.35x | 7.5 s | 1.34x |
| extract `SELECT *` to CSV | 56.8 s | 1.19x | 88.9 s | 1.29x |
| `save` to a database file | 9.1 s | 1.31x | 5.3 s | 1.27x |
| re-attach that saved file | 1.6 s | 3.03x | 0.5 s | 2.91x |

**§10.2's `count(*)` of 1.15 s and this 13.4 s are not the same operation.** Relief is lazy —
`relieve_memory` runs on the way *into* an operation — so at a spilling budget the count is the
step the spill happens in, and the spill is inside its timing. §10.2's figure was a resident slot
at a large budget. Neither number is wrong; the step is simply not a read-throughput baseline at a
budget that spills, and it was described as one.

**§10.2's unexplained 149.6 s no longer needs a code explanation.** Attach here spans 101.5–124.4 s
and **every repetition paged** — the first pushed **626,278 pages (~2.4 GB) to swap** during the
attach alone. A paging attach lands squarely in the 149.6 s range. The likeliest reading is now a
measured mechanism rather than a guess, and **the `d4abba1b` A/B is probably unnecessary.**

#### The format sweep

Extract, then read the file back through `attach`. Every byte count reproduced **exactly** across
repetitions except `.xlsx` and `.ods`, which embed a timestamp (that check is what caught §10.1's
`.ods` defect).

| | Wide extract | Wide read back | Tall extract | Tall read back |
|---|---|---|---|---|
| `.feather` | **19.0 s** | **74.8 s** | **55.9 s** | 218.1 s |
| `.orc` | 22.5 s | 79.3 s | 61.0 s | **205.8 s** |
| `.parquet` | 26.2 s | 79.1 s | 58.7 s | 208.4 s |
| `.jsonl` | 39.2 s | 107.3 s | 110.9 s | 293.4 s |
| `.json` | 40.3 s | 116.7 s | 130.5 s | 318.9 s |
| `.csv` | 56.8 s | 111.0 s | 88.9 s | 290.4 s |
| `.txt` | 57.1 s | 109.8 s | 83.7 s | 262.5 s |
| `.tsv` | 57.6 s | 107.0 s | 88.5 s | 283.6 s |
| `.xml` | 26.6 s | 132.9 s | — | — |
| `.html` | 24.6 s | **refused** | — | — |
| `.md` | 226.0 s | write-only | — | — |
| `.yaml` | 1,619.2 s → 1.39 GB | **could not complete** | — | — |
| `.xlsx` | *refused above 65,535 rows — see below* | | | |
| `.ods` | *refused above 65,535 rows — see below* | | | |

**`.yaml` cannot be round-tripped at 1M rows on this machine.** It writes in **1,619 s (27
minutes)** producing **1.39 GB**, and then **exceeds 16 GB reading it back**. That ceiling is a
chosen bound rather than a property of the machine (64 GB installed), so what is measured is
*"exceeds 16 GB"* and not the point at which it would succeed. `yaml.safe_dump` over a million
records was already known to be 20+ minutes; the read side is the new part. **YAML has no cap and
no decision behind it yet** — unlike the spreadsheets below, there is no "it is a format for
reading" argument to bound it with.

> **Decided 2026-07-28: no cap, and the writer streams instead.** The write side was the half
> this server controls, and it is now bounded — `_write_yaml` dumps a chunk at a time, byte-for-byte
> what one dump would have written, holding **2.46 MB at 200,000 rows where it held 540 MB**
> (§9.2).
>
> **Re-run on this same corpus and budget** (`tmp/perf/probe_streamed_export.py`, 1M × 11,
> 100 MB budget), with `.csv` alongside as the control for what "streaming" looks like here:
>
> | | Rows | Seconds | File | Peak RSS over the step's baseline |
> |---|---|---|---|---|
> | `.csv` | 1,000,000 | 55.5 s | 1.21 GB | **+0 MB** |
> | `.yaml` | 1,000,000 | **237.6 s** | 1.39 GB | **+0 MB** |
>
> **The memory result is the clean one**: where the old writer held ~5.4 GB (§9.2) the new one adds
> nothing measurable over the baseline it started the step at, on the corpus that used to defeat it.
> **The 1,619 s → 238 s is not clean and must not be quoted as a 6.8x code speedup** — the earlier
> figure was drawn from a sweep in which steps paged (§10.7's opening), so the gap mixes the writer
> change with the conditions it was measured under. What can be said is that both the peak and the
> wall clock moved in the same direction and that YAML is still, by a wide margin, the slowest
> writer here: 237.6 s against `.csv`'s 55.5 s on the identical result.
>
> **A cap was considered and rejected.** The spreadsheet argument does not transfer: 65,535
> is the older worksheet's own limit and a spreadsheet is a thing a person opens, whereas YAML has
> no such number and is read by programs as often as by people. Inventing a bound for it would have
> been an arbitrary refusal dressed as a format fact.
>
> **What remains is the read side, and it is not a YAML defect.** `read_file` builds the whole
> pandas frame before inserting a row, so the 16 GB is the load path (task 21's other half, §10.6);
> YAML is simply the format that reaches it soonest, being the bulkiest on disk. Until that is
> closed, a very large YAML this server writes is one it may not read back — **recorded here as a
> documented cliff rather than papered over with a limit**. The 27 minutes is unchanged and is
> PyYAML serialising, not memory: YAML remains the slowest writer by an order of magnitude, and
> `.jsonl` is the format to ask for when the result is large and the shape is the same.

#### Spreadsheets are capped, and measured at the cap

`export.SPREADSHEET_ROW_LIMIT` **refuses more than 65,535 rows**, so timing a spreadsheet at a
million measures the refusal. Measured separately at the cap, same corpus and the same eleven
columns:

| | Rows | Seconds | File | Peak RSS added |
|---|---|---|---|---|
| `.xlsx` at the cap | 65,535 | **40.7 s** | 43.7 MB | **+0.3 MB** |
| `.ods` at the cap | 65,535 | **551.4 s** | 48.2 MB | +1,179 MB |
| `.xlsx` over the cap | refused | 0.88 s | none | **+0 MB** |

**The cap is what bounds the memory.** Uncapped, `.xlsx` on the same corpus added **4,832 MB** and
the run peaked at **12.9 GB**; at the cap it adds nothing measurable over the attach baseline. The
refusal costs 0.88 s and no memory at all — it is settled before the frame is built, so declining a
million rows does not pay what writing them would have, and it leaves no partial file.

**What the cap does not fix is `.ods` being slow.** 551 s for 65,535 rows is **13.5x `.xlsx`'s
40.7 s** on identical input — the same ratio measured at 50,000 rows, so it is a property of odfpy
rather than of scale. Nine minutes for a file a spreadsheet opens in seconds is the honest cost of
the format, and it is only visible now: before §10.1's defect was fixed, `.ods` was writing XLSX,
so every ODS timing this document ever carried was an XLSX timing. Uncapped, `.ods` crossed 16 GB
about six minutes into the export without producing a file at all.

> **Decided 2026-07-28: `.ods` stays, and the cost is told to the caller instead.** Being slow is
> not being wrong — the format is correct, the cap bounds its memory, and OpenDocument is what some
> people actually need. Dropping a working format because it is slow would remove real capability
> to save a wait the caller can decide about for themselves, and the comparison with `.xls` does
> not hold: `.xls` is read-only here because no maintained writer exists, not because a writer was
> judged too slow. So the 13.5x is now stated where it can be acted on — `query`'s own docstring
> and the shipped skill both say to reach for `.xlsx` unless OpenDocument was specifically wanted.
> This follows the standing shape of the server: it offers the primitive and reports the cost, and
> the judging is the caller's.

`.xlsx`, `.ods` and `.yaml` were three of the eight suffixes §9.2 listed as materialising every row
before writing any of it, measured at the scale where that stops being a footnote. **Two of the
three still are** — the workbooks, now bounded by the cap. `.yaml` streams as of 2026-07-28, and
§9.2's corrected table puts the boundary at eleven streaming suffixes against six materialising.

**A columnar format is a large win on writing and a small one on reading, and the gap between
those two is the result worth keeping.** Against CSV, on the same rows:

| | Wide | Tall |
|---|---|---|
| extract | **2.2–3.0x faster** | **1.5–1.6x faster** |
| read back | **1.40–1.48x faster** | **1.41x faster** |

The read-back figure reproduces on both corpora, which is what makes it worth trusting. It is also
the one that contradicts the obvious expectation: re-attaching Parquet does no text parsing at all
and still costs 71% of re-attaching the CSV of the same rows. **Read-back is dominated by inserting
rows into SQLite, not by decoding the file**, so choosing a columnar export buys much less on the
load path than on the write path.

The write advantage is a property of the *data*, not of the format: 3.0x on the wide corpus and
1.6x on the tall. Wide carries a 1000-character text column, and serialising long text to CSV is
exactly what a columnar writer avoids. Neither figure generalises without saying which corpus it
came from.

**HTML stops round-tripping at roughly 417,000 rows of 11 columns.** The export succeeds at 1M rows
(24.6 s) and the read back is refused: at about `2 x columns + 2` document nodes per row, 1M rows is
~24M nodes against libxml2's 10,000,000 ceiling. That is the improved message from `af95ce79`
working — the same condition used to report only `"unknown error"`.

> **Resolved 2026-07-28 by removing the format.** This measurement is what settled it: HTML was the
> only suffix in the catalogue that would write a file the server could not read back, so it was the
> only one breaking the round-trip property the reader/writer overlap is supposed to state. The
> writer also produced a bare `<table>` fragment rather than a document, leaving it worse than `.md`
> for a person and worse than `.csv` for a program. Capping the writer at the node ceiling was
> considered — the bound is computable, `10,000,000 / (2 x columns + 2)`, and would have scaled with
> column count rather than guessing a row count. **Both sides were dropped instead**, on Chris's
> call: a reader kept for inputs that are rarely table-shaped documents was earning its place by
> history rather than by use. lxml was its only dependency, and the `html` extra went with it. The
> catalogue is now **18 read, 15 written, 14 both**.

#### The budget bounds the slot, not the process

**Peak RSS reached 14.16 GB during the wide sweep, at a 100 MB budget, over a 1.22 GB source** —
within 1.8 GB of the run's 16 GB abort ceiling. `.xlsx` alone held **12.9 GB** while writing.

Nothing malfunctioned: the arm read `disk` throughout, so `relieve_memory` had spilled the slot
exactly as designed. **The DataFrame a materialising writer builds is not the slot**, and the budget
governs slot residency. This is §10.6's gap restated on the *write* side — §10.6 records that the
load peak tracks the file and no configuration bounds it; the same is true of the export peak.
Task 21 covers the read side. Whether the export path should be admitted against the budget is a
design question, not a defect in what the budget claims to do.

#### The two arms, and what spilling actually costs

The same wide corpus at a **4,000 MB** budget, where the slot stays resident, against the 100 MB
budget above, where it spills. Both are bounded — §5.2's rule is that a local slot is only
comparable with an endpoint database when it is on disk, not that the resident arm may go
unmeasured.

| Step | Disk arm (100 MB) | spread | Resident arm (4 GB) | spread | shown? |
|---|---|---|---|---|---|
| `attach` | 107.3 s | 1.23x | 80.5 s | 1.17x | 1.33x — **yes** |
| `SELECT count(*)` | 13.4 s | 1.35x | **0.54 s** | 1.14x | **24.9x — yes** |
| extract → csv | 56.8 s | 1.19x | 44.6 s | 1.03x | 1.27x — **yes** |
| read back csv | 111.0 s | 1.24x | 87.8 s | 1.05x | 1.26x — marginal |
| extract → parquet | 26.2 s | 1.15x | 23.0 s | 1.02x | 1.14x — **no** |
| read back parquet | 79.1 s | 1.33x | 73.3 s | 1.05x | 1.08x — **no** |
| `save` | 9.1 s | 1.31x | 7.6 s | 1.08x | 1.19x — no |

**Spilling costs a one-off spill and roughly a quarter on the operations after it** — and on the
Parquet steps the difference does not clear its own noise, so it is recorded as not shown rather
than as a small effect. The `count(*)` column is the spill itself, and 0.54 s is where §10.2's
1.15 s belongs: a resident slot answering a cheap aggregate.

**The more useful result is in the spread column.** On the resident arm the spreads collapse to
**1.02–1.17x**, against 1.15–1.35x on the disk arm, and only **3 of 27 steps paged** against **32 of
63**. The variance §10 has been carrying since session 35 — the ±25% band, the 1.50x within-condition
span, the "unattributed speedup" of session 39 — **is very largely the machine paging, not the code
varying.** Measure on an arm that does not page and the same operations reproduce to a few percent.

#### The conclusions of §10.3 and §10.4 survive re-measurement

Re-opening a saved database still skips the work rather than doing it faster: **107.3 s → 1.6 s**
(67x) for the wide file, **261.0 s → 0.5 s** (568x) for the tall. `create(type="table")` reading the
10M-row file cost **231.3 s** against `attach`'s 261.0 s — 0.89x, where §10.3 measured 1.11x. Both
sit either side of 1.0, which is the point: it is the same operation with the same shape.

The index still pays for itself on the first join:

| | s40 (median of 3) | spread | §10.4 |
|---|---|---|---|
| Join, no index | 31.4 s | 1.12x | 124.8 s |
| Build index on `tall.foreign_id` | 14.1 s | 1.24x | 47.4 s |
| Join, indexed | 2.2 s | 2.37x | 10.3 s |

14.1 + 2.2 = 16.3 s against 31.4 s, so the index wins by **1.93x counting the build** (§10.4 said
2.16x) and by **14.2x on every join after** (§10.4 said 12.1x). Every absolute number moved and
**both conclusions held**, which is the useful part: the ordering in §10 was never the fragile
thing, the absolute seconds were.

## §11 — ClickHouse, the first backend with no transactions (2026-07-28)

> **Removed on 2026-07-29 and restored the same day, under the eligibility rule.** It was dropped on
> the judgement that four driver defects and three single-user seam axes were not worth carrying.
> Chris then stated the rule that governs the catalogue — **a database is eligible if an open-source
> SQLAlchemy adapter exists for it, and not otherwise** — and ClickHouse plainly passes it:
> `clickhouse-connect` is ClickHouse's own, Apache-2.0, Production/Stable, committed to daily. The
> quality objection also does not survive contact with the mitigation: the silent-empty-string defect
> in §11.2 **cannot occur through this server's load path**, because `column_type` returns
> `Nullable` for every loaded column. And the "one user forever" claim about the three axes was
> weak — Greenplum requires a `DISTRIBUTED BY` clause on every `CREATE TABLE`, which is
> `table_options()` again, and several remaining columnar engines are unlikely to have reflectable
> indexes.
>
> Recorded rather than quietly reversed, because the reasoning that was wrong is worth seeing: an
> eligibility question was answered with a quality judgement, and the two are not the same test.

The first entry from the backend catalogue. ClickHouse was taken first because it removes the thing
every other backend here has: transactions. What that removes turns out to be the *floor* the
read-only guarantee stands on, and most of what follows is a consequence.

All 18 endpoint tests pass against `clickhouse/clickhouse-server:25.3`. Four of the findings below
are defects in the driver rather than facts about the database, and they are marked as such so that
nobody later "fixes" ClickHouse for them.

### 11.1 The live dialect is inside `clickhouse-connect`, not `clickhouse-sqlalchemy`

The obvious candidate is the wrong one. `clickhouse-sqlalchemy` carries a `4 - Beta` classifier, is
sdist-only, has **no GitHub releases at all**, and its last commit is 2025-11-24. Its release
ordering on PyPI is also inverted — 0.2.8 and 0.2.9 were published *after* 0.3.2.

`clickhouse-connect` is ClickHouse's own driver: `5 - Production/Stable`, v1.6.0 published five days
before this measurement, commits landing daily, and `sqlalchemy>=1.4.40,<3.0`. It registers the
dialect under the name **`clickhousedb`** (and `clickhousedb.connect`), speaks HTTP on 8123 rather
than the native protocol on 9000, and ships a `MIGRATING_FROM_CLICKHOUSE_SQLALCHEMY.md` in-tree —
upstream itself treats the third-party package as the thing to move away from.

This is the second time the dialect a search finds first was not the live one; §-note in the s42
handover records the same for Trino. **The registered dialect name is also the backend key**, so
`backend_for` looks up `clickhousedb`, not `clickhouse`.

### 11.2 A missing value silently becomes an empty string — in one code path, and raises in the other

A ClickHouse column is `NOT NULL` unless declared `Nullable`, and the portable Core types render as
aliases (`TEXT` → `String`, `INTEGER` → `Int32`, `DOUBLE` → `Float64`) that inherit that. Inserting
`None` into one behaves **two different ways depending on the batch size**:

| Insert | Result |
|---|---|
| One row, `{'name': None}` | **stored as `''`, reported as success**, `isNull()` = 0 |
| Two rows, one with `None` | `DataError: Invalid None value in non-Nullable column` |

The single-row case is the dangerous one and it is the fail-open shape this project keeps meeting: a
value that was absent comes back present, and nothing anywhere says so. A loaded file has gaps as a
matter of course, so this is not an edge case on the load path — it *is* the load path.

`column_type` therefore returns `Nullable(String)` / `Nullable(Int64)` / `Nullable(Float64)`. With
those, a missing value round-trips as missing and an aggregate skips it rather than counting an empty
string as a value. This is the fourth dialect to need a `column_type` override and the first to need
one for a reason that is about *nullability* rather than about width.

### 11.3 The driver's exceptions are not its DBAPI's exceptions, so SQLAlchemy never wraps them

`clickhouse_connect.dbapi` exports an `Error` as PEP 249 requires, but nothing the driver raises
inherits from it — `driver.exceptions.ClickHouseError` and `dbapi.Error` are unrelated class trees.
SQLAlchemy decides what to wrap by `isinstance(e, dialect.loaded_dbapi.Error)`, so a ClickHouse
failure is **not** a `SQLAlchemyError`, carries **no** `.orig`, and passes untouched through every
`except SQLAlchemyError` in this codebase.

Measured consequence: the read-only refusal reached the caller as
`DatabaseError: Received ClickHouse exception, code: 164 …` instead of the words naming the verb to
use instead. The statement was still refused — the guarantee held — but the agent was told only
"no", which is precisely the message it cannot act on.

The fix is a named set, `Backend.driver_errors()`, added to the guards that translate a driver
failure into words. **Not** a widening to `except Exception`: that would pull unrelated failures into
an explainer written for driver errors, which is the mistake recorded in the carried gotchas.

The error code itself is clean — the exception carries `code = 164` as an attribute, so
`denies_write` matches on the code and never on the sentence, which here contains both a version
string and a URL.

### 11.4 Two Core types the driver cannot bind

Neither is a ClickHouse limitation.

* **`LargeBinary`.** `clickhouse_connect.dbapi` does not define the `Binary` constructor PEP 249
  requires. SQLAlchemy's bind processor calls it and gets `AttributeError: module
  'clickhouse_connect.dbapi' has no attribute 'Binary'` before any statement is sent. ClickHouse
  stores binary perfectly well; `String` holds arbitrary bytes.
* **`Time`.** The column *is* created — `TIME` is a real type, stored as an integer number of
  seconds — but a Python `time` reaches the server as the bare literal `14:30:00` where an `Int64`
  was expected, and the insert fails to parse. A column that exists and refuses every value is worse
  than one that does not exist.

Both are recorded on the backend as `unstorable_column_types()`. That axis also absorbed Oracle's
"no time-of-day type", which had been a literal dialect-name branch in a test fixture — the one place
standing instruction 1 says a dialect fact may never be stated.

### 11.5 `readonly=1` is the whole read-only guarantee, because there is no floor beneath it

Every other backend either refuses a write or declines to keep it. ClickHouse does neither: with no
transactions there is nothing to leave uncommitted, and an `INSERT` sent through a read connection is
simply applied — measured, and the row was still there on the next connection.

`readonly=1` carried in the URL supplies the posture the database has no other way to hold. Measured
against the container: it refuses `INSERT` **and** `CREATE TABLE` with code 164, `SELECT` continues
to work, and the orphan table did not land. `readonly=2` behaves identically for these.

Two consequences worth stating plainly:

* **`ddl_survives_refusal()` is `False` here**, and ClickHouse is therefore *stronger* than Oracle on
  this axis despite having no transactions at all. The refusal happens before the statement runs, so
  there is no caveat to carry.
* **`read_only_query` now applies to a server URL, not only to a file.** `Backend.open` was building
  both engines from the same URL and leaving the posture to the transactional floor. That was
  adequate while every endpoint had one. A dialect that can be told read-only in the URL should be
  told so however it was reached, and here the URL is the *only* place the posture can be stated.

### 11.6 The index verb does not apply, and saying so is the answer

Three separate things fail, and only the first is about syntax:

1. `CREATE INDEX` without a `TYPE` is refused outright — code 80, `CREATE INDEX without TYPE is
   forbidden`.
2. What ClickHouse has instead is a **data-skipping** index
   (`ALTER TABLE … ADD INDEX … TYPE minmax GRANULARITY 1`, which does work). It prunes granules that
   cannot match. It is not a point lookup and offers no uniqueness — it does not answer the question
   a caller asking for an index is asking.
3. **The dialect reflects no indexes at all.** After creating one, `system.data_skipping_indices`
   shows `('probe4_ix', 'minmax', 'salary')` while `get_indexes()` and a reflected `Table.indexes`
   both return empty. So one created here could afterwards be neither listed by `info` nor found by
   `drop`.

Creating one anyway, under a name handed back to the caller, would produce an answer that reads as
done and cannot be acted on. `build_index` therefore raises `UnsupportedOperation` naming what
actually orders a ClickHouse table — the ordering key, chosen when the table is created — and
`builds_indexes()` is `False` so the endpoint test asserts the refusal rather than skipping the case.

### 11.7 What needed no override, which is the result that matters

The seam generalised. Against nine overridable axes, ClickHouse needed six answers, and the ones it
did **not** need are the point:

| Axis | ClickHouse |
|---|---|
| `read_posture` | **generic** — the URL carries it |
| `resident_bytes` | **generic** — `None`, a server holds nothing here |
| `snapshot` | **generic** — refused, nothing local to write |
| `storage_classes` | **generic** — real column types |
| `read_only_query` | `readonly=1` |
| `denies_write` | code 164 |
| `column_type` | `Nullable(…)` |
| `rename_table` | `RENAME TABLE` |
| `build_index` | refused |
| `table_options` | `MergeTree ORDER BY tuple()` |

`ALTER TABLE … RENAME TO` is not merely unsupported: ClickHouse parses `ALTER TABLE … RENAME` as the
start of `RENAME COLUMN` and fails at the `TO`, so the generic spelling produces a syntax error
naming a clause the caller never wrote. `RENAME TABLE` is the statement, and it keeps case through
the dialect's own preparer.

Two axes are new, and both were added because ClickHouse has something no previous backend had rather
than because it is awkward: `table_options()` (it has no default table engine, so the DDL does not
fail at the database — it fails at compile time with nothing created) and `driver_errors()` (11.3).

### 11.8 Not measured here

`ORDER BY tuple()` means a loaded table has **no ordering key**, which is the honest default — a file
has no natural key, and picking one would silently decide the physical layout and the primary index
on the caller's behalf. What that costs on a large table, and whether a caller should be offered the
choice, is unmeasured. The benchmark corpus in §10 has never been run against an endpoint.

Nor is the no-auth case covered: ClickHouse's default posture is the `default` user with an empty
password, and every endpoint here still uses one auth mode — username and password in the URL.

## §12 — CockroachDB, the dialect that needed nothing (2026-07-28)

Second entry from the backend catalogue, and the first from the wire-compatible tier. The reason
that tier is in the worklist at all is that its members are cheap *because* they are compatible —
one `endpoints.py` entry, the generic `Backend`, no subclass — and **if they pass unchanged that is
itself the result**, because it shows the seam generalises rather than having been fitted to the
engines it was built against.

CockroachDB passes unchanged. 17 of 18 endpoint tests green against `cockroachdb/cockroach:latest`,
the eighteenth skipped for a reason that is about the auth mode rather than the database.

### 12.1 No `Backend` subclass, and that is the finding

The question this endpoint was taken to answer was whether **"postgresql" names the dialect or the
engine** — whether the answers the seam gives for PostgreSQL are really about the wire protocol or
about that specific server. Every axis came back generic:

| Axis | CockroachDB |
|---|---|
| `read_posture` | generic — the transactional floor holds |
| `denies_write` | generic — nothing to recognise, the floor does not refuse |
| `ddl_survives_refusal` | generic `False` — DDL is transactional, so a read connection's `CREATE` rolls back |
| `column_type` | generic — the portable types render correctly |
| `rename_table` | generic — `ALTER TABLE … RENAME TO` is accepted verbatim |
| `build_index` | generic — Core's `Index` is enough, and it reflects |
| `resident_bytes` / `snapshot` / `storage_classes` | generic |

So `BACKENDS` gains no entry, `backend_for("cockroachdb")` returns a plain `Backend` carrying its own
name, and every verb works. This is the second dialect after PostgreSQL to need nothing at all, and
it is a stronger result than PostgreSQL's: PostgreSQL needing nothing could mean the generic answers
*are* PostgreSQL's answers. A different engine on the same wire needing nothing means they are not.

**It is addressed as `cockroachdb+psycopg`, not as `postgresql+psycopg`.** psycopg reaches it either
way, but the scheme decides which dialect SQLAlchemy loads and therefore which backend answers —
the same reason MariaDB is registered in its own right rather than aliased to MySQL. Addressing it
as PostgreSQL would have tested PostgreSQL's answers against CockroachDB's behaviour, which is the
one thing this endpoint exists not to do.

The dialect is `sqlalchemy-cockroachdb` 2.0.4, maintained by Cockroach Labs, requiring
`SQLAlchemy>=2.0.47,<2.1`. It registers four entry points; `cockroachdb.psycopg` rides the psycopg 3
driver the PostgreSQL endpoint already carries, so the extra is the dialect and nothing else.

### 12.2 The first endpoint reached with no password, and two tests assumed there would be one

The container runs `start-single-node --insecure`, so `root` connects with **no password at all**.
That is a genuinely different auth mode from the five that came before — all of which embed a
username and password in the URL — and it broke two tests that had quietly assumed otherwise:

* **`test_an_endpoint_attaches_as_an_engine_and_keeps_its_password`** asserted `"***" in source`.
  With no credential there is nothing to redact, and rendering a `***` for an absent password would
  tell the caller a secret was carried when none was. It now asserts the redaction where there is a
  password and its *absence* where there is not.
* **`test_a_failed_open_does_not_echo_the_password`** sets a deliberately wrong password and expects
  the open to fail. **Insecure mode accepts any password for `root`**, so the open succeeded and the
  test failed on its own premise. It now skips, naming that.

Neither was a defect in the server; both were the harness generalising from five endpoints that
happened to share an auth mode. This is the coverage gap task 23 exists for, meeting the code from
the other direction — and it is worth noting that adding a *backend* is what surfaced it, not adding
an auth test.

`_password()` now returns `str | None` rather than `str(...)`. It had been stringifying an absent
password into the literal `"None"`, which was then searched for in the payload — an assertion that
would have passed for the wrong reason.

`URL.create(password=None)` omits the `:` entirely rather than rendering an empty one, which is the
form a trust-authenticated server expects. That distinction is why the harness helper takes
`password: str | None` rather than defaulting to `""`.

### 12.3 What this did not test

CockroachDB's interesting properties — serializable isolation by default, retryable transaction
errors under contention, distribution across nodes — are invisible to a single-node harness running
one statement at a time. Nothing here says how the server behaves when CockroachDB returns a
retryable error (`40001`), which is the failure mode a real deployment meets and which no other
backend in this harness produces. That is unmeasured, and it is the one thing about this dialect
worth measuring later.

## §13 — TiDB, and the assumption underneath `BACKENDS` (2026-07-28)

Third from the backend catalogue, and the first that was **backed out rather than taken**. Nothing
about TiDB is committed: no compose service, no URL builder, no `ENDPOINTS` entry. What follows is
measured, and it is recorded because the measurement is about the *seam* rather than about TiDB.

The worklist predicted this one would need no subclass and that the absence would be the finding.
The opposite happened, and it is a better finding.

### 13.1 TiDB refuses the statement MySQL's read-only posture is built on

TiDB has no SQLAlchemy dialect of its own — PingCAP's documentation says to connect with
`mysql+pymysql`, and there is no `sqlalchemy-tidb` on PyPI. So `backend_for("mysql")` returns
`MySQLBackend`, which opens a read-only session on every read connection. TiDB will not:

```
pymysql.err.NotSupportedError: SET SESSION TRANSACTION READ ONLY has only noop
implementation in tidb now, use tidb_enable_noop_functions to enable these functions
```

The statement runs in a `connect` event listener, so this is not a degraded read posture — **every**
connection fails and `attach` fails outright. 16 of 18 endpoint tests never get past the fixture.

**The workaround the error suggests fails open, and must not be used.** `tidb_enable_noop_functions`
does not implement the statement; it makes it a *no-op*. Enabling it buys a successful attach and a
read-only posture that silently does nothing — the exact shape §11.2, §5 and the memory-admission
work all record this project being bitten by. A guarantee that reports itself as installed and
enforces nothing is worse than one that refuses.

### 13.2 The shape TiDB wants already exists, and cannot be reached

Measured directly against `pingcap/tidb:latest` (v7.5.1):

| | TiDB |
|---|---|
| DDL survives an uncommitted transaction | **yes** |
| DML rows surviving an uncommitted insert | **0** |

So TiDB wants precisely **Oracle's shape**: keep the generic transactional floor, which holds for
DML, and answer `ddl_survives_refusal() == True`, because a `CREATE` through `query` really does
land and the refusal has to say so.

That shape is already in the seam. It is simply unreachable — MySQL and TiDB both resolve to
`MySQLBackend`, which installs a posture TiDB rejects and answers `ddl_survives_refusal() == False`,
the opposite of the truth.

### 13.3 The assumption, stated plainly

`BACKENDS` is keyed by SQLAlchemy dialect name and `backend_for()` resolves on it. That encodes:

> a dialect name identifies the engine on the other end

**It does not.** A dialect names a *wire protocol and a driver*, and several engines answer on each.
This is the production-code twin of the harness defect fixed at `d7cb14d8`, where the probe cache
and the pytest ids made the same wrong assumption and would have run one container's suite under
another container's name.

It is a tier of the worklist rather than one database. TiDB and OceanBase inherit `MySQLBackend`,
which carries real engine-specific behaviour, and that is where it breaks. YugabyteDB, Greenplum and
OpenGauss inherit the *generic* `Backend` and are probably unaffected — which is not a guess but the
CockroachDB result from §12 read forward: a different engine on PostgreSQL's wire needed nothing,
because PostgreSQL itself needs nothing.

The decision this needs — whether to resolve a backend by asking the server what it is, which
requires a connection *before* the backend is chosen and so inverts the current
`backend_for` → `open` order — is issue #45 and task 24. It was not made here, because making it
silently mid-worklist is how an architecture drifts.

One thing worth doing whichever way that goes: a read posture that cannot be installed should reach
the caller as "this datasource cannot be opened read-only" rather than as the driver's own sentence
about noop functions.

### 13.4 An operational limit, found the hard way

Running eight containers at once had CockroachDB **killed** by the Docker VM:

```
WARNING: disk slowness detected: unable to sync log files within 10s
/cockroach/cockroach.sh: line 265: 59 Killed "${start_node_query[@]}"
```

The endpoint suite went from 1m25s to 6m02s with ten failures and ten errors, none of which were
code. The catalogue is approaching what this machine holds concurrently, and a worklist with sixteen
entries will not fit at all — containers will need bringing up per-dialect rather than all at once.
Recorded because a killed container looks exactly like a broken commit until the logs are read.

## §14 — The MCP 2026-07-28 specification, measured against this server (2026-07-29)

The fifth MCP spec release landed on 2026-07-28: a **stateless protocol core**, Multi Round-Trip
Requests, header-based routing, cacheable list results, hardened OAuth 2.1, and a formal extensions
framework (Tasks, MCP Apps) with a twelve-month deprecation window. `initialize`/`initialized` and
the `Mcp-Session-Id` header are retired.

**This server needs no architectural change, and that is not luck.** It was measured, not assumed.

### 14.1 The handle-based design is already what the spec prescribes

The specification's own guidance for a server that must carry state:

> Dropping the protocol-level session doesn't force your application to be stateless. If your server
> needs to carry state across calls, mint an explicit handle from a tool and have the model pass it
> back as an argument.

That is exactly `attach` → **nickname**. Every verb takes the nickname as an argument, and `attach`
returns the one it actually used. Checked against the code rather than remembered:
`server._registry` is a **module-level global**, one per *process*, and `_session()` is only a name
for "the process's registry" — it is not keyed to any protocol session identifier. Nothing anywhere
reads `Mcp-Session-Id`.

The server also uses **no `Context` parameter**, so it uses none of sampling, MCP-level logging or
progress — the capabilities the stateless core removes because they push a request down a live
connection. (`Config(roots=...)` is our own filesystem allowlist and has nothing to do with protocol
Roots.)

### 14.2 Run against the new stack: 113 of 114, and the one failure is ours

`fastmcp 4.0.0b1` (published the same day as the spec, on `mcp 2.0.0`) implements it and, in its own
words, "answers both the sessionless `2026-07-28` protocol and the older session-based handshake,
negotiated per connection". The tool-surface suite was run against it directly:

| | Result |
|---|---|
| `tests/test_server.py` on fastmcp 3.2.0 / mcp 1.27.0 | 114 passed |
| `tests/test_server.py` on fastmcp 4.0.0b1 / mcp 2.0.0 | **113 passed, 1 failed** |

**No source change was needed for either.** Both problems are in the tests:

1. `Tool.inputSchema` is renamed `input_schema` — a deprecation warning, at `test_server.py:104`.
2. `test_the_tool_descriptions_name_exactly_the_formats_that_exist` fails, and the reason is worth
   stating precisely because it looks alarming and is not.

### 14.3 The `Args:` block moves out of the description and into the schema

Measured, for `attach`:

| | `description` length | carries the format list |
|---|---|---|
| fastmcp 3.2.0 | 1832 chars | yes |
| fastmcp 4.0.0b1 | **555 chars** | no |

fastmcp 4 stops folding the docstring's `Args:` section into the tool description and puts each
parameter's prose into the **input schema's per-property `description`** instead. Confirmed by
reading the schema back: `database` still carries the full reader catalogue, `nickname`, `writable`
and `delimiter` each carry their own text.

**Nothing is lost, and the placement is better** — a client can render per-parameter help, and the
format catalogue reaches the agent attached to the parameter it governs rather than buried in one
long string. What breaks is only our test, which greps the description for the list. Under fastmcp 4
it must grep the parameter schema.

This matters more here than it would elsewhere: the format catalogue *is* the tool description for
`attach` and `query`, and the test exists so a format cannot land without the agent being told. It
must keep doing that job against whichever field carries it.

### 14.4 What is deliberately not being done yet

`fastmcp 4.0.0b1` is a beta. The pin stays `fastmcp>=3.0.0` and is deliberately **not** capped: the
server genuinely runs on both majors, so capping would refuse users a working combination to protect
a test. Adoption waits for a stable 4.x — task 25 names the two test changes it needs.

Three parts of the new spec are worth a second look then, none urgent:

* **Cacheable list results** (`ttlMs`, `cacheScope`). This tool list is static for the life of the
  process, so it can advertise a long TTL — a small, free win, and fastmcp's to implement.
* **MRTR** (`resultType: "input_required"`) would let a tool ask mid-call instead of refusing. It is
  tempting for the ambiguous-source case, where a file with two candidate tables is currently refused
  naming both. **It should be resisted by default**: standing instruction 4 says the server offers
  primitives and the LLM does the judging, and a refusal that names both candidates already gives the
  caller everything it needs to choose.
* **The stateless core bounds where this server could ever be deployed**, not how it behaves today. A
  slot is memory or a temp file in *this* process, so several instances behind a load balancer would
  not share slots. That is fine for a local stdio server, which is what this is, and it is the reason
  hosting was dropped as a concern rather than a gap to close.

## §15 — YugabyteDB, and the retryable error §12.3 asked for (2026-07-29)

Third entry from the backend catalogue, and the first to **fail** against the seam rather than pass
through it. It also answers the question §12.3 left open — what this server does when a distributed
engine returns a retryable `40001` — which arrived here rather than from CockroachDB because
YugabyteDB raises it for an ordinary rename, not only under contention.

`yugabytedb/yugabyte:2.25.2.0-b359`, `yugabyted start --background=false`, YSQL on 5433. 18 of 18
endpoint tests green — but only after a defect that two of them found.

### 15.1 The rename succeeded and the server said it had failed

`update(type='table', name=…, to=…)` renamed a table and then reported an error:

```
(psycopg.errors.SerializationFailure) The catalog snapshot used for this transaction has been
invalidated: expected: 42, got: 41: MISMATCHED_SCHEMA
CONTEXT:  Catalog Version Mismatch: A DDL occurred while processing this query. Try again.
```

The failing statement is not the `ALTER TABLE`. It is the `pg_catalog` reflection **afterwards**:

```sql
SELECT pg_catalog.pg_class.relname FROM pg_catalog.pg_class JOIN pg_catalog.pg_namespace …
```

YugabyteDB caches the catalog per connection. A DDL committed on the write engine's connection
leaves every *other* pooled connection — here the read engine's, which `table_names` uses — holding
a snapshot the cluster has moved past, and its next catalog read is refused. Isolated directly:

| Step | Result |
|---|---|
| `CREATE TABLE yb_a`, insert 2 rows | ok |
| reflect on the read engine (warms its snapshot) | sees `yb_a` |
| `ALTER TABLE yb_a RENAME TO yb_b` on the write engine | **committed** |
| reflect again on the read engine | `SerializationFailure`, SQLSTATE `40001` |
| reflect once more, no delay | `['yb_b']` |
| `SELECT count(*) FROM yb_b` | `2` |

**The rename had happened.** The rows were under the new name and the caller was told the operation
failed — the worst answer available, and worse than the raw error, because the obvious next move is
to rename again and be told there is no such table. Being refused is itself what refreshes the
snapshot, so the immediately following read succeeds with no sleep: this is a stale snapshot being
replaced, not contention being waited out, which is why one retry is the whole remedy and a backoff
would add nothing.

### 15.2 The fix is generic, and deliberately not a `Backend` axis

`loader._run_again_once` runs a catalog read again, once, when the driver reports SQLSTATE `40001`.
It is **not** a seam axis, for a reason worth stating because every prior backend finding became one:

* **`40001` is standard.** `serialization_failure` means "this transaction was aborted, run it
  again" in every engine that raises it — PostgreSQL under SERIALIZABLE, CockroachDB, and every
  distributed SQL engine routinely. There is no per-dialect answer to override, so a dialect branch
  would be the shape standing instruction 1 forbids.
* **There is nowhere to put one anyway.** YugabyteDB is reached through PostgreSQL's dialect, so an
  entry in `BACKENDS` keyed `postgresql` would change *PostgreSQL's* behaviour to serve YugabyteDB.
  That is issue #45 biting for real rather than in principle — see §15.4.

Scope is deliberately narrow. **Once**, because a second stale snapshot is a real failure rather
than a slow one, and the test asserts the read is called exactly twice so it cannot become a loop.
**Reads only**: a catalog read is idempotent, so running it again carries no consequence, whereas
whether a failed *write* is safe to send again depends on what it was — the caller's judgement, not
this server's. The code is read from `sqlstate` (psycopg 3) or `pgcode` (psycopg 2); a driver
publishing neither simply does not match, and the caller then gets the error it would have got.

Four tests in `test_loader.py` pin the decision without needing a container, and all four were
proved able to fail — two by removing the retry, two by making the classifier always retry.

### 15.3 Addressed as PostgreSQL, and why that is not the pattern break it looks like

The other two PostgreSQL-wire endpoints are addressed as themselves. This one is not, and the reason
is the adapter rather than the database.

YugabyteDB **is eligible**: `sqlalchemy-yugabytedb` 1.0.0.1 exists and is Apache-2.0, Yugabyte's own.
What it cannot do is be reached from here. It registers **psycopg2 entry points only** and
hard-requires `psycopg2-yugabytedb`, a fork of psycopg2 pinned at 2.9.3 publishing wheels for
**macOS arm64 and nothing else** — every Linux and Windows user compiles it against libpq. Adopting
it would put a second PostgreSQL driver family in this project for one database, beside the psycopg 3
the `postgres` extra already carries.

Against that cost, the dialect is 81 lines and adds nothing the seam asks about: it narrows the
isolation-level lookup, and overrides `initialize` with a call to `super(PGDialect, self)` — which
*skips* PGDialect's own initialisation rather than extending it. Plain `postgresql+psycopg` connects
and reads the version correctly: `PostgreSQL 15.12-YB-2.25.2.0-b0` parses to `(15, 12)`, where
CockroachDB's `CockroachDB CCL v26.2.4 …` could not be parsed at all, which is why *that* endpoint
genuinely needs a dialect of its own.

**The driver's distribution is a quality judgement and it decides only the addressing, never the
eligibility.** Those are different tests, and answering one with the other is exactly the mistake
§11 records ClickHouse being removed and restored over. YugabyteDB is in the catalogue; only its
scheme is PostgreSQL's.

So it costs **no new dependency at all** — it reuses the `postgres` extra, and `pyproject.toml` is
untouched. That makes it the cheapest entry the catalogue has taken and the most expensive to
reason about, which are not the same axis.

### 15.4 Issue #45 now blocks two items, not one

The standing assessment was that `BACKENDS` being keyed by dialect name blocks **OceanBase alone**,
and that the PostgreSQL-wire entries were "almost certainly unaffected" because CockroachDB needed
nothing. **That was true only for as long as a PostgreSQL-wire engine needed nothing.** YugabyteDB
needs something, and the two harms differ in kind:

| Item | Reached as | Harm |
|---|---|---|
| OceanBase | `mysql` | inherits `MySQLBackend`'s **overrides** — wrong behaviour |
| YugabyteDB | `postgresql` | `Backend(name="postgresql")`, so a refusal names the **wrong database** to the caller, and any answer of its own would have to change PostgreSQL's |

The name is user-facing: `Backend.name` is what "A {name} datasource is reached over its own
connection…" prints, so someone who opened YugabyteDB is told about PostgreSQL. That is mild next to
OceanBase's, and it is the same root — a dialect names a wire protocol and a driver, never an engine.

Here it was dodged rather than solved, because `40001` genuinely is generic and belonged in the
generic path regardless. **The next PostgreSQL-wire engine needing something that is not generic has
no such escape**, and Greenplum is on the worklist already needing `table_options()` for its
`DISTRIBUTED BY` clause. Issue #45 is updated with this.

### 15.5 The healthcheck that lied, and what it cost

`yugabyted` binds YSQL to the address it advertises — the container's own interface — so a
healthcheck probing `localhost` is refused while the database answers perfectly well on the
published port. The container sat `unhealthy` for six minutes with a fully working database behind
it. The compose entry uses `$(hostname)`, and this is the same fail-open shape `endpoints.py` was
written to avoid: **a harness looking in the wrong place reports absence, not error.** The remedy
was the standing one — read `docker logs` before believing a red container.

### 15.6 What this did not test

Single node, so YugabyteDB's distribution, its sharding and its cross-node latency are all invisible
here, exactly as §12.3 says of CockroachDB. The `40001` measured is the *catalog* form; the
contention form — two transactions genuinely conflicting — is still unmeasured, and it is the one a
write would raise, which is precisely the case `_run_again_once` deliberately declines to retry.

## §16 — Trino, the backend that owns no data (2026-07-29)

The fifth entry from the backend catalogue, and the first that is not a database. Trino is a *query
engine*: it holds no storage of its own and reads everything through a **catalog**, a configured
connector onto some other system. Several of the seam's questions therefore have answers that are
about Trino's position in the stack rather than about a feature it lacks — it has no indexes because
it has nothing to index, not because indexing was left out.

All 18 endpoint tests pass against `trinodb/trino:476`, addressing the `memory` catalog. Five of the
findings below needed an answer; two of those five are defects in the client library rather than
facts about Trino, and they are marked as such so that nobody later "fixes" Trino for them.

### 16.1 The live dialect is inside `trino`, and the standalone package has become a shim

The same shape as ClickHouse's §11.1, and this time the evidence is unusually direct. The obvious
candidate `sqlalchemy-trino` last shipped **0.5.0 on 2022-05-05** — and its own metadata now
declares `trino[sqlalchemy] (>=0.310)` as a dependency. It does not compete with the official
dialect; it *requires* it. Installing it would add a package to pull in the one already chosen.

`trino` is Trino's own client: **0.338.0, uploaded 2026-06-29**, Apache-2.0, and the dialect lives at
`trino.sqlalchemy`, registered under the name `trino`. Its `Development Status` is still `4 - Beta`,
which is a cost to record rather than grounds to exclude — the eligibility rule §11 settled asks only
whether an open-source SQLAlchemy adapter exists, and this is the engine's own.

### 16.2 The driver connects in autocommit, which deletes the floor rather than weakening it

The generic read-only guarantee is transactional: `query` opens a connection, never commits, closes
it, and whatever the statement changed is rolled back. **The Trino client's default isolation level
is `AUTOCOMMIT`**, so every statement commits itself as it runs. Measured: an `INSERT` sent through
a read connection that never commits was still there on the next connection.

| Read engine | `SELECT` | `INSERT` | `CREATE TABLE` |
|---|---|---|---|
| default (`AUTOCOMMIT`) | served | **applied and kept** | **applied and kept** |
| `SERIALIZABLE` | served | refused, `AUTOCOMMIT_WRITE_CONFLICT` | refused, `AUTOCOMMIT_WRITE_CONFLICT` |

This is ClickHouse's §11 situation reached from the opposite direction. There the database has no
transactions at all and the posture had to come from the server (`readonly=1`); here the database has
them perfectly well and the *driver* declines to use them. Naming any real isolation level on the
read engine is the whole remedy, and `TrinoBackend.read_posture` is where it is named.

What happens after that is the **catalog's** business rather than this server's, and both outcomes
are safe:

* A catalog that writes transactionally accepts the statement and has it rolled back on close — the
  generic floor, working exactly as designed.
* A catalog that writes only in autocommit refuses it outright with `AUTOCOMMIT_WRITE_CONFLICT`
  (*"Catalog only supports writes using autocommit: memory"*), which `denies_write` recognises so the
  refusal names `create` instead of quoting the server. `memory`, which this harness uses, is one.

Reflection, column inspection and ordinary reads were all re-checked on a clean non-autocommit
connection and all work. What does *not* survive is a connection that has already had a statement
refused: everything after it fails with `TRANSACTION_ALREADY_ABORTED` until the connection is closed.
That costs nothing here, because `query` closes its connection either way.

### 16.3 Only one isolation level survives the round trip — a client defect

`SERIALIZABLE` is not a strictness decision. It is the only level that can be reached at all:

| Asked for | Result |
|---|---|
| `SERIALIZABLE` | accepted |
| `READ UNCOMMITTED` | `KeyError: 'READ UNCOMMITTED'` at connect |
| `READ COMMITTED` | `KeyError: 'READ COMMITTED'` at connect |
| `REPEATABLE READ` | `KeyError: 'REPEATABLE READ'` at connect |

SQLAlchemy normalises an isolation level to **spaces**; the dialect looks it up in an enum keyed with
**underscores** (`IsolationLevel.READ_UNCOMMITTED`). Every level whose name is two words therefore
raises at connect time, and the one-word name is the only one that matches by accident. Nothing here
wants a stricter snapshot — only a transaction — so if this is ever fixed upstream the weakest level
becomes reachable and is the better choice.

`isolation_level` is also **ignored as a URL query parameter** (the connection still reports
`AUTOCOMMIT`), which is why the posture is set on the engine rather than carried in the URL the way
`read_only_query` carries ClickHouse's and DuckDB's.

### 16.4 Trino folds every identifier, and quoting does not stop it

Every other dialect here keeps a *quoted* identifier verbatim — that is precisely why
`rename_table` puts both names through the dialect's own preparer. Trino folds anyway, at the
connector rather than in the parser:

| Asked for | Stored as | Resolves as |
|---|---|---|
| `CREATE TABLE "probe_Mixed"` | `probe_mixed` | `"probe_Mixed"`, `"probe_mixed"` and `probe_Mixed` all work |
| column `"Dept"` | `dept` | — |

Nothing breaks: both spellings still find the table. What breaks is the *answer*. `update` reported
the table as `Mixed` while `info`, in the very next payload, listed it as `mixed` — one response
naming a table the other says is not there.

**The fix is generic and needed no dialect fact at all.** `Workspace.landed_as` asks the database
what the table ended up called and returns that; `rename_table` and `insert_frame` both report it,
and the cached description is keyed under it. A name is observable, so it is observed — a table of
which backends fold would be a dispatch on dialect name, and one that went stale would fail silently.
On every non-folding backend the first line matches and the answer is unchanged.

`Backend.folds_identifiers()` exists **only so a test can tell the two outcomes apart**, and it is in
the seam rather than in the fixture for the reason standing instruction 1 gives. Asserting merely
that the reported name is findable would let a genuine folding regression through on PostgreSQL;
asserting case-insensitively would let all of them through.

### 16.5 No indexes whatsoever, which is not ClickHouse's answer

ClickHouse has an index of a *different kind* — data-skipping, unreflectable, answering a different
question. Trino has none at all, and the reason is structural: it stores nothing, so there is nothing
of its own to index. A filter is made fast by being pushed down into the catalog, where the
underlying system's own layout decides what it costs.

So `builds_indexes()` is `False` and `build_index` refuses, naming where indexing actually lives —
the system behind the catalog. This is the second user of both axes, which retires the "one user
forever" objection §11 recorded against them.

The endpoint test asserted ClickHouse's own wording (`"ordering key"`), which is a dialect fact
stated in a fixture and would have had to grow a branch per backend. It now asserts the property that
is actually required of any such refusal — **that it names the database which declined** — and lets
each backend supply its own words.

### 16.6 `bytes` cannot be bound — a client defect, not a Trino one

Trino has `VARBINARY` and stores binary perfectly well. `trino.dbapi` does not: its literal formatter
calls `.encode` on the value it was handed, which is what one does to a `str`, so a `bytes` raises
`AttributeError: 'bytes' object has no attribute 'encode'` before any statement is sent. The column
is created and cannot be written to, which is worse than not having it — the same shape as
ClickHouse's missing PEP 249 `Binary` constructor (§11.4), and recorded the same way.

Everything else in the typed-value round trip works, including `Time`, which both Oracle and
ClickHouse could not hold: `Numeric`, `Date`, `DateTime`, `Time` and `Boolean` all came back as the
values that went in.

### 16.7 The compose entry needed no catalog file after all

Recorded because the groundwork said otherwise. The expectation was that `trinodb/trino:476` ships
`tpch` and `jmx` but not `memory`, so the compose service would have to write
`/etc/trino/catalog/memory.properties` from an `entrypoint:` before starting. **The stock image
already ships all four** — `memory`, `tpch`, `tpcds` and `jmx` — so the service is an image, a port
and a healthcheck, with nothing written into it.

It also ships its own `/usr/lib/trino/bin/health-check`, which is used verbatim. It asks `/v1/info`
and — the part a naive probe misses — insists on `"starting": false`. Trino answers HTTP long before
it will accept a query, so a probe checking only the port reports ready while every statement is
still refused with `SERVER_STARTING_UP`. That is the §15.5 fail-open shape again, and this time the
image had already solved it.

No credentials of any kind: with no authenticator configured Trino accepts whatever username the
client offers and asks for no password. That makes it the **third** endpoint here reached with no
password, after CockroachDB's `--insecure` and YugabyteDB's trust — so task 23's first auth mode is
now covered three times over and the remaining ones are still untouched.

### 16.8 What this did not test

One coordinator, one catalog, and that catalog the simplest one there is. Trino's actual subject —
federating a query across several catalogs at once — is invisible here, and so is everything about
distribution: no workers, no split scheduling, no cross-node exchange. The `memory` connector also
happens to be the one that refuses transactional writes, so §16.2's *other* branch — a catalog that
accepts the write and has it rolled back — is reasoned from the transaction semantics rather than
measured. Measuring it needs a catalog with a real system behind it, which is a much larger fixture
than one container.

## §17 — A dialect names a wire protocol, never an engine (2026-07-29)

Issue #45, opened when TiDB could not be attached at all, closed here. The assumption it named:

> a dialect name identifies the engine on the other end

It does not, and the catalogue is where that stops being academic. Five of sixteen entries answer on
a dialect another engine wrote:

| Engine | Reached as | Was handed |
|---|---|---|
| TiDB | `mysql` | `MySQLBackend` — a posture it rejects outright |
| OceanBase | `mysql` | `MySQLBackend` — overrides written for MySQL |
| YugabyteDB | `postgresql` | `Backend(name="postgresql")` — **the wrong name, to the caller** |
| Greenplum | `postgresql` | same, and no way to answer `DISTRIBUTED BY` without changing PostgreSQL's |
| OpenGauss | `postgresql` | same |

§15.4 recorded YugabyteDB dodging this rather than solving it — its finding (`40001` wants a retry)
turned out to be generic and belonged in the generic path anyway. **That was luck.** Greenplum has no
such escape, and it is the next entry on the worklist.

### 17.1 The resolution, and where it does not happen

`backend_for_url` asks the server what it is, and asks **only where the question can have a second
answer**. A backend declares `impostors` — banner fragments mapped to the name each engine should be
known by — and a dialect with none returns immediately, connecting to nothing. Eight of the nine
endpoints in the harness take that path and are bit-for-bit unchanged.

Where a probe does happen it costs one short-lived connection on a path that is about to open two
engines and reflect a table list, so it is not a round trip the caller would otherwise have avoided.

`Backend.named_by(banner)` — the matching — is pure and separate from reading the banner, because
which engine a version string names is the part worth pinning and it needs no database to pin.

**Every failure resolves to the dialect's own backend.** A refused `SELECT version()`, a permission
the credentials lack, a driver raising something unrelated: none is a reason to refuse a datasource
that would otherwise open, and all land on exactly the behaviour that preceded this. That is the one
place in this codebase where a bare `except Exception` is the correct width — the question is
optional, so nothing it can raise may propagate. It is the opposite of the fail-open shape recorded
elsewhere: nothing is *guessed*, the answer simply stays what it already was.

### 17.2 The fragment is measured, and tighter than the obvious one

Read from the live containers:

| Engine | Banner |
|---|---|
| PostgreSQL 16 | `PostgreSQL 16.14 on x86_64-pc-linux-musl, compiled by gcc (Alpine 15.2.0)…` |
| YugabyteDB | `PostgreSQL 15.12-YB-2.25.2.0-b0 on x86_64-pc-linux-gnu, compiled by clang version 19.1.0 (https://github.com/yugabyte/llvm-project.git …)` |

The fragment is **`-YB-`**, not `yugabyte`. Both match — YugabyteDB's banner says `yugabyte` in the
compiler's source URL — but the *version* is what identifies the engine, and a fragment leaning on a
build detail is one waiting to stop matching. `-YB-` is the part real PostgreSQL can never carry.

A guessed fragment fails in both directions: too loose and the real engine matches its own impostor,
too tight and nothing does. So **Greenplum and OpenGauss are deliberately absent** — neither has a
container, so neither has a measured banner, and an entry taken from documentation is precisely what
the table exists to prevent. TiDB and OceanBase are absent for the same reason; TiDB is out of the
catalogue anyway under the §13 eligibility rule, since no SQLAlchemy adapter for it exists.

### 17.3 What was actually wrong, and how it passed for a whole session

YugabyteDB shipped in §15 with all 18 endpoint tests green while carrying PostgreSQL's name. Nothing
was wrong with the tests except what they did not ask: **no assertion anywhere named the backend the
server had chosen.** The endpoint table now states it — `Endpoint.engine`, `None` where the dialect
and the engine agree — and one test compares it against what `backend_for_url` resolves. Proved able
to fail: emptying `impostors` reddens `[yugabytedb]` with `postgresql`, and leaves the other four
endpoints in that batch passing.

The test helpers were resolving by dialect too, so they were consulting a *different* backend than
the code under test used — harmless while the two agreed and exactly the shape of #44. They now go
through `backend_for_url` like the server does.

`PostgreSQLBackend` exists solely to hold that table. Every answer in it is the generic one, and five
sessions of endpoint work have not turned up anything PostgreSQL needs said for it — it is registered
for the *other* reason a dialect earns an entry: three engines borrow it.

## §18 — MonetDB, a column store that needed nothing but cost a version ceiling (2026-07-29)

Tenth endpoint dialect, and the seventh entry from the backend catalogue (task 22, worklist item 7).
MonetDB is a **column store** — the first storage model in this harness that is neither a row store
nor, like Trino, an absence of storage — and the question it was taken to answer is whether the seam's
generic answers are about SQL or about how a database keeps its bytes.

They are about SQL. **All 19 endpoint tests pass unchanged, `BACKENDS` gains no entry, and no test
assertion needed generalising.** Measured against `monetdb/monetdb:latest`, server version
**11.55.7**, through `sqlalchemy-monetdb` 2.0.0 on `pymonetdb` 1.9.1.

### 18.1 The adapter cannot be imported without `setuptools`, and says so nowhere

This is the cost, and it is a defect in the dialect rather than in the database — recorded so nobody
later "fixes" MonetDB for it. Issue #50.

`sqlalchemy_monetdb/__init__.py` imports `pkg_resources` at package import, unconditionally, **only
to read its own version string**:

```python
import pkg_resources
try:
    __version__ = pkg_resources.require("sqlalchemy-monetdb")[0].version
except pkg_resources.DistributionNotFound:
    ...
```

`pkg_resources` is supplied by `setuptools`, which the distribution does not declare — its
`requires_dist` is `pymonetdb>=1.8.2`, `sqlalchemy>=2.0.34` and test extras, nothing more. A modern
environment carries no `setuptools` unless something asks for it, so on a clean install **every one of
the 19 endpoint tests errors at setup** with `ModuleNotFoundError: No module named 'pkg_resources'`.
There is no way around it from here: SQLAlchemy loads the dialect through the entry point
`monetdb -> sqlalchemy_monetdb.dialect:MonetDialect`, which runs the package `__init__`.

So the `monetdb` extra carries `setuptools`, and it is bounded. `pkg_resources` shipped inside
`setuptools` up to and including **81.0.0** and was removed in **82.0.0** — measured by installing
each version to a clean target and looking, not read from a changelog:

| setuptools | ships `pkg_resources` | warns on import |
|---|---|---|
| 80.10.2 | yes | yes |
| 81.0.0 | yes | yes |
| 82.0.1 | no | — |
| 83.0.0 | no | — |

`setuptools<82` is therefore the **only version ceiling in `pyproject.toml`**, and because `monetdb`
is in the `databases` aggregate it is inherited by `localdata-mcp[databases]` and
`localdata-mcp[all]`. One `UserWarning` per session survives it: both surviving versions deprecate
`pkg_resources` on import, and the warning's own advice — "pin to Setuptools<81" — does not silence
it, because 80.10.2 warns too. The upstream fix is one line of `importlib.metadata`, stdlib since
3.8.

**Eligibility was never in question.** An open-source SQLAlchemy adapter exists, published under the
vendor's own `github.com/MonetDB` organisation and MIT-licensed, so MonetDB is eligible under the
§13 rule. Adapter quality is a cost to record. This is what the cost turned out to be.

### 18.2 Every axis generic, on a storage model nothing here had exercised

| Axis | MonetDB | Measured by |
|---|---|---|
| `read_posture` | generic — the transactional floor holds | an `INSERT` on a never-committed connection left **0 rows** behind |
| `denies_write` | generic — nothing to recognise; the floor declines to keep a write, it does not refuse one | — |
| `ddl_survives_refusal` | generic `False` | DDL is transactional here |
| `unstorable_column_types` | generic — **empty** | `LargeBinary`, `Time`, `Date`, `DateTime`, `Numeric`, `Boolean`, `Float`, `String`, `Integer` all stored and read back identical |
| `folds_identifiers` | generic `False` | a quoted `Mixede0a6` came back from reflection **verbatim**; its lowered form was absent |
| `builds_indexes` | generic `True` | `CREATE INDEX` accepted, reflected by `get_indexes`, `DROP INDEX` accepted |
| `rename_table` | generic — `ALTER TABLE … RENAME TO` accepted verbatim | `RENAME TABLE …` is a syntax error: `42000!syntax error, unexpected RENAME` |
| `impostors` | none — MonetDB answers on its own dialect and borrows nobody's | — |
| `resident_bytes` / `snapshot` / `storage_classes` | generic | server-side database, no local file to weigh |

Two of those are worth stating rather than tabulating.

**`Time` and `LargeBinary` both store.** They are the two types this harness has watched fail
elsewhere — Oracle and ClickHouse both refuse one or the other, and Trino refuses `LargeBinary` (§16).
A column store had no obligation to keep them and does.

**Case survives.** Trino folds every identifier to lower case at the connector whether quoted or not,
which is what made issue #48 the bad kind of defect — both spellings resolved, so nothing failed while
`update` and `info` disagreed about a table's name one payload apart. MonetDB preserves what it is
given, so `landed_as` reports back the name that was asked for, and the assertion that a rename keeps
its case holds without the seam being consulted.

### 18.3 Isolation is `SERIALIZABLE` and there is nothing else to choose

The dialect offers exactly two levels — `AUTOCOMMIT` and `SERIALIZABLE` — and connects at
`SERIALIZABLE`. Naming any other raises before a connection is made:

```
ArgumentError: Invalid value 'READ COMMITTED' for isolation_level.
Valid isolation levels for 'monetdb' are AUTOCOMMIT, SERIALIZABLE
```

This is the same *shape* as Trino's single reachable level (§16.3) and the opposite *outcome*. There,
the driver connected in `AUTOCOMMIT` and the transactional floor was not weakened so much as deleted,
which is why `TrinoBackend.read_posture` is load-bearing. Here the default is the strict end, the
floor holds without anything being set, and `read_posture` stays generic. **A short list of isolation
levels is not by itself a finding; which end of it the driver defaults to is.**

### 18.4 The credential sweep hardcoded its own environment — issue #49

Adding the builder did not fail against the database. It failed against a test, and the test was
`test_every_endpoint_builder_round_trips_its_own_credentials`, whose docstring says it sweeps the
endpoint table "so a builder added later is covered the day it appears".

It was not covered; it **errored**, with `KeyError: 'MDB_DB_ADMIN_PASS'`. The test synthesised its
environment from a hardcoded list of the variable names the existing builders happened to read — a
per-endpoint fact living in a fixture, which is the shape standing instruction 1 forbids in test
fixtures as firmly as in shared code. Every password-bearing endpoint had silently extended that
list; the four no-password endpoints exempted themselves through a `continue` and hid how much it had
been growing.

The list is removed rather than lengthened. The environment is now a `dict` whose `__missing__`
answers **any** variable with the hostile password, so a builder reading a name nobody anticipated is
covered instead of fatal — usernames and database names come back hostile too, which only widens the
demand, since the assertion is that a credential survives the round trip as a *value* rather than as
text. Proved still able to fail: an interpolating builder handed the same environment dies in
`make_url` with `invalid literal for int() with base 10: 'w'`, because `p@ss:w/rd?x#y` re-parses as a
host and a port.

The failure was loud, so no dialect was ever tested against a credential rule it did not meet. What
was wrong was the claim.

### 18.5 The image demands a password before it will start, which is the good failure

`monetdb/monetdb` creates no database unless `MDB_DB_ADMIN_PASS` is set — its entrypoint exits
rather than coming up with an unreachable server. The database it then creates is named by
`MDB_CREATE_DBS`, defaulting to `monetdb`, and the password is set on the `monetdb` user, so the
dialect, the user and the database all carry the same name and only their positions distinguish them.

The healthcheck runs a **real query** rather than asking `monetdbd` about itself. `monetdb status`
answers over the farm's local socket and reports a database healthy before anything has proved the
SQL layer will accept a statement; `mclient` authenticates and selects the way a client does.
`mclient` takes its credentials from a file named by `DOTMONETDBFILE` rather than from an argument,
which also keeps the password out of a process list. The container reaches healthy in **~9 s** —
the fastest endpoint in this harness.

Default schema is `sys`, which is also where user tables land, and `get_table_names()` returns **only
user tables** — 0 on a fresh database — so nothing has to filter system objects out.

### 18.6 What this did not test

A column store's interesting properties are all about scale: MonetDB's advantage is vectorised
execution over columns, and every table here is a handful of rows written one statement at a time.
Nothing measured says how it behaves under the sizes §9 and §10 put through DuckDB and SQLite, and it
is the one endpoint where that comparison would mean something — it and DuckDB are the two columnar
engines in this project, one remote and one local. Unmeasured, and worth measuring if the load half
of task 21 is taken up.

Its concurrency story is equally untouched: `SERIALIZABLE` with no other level available says a
single-statement harness will never see a conflict, not that conflicts resolve well.

## §19 — CrateDB, where a write is durable before it is readable (2026-07-29)

Eleventh endpoint dialect, eighth from the backend catalogue (task 22, worklist item 8), and the
first from the **search-engine lineage** rather than the database one — a distributed SQL layer over
Lucene. Two of its properties have no precedent in this seam, and one of them found a fail-open in
shared code that every dialect before it had been passing by luck.

Measured against the official `crate` image, server **6.4.1**, through `sqlalchemy-cratedb` 0.43.1 on
the `crate` client 2.2.1 — Crate.io's own, Apache-2.0, released 2026-06-22. This is the best-provenance
image in the harness: `crate` is an *official* Docker Hub library image, which no other entry in the
catalogue has been.

### 19.1 An INSERT that reported rows, and the floor that believed it

The finding, and it was never about CrateDB.

`Workspace.query_stream` refuses a statement that is not a read, and decided it this way:

```python
if not result.returns_rows:
    raise LoadError(_not_a_read(entry))
```

Every dialect before this one answered `returns_rows = False` for an `INSERT`. CrateDB answers
**`True`** — one row, of **zero columns** — so the statement passed the floor, `query` reported
`ok: True`, and the row stayed inserted, there being no transaction to withhold it. A caller who was
told `query` refuses writes was told the truth about the *policy* and a lie about *this statement*.

The fix is generic and names no dialect:

```python
names = list(result.keys()) if result.returns_rows else []
if not names:
    raise LoadError(_not_a_read(entry))
```

A `SELECT` returns rows even when it matches none, and it always projects at least one column, so the
test stays exact rather than heuristic. **The assumption that failed was in shared code**: that a
driver saying "rows" meant a caller was reading. Asking for a column as well is the same question
asked completely.

Proved able to fail in the only way that counts — it *did* fail, before the fix, as
`test_query_refuses_a_write_even_where_the_datasource_permits_it[cratedb]` reporting
`{'columns': [], 'ok': True, 'row_count': 1, 'rows': [[]]}`.

### 19.2 No transactions, and unlike ClickHouse nothing to put in their place

ClickHouse (§11) was "the first backend with no transactions", and the phrase turned out to be doing
two jobs. ClickHouse has no transactions *and* has `readonly=1`, a posture the server enforces before
a statement reaches the data. CrateDB has neither:

| | ClickHouse | CrateDB |
|---|---|---|
| Transactions | none | none |
| Isolation levels offered by the dialect | — | `()` — the tuple is empty |
| Read-only session | `readonly=1` in the URL | none; its read-only setting is a **cluster-wide** block that would stop the write engine too |
| DML through a read connection | refused | **applied** — measured, the row was there afterwards |
| DDL through a read connection | refused | **applied** — measured, the table was there afterwards |

So CrateDB is the first backend where a refused write really happened, and `Backend` gains
`dml_survives_refusal()` to say so — the sibling of `ddl_survives_refusal()`, which Oracle has
answered `True` since §5. They are split rather than merged because they are independent: Oracle's DDL
survives while its DML does not, and one axis would have to lie about one of them.

**The refusal is still issued, and that is not a formality.** What it now says is true in both
directions: `query` does not write here, *and* this particular statement was not undone. A refusal
claiming a statement did not happen, when it did, is the same lie as reporting a rolled-back write as
a success — the judgement §5 recorded for Oracle, reaching rows for the first time.

### 19.3 Durable before readable, and why waiting was the wrong answer

Rows land in a Lucene index that refreshes on a timer. Measured, on a fresh table:

| Moment | `SELECT COUNT(*)` |
|---|---|
| immediately after the write commits | **0** |
| after `REFRESH TABLE` | 1 |
| without refreshing, polling | 1, after **0.92 s** |

`insert_frame` counts the rows it just wrote and puts that number in the payload, so left alone this
is not a flaky test but a **wrong payload for a reason the caller cannot see** — 0 on a fast machine,
correct on a slow one, and nothing to tell them which they were handed.

`Backend.settle(conn, table)` is the new axis: a no-op for every transactional backend, because there
the commit *is* the moment rows become visible, and `REFRESH TABLE` on CrateDB. It runs on the same
connection and inside the same block as the insert, so a write and the visibility of that write cannot
be separated by a failure between them.

Waiting instead was considered and rejected twice over: it trades a wrong answer for a slow one, and
it picks a timeout by guessing at a server setting that is free to change. The measured 0.92 s is
what the default happens to be here, not a contract.

Two tests write *below* the verbs and therefore had to ask for it themselves —
`_build_typed_table` and the refused-write count. Both consult the seam rather than naming the
dialect, which is the precedent `unstorable_column_types` already set.

### 19.4 A date that arrived as a number

CrateDB's HTTP protocol carries values untyped, with the column types beside them, and the driver
spells a value as a Python object only if it is asked to. Unasked, a `TIMESTAMP` reached the caller as
`1709251200000` — epoch milliseconds. JSON carries that perfectly happily and an agent reads it as a
quantity, which is precisely the failure standing rule 7 exists to prevent: **dates are canonical UTC
ISO 8601 text, never epoch integers.**

It is not reachable through SQLAlchemy's typing. A Core `select()` over a reflected table converts
correctly, because SQLAlchemy knows the column types; `query` runs the caller's own text, where it
knows nothing — and the DBAPI cursor description supplies **no type codes at all**, every field
`None`:

```
(('d', None, None, None, None, None, None), ('n', None, None, None, None, None, None))
```

The driver's own `DefaultTypeConverter` is the public answer, working off the `col_types` the server
sends alongside the rows. `Backend.connect_args()` is the new axis that delivers it — driver arguments
a URL cannot express, given to the read and write engines alike so the two cannot disagree about how a
value is spelled. With it, the same column comes back `2024-03-01T00:00:00Z`.

The `Z` is information, not noise: CrateDB has no date type, so a date is a `TIMESTAMP`, and its
instants are UTC. Three spellings now satisfy that assertion — a bare date, a naive instant, and a
UTC-marked one — and each is the truth its dialect can tell. None is an epoch integer.

### 19.5 `Numeric` is silently truncated, and the database is not at fault

The dangerous one, because nothing fails.

`sqlalchemy-cratedb` renders `Numeric` as **`BIGINT`**. `Decimal("12345.6789")` is therefore stored as
`12345`, and read back as `Decimal("12345.0000")` — SQLAlchemy's own type re-applies the scale on the
way out, which is what makes the loss invisible. Measured three ways:

| Declared as | Value in | Value out |
|---|---|---|
| `Numeric(10, 2)` via SQLAlchemy | `Decimal("1.25")` | `Decimal('1.00')` |
| `Numeric(38, 4)` via SQLAlchemy | `Decimal("1.25")` | `Decimal('1.0000')` |
| `NUMERIC(10,2)` in raw SQL, literal | `1.25` | `1.25` |
| `NUMERIC(10,2)` in raw SQL, bound `Decimal` | `Decimal("1.25")` | `1.25` |

`SHOW CREATE TABLE` confirms it directly: the column SQLAlchemy declared `Numeric(10,2)` is created as
`"c" BIGINT`. **The database supports the type; the dialect does not use it.** Recorded as the
dialect's defect so nobody later fixes CrateDB for it — issue #52.

Nothing this server writes reaches it: `_declared_type` emits only `INTEGER`, `REAL` and `TEXT`, so
the exposure is a caller's own table. `Numeric` therefore joins `unstorable_column_types()`, alongside
`LargeBinary` and `Time` — a column that silently corrupts is worse than one that cannot be created,
which is the judgement §11.4 already recorded for ClickHouse's `Time`.

`LargeBinary` and `Time` are here for a different reason from ClickHouse's two of the same name: those
are the *driver's* gaps, where the type exists and cannot be bound. These are the database's absences.
`CREATE TABLE` is refused at parse time — `Cannot find data type: blob`, `Cannot find data type:
time` — so the column is never made.

### 19.6 Everything else, and a healthcheck that repeated a known gotcha

| Axis | CrateDB |
|---|---|
| `builds_indexes` | `False` — `CREATE INDEX` is *unparseable* (`no viable alternative at input 'CREATE INDEX'`), because every column is already indexed on write unless the table says `INDEX OFF` |
| `rename_table` | generic — `ALTER TABLE … RENAME TO` accepted verbatim |
| `folds_identifiers` | generic `False` — a quoted `Mixed902b` reflects back verbatim |
| `impostors` | none; it answers on its own dialect |
| `resident_bytes` / `snapshot` / `storage_classes` | generic |

`builds_indexes` now has **three** users — ClickHouse, Trino and CrateDB — refusing for three
different reasons: an index of another kind, no data to index, and an index already there. The axis
carries the fact and only the sentence differs, which is what an axis is for. §11 recorded an
objection that it might have one user forever; that is now twice retired.

The healthcheck repeated the `yugabyted` gotcha exactly. `network.host=_site_` binds the container's
own interface, so `crash --hosts http://localhost:4200` returned `CONNECT ERROR` for two minutes while
the database answered perfectly well — `curl` against the container's address returned the node banner
and the same `crash` command succeeded. **A healthcheck must probe the address the server actually
binds**, and this is the second endpoint to prove it. The entry uses `$(hostname -i)`.

The image reaches healthy in ~10 s. It publishes three ports — 4200 HTTP, 4300 transport, 5432
PostgreSQL-wire — and **only 4200 is published**, deliberately: addressing the compatibility layer
would load PostgreSQL's dialect and answer PostgreSQL's questions about CrateDB, the mistake §12
records for CockroachDB.

### 19.7 What this did not test

The whole distributed half. One node with `discovery.type=single-node` says nothing about sharding,
replication, or what a partial write looks like when a node is lost — and CrateDB's answers there are
the reason anyone chooses it. Its eventual-consistency window was measured at rest, with one writer;
nothing here says what `settle` costs under load, or whether a refresh forced after every insert is
the right trade at a million rows rather than at five.

## §20 — Firebird, whose strictness costs more than any laxity here (2026-07-30)

Twelfth endpoint dialect, ninth from the backend catalogue (task 22, worklist item 9), and the oldest
engine in it — an InterBase descendant whose lineage predates everything else in this harness. It is
also the first entry whose difficulties come from a database being **stricter** than its neighbours
rather than looser, and the first whose dialect is named after neither its engine nor another
engine's.

Measured against the Firebird Project's own `firebirdsql/firebird:5` image, server **LI-V5.0.4.1812
Firebird 5.0**, engine version `5.0.4` read from `RDB$GET_CONTEXT('SYSTEM','ENGINE_VERSION')`, through
`sqlalchemy-firebirdsql` 0.1.0 on the pure-Python `firebirdsql` 1.4.6.

Three of its four findings are defects in shared code or in the client library. Only one is a genuine
capability limit of the database, and it is the least interesting of them.

### 20.1 Two dialects, and the adapter that cannot be reached from here

Firebird has two SQLAlchemy dialects on PyPI, and the eligibility rule (standing instruction 10) is
satisfied twice over. Which one to use is a question about **addressing**, not eligibility, and it was
decided by measurement rather than by maturity:

| Distribution | Version | Registers | Driver | License | Published |
|---|---|---|---|---|---|
| `sqlalchemy-firebird` | 2.2.0 (2026-05-21) | `firebird` | `firebird-driver` 2.0.3 | MIT | fdcastel |
| `sqlalchemy-firebirdsql` | 0.1.0 (2026-05-23) | `firebirdsql` | `firebirdsql` 1.4.6 | MIT | Nakagami / fdcastel |

The first is the older and better-established, and **cannot be used on this machine**.
`firebird-driver` is a ctypes binding to Firebird's own `libfbclient`, a native library with no wheel,
no Homebrew formula (`brew info firebird` → `No available formula`), and nothing but a system-wide
`.pkg` installer to supply it. Measured rather than inferred — with `firebird-driver` installed and
nothing else:

```
Exception: The location of Firebird Client Library could not be determined.
```

The second needs no native library at all: `firebirdsql` speaks the wire protocol in Python. **It is
not a shim** — step 1 of the adoption procedure exists to ask that question, and the answer here is
that it is the same dialect base ported, co-authored by `sqlalchemy-firebird`'s own author and
maintained by the driver's. The cost it carries instead is youth: version 0.1.0, sdist only, no wheel.

**The server's own security defaults were not weakened to reach it, and that was checked rather than
assumed.** Firebird 4+ defaults `WireCrypt` to `Required`, and the image's entrypoint offers
`FIREBIRD_USE_LEGACY_AUTH`, which would reduce it to `Enabled` as a side effect — so the tempting move
is to set it pre-emptively. Measured instead: the pure-Python driver completes `Srp256` authentication
against `WireCrypt=Required` unmodified. The compose entry sets no crypto or auth configuration at all.

### 20.2 DDL and DML cannot share a transaction (issue #53)

The highest-value finding, and it was never about Firebird.

`insert_frame` created its target table and inserted the rows in one `write.begin()` block. The single
transaction is deliberate and its reason is in the code — a write and the visibility of that write must
not be separable by a failure between them. The **assumption underneath** it was not deliberate: that
a table this server just created is addressable by the next statement on the same connection.

Firebird's DDL is genuinely transactional, and statements are prepared against *committed* metadata.
So the new table is invisible to the very transaction that made it:

```
firebirdsql.err.OperationalError: Dynamic SQL Error
SQL error code = -204
Table unknown
MIX_0A5C31
[SQL: INSERT INTO mix_0a5c31 VALUES (1)]
```

One server, one variable — whether the DDL commits first:

| Sequence | Result |
|---|---|
| `CREATE` then `INSERT`, one transaction | `-204 Table unknown` |
| `CREATE`, commit, then `INSERT` | the rows land |
| `DROP … checkfirst` then `CREATE`, one transaction | accepted |

The third row is why the split is at the DDL→**DML** boundary and nowhere else: schema statements may
share a transaction with each other, so the drop-and-create pair stays atomic.

**Eleven dialects had agreed with the assumption, which is one observation repeated eleven times** —
the same shape as §19.1, where ten dialects agreed that `returns_rows` meant a read. Note *why* they
agreed, because it is not a point in their favour: Oracle, MySQL, MariaDB and SQL Server pass this
test by committing DDL behind the caller's back. Firebird refuses to do that. **The backend that
looks broken here is the only one keeping the promise the others quietly break.**

The remedy is `Backend.sees_new_tables_in_transaction()`, and **it costs something that is worth
stating rather than burying**: with the `CREATE` committed first, a failure part-way through the rows
leaves an empty table behind where the single transaction would have left nothing at all. That is
strictly worse, and it is accepted only because the alternative on this backend is that the write
cannot happen.

### 20.3 `Text` becomes a BLOB, and a BLOB groups by identity (issue #56)

The dangerous one, because nothing fails.

`_PORTABLE_TYPES` maps a loaded text column to Core's `Text`, which this dialect renders `BLOB`.
Firebird accepts it, stores every byte, and reads each value back exactly — and then treats the column
as an **opaque handle** for every set operation. The five-person fixture answers
`GROUP BY department` with five groups:

```
[['engineering', 75000], ['sales', 65000], ['marketing', 70000], ['engineering', 80000], ['hr', 55000]]
```

Same rows, same server, one variable — the column type:

| Column type | declared | `GROUP BY department` | `DISTINCT department` | `ORDER BY` sorts |
|---|---|---|---|---|
| `Text` | `BLOB` | 5 groups, engineering split | 5 rows for 4 values | **no** |
| `String(11)` | `VARCHAR(11)` | 4 groups, engineering = 155000 | 4 rows | yes |

A BLOB also cannot be indexed at all — `CREATE INDEX … ON t (dept)` fails with `unsuccessful metadata
update`, which is what broke the index test here.

This is Oracle's problem (§ on `column_type`) by a different mechanism and with the same remedy:
there `CLOB` refuses to be grouped **out loud**, here `BLOB` agrees and gets it wrong. The fix is
`VARCHAR` sized from the widest value the column actually holds.

**A warning about how this was measured, because the first measurement was wrong.** An earlier probe
of the same question returned the *correct* `[('engineering', 155000), ('sales', 65000)]` from a BLOB
column — three rows, a different table — and that answer was luck rather than a result. Taken at face
value it would have concluded that BLOB grouping works and shipped the silent corruption. An
intermittently-correct answer is worse than a consistently wrong one, and **one observation of a set
operation over opaque handles is not a measurement of it.** What made the truth visible was putting
the two column types side by side over identical rows, rather than asking the same question twice.

The ceiling is **8,191 characters, not the 32,765 this container accepts**, and the difference is the
point. Firebird's limit is 32,765 **bytes**; a database created with charset `NONE` — which this
container is, confirmed by `RDB$CHARACTER_SET_NAME` — spends one byte a character, while a UTF8
database spends up to four. 8,191 is the widest width that fits under *any* character set, so the
declaration cannot fail for a reason belonging to how somebody else created their database. Using the
measured 32,765 would have been a measurement from one configuration presented as a property of the
engine.

Beyond the ceiling it falls back to `Text`: the grouping is lost, which is bad, and the values are kept
whole, which matters more than truncating them to fit.

### 20.4 The dialect cannot spell a 64-bit float (issue #55)

`_PORTABLE_TYPES` maps a loaded float64 column to Core's `Double`, which both Firebird dialects render
as bare `DOUBLE` — a keyword Firebird does not have. The parser is waiting for `PRECISION` and fails on
whatever follows, so no table is made:

```
SQL error code = -104
Token unknown - line 4, column 1
)
```

Compiled against both dialects, so this is the shared `base.py` lineage rather than either port:

| Type asked for | `sqlalchemy-firebirdsql` 0.1.0 | `sqlalchemy-firebird` 2.2.0 | stored/read back | reflected as |
|---|---|---|---|---|
| `Double` | `DOUBLE` | `DOUBLE` | *table not created* | — |
| `DOUBLE_PRECISION` | `DOUBLE PRECISION` | — | `0.1` → `0.1`, exact | `DOUBLE PRECISION` |
| `Float(53)` | `FLOAT(53)` | — | `0.1` → `0.1`, exact | `DOUBLE PRECISION` |

`DOUBLE_PRECISION` is used over `Float(53)`: both land in the same Firebird column, and only one says
what it means. **A client-library defect, not a database limit** — Firebird holds an 8-byte float
perfectly well once asked in its own words. Recorded so nobody later "fixes" Firebird for it.

### 20.5 There is no way to rename a table

The one genuine capability limit, and the least consequential.

`ALTER TABLE … RENAME TO` is not merely unsupported but unparseable — `-104 Token unknown - line 1,
column 24 RENAME` — and unlike SQL Server, which needs `sp_rename`, Firebird has no vendor-specific
alternative. A *column* can be renamed here; a table cannot, in any version.

Faking it was considered and refused. Copying the rows into a table of the new name and dropping the
old one reads like a rename and is not one: `rename_table` promises rows, types **and indexes**, and a
copy keeps only the first. Handing back a name that is a table missing its indexes would be a lie the
caller then builds on — the judgement `snapshot` already makes about a database this server does not
hold. So `Backend.renames_tables()` answers `False` and the refusal says what the alternative costs.

### 20.6 Everything else, and the measurements that found nothing wrong

| Axis | Firebird |
|---|---|
| transactional floor, DML | generic — an `INSERT` on a connection that never commits leaves 0 rows |
| transactional floor, DDL | generic — a `CREATE TABLE` that never commits leaves no table |
| `ddl_survives_refusal` / `dml_survives_refusal` | both generic `False`. Worth noting beside 20.2: the DDL does **not** survive a refusal *and* cannot be used by its own transaction. The two axes look adjacent and are opposite here, which is why they are separate |
| `folds_identifiers` | generic `False` — a quoted `ProbeEfaa` reflects back verbatim |
| `Numeric(10,2)` | exact — `Decimal("1.25")` round-trips, where CrateDB truncated it to `BIGINT` (§19.5) |
| `Date` / `DateTime` / `Time` / `LargeBinary` | all store and read back; `unstorable_column_types` is empty |
| `builds_indexes` | generic `True` — create, reflect and drop all work on a `VARCHAR` column |
| `driver_errors` | generic — `firebirdsql` errors arrive properly wrapped as `SQLAlchemyError` with `.orig` set |
| `impostors` / `banner_query` | none; only Firebird speaks this dialect |
| `resident_bytes` / `snapshot` / `storage_classes` / `table_options` / `connect_args` | generic |

`server_version_info` reads **`None`** — the dialect never populates it. Harmless here, because
nothing shares this dialect and so no identity probe is needed, but it would matter to any future
`impostors` entry and is recorded for that reason.

**A dialect named after a driver.** `sqlalchemy-firebirdsql` registers itself as `firebirdsql`, so a
URL resolves to that and `backend_for` would hand back `Backend(name="firebirdsql")` — putting the name
of a Python package in front of somebody who opened a database. This is issue #45 in a third form: the
loud cases were two engines sharing one dialect (TiDB on MySQL's, YugabyteDB on PostgreSQL's), resolved
by asking the server for its banner. This one needs no probe, because only Firebird speaks the dialect;
it needs only the entry to be keyed on the dialect while *naming* the engine. Hence the one `BACKENDS`
key that is deliberately not an engine's name.

**Index key ceiling, for the record.** On the default 8,192-byte page: `VARCHAR(8100)` indexes,
`VARCHAR(8191)` does not (`unsuccessful metadata update`). `VARCHAR(32765)` is the widest column the
container accepts; `VARCHAR(32766)` is refused.

**The healthcheck repeats a known gotcha in a new disguise.** `isql` against a bare path opens the
database **in embedded mode**, inside the healthcheck's own process — which succeeds while the TCP
listener is still starting, reporting a database ready that no client can reach. Prefixing
`localhost:` forces the connection through the listener. This is the third form of "a healthcheck must
probe the address the server actually binds", after `yugabyted` and CrateDB's `network.host=_site_`;
the first two were the wrong *interface*, this one is the wrong *transport*. The image ships no
healthcheck of its own and reaches healthy in ~10 s.

### 20.7 The container ceiling is now exceeded by the catalogue (issue #46)

Twelve dialects and a proven ceiling of six means **no two runs can cover the catalogue** any more.
Batch A and batch B each held six with PostgreSQL in both, which covered eleven; the twelfth needs a
third batch. This session ran three — 579, 576 and 502 passed with no failures — and every one of them
reported a green suite while five or six dialects stayed silent. The live half of #46 is now
arithmetic rather than a risk.

### 20.8 What this did not test

Firebird's own distinctive machinery, all of it. Multi-generational architecture means readers never
block writers and a long transaction pins old record versions; nothing here says what that costs, and
the single-statement harness cannot produce the sweep-and-garbage-collect behaviour that makes it
interesting. `SuperServer` versus `Classic` versus `SuperClassic` — the image's `changeServerMode.sh`
offers all three — was left at the default. Nothing exercised its embedded mode, which is the form
most Firebird deployments actually use and which this server would reach as a *file* rather than as an
endpoint. Events, external tables, and `PSQL` stored procedures are all untouched.

## §21 — openGauss, and a banner that stopped the dialect before the query (2026-07-30)

Thirteenth endpoint dialect, tenth from the backend catalogue (task 22, worklist item 11 — taken
ahead of Db2, which the procedure permits). A PostgreSQL fork, and the **third** engine here on that
wire after CockroachDB and YugabyteDB — but the first that cannot be addressed as PostgreSQL at all.

Measured against the openGauss project's own `opengauss/opengauss-server:7.0.0-RC3.B025`, server
**openGauss 7.0.0-RC3 build 01b7e318**, through `opengauss-sqlalchemy` 2.4.0 on `psycopg2-binary`
2.9.12. The widely-cited `enmotech/opengauss` was passed over: its newest tag is from 2024, and a
vendor image exists.

Its own behaviour turned out to be almost entirely generic. Both findings are in **our** code, and
both were invisible rather than loud.

### 21.1 Authentication succeeds and the dialect kills the connection anyway

The expectation going in — from YugabyteDB, §15 — was that a PostgreSQL fork is reached as
`postgresql+psycopg` and the only question is what it calls itself. That is wrong here, and the
measurement is unambiguous:

```
FAIL testuser@testdb -> AssertionError: Could not determine version from string
  '(openGauss 7.0.0-RC3 build 01b7e318) compiled at 2026-03-25 18:12:24 commit 0 last mr 9114 ...'
```

**psycopg connects and authenticates perfectly well.** What fails is SQLAlchemy's own `PGDialect`
immediately afterwards: it asserts that `version()` matches `PostgreSQL x.y`, and openGauss's banner
begins with a parenthesis. The failure lands during connection *initialisation*, so it is not a
degraded read or a wrong name — no statement ever runs.

This is the distinction §15 drew and could not yet demonstrate. YugabyteDB's banner is
`PostgreSQL 15.12-YB-2.25.2.0-b0`, which parses to `(15, 12)`, so plain PostgreSQL reaches it;
CockroachDB's does not parse, which is why it has its own dialect. openGauss is a second instance of
the CockroachDB case, and the two together retire the idea that PostgreSQL-lineage implies
PostgreSQL-addressable.

| Engine | Banner starts | `postgresql+psycopg` reaches it | Addressed as |
|---|---|---|---|
| YugabyteDB | `PostgreSQL 15.12-YB-…` | yes | `postgresql+psycopg`, with an `impostors` entry |
| CockroachDB | `CockroachDB CCL v…` | no | `cockroachdb+psycopg` |
| openGauss | `(openGauss 7.0.0-RC3 …` | **no — AssertionError on connect** | `opengauss+psycopg2` |

**So the cost YugabyteDB declined is paid here, and paid because there is no alternative.**
`opengauss-sqlalchemy` is the openGauss project's own (MIT) and registers psycopg2 drivers only,
making psycopg2 the **second PostgreSQL driver family** in this project. It is a milder cost than the
one that decided YugabyteDB's addressing: `psycopg2-binary` publishes ordinary wheels, where
`psycopg2-yugabytedb` was a fork pinned at 2.9.3 publishing wheels for one platform. A second driver
family with real wheels is a cost to record; a second driver family that must be compiled is a reason
to choose differently.

**No `impostors` entry, and that is the point of shipping a dialect.** The identity problem #45
describes does not arise: the dialect registers under its own name, so the backend already calls
itself `opengauss` and there is nothing to resolve from a banner. Firebird (§20) is the mirror image —
its dialect is named after the *driver*, so the entry is keyed on the dialect while naming the engine.

Two container facts, both from the entrypoint rather than from documentation. `GS_PASSWORD` is checked
against a complexity rule — eight characters, a lower, an upper, a digit and one of `#?!@$%^&*-` —
and initialisation is refused without one that passes, **so every password this database accepts
contains a URL delimiter.** An endpoint that formatted credentials into a URL could not reach
openGauss at all; the `@` in the compose file is load-bearing, not decorative. And the initial user
`omm` is refused over TCP outright — `FATAL: Forbid remote connection with initial user` — so
`GS_USERNAME` must create a normal user or nothing connects.

### 21.2 A BYTEA that arrived as a process address (issue #57)

`_ON_THE_WIRE` spells binary for JSON by exact type and named two of the three spellings, `bytes` and
`bytearray`. **psycopg2 returns a `BYTEA` as `memoryview`**, where psycopg 3 returns `bytes` — so
which type arrives is a property of the *driver*, and this project now carries two drivers for one
wire protocol.

```
AssertionError: assert '<memory at 0x1192c7640>' == '0x00ff'
```

Worth separating from an ordinary missing spelling. `_wire_value`'s fallback is `str(value)`, and that
fallback is a good decision — it beats refusing a whole result over one unrecognised column, and for a
PostGIS geometry it produces something readable. `memoryview` is where the reasoning breaks: its
`repr` is **not a lossy rendering of the value** but a process address. None of the bytes are in it, it
differs every run so it cannot even be compared, and it looks like data. That is the fail-open shape
the table exists to prevent, arriving *through* the escape hatch rather than around it.

The general note, recorded because `memoryview` is unlikely to be the only one: a type whose `repr`
carries an address rather than its contents is silently unserialisable, so the `str()` fallback cannot
be assumed harmless.

### 21.3 A refused write reported as a syntax error (issue #58)

The second invisible one, and it is the same class as #45 reached through a string match instead of a
dialect name.

No PostgreSQL-lineage engine will declare a server-side cursor over anything but a query, so a
streamed write comes back as a syntax error naming the statement's own first word.
`_objected_to_the_leading_verb` exists to recognise that shape and replace the diagnosis. Its guard
required the literal `cursor for`:

| Engine | Cursor declaration in the error | contains `cursor for` |
|---|---|---|
| PostgreSQL | `DECLARE "c_1" CURSOR FOR INSERT INTO …` | yes |
| openGauss | `DECLARE "c_1" CURSOR WITHOUT HOLD FOR INSERT INTO …` | **no** |

So openGauss's refusal reached the caller as `syntax error at or near "INSERT"`, sending an agent to
hunt for a typo in a statement that has none. **The write was still refused** — nothing reached the
data, so this was a diagnosis defect and not a safety one, which is exactly why it could sit there
unnoticed: the test that caught it asserts the message names `create`, not that the write failed.

The helper's docstring calls itself "narrow on purpose", and it was right about *which* narrowness
matters — that the objected-to token is the statement's own first word, which is what separates
"wrong kind of statement" from "malformed query". The `cursor for` literal was not that. It pinned one
engine's phrasing of a construct the whole lineage emits differently. Requiring `declare` **and**
`cursor` keeps the context without pinning the words between them, and the leading-word check, which
is the real guard, is unchanged.

Only openGauss exercises this path today: CockroachDB and YugabyteDB are reached through psycopg 3,
which streams without declaring a server-side cursor at all.

### 21.4 Everything else, and it really was everything

| Axis | openGauss |
|---|---|
| transactional floor, DML and DDL | generic — both roll back on a connection that never commits |
| `rename_table` | generic — `ALTER TABLE … RENAME TO` accepted verbatim |
| `folds_identifiers` | generic `False` |
| `column_type` | generic — the portable types render usably |
| `builds_indexes` | generic `True` — create, reflect, drop |
| `unstorable_column_types` | empty — `Numeric`, `Date`, `DateTime`, `Time` and `LargeBinary` all round-trip |
| `driver_errors` | generic — psycopg2's errors arrive wrapped |
| `impostors` / `banner_query` | none; it ships its own dialect |
| everything else | generic |

**No `BACKENDS` entry.** Like MonetDB (§18), openGauss answers every axis the generic way, and the
work it cost was entirely in code shared by every dialect. `server_version_info` reads `(9, 2, 4)` —
the PostgreSQL version openGauss reports compatibility with, not its own 7.0.0.

### 21.5 What this did not test

Its distributed and enterprise halves, which are most of why openGauss exists: primary/standby
replication, the `dcf` consensus mode, column-store tables, and the in-place update storage engine
(`ustore`) were all left at their defaults. The `dc_psycopg2` and `asyncpg` drivers its dialect also
registers are untried — only the synchronous psycopg2 one is exercised. Nothing here touches its
compatibility modes: an openGauss database can be created in `A` (Oracle), `B` (MySQL) or `PG` mode,
and the container's default is the only one measured.

## §22 — YDB, and a rollback that reports success over a write that stands (2026-07-30)

Fourteenth endpoint dialect, eleventh from the backend catalogue (task 22, worklist item 16 — taken
ahead of Db2, OceanBase, Exasol, Databend and HyperSQL, which step 4 of the procedure permits). A
distributed OLTP store from Yandex, and the first backend here reached through a dialect named after
neither its engine nor its driver.

Measured against `ydbplatform/local-ydb:26.1.1.22`, server version **26.1.1.22** (`SELECT version()`,
returned as `b'26.1.1.22'` — bytes, since YDB's `String` is a byte string), through `ydb-sqlalchemy`
0.1.22 on `ydb-dbapi` 0.1.22 and the `ydb` SDK 3.31.1. All three are Yandex's own, Apache-2.0, and
pure Python: there is no rival package and nothing native to install, which after Firebird's
`libfbclient` (§20.1) is worth noticing rather than assuming.

**One rule about transactions explains almost every measurement below: a schema operation may not be
inside a transaction.** It accounts for the rename, the drop, the index, the harness's own teardown,
and the shape of the refusal `query` produces for DDL. The findings that are *not* downstream of it
are the missing primary key, the absent time-of-day type, and the rollback that does not roll back.

Three findings are in **our** code or the harness's, one is a client-library defect, and one is a
property of the test image rather than of YDB.

### 22.1 Reaching it at all: the client does not talk to the endpoint it was given

A YDB client resolves the endpoint it is handed into the cluster's *own* node list and connects to
what comes back. Inside Docker that is the container's hostname on its internal port, which resolves
nowhere on the host:

```
InterfaceError: Resolved endpoints for database /local: DiscoveryResult
  <self_location: 1, endpoints [<Endpoint d7e126e4eca8:2136, location 1, ssl: False>]>
```

The handshake to the published port succeeds; discovery is what fails, immediately after.

Three routes were measured, and the choice between them is a design decision rather than a
workaround, so it is recorded as one:

| Route | Result | Why not |
|---|---|---|
| URL query `?disable_discovery=true` | **ignored**, original error unchanged | the driver swallows unknown connect keywords (#62) |
| `connect_args={"driver_config_kwargs": {"disable_discovery": True}}` | works | puts a fact about *this compose file* into the code path every YDB user runs |
| compose `hostname: localhost` + port mapped 2136→2136 | works | **chosen** |

The third makes the advertised address *true* rather than routing around it: the node advertises
`localhost:2136`, and on the host that address reaches the container. Discovery then behaves exactly
as it does against a real cluster, which is the point — the second route would have disabled a
production code path in order to fix a test harness.

Its cost is real and is the reason this is written down: **YDB is the only service in this harness
whose published port may not be offset.** Every other one publishes well away from its engine's
default so a locally-installed copy cannot be reached by accident. Here the published port must equal
the advertised one, so it is 2136 on both sides.

The measured fact underneath: `SELECT * FROM \`.sys/nodes\`` reports the node's host as `localhost`
once the container is given that hostname, which is what makes the address reachable.

### 22.2 A rollback that reports success over a write that stands (issue #61)

**The transactional floor does not exist on YDB as SQLAlchemy drives it.** An uncommitted write
survives. Measured three ways, one variable changed:

| Sequence | Rows afterwards |
|---|---|
| `INSERT`, connection closed without committing | 1 |
| `INSERT`, then an explicit `Connection.rollback()` | 1 |
| `INSERT` inside a `begin()` block that raises | 1 |

`rollback()` raises nothing in any of them. It returns normally.

**Trino's remedy was tried first and does not work here**, and the difference matters because the
two look identical from outside. In §16.2 the driver defaults to autocommit and naming a real
isolation level restores the floor. Here `SERIALIZABLE` is genuinely in force —
`get_isolation_level` reads back `IsolationLevel.SERIALIZABLE`, `interactive_transaction` is `True` —
and the write still lands.

The cause is **call order inside the client library**, not the isolation level.
`ydb_dbapi.Connection.cursor()` captures the connection's current `_tx_context` into the cursor it
builds. SQLAlchemy creates the cursor *before* it calls `do_begin`. Traced with both methods
instrumented:

```
call order: ['cursor(tx_context=None)', 'begin()']
rows after SQLAlchemy rollback: 1
```

So the cursor captures "no transaction", the statement runs in implicit autocommit, and the
transaction opened a moment later is empty — which is what `rollback()` then dutifully rolls back.

Driving the DBAPI directly, varying only the order:

| Order | Rows after `rollback()` |
|---|---|
| `begin()` → `cursor()` → `execute` → `rollback()` | **0** |
| `cursor()` → `begin()` → `execute` → `rollback()` | 1 |

**This is a client-library defect, not a property of YDB.** YDB's transactions work; the driver
exposes them; the cursor binds them at the wrong moment. Mark it as such before anybody "fixes" the
database for it.

Unreachable from here, so the posture is **raised rather than restored** — see 22.3.

### 22.3 A read-only isolation level, which refuses instead of undoing

Since there is no rollback to build a floor on, the read connection is given a read-only isolation
level and the server refuses the write before it reaches the data. Measured across every level the
dialect declares, same table, same statement:

| Isolation level | `SELECT` | write | rows before → after |
|---|---|---|---|
| `ONLINE READONLY` | works | refused | 1 → 1 |
| `SNAPSHOT READONLY` | works | refused | 1 → 1 |
| `STALE READONLY` | works | refused | 1 → 1 |
| `ONLINE READONLY INCONSISTENT` | works | refused | 1 → 1 |
| `SERIALIZABLE` | works | **accepted** | 1 → 2 |
| `AUTOCOMMIT` (the default) | works | **accepted** | 1 → 2 |

The refusal: `Operation 'InsertAbort' can't be performed in read only transaction`, carrying
`issue_code: 2008` on a nested issue. That is the shape SQLite and ClickHouse already have, and it is
a **stronger** guarantee than the transactional floor it replaces — the write never happens, rather
than happening and being undone.

`SNAPSHOT READONLY` is the one chosen, from four that all work. It is the level whose *meaning*
matches what a read connection should be: the latest committed state, consistent for as long as the
connection holds it. Picking one of the others for its behaviour under #61 would bake that defect
into the choice.

**Unlike Trino, the choice was free.** §16.2 records that SQLAlchemy normalises an isolation level to
spaces while Trino's dialect looks it up in an enum keyed with underscores, so every two-word level
raised `KeyError` and `SERIALIZABLE` was the only reachable one. This driver's enum *values* carry the
spaces, so all six round-trip.

### 22.4 A table must declare a primary key (the one new axis)

YDB has no heap tables. A keyless `CREATE TABLE` is refused at parse time:

```
message: "Pre type annotation" issue_code: 1020
  issues { message: "Primary key is required for ydb tables." }   (server_code: 400080)
```

Nothing is created, so this is a loud failure — which is the only reason it was cheap to find.

A file has no key to offer: nothing in a CSV is guaranteed unique, so any column nominated would be a
constraint this server invented on the caller's data. Three responses were considered and the choice
is recorded because it changes the shape of a table the caller gets back:

| Response | Why not |
|---|---|
| nominate every column as the key | YDB resolves a primary-key collision by **merging**, so two identical rows in a CSV would land as one, silently |
| refuse to load the file at all | the rows *can* be stored exactly; only the table's shape has to give |
| **add a surrogate key holding the row's position** | chosen |

So `insert_frame` prepends `_row`, an integer key holding each row's position in the file, and
**reports it**: the column appears in `info` like any other, and the table's notes say why it is
there. A column the caller did not ask for and cannot account for would be a table shape they have to
reverse-engineer.

`_row` cannot collide with a column the file brought: `_sanitize` strips leading underscores from
every header it cleans, and its fallback names are `column_N`.

The axis is `Backend.requires_primary_key()` — a bare fact, `True` only here. The *response* lives in
the loader, so a second engine that ever states the fact inherits the whole answer without writing
any of it.

**What this costs:** the write path has no protection at all here, per 22.2. A failure part-way
through `insert_frame` leaves the rows that already landed, and the drop-and-create pair is not atomic
either. Recorded as a known limit rather than worked around.

### 22.5 A schema statement stated as text is a different statement (issue #59)

`Backend.rename_table` is the one place this server must state SQL, because Core has no rename
construct — and it stated it as `text(...)`. Every other schema statement it issues is a Core
construct. On thirteen dialects the two are interchangeable. Here they are not:

| How the statement was issued | Result |
|---|---|
| `conn.execute(text("ALTER TABLE a RENAME TO b"))` | `Scheme operations cannot be executed inside transaction` (400120) |
| `conn.execute(DDL("ALTER TABLE a RENAME TO b"))` | renamed, rows kept |
| `conn.execute(text("DROP TABLE a"))` | refused, same message |
| `conn.execute(DropTable(a))` | dropped |
| `conn.execute(text("CREATE TABLE …"))` | refused, same message |
| `conn.execute(CreateTable(t))` | created |

Both render the same string. Only `DDL` is an `ExecutableDDLElement`, so only that travels the DDL
execution path where a dialect can say a schema statement does not belong inside the surrounding
transaction.

**The defect is in shared code and predates YDB.** Thirteen dialects agreeing that `text` and `DDL`
are the same thing is one observation repeated thirteen times.

The **harness's own teardown had it too** — `_drop_everything_named` dropped with `text`, was refused,
and left every YDB test's tables behind. Fixed the same way, and noted here because a fixture stating
a dialect fact is the same defect as shared code stating one.

The refusal that reaches a caller who sends DDL through `query` is now recognised too. It arrives the
opposite way round from the row refusal in 22.3 — `PRECONDITION_FAILED` (400120) on the *status*, with
`issue_code: 0` on the issue — so `denies_write` matches both, and `query` answers a `CREATE` with the
verb that does it rather than with the driver's sentence. Without that branch the outcome was already
right and only the explanation was wrong, which is the class §21.3 named.

### 22.6 No time-of-day type

`Unknown simple type 'TIME'`, raised while compiling the column, so there is no column for a value to
go into. Oracle's gap (§11-era, `OracleBackend.unstorable_column_types`) reached independently, and
the reason that axis is a set of names rather than a flag: the two backends share exactly one entry.

### 22.7 The test image survives being created but not being restarted (issue #63)

`YDB_USE_IN_MEMORY_PDISKS=true` works, once. `local_ydb deploy` writes the cluster configuration into
the container's filesystem, which survives a restart; the RAM-backed disks it names do not.

| pdisk setting | first boot | after `stop` then `up` |
|---|---|---|
| `YDB_USE_IN_MEMORY_PDISKS=true` | healthy ~15s | **unhealthy, permanently** |
| default (disk-backed) | healthy ~15s | healthy ~15s |

The restarted cluster refuses every `CREATE TABLE`, including the one in the image's own health
check: `database doesn't have storage pools at all to create tablet channels to storage pool binding
by profile id`.

This harness rotates its containers in batches by stopping and starting them, so the variable is a
trap laid for the next session rather than a saving. It is dropped, and the disk-backed default costs
**6.17 MB** of container writable layer (`docker ps --size`) — so what it was saving was not disk.

### 22.8 Everything else, and it really was almost everything

| Axis | YDB |
|---|---|
| `requires_primary_key` | **`True`** — the one new axis (22.4) |
| `read_posture` | **read-only isolation level** — a refusal, not a rollback (22.3) |
| `denies_write` | **two codes** — `issue_code 2008` for rows, status `400120` for schema (22.3, 22.5) |
| `unstorable_column_types` | **`{"Time"}`** (22.6) |
| `ddl_survives_refusal` / `dml_survives_refusal` | generic `False` — both refused before touching anything |
| `sees_new_tables_in_transaction` | generic `True` — schema is committed as it runs, so a new table is addressable |
| `rename_table` | generic, **once the generic one was fixed** (22.5); rows kept |
| `renames_tables` / `builds_indexes` | generic `True` — index created, reflected by name, dropped |
| `folds_identifiers` | generic `False` — `Probe9E9C` created and reflected verbatim |
| `column_type` | generic — the portable types render usably |
| `table_options` | generic — empty |
| `driver_errors` | generic — the driver's errors arrive wrapped as `SQLAlchemyError` |
| `impostors` / `banner_query` | none; it ships its own dialect |
| `settle` | generic — a committed write is immediately readable |

Types measured one column at a time, so one failure could not hide the rest:

| Core type | Rendered | Round trip |
|---|---|---|
| `Integer` | `Int32` | `7` → `7` |
| `String(50)` | `Utf8` | `'abc'` → `'abc'` |
| `Text` | `Utf8` | `'abc'` → `'abc'`, and **groups by value** |
| `Numeric(12,2)` | `Decimal(12,2)` | `Decimal('12.34')` → `Decimal('12.34')`, exact |
| `Double` | `Double` | `0.1` → `0.1` |
| `Date` | `Date` | exact |
| `DateTime` | `Timestamp` | exact |
| `LargeBinary` | `String` | `b'\x00\x01'` → `b'\x00\x01'` |
| `Boolean` | `Bool` | `True` → `True` |
| `Time` | — | **no such type** (22.6) |

`Numeric` round-trips exactly, which CrateDB did not (§19.5), and `Text` groups by value, which
Firebird did not (§20.3).

### 22.9 What this did not test

Everything distributed, which is most of why YDB exists. This is a single-node `local-ydb` container:
no partitioning across nodes, no replication, no failover, and therefore nothing about the
`40001`-style write-contention retry that §15.3 and issue #47 leave unmeasured — a single node cannot
produce genuine contention any more than the other single-node endpoints here can.

Also untried: the `ydb_async` dialect the same package registers (only the synchronous driver is
exercised); YDB's column-oriented tables (`STORE = COLUMN`), where the primary-key requirement and the
type mapping may both differ; secondary indexes beyond the plain global one, in particular
`GLOBAL ASYNC` and covering indexes; topics and the Kafka proxy the image also exposes on 9092;
authentication of any kind, since the image configures none; and TLS, which the image enables on 2135
while everything here goes over plaintext 2136.

The transaction findings in 22.2 are measured **through SQLAlchemy**, which is how this server reaches
every database. The DBAPI-direct measurements are there to locate the defect, not to describe a
supported path — nothing here uses the driver directly.
