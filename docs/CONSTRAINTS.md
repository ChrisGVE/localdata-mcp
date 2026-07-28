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
