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
over-correction that fails the flagship path for a condition the caller can work around once told
(`WHERE typeof(col)='integer'`, or an explicit `CAST`). Make strictness a config knob defaulting to
signal-and-continue.

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

**Sync tool bodies are dispatched to worker threads.** This is real parallelism, not async
interleaving. Any "only one thing happens at a time" assumption is false from the first tool.

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

### 4.2 Refusing an existing target is a feature — design *for* it

`VACUUM INTO` fails with `output file already exists` rather than overwriting. That removes a
silent-data-loss class and is the behaviour to copy: **an export must not silently overwrite.**
Require an explicit `overwrite=True`, and treat the target path as a trust boundary (traversal,
symlink-swap between check and write). Prior work in this codebase already treated file-identity
races as real; keep that posture.

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
