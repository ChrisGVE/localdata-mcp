# Level 0 — SQLite and flat files, done well

Level 0 is the gate. Until the surface below is built and behaving, nothing else is
started: not more input formats, not more backends, not endpoint datasources. The
reason is that everything above level 0 is the *same* building blocks pointed at more
kinds of source, so a block that is wrong here is wrong everywhere later.

## The premise

**A slot is a database.** Not a table, not a file — a database, addressed by a
nickname. A CSV becomes a fresh in-memory database holding one table named after the
file; a SQLite file arrives with the tables it already has; a service URL becomes its
own engine. Because all three are databases, addressing is uniformly `nickname.table`
and a join across two datasources is ordinary SQL.

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
3. **Check whether the join is complete**, and say so in the user's terms: *"X, Y and Z
   have no match in that file."*

Adding to the existing database rather than attaching a second one is not only the
friendlier mental model — it is **what makes the result keepable**. `save` writes one
database, not a join, so a lookup meant to outlive the session needs both sides inside
the slot being saved.

So the two halves behave differently and are easy to conflate. Joining across slots
works and is worth doing for a one-off answer. Keeping that relationship — coming back
to it next session — requires both tables in one database, which is what `add_table` is
for.

> **Superseded (2026-07-26).** An earlier draft made step 2 *"build a view over the
> join"* and justified `add_table` by SQLite's refusal of a cross-database `CREATE VIEW`
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

## The seven verbs

| Verb | Shape | Notes |
|---|---|---|
| `attach` | `database`, `nickname?`, `writable?` | Multipurpose — flat file, SQLite file, later an endpoint. Returns the nickname **actually used**. |
| `detach` | `nickname` | Drop a slot deliberately instead of waiting for FIFO to guess. Deletes the temp file if spilled. |
| `query` | `nickname`, `sql`, `path?` | **Reads only** — every write is refused by SQLite's authorizer, whatever the slot allows. The optional path is where results are written, which **absorbs `export_query`**. |
| `info` | — \| `nickname` \| `nickname`+`table` | Polymorphic: bare → every slot; nickname → its tables; nickname+table → schema and row count. **Absorbs `list_tables` + `describe_table`.** |
| `add_table` | `nickname`, `source`, `table?` | Reads a datasource in beside the tables already there. This is what makes arc 2 possible. |
| `drop_table` | `nickname`, `table` | Composition needs both directions. |
| `save` | `nickname`, `path` | Relocate an in-memory or spilled database to a path the user chose — the "actually, keep this" escape from ephemerality. |

### Write is not the default

Only databases **the MCP created** are read/write. A flat file becomes our own in-memory
database, so it is writable by construction. Everything attached from outside is
**read-only**. The caller can grant write on an external database at attach time
(`writable=true`), and that grant is per-attach.

The grant governs `add_table` and `drop_table` — and only those, because **`query` never
writes to anything**. A query reads: `INSERT`, `CREATE TABLE`, `CREATE VIEW`, `PRAGMA`
and the rest are refused there even on a database the caller owns outright. So there is
exactly one way to change a slot, and it is a named verb rather than a clause buried in
a statement.

That is deliberately not a permission model. The enforcement is SQLite's own: a
read-only attach carries `mode=ro` in the connection URI, so a write fails in the engine
rather than in a check that could be reached around.

A database that was `save`d and is attached again later is, like any other external
database, **read-only by default** until the caller says otherwise.

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
completeness is the clearest case: `add_table` returns unmatched key counts and sample
values in both directions, and the skill turns that into the user's own words rather
than into the phrase "anti-join".

The skill ships in this repo and is versioned with the server, because the two are only
correct against each other.

## What comes after

Level 0 first, and only then: more input and output formats, then more backends —
file-based and endpoint-based, anything SQLAlchemy speaks — and after that, whether
SQLAlchemy needs extending for the formats it does not cover.

Building blocks first: **simple, composable, multi-faceted, and where possible
transparent even to the LLM.**
