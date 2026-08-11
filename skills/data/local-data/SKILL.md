---
name: local-data
description: Answer questions about local data files and databases with SQL. Attach a spreadsheet, CSV, Parquet file, SQLite or DuckDB file, or a database URL, then look one source up against another and keep the result. Use whenever someone points at a data file or a database and asks what is in it.
allowed-tools: mcp__localdata__attach mcp__localdata__detach mcp__localdata__info mcp__localdata__query mcp__localdata__create mcp__localdata__update mcp__localdata__drop mcp__localdata__save
argument-hint: "<file-path> [and what you want to know]"
---

# Working with local data

The tools are few and mechanical on purpose. What they do not carry is the way
people actually talk about their data — that lives here.

## The one thing to understand first

**Every datasource becomes a database, and a database holds tables.** Even one
lonely CSV. The call names the database; the SQL names the table inside it:

```
query(nickname="shop", sql="SELECT sum(qty) FROM sales")
```

`shop` is the nickname of the database; `sales` is the table the file became.
Write the table's own name — `FROM sales`, never `FROM shop.sales`, which reads
as a table called `sales` in a database called `shop` that this statement was
never pointed at.

**One statement reaches one datasource.** There is no join across nicknames; to
put two files together, land one inside the other with `create` first.

Users will not talk this way — they say *"the sales file"* — and the translation
is yours to do silently. Never make someone say "table".

**`attach` returns the nickname it actually used, and it may not be the one you
asked for.** Read it back from the response every time. If a nickname collided,
`collided_with` tells you which datasource already had it.

## Speaking the user's language

They speak spreadsheet. Translate, and answer in their words.

| They say | You do |
|---|---|
| "the sales file", "that spreadsheet" | the nickname you got back from `attach` |
| "column", "field", "header" | a column |
| "row", "record", "line", "entry" | a row |
| "look up", "match against", "VLOOKUP", "cross-reference" | `create` the second file into the same datasource, then a join |
| "the total", "sum it up", "how many" | `sum()`, `count()` |
| "group by region", "break it down by" | `GROUP BY` |
| "filter out", "only show" | `WHERE` |
| "export it", "send me the file", "save as CSV" | `query` with `path=` |

Answer with the number and what it means, not with the SQL. Show the SQL only
when asked, when the result is surprising, or when they need to trust it.

## The things people ask for

### 1. "Have a look at this file"

```
attach(database="/path/sales.csv")        → note the nickname it returns
info(nickname, table)                      → columns, types, row count
query(nickname, "SELECT …")                → the answer
```

`attach` already returns the columns and row count for a file, so do not call
`info` straight after it — you have that. Go and answer the question.

**Nothing here is permanent.** It lives until `detach`, or until this server
stops. If the work is worth keeping, say so and offer to keep it — do not let
someone spend twenty minutes building something that evaporates. How you keep it
depends on where it came from: `save` for anything built from a file, and
`query(path=…)` for a database reached over a URL. See §6.

### 2. "Can you look it up against this other file?"

This is the common one, and the mistake is attaching the second file as its own
datasource. **Read it into the database that is already open:**

```
create(nickname="shop", type="table", source="/path/prices.csv")
```

Two reasons, and the second is the one that bites later:

- One database, so `sales` and `prices` join in plain SQL — which is the
  only way to query them together, since a statement reaches one nickname.
- **`save` writes one database, not a join.** Attach the two files separately
  and you get an answer now and nothing to keep — the moment the session ends,
  the relationship between them is gone. Landing the second file inside the
  first is what makes the lookup survivable.

**One case does not go this way: a second file holding more than one table.**
`create` makes one table, so a two-sheet workbook — or a JSON document with two
candidate keys — is refused, naming what it holds, and `table=` does not select
a sheet. The refusal says to attach the file as its own datasource, which is the
arrangement this section calls the mistake; it is the right answer only if you
wanted the whole workbook. To get one sheet in beside what you already have,
take it out through a flat file:

```
attach("/path/book.xlsx")                                   # → "book"
query("book", "SELECT * FROM prices", path="/path/prices.csv")
create(nickname="shop", type="table", source="/path/prices.csv")
```

The sheet is addressed by the snake_cased name it arrived under, which the
`attach` response lists — `Q2 Prices` is `q2_prices`. The directory `path`
writes into has to exist already.

The response describes the table it read in, so — as with `attach` — there is
nothing for an `info` call straight afterwards to add.

If the join is slow because both sides are large, index the column you are
joining on first. Nothing guesses this for you, because which query is coming is
yours to know:

```
create(nickname="shop", type="index", table="prices", columns=["sku"])
```

`info(nickname, table)` lists the indexes already there — cheaper than asking
for one twice. The name comes back, and that is the name `drop` wants.

**Five backends refuse an index** — ClickHouse, Trino, CrateDB, Databend and
Exasol — because on those engines the request means nothing: every column is
indexed already, or there is no storage to index, or the engine keeps its own.
The refusal says which. It is not a permission problem and not something to
retry; run the join as it stands, and if it is genuinely slow, narrow it in SQL.

**`query` will not write.** Not a permission you can ask for — a property of the
verb. If you find yourself reaching for `INSERT` or `CREATE TABLE`, the answer is
`create`; for removing one, `drop(nickname, type="table", name=…)`.

**Whether the match is complete is yours to check, and nothing checks it for
you.** It is an anti-join — ordinary SQL over two tables in one database, and
you must run it in **both** directions, because the rows in the new file that
nothing refers to are usually the surprise:

```sql
SELECT sku FROM sales  WHERE sku NOT IN (SELECT sku FROM prices)
SELECT sku FROM prices WHERE sku NOT IN (SELECT sku FROM sales)
```

**Then say what you found, in their terms**, and never as SQL:

> "Two products in the sales file have no price listed — `b` and `c`. And
> there's a price for `z`, which never appears in sales. Want me to leave those
> out, or treat the missing prices as zero?"

Never say "anti-join" to them. A join that silently dropped rows is the single
most likely way to hand back a confident wrong number, so do not skip this
because the totals looked plausible.

### 3. "That sheet is called Sheet1"

A workbook's tables arrive under the names the *spreadsheet* chose, lowercased
and snake_cased — `Sheet1` becomes `sheet1`, `Q2 Prices` becomes `q2_prices`.
**Use the name `attach` reported, not the one on the tab**, or the rename is
refused for naming no table. Then rename rather than re-reading the file; the
rows, types and any index stay put:

```
update(nickname="shop", type="table", name="sheet1", to="q2_sales")
```

A workbook whose sheets are all called `Sheet1`, `Sheet2`, `Sheet3` is worth
renaming before you do anything else — every query you write afterwards reads
better for it, and so does the file if they `save` it.

### 4. "It's in our Postgres, not a file"

A database URL attaches exactly like a file, and the same eight verbs address it:

```
attach(database="postgresql://user:pass@host:5432/sales")
```

Eighteen backends are supported — SQLite, DuckDB, PostgreSQL, MySQL, MariaDB,
SQL Server, Oracle, ClickHouse, CockroachDB, YugabyteDB, Trino, MonetDB, CrateDB,
Firebird, openGauss, YDB, Databend and Exasol. Three things differ from a file:

- **A URL is refused unless `network.enabled = true`** is set in the user's
  configuration file. The refusal says so and masks the password. That is a
  decision for them to make, not a flag to hunt for.
- **It arrives read-only**, like anything from outside, and it is somebody's
  production database. Do not ask for `writable=true` because a `create` failed —
  ask the user whether they meant to change the database itself.
- **`save` does not work here.** It writes out a database this server is
  holding, and a datasource reached over its own connection has none — the rows
  live in the engine. **Never offer `save` over a URL-attached datasource.** This
  also covers a DuckDB *file*, which is reached over DuckDB's own connection and
  is refused for the same reason. Offer `query(nickname, "SELECT …",
  path="/path/result.parquet")` instead, or `create` the rows into a datasource
  of your own — attach a small local file, land the result in it — and `save` it.
  §6 has the rule in full.

Two smaller refusals live here too: `create(type="index")` on the five engines
listed in §2, and `update(type="table")` on **Firebird**, which has no
rename-table statement. Both name the reason.

`create(nickname, type="table", source="./local.csv")` reads a local file *into*
that database, so a lookup against a server-side table is the same move as
against a second file. On **two** backends a statement `query` refuses can still
have happened, and you should say so rather than reassure: on **Oracle** a
refused `CREATE` or `DROP` stands, because Oracle commits DDL as it runs it,
while DML still rolls back; on **CrateDB** there are no transactions at all, so
a refused `INSERT` stands too. The refusal names `CREATE`/`DROP` in both cases,
so on CrateDB do not read it as meaning nothing happened.

### 5. "Send me the result"

A result they want as a *file* — to open in Excel, to mail on, to feed something
else — goes straight to disk instead of coming back through you:

```
query(nickname, "SELECT …", path="/path/result.csv")
```

**The suffix chooses the format**, and one this server cannot write is refused
by name rather than written as something else. Choose it rather than defaulting:

| They want | Ask for | Why |
|---|---|---|
| to open it in Excel or Numbers | `.xlsx` | **refused above 65,535 rows** — narrow it with `LIMIT` or send `.csv` |
| to open it in LibreOffice specifically | `.ods` | same cap, and slow — **5.8× `.xlsx`** at 20,000 rows of eleven ordinary-width columns, **~13×** at 50,000; on a wide result it had not produced a file after half an hour (below). Use `.xlsx` unless OpenDocument was asked for |
| a normal file, any size | `.csv`, `.tsv`, `.jsonl` | written row by row, so size costs nothing |
| something big, for another program | `.parquet` | the safe default, not a size winner: over seven shapes driven it was smallest on three — **by 100× or more where a column repeats few distinct values** — and eighth of fifteen on high-entropy text, where `.orc` won by about 1.2×. `.orc` and `.parquet` write within 8% of each other; `.feather` writes faster, and was the larger on every text shape but the smaller on both float shapes |
| it pasted into a document | `.md` | small results only — it builds the whole table in memory |

`.yaml` is available and is **4.3× `.csv`** on the same million-row result,
237.6 s against 55.5 s. It holds nothing in memory, so size is not the problem;
time is. Reach for `.jsonl` unless YAML was specifically wanted.

`.yaml` is not the slowest writer, though. **Which writer is slowest depends on
the shape of the result, not only its size**, and the spreadsheet writers are in
the same race rather than a separate one. On a wide result — 50,000 rows × 40
columns — `.md` takes 20.3 s against `.yaml`'s 15.1 s, and `.ods`, still legal
under its cap at that many rows, ran for over half an hour without producing a
file; on a narrow one `.yaml` comes first, by about a quarter rather than by a
wide margin. Under the 65,535-row cap, at 20,000 rows of eleven ordinary-width
columns: `.ods` 26.5 s, `.xlsx` 4.6 s, `.yaml` 2.0 s, `.md` 1.5 s, `.csv`
0.13 s — so on a result an agent can actually ask for, `.ods` is the slowest,
**5.8× `.xlsx` and 13× `.yaml`**, and `.yaml` is only the third slowest. Wide
text columns move these apart: `.csv`'s cost tracks bytes while the other four
track cells, so on a corpus with a 1,000-character column `.csv` alone is
several times slower.

This is also the answer when a result is simply too big to return — say so and
offer it, rather than returning tens of thousands of rows through the
conversation. The path is theirs: ask for it, and treat "the file already
exists" as a question for them, exactly as with `save` below.

### 6. "Keep this"

**Check where the datasource came from before you offer anything.** `save` writes
out a database this server is holding, so it works on one built from a file — a
CSV, a workbook, a Parquet file — and on SQLite, and **only** on those. Every
other backend is reached over its own connection and has no local database to
write out, so `save` is refused there. A DuckDB file is one of those: it is a
file on disk, and it is still reached over DuckDB's own connection.

`attach` reports `"kind": "file"` for anything read out of a data file, and
those can always be saved. `"kind": "database"` covers both SQLite (saveable)
and everything else (not), and **the response does not say which engine it is**
— so for a `"database"` datasource, go by the URL or the suffix they gave you, and
otherwise treat a `save` refusal as the answer rather than as something to
retry.

```
save(nickname, path="/path/analysis.db")
```

Writes the whole database — every table added — to a file they own. It stays
open afterwards. Attaching it again another day is an ordinary `attach`, so it
comes back **read-only** unless they pass `writable=true`.

Where `save` is refused, the answer is one of two things, and the refusal names
the first:

- **Land the rows in a datasource of your own and save that.** Attach or build a
  local one, `create(…, type="table", source=…)` the pieces you want into it, and
  `save` that. This is what to do when they want the *relationship* — several
  tables they can come back to.
- **Send the result to a file** with `query(nickname, "SELECT …", path=…)`. This
  is what to do when they want one answer in a form they can open or mail on.
  The answer comes back as `rows_written` and the column names rather than the
  rows themselves — `rows_written` is what to report back.

**The path is theirs, not yours.** Ask for the name rather than inventing one.
If `save` reports the file already exists, that is a question for them, not a
retry for you — say what is in the way and ask. Once they say replace it, pass
`force=true`. Never set it because a first attempt failed.

`save` also refuses to overwrite a file that any open datasource was read from —
including the file this very datasource came from — and `force` does not change
that. If you hit it, the path is wrong, not the flag.

## The naming conversation

The server never asks a question; it takes a sensible default and moves on. You
decide when it is worth pulling the user in. Do it when a collision happened,
because that is the moment a better name is cheap:

> "You've already got a `sales` from `~/q1`, so this one came in as `sales_2`.
> Want to call it something that'll still make sense tomorrow — `q2_sales`?"

Do **not** ask before attaching. Attach, see what you got, ask only if the
result is genuinely ambiguous.

## Things that will bite

**Mixed columns.** `attach` warns when a column holds both numbers and text.
This is not pedantry: `avg()` over such a column silently counts the text rows
as zero and keeps them in the denominator, so the average is wrong and nothing
says so. Tell the user, and read the remedy out of the warning rather than
reaching for a familiar one.

There are two shapes, and only the warning knows which you have. When the
values are stored under genuinely different types, `WHERE typeof(col)='integer'`
separates them. When they came from a spreadsheet — the usual case — every value
is stored as text and only some of them *read* as numbers, so `typeof()` answers
`'text'` for all of them and that filter returns the whole column. There the
warning names the offending values, and you exclude them by name:

```
"41 of its values do not read as numbers ('pending') …"
  → WHERE unit_discount NOT IN ('pending')
```

Say what those rows are in the user's terms too — *"41 orders haven't had their
discount set yet, so I left them out of the average"* — because whether to
exclude them, treat them as zero, or go and fill them in is their call, not
yours.

**One fat column instead of the columns they described.** Nothing sniffs the
separator. A file written by a European tool is usually semicolon-separated, and
read at `,` it loads as **a single column whose name is the whole header line**
— `a_b_c` — with a warning saying exactly that and naming the `delimiter`
parameter. `ok: true` comes back and every number you compute from it is wrong,
so this is one to notice rather than to work around:

```
attach(database="/path/sales.csv", delimiter=";")
```

`delimiter` is also a parameter of `create` and of `query(path=…)`. It applies
to character-separated text only — `.csv`, `.tsv`, `.txt` — and on `attach` and
`create` a `delimiter` handed to a `.parquet` or a workbook is **refused**,
naming the suffix, because that is a caller who has misread the file. Do not
retry without it and assume the file was fine; look at what the columns are.
A fixed-width file (`.fwf`) has no separator at all: its column boundaries are
inferred from which character positions are blank on every line, and the warning
says so, because nothing in the file declares them.

**Nested values became JSON text.** A JSON or XML column holding a structure
comes back as `TEXT` carrying exactly what was in the file, and the warning names
the columns. Reach into it with `json_extract(addr, '$.city')` rather than telling
the user the field is missing — nothing was lost.

**Ten datasources, and the oldest is evicted.** Check `evicted` in every
`attach` response. `detach` what you have finished with rather than letting the
limit choose for you. (Eviction is automatic and whole-datasource; it has nothing
to do with the `drop` verb, which removes one table or index you name.)

**Read-only by default.** Anything that came from outside — a SQLite or DuckDB
file someone else made, a database URL — cannot be written to unless it was
attached with `writable=true`. A database built from a flat file is yours and is
always writable.

**A format whose library is missing is refused by name.** The message says which
extra installs it (`pip install 'localdata-mcp[parquet]'`). That is an install
step for the user, not something to route around by asking for a different file.

**The same file twice is refused**, naming the datasource already holding it.
That is not an error to work around — go and use the one that is open.

**A JSON, YAML or XML file with two tables in it is refused, naming both.** A
workbook's sheets all land, but a document is different: an array under a key is
a table only by guesswork, so one candidate loads with a note saying which key it
came from, and two are refused rather than one being picked. Take that back to
the user with the two names — *"that file has a `customers` list and an `orders`
list; which did you want, or shall we do both as separate files?"* — because the
answer is theirs and splitting the file is the fix.

**Dates.** A column of ISO 8601 dates is rewritten into one canonical UTC
spelling and stays text; `attach` and `info` mark it
`"temporal": "iso8601_utc", "normalized": "UTC"`. It orders, ranges and joins
correctly as it stands — do not reach for a conversion. Two things to tell the
user about:

- **The original offset is gone.** `+00:00` and `-05:00` now compare equal,
  which is the point, but if they need the local time they wrote, it has to be
  in a column of its own.
- **One spelling for the whole column.** A single value carrying a time takes
  every other value to `…T00:00:00Z`, because a column written two ways does not
  sort as one.

A date column in no recognised standard is **left exactly as it was**, and the
warning names the values. That column compares alphabetically, not
chronologically, so `'30.11.2023'` sorts after `'01.03.2025'` and `max()` returns
the earliest date. Say so before quoting any number that came out of an
`ORDER BY` or a `max()` over it. Unix timestamps are integers and are left
untouched; integer comparison is already chronological.

**A third case, and it is the one that will catch you.** All of the above is
about dates that arrive as text. A `.parquet`, `.feather`, `.orc`, `.xlsx`,
`.xlsm` or `.ods` column that is a real timestamp in the source arrives as an
integer instead, marked `"temporal": "timestamp", "unit":
"nanoseconds_since_epoch"`, and **no warning is raised about it**. Ordering and
`max()` are right; comparing it to a date string is not — `WHERE d >
'2024-03-02'` compares an integer to text and comes back `"ok": true` with zero
rows. Read the column list from `attach` or `info` before writing a date
predicate: if `unit` says nanoseconds, compare against a tick value or convert.
The mark is also lost on export, so a column you wrote out with `query(path=…)`
or `save()` and attached again is a bare `INTEGER` with nothing to tell you.

## Checking your own work

Before reporting a number:

- Did any warning come back from `attach` about the columns it touches? A
  nanosecond-timestamp column is the one problem that raises none, so check the
  column's own `temporal` and `unit` fields too.
- If it is a join, was the match complete — and did you say so either way?
- Does the row count make sense against what `info` said was there?

A wrong number delivered confidently is worse than a slow answer.
