---
name: local-data
description: Answer questions about local data files and SQLite databases with SQL — attach a spreadsheet or CSV, look one file up against another, check the match is complete, and keep the result. Use whenever someone points at a data file and asks a question about what is in it.
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
stops. If the work is worth keeping, say so and offer `save` — do not let
someone spend twenty minutes building something that evaporates.

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

### 2b. "That sheet is called Sheet1"

A workbook's tables arrive under the names the *spreadsheet* chose. Rename
rather than re-reading the file — the rows, types and any index stay put:

```
update(nickname="shop", type="table", name="Sheet1", to="q2_sales")
```

### 3. "Send me the result"

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
| a normal file, any size | `.csv`, `.tsv`, `.jsonl` | written row by row, so size costs nothing |
| something big, for another program | `.parquet` | fastest and smallest of all of them |
| it pasted into a document | `.md` | small results only — it builds the whole table in memory |

`.yaml` is available and is roughly an order of magnitude slower than anything
else here; reach for `.jsonl` instead unless YAML is specifically wanted.

This is also the answer when a result is simply too big to return — say so and
offer it, rather than returning tens of thousands of rows through the
conversation. The path is theirs: ask for it, and treat "the file already
exists" as a question for them, exactly as with `save` below.

### 4. "Keep this"

```
save(nickname, path="/path/analysis.db")
```

Writes the whole database — every table added — to a file they own. It stays
open afterwards. Attaching it again another day is an ordinary `attach`, so it
comes back **read-only** unless they pass `writable=true`.

**The path is theirs, not yours.** Ask for the name rather than inventing one.
If `save` reports the file already exists, that is a question for them, not a
retry for you — say what is in the way and ask. Once they say replace it, pass
`force=true`. Never set it because a first attempt failed.

A file that some attached datasource is sitting on is refused even with
`force`, including the one the slot was built from. If you hit that, the name
is wrong, not the flag.

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

**Ten datasources, and the oldest is dropped.** Check `evicted` in every
`attach` response. `detach` what you have finished with rather than letting the
limit choose for you.

**Read-only by default.** A SQLite file someone else made cannot be written to
unless it was attached with `writable=true`. A database built from a flat file
is yours and is always writable.

**The same file twice is refused**, naming the datasource already holding it.
That is not an error to work around — go and use the one that is open.

**Dates.** A temporal column is stored as integer ticks, and `info` says so.
Format it in SQL when a human is going to read it.

## Checking your own work

Before reporting a number:

- Did any warning come back from `attach` about the columns it touches?
- If it is a join, was the match complete — and did you say so either way?
- Does the row count make sense against what `info` said was there?

A wrong number delivered confidently is worse than a slow answer.
