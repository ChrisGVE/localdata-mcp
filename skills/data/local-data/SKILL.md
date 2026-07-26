---
name: local-data
description: Answer questions about local data files and SQLite databases with SQL — attach a spreadsheet or CSV, look one file up against another, check the match is complete, and keep the result. Use whenever someone points at a data file and asks a question about what is in it.
allowed-tools: mcp__localdata__attach mcp__localdata__detach mcp__localdata__info mcp__localdata__query mcp__localdata__add_table mcp__localdata__drop_table mcp__localdata__save
argument-hint: "<file-path> [and what you want to know]"
---

# Working with local data

The tools are few and mechanical on purpose. What they do not carry is the way
people actually talk about their data — that lives here.

## The one thing to understand first

**Every datasource becomes a database, and a database holds tables.** Even one
lonely CSV. So a question about `sales.csv` becomes SQL naming a table inside a
database:

```sql
SELECT sum(qty) FROM shop.sales
```

`shop` is the nickname of the database; `sales` is the table the file became.
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
| "look up", "match against", "VLOOKUP", "cross-reference" | `add_table` with `join_on`, then a join |
| "the total", "sum it up", "how many" | `sum()`, `count()` |
| "group by region", "break it down by" | `GROUP BY` |
| "filter out", "only show" | `WHERE` |

Answer with the number and what it means, not with the SQL. Show the SQL only
when asked, when the result is surprising, or when they need to trust it.

## The three things people ask for

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
datasource. **Add it to the database that is already open:**

```
add_table(nickname="shop", source="/path/prices.csv", join_on="sku")
```

Two reasons, and the second is the one that bites later:

- One database, so `shop.sales` and `shop.prices` join in plain SQL.
- **`save` writes one database, not a join.** Attach the two files separately
  and you get an answer now and nothing to keep — the moment the session ends,
  the relationship between them is gone. Landing the second file inside the
  first is what makes the lookup survivable.

`join_on` names the column the two files share. Pass it, and the response tells
you whether the match is actually complete.

**Then say what you found, in their terms.** The response gives facts; the
sentence is yours:

> `missing_from_added: {values: ["b", "c"], total: 2}`

means two things in *their* file have no match. Say:

> "Two products in the sales file have no price listed — `b` and `c`. And
> there's a price for `z`, which never appears in sales. Want me to leave those
> out, or treat the missing prices as zero?"

Never say "anti-join". Never hand back the raw payload. And check **both**
directions — the rows in the new file that nothing refers to are usually the
surprise.

### 3. "Keep this"

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
says so. Tell the user, and filter with `WHERE typeof(col)='integer'` or `CAST`.

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
