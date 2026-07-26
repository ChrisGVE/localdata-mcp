# Flat Data Files

LocalData MCP treats flat files as queryable SQL tables. Connect to a file, then query it with standard SQL using `execute_query`. The connection remains active until you disconnect or the session ends.

## Supported formats

### CSV and TSV

Delimiters are detected automatically. The file contents become a single SQL
table named `data_table` — the name is fixed and does not follow the file name or
the connection name.

```python
connect_database("sales", "csv", "./sales_data.csv")
execute_query("sales", "SELECT region, SUM(amount) FROM data_table GROUP BY region")
```

TSV files use the same connection type:

```python
connect_database("logs", "csv", "./access_log.tsv")
execute_query("logs", "SELECT status_code, COUNT(*) FROM data_table GROUP BY status_code")
```

For files larger than 100 MB, LocalData MCP automatically loads the data into a temporary SQLite database on disk. Queries behave identically — the storage strategy is transparent.

### Excel (.xlsx, .xls)

The connection type is `"excel"` for both `.xlsx` and `.xls`. `"xlsx"` is not a
valid type and is rejected with `Unsupported db_type: xlsx`.

Each worksheet becomes a separate SQL table. Sheet names are sanitized: runs of
spaces and hyphens become a single underscore, any other non-word character
becomes an underscore, and a name that does not start with a letter or underscore
is prefixed with `sheet_`. A sheet named `Q1 Results` becomes the table
`Q1_Results`; a sheet named `2024 Data` becomes `sheet_2024_Data`.

```python
connect_database("report", "excel", "./annual_report.xlsx")
describe_database("report")   # lists all sheets as tables
execute_query("report", "SELECT * FROM Q1_Results LIMIT 20")
```

To load a single sheet, pass its name as the fourth argument to
`connect_database`. There is no query-string form — a path ending in
`?sheet=Name` is treated as a literal file name and fails with "File not found".

```python
connect_database("q1", "excel", "./annual_report.xlsx", "Q1 Results")
execute_query("q1", "SELECT * FROM Q1_Results")
```

Selecting a single sheet does not rename its table: it keeps the sanitized sheet
name, and it is the only table on the connection.

### LibreOffice Calc (.ods)

Behavior is identical to Excel. Each sheet becomes a table with the same name sanitization rules.

```python
connect_database("budget", "ods", "./budget.ods")
execute_query("budget", "SELECT category, SUM(amount) FROM Expenses GROUP BY category")
```

### Apple Numbers (.numbers)

Behavior is identical to Excel and ODS. Each sheet becomes a table.

```python
connect_database("plan", "numbers", "./project_plan.numbers")
describe_database("plan")
execute_query("plan", "SELECT milestone, due_date FROM Timeline WHERE status = 'pending'")
```

### Parquet, Feather, and Arrow

These binary columnar formats are read directly and queried with SQL. They are well-suited to large analytical datasets because column pruning reduces memory usage when only a subset of columns are selected.

```python
connect_database("events", "parquet", "./events.parquet")
execute_query("events", "SELECT event_type, COUNT(*) FROM data_table GROUP BY event_type ORDER BY 2 DESC")
```

```python
connect_database("features", "feather", "./model_features.feather")
execute_query("features", "SELECT * FROM data_table WHERE label = 1 LIMIT 1000")
```

Arrow IPC files use the same pattern with type `"arrow"`.

### HDF5

HDF5 files store datasets in a hierarchical group structure. Each top-level dataset is exposed as a SQL table.

```python
connect_database("experiment", "hdf5", "./results.h5")
describe_database("experiment")   # shows available datasets
execute_query("experiment", "SELECT * FROM measurements WHERE temperature > 37.5")
```

## Available tools

All flat file formats share the same tool set.

| Tool | Description |
|---|---|
| `connect_database(name, type, path)` | Open a file and register it under `name`. A fourth argument selects one sheet of a spreadsheet |
| `disconnect_database(name)` | Close the connection and release resources |
| `list_databases()` | Show all active connections and their types |
| `describe_database(name)` | List tables (sheets/datasets) and row counts |
| `describe_table(name, table)` | Show column names and types for a table |
| `find_table(table_name)` | Report which connections hold a table of that name. It searches every active connection and takes no connection argument |
| `execute_query(name, sql)` | Run a SQL SELECT statement |
| `next_chunk(query_id, start_row, chunk_size)` | Retrieve a further slice of a buffered result |

## Large result handling

When a result is large enough that returning it whole would be wasteful,
LocalData MCP returns a first chunk and buffers the remainder. The chunk size is
derived from the estimated size of the result, capped at 2,000 rows once the
estimate passes 10 MB and at 500 rows past 100 MB.

The response carries the rows under `data` and the bookkeeping under `metadata`:

```python
result = execute_query("sales", "SELECT * FROM data_table")
result["metadata"]["total_rows"]    # 200000 — the full result, not the chunk
result["metadata"]["showing_rows"]  # "1-2000"
result["metadata"]["query_id"]      # "sales_1784566779_95f7"
result["data"]                      # the first 2000 rows
```

There is no `has_more` flag. Compare `showing_rows` against `total_rows`, or read
the `pagination` block, which hands back the exact call to make next:

```python
result["pagination"]["get_all_remaining"]
# "next_chunk(query_id='sales_1784566779_95f7', start_row=2001, chunk_size='all')"
```

`next_chunk` is keyed by the query, not by the connection, and takes three
arguments — the `query_id`, the 1-based row to start at, and a chunk size as a
string. Pass `"all"` for the rest of the result:

```python
more = next_chunk(result["metadata"]["query_id"], 2001, "1000")
more["metadata"]["showing_rows"]   # "2001-3000"
more["data"]                       # the next 1000 rows
```

Each response repeats the `pagination` block with the following call already
filled in, so a loop can follow it until `showing_rows` reaches `total_rows`.
Buffers expire on their own; `clear_streaming_buffer(query_id)` releases one
early.

## Notes

- File access is restricted to the current working directory and its subdirectories. Paths containing `../` are rejected.
- SQL queries are read-only. `INSERT`, `UPDATE`, `DELETE`, and `DROP` are not permitted.
- Column names are taken directly from the file header row. Names with special characters may need quoting in SQL: `SELECT "unit price" FROM data_table`.
- For CSV files without a header row, columns are named `col0`, `col1`, and so on.
