# Flat Data Files

LocalData MCP treats flat files as queryable SQL tables. Connect to a file, then query it with standard SQL using `execute_query`. The connection remains active until you disconnect or the session ends.

## Supported formats

### CSV and TSV

Delimiters are detected automatically. The file contents become a single SQL table named `data`.

```python
connect_database("sales", "csv", "./sales_data.csv")
execute_query("sales", "SELECT region, SUM(amount) FROM data GROUP BY region")
```

TSV files use the same connection type:

```python
connect_database("logs", "csv", "./access_log.tsv")
execute_query("logs", "SELECT status_code, COUNT(*) FROM data GROUP BY status_code")
```

For files larger than 100 MB, LocalData MCP automatically loads the data into a temporary SQLite database on disk. Queries behave identically — the storage strategy is transparent.

### Excel (.xlsx, .xls)

Each worksheet becomes a separate SQL table. Sheet names are sanitized: spaces become underscores, and names starting with a digit get an underscore prefix. A sheet named `Q1 Results` becomes the table `Q1_Results`; a sheet named `2024 Data` becomes `_2024_Data`.

```python
connect_database("report", "xlsx", "./annual_report.xlsx")
describe_database("report")   # lists all sheets as tables
execute_query("report", "SELECT * FROM Q1_Results LIMIT 20")
```

To connect to a single sheet, append `?sheet=SheetName` to the path:

```python
connect_database("q1", "xlsx", "./annual_report.xlsx?sheet=Q1 Results")
execute_query("q1", "SELECT * FROM data")
```

When a single sheet is selected this way, the table is named `data` rather than the sheet name.

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
execute_query("events", "SELECT event_type, COUNT(*) FROM data GROUP BY event_type ORDER BY 2 DESC")
```

```python
connect_database("features", "feather", "./model_features.feather")
execute_query("features", "SELECT * FROM data WHERE label = 1 LIMIT 1000")
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
| `connect_database(name, type, path)` | Open a file and register it under `name` |
| `disconnect_database(name)` | Close the connection and release resources |
| `list_databases()` | Show all active connections and their types |
| `describe_database(name)` | List tables (sheets/datasets) and row counts |
| `describe_table(name, table)` | Show column names and types for a table |
| `find_table(name, pattern)` | Search table names by substring or pattern |
| `execute_query(name, sql)` | Run a SQL SELECT statement |
| `next_chunk(name, buffer_id)` | Retrieve the next chunk from a buffered result |

## Large result handling

When a query returns more than 100 rows, LocalData MCP returns the first chunk immediately and buffers the remainder. The response includes a `buffer_id` and a `has_more` flag.

```python
result = execute_query("sales", "SELECT * FROM data")
# result["has_more"] == True
# result["buffer_id"] == "buf_abc123"
# result["rows"] contains the first chunk

while result.get("has_more"):
    result = next_chunk("sales", result["buffer_id"])
    # process result["rows"]
```

Call `next_chunk` repeatedly until `has_more` is `False`. Buffers are released automatically when exhausted. If you stop consuming a buffer early, call `disconnect_database` to release the resources, or the buffer will expire after the session ends.

## Notes

- File access is restricted to the current working directory and its subdirectories. Paths containing `../` are rejected.
- SQL queries are read-only. `INSERT`, `UPDATE`, `DELETE`, and `DROP` are not permitted.
- Column names are taken directly from the file header row. Names with special characters may need quoting in SQL: `SELECT "unit price" FROM data`.
- For CSV files without a header row, columns are named `col0`, `col1`, and so on.
