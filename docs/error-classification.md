# Error classification

LocalData MCP classifies every database error into a structured response that
tells LLM agents *what went wrong*, *whether it makes sense to retry*, and
*what to do next*. Instead of parsing raw driver messages, agents receive a
uniform JSON object they can act on programmatically.

## Error categories

Every classified error carries one of these categories:

| Category | Value | Description | Example |
|---|---|---|---|
| `AUTH_ERROR` | `auth_error` | Authentication or authorization failure | Wrong password, missing privileges |
| `SCHEMA_ERROR` | `schema_error` | Missing or invalid database objects | Non-existent table or column |
| `SYNTAX_ERROR` | `syntax_error` | Malformed SQL | Typo in a keyword, unbalanced parentheses |
| `TRANSIENT_ERROR` | `transient_error` | Temporary failure that may resolve itself | Lock timeout, deadlock, busy database |
| `RESOURCE_ERROR` | `resource_error` | System resource exhaustion | Out of memory, disk full |
| `CONSTRAINT_ERROR` | `constraint_error` | Data integrity violation | Duplicate key, foreign key mismatch |
| `CONNECTION_ERROR` | `connection_error` | Unable to reach the database | Network unreachable, server down |
| `QUERY_EXECUTION` | `query_execution` | General execution failure (fallback) | Division by zero, type mismatch |

## Retryability

The `is_retryable` flag signals whether repeating the same operation is likely
to succeed:

| Retryable | Categories | Recommended action |
|---|---|---|
| **Yes** | `TRANSIENT_ERROR`, `CONNECTION_ERROR` | Wait briefly and retry (with backoff) |
| **No** | All others | Fix the query, schema, credentials, or data before retrying |

Agents should treat `is_retryable: false` as a hard stop for the current
request. Retrying a syntax error or constraint violation will always produce
the same failure.

## Structured error response format

```json
{
  "error": true,
  "error_type": "schema_error",
  "is_retryable": false,
  "message": "no such table: orders",
  "suggestion": "Verify table/column names in the SQLite database.",
  "database_error_code": null,
  "database": "sales.db"
}
```

| Field | Type | Description |
|---|---|---|
| `error` | `bool` | Always `true` for error responses |
| `error_type` | `string` | One of the category values listed above |
| `is_retryable` | `bool` | Whether the agent should retry |
| `message` | `string` | Human-readable description of the error |
| `suggestion` | `string` | Actionable next step |
| `database_error_code` | `string \| null` | Backend-specific code (SQLSTATE, errno) when available |
| `database` | `string \| null` | Name of the database where the error occurred |

## Database-specific mappings

Each supported backend has a dedicated mapper that uses backend-native signals
(message patterns, SQLSTATE codes, error numbers, or exception class names) to
classify errors. When no backend-specific mapper matches, a generic
keyword-based fallback is used.

### SQLite

SQLite errors are classified by scanning the exception message for known
substrings:

| Pattern | Category |
|---|---|
| `authorization`, `readonly` | `AUTH_ERROR` |
| `locked`, `busy` | `TRANSIENT_ERROR` |
| `disk`, `full`, `no space` | `RESOURCE_ERROR` |
| `constraint`, `unique`, `foreign key`, `not null` | `CONSTRAINT_ERROR` |
| `no such table`, `no such column` | `SCHEMA_ERROR` |
| `syntax`, `near` | `SYNTAX_ERROR` |

### PostgreSQL

PostgreSQL errors are classified by their SQLSTATE code (`pgcode`):

| SQLSTATE prefix | Category | Notes |
|---|---|---|
| `42601` | `SYNTAX_ERROR` | Exact match for syntax errors |
| `42xxx` (other) | `SCHEMA_ERROR` | Undefined table, column, etc. |
| `28xxx` | `AUTH_ERROR` | Invalid authorization |
| `08xxx` | `CONNECTION_ERROR` | Connection exception (retryable) |
| `23xxx` | `CONSTRAINT_ERROR` | Integrity constraint violation |
| `40xxx` | `TRANSIENT_ERROR` | Transaction rollback (retryable) |
| `53xxx` | `RESOURCE_ERROR` | Insufficient resources |

### MySQL / MariaDB

MySQL errors are classified by their numeric error code (`errno`):

| Error code | Category | Notes |
|---|---|---|
| 1044, 1045 | `AUTH_ERROR` | Access denied |
| 1146, 1054 | `SCHEMA_ERROR` | Unknown table or column |
| 1064 | `SYNTAX_ERROR` | SQL syntax error |
| 1205 | `TRANSIENT_ERROR` | Lock wait timeout (retryable) |
| 1114 | `RESOURCE_ERROR` | Table is full |
| 1062, 1452 | `CONSTRAINT_ERROR` | Duplicate entry, FK violation |
| 2002, 2003, 2006 | `CONNECTION_ERROR` | Connection failures (retryable) |

### DuckDB

DuckDB errors are classified by the Python exception class name:

| Exception class | Category |
|---|---|
| `ParserException` | `SYNTAX_ERROR` |
| `CatalogException` | `SCHEMA_ERROR` |
| `BinderException` | `SCHEMA_ERROR` |
| `ConstraintException` | `CONSTRAINT_ERROR` |
| `OutOfMemoryException` | `RESOURCE_ERROR` |
| `IOException` (memory/space) | `RESOURCE_ERROR` |
| `IOException` (other) | `CONNECTION_ERROR` |

## Integration guide

To add a mapper for a new database backend:

1. Create a class that inherits from `DatabaseErrorMapper` and implements
   `map_error(self, exception) -> StructuredErrorResponse`.

2. Register the mapper with `ErrorMapperRegistry`:

```python
from localdata_mcp.error_classification import (
    DatabaseErrorMapper,
    ErrorMapperRegistry,
    StructuredErrorResponse,
)
from localdata_mcp.error_handler import ErrorCategory


class MyDBErrorMapper(DatabaseErrorMapper):
    def map_error(self, exception: Exception) -> StructuredErrorResponse:
        msg = str(exception).lower()
        if "duplicate" in msg:
            return StructuredErrorResponse(
                error_type=ErrorCategory.CONSTRAINT_ERROR,
                message=str(exception),
                suggestion="Check for duplicate values.",
            )
        # Fall back to the generic mapper for unrecognised errors.
        from localdata_mcp.error_classification import GenericDatabaseErrorMapper
        return GenericDatabaseErrorMapper().map_error(exception)


ErrorMapperRegistry.register("mydb", MyDBErrorMapper())
```

3. All calls to `classify_error(exc, db_type="mydb")` will now use the new
   mapper automatically.

## Helper functions

Three convenience functions in `localdata_mcp.error_classification` cover the
most common agent needs:

### `classify_error(exception, db_type="generic")`

Returns the full `StructuredErrorResponse` for an exception. Use this when the
agent needs all fields (category, retryability, suggestion, error code).

```python
from localdata_mcp.error_classification import classify_error

resp = classify_error(exc, db_type="sqlite")
print(resp.error_type)   # ErrorCategory.SYNTAX_ERROR
print(resp.suggestion)   # "Review the SQL statement for syntax errors."
```

### `is_error_retryable(exception, db_type="generic")`

Returns `True` if the error is transient and worth retrying. A shorthand for
`classify_error(...).is_retryable`.

```python
from localdata_mcp.error_classification import is_error_retryable

if is_error_retryable(exc, db_type="postgresql"):
    # back off and retry
    ...
```

### `get_error_suggestion(exception, db_type="generic")`

Returns the actionable suggestion string. A shorthand for
`classify_error(...).suggestion`.

```python
from localdata_mcp.error_classification import get_error_suggestion

hint = get_error_suggestion(exc, db_type="mysql")
```

All three functions use the same classification pipeline internally, so their
results are always consistent for the same exception and database type.
