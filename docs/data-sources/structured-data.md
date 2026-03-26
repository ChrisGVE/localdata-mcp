# Structured Data

LocalData MCP handles JSON, YAML, and TOML files as navigable trees, and XML and INI files as flat SQL
tables. The connection interface is the same for all five formats; the storage model determines which tools
are available after connection.

## Tree storage (JSON, YAML, TOML)

When connected, these files are parsed into a tree where each node is individually addressable. Nodes can
hold key-value properties and any number of child nodes. An LLM agent can navigate, query, edit, and export
the tree without loading the entire file into context.

```python
connect_database("cfg",       "json", "./config.json")
connect_database("settings",  "yaml", "./settings.yaml")
connect_database("pyproject", "toml", "./pyproject.toml")
```

The connection response includes a tree summary: root nodes, maximum depth, and total property count.

### Tree tools

| Tool | Description |
|------|-------------|
| `get_node(name, path)` | View a node's properties and whether it has children |
| `get_children(name, path)` | List child nodes with pagination |
| `set_node(name, path)` | Create a node; missing ancestors are created automatically |
| `delete_node(name, path)` | Remove a node and all its descendants |
| `list_keys(name, path)` | List key-value pairs at a node |
| `get_value(name, path, key)` | Read a single property value |
| `set_value(name, path, key, value)` | Create or update a property; type is inferred automatically |
| `delete_key(name, path, key)` | Remove a property from a node |
| `export_structured(name, format)` | Export the tree as JSON, YAML, or TOML |

The `path` argument uses dot notation to address nodes: `"server"` addresses a top-level node,
`"server.tls"` addresses a child node named `tls` under `server`.

### Navigation

```python
connect_database("cfg", "toml", "./config.toml")

get_node("cfg")               # Root summary: node count, depth, property count
get_node("cfg", "server")     # Properties of the "server" node and child count
get_children("cfg", "server") # Paginated list of child nodes
get_value("cfg", "server", "port")  # → 8080
```

### Editing

```python
set_value("cfg", "server", "port", "9090")

# Dot-notation path; "monitoring" and "alerts" are created if absent
set_value("cfg", "monitoring.alerts", "enabled", "true")

delete_key("cfg", "server", "deprecated_setting")
delete_node("cfg", "monitoring.alerts")  # Removes the node and all children
```

### Format conversion

`export_structured` converts the in-memory tree to any of the three supported formats, regardless of the
source format:

```python
connect_database("cfg", "toml", "./config.toml")
export_structured("cfg", "json")   # Emit as JSON
export_structured("cfg", "yaml")   # Emit as YAML
```

This lets an agent convert between TOML, JSON, and YAML without writing any conversion logic.

### Type inference

`set_value` infers the type of the supplied string before storing it:

- `"true"` / `"false"` → boolean
- Numeric strings → integer or float
- `"null"` → null
- Anything else → string
- Lists and dicts must be supplied as JSON-serialized strings; they are stored as-is and returned in that
  form.

## XML and INI (flat handling)

XML and INI files connect with the same call but are stored as flat SQL tables rather than trees. After
connection, use the standard database tools (`execute_query`, `describe_table`, and so on).

```python
connect_database("config_xml", "xml",  "./config.xml")
connect_database("app_ini",    "ini",  "./app.ini")
```

### XML table layout

Each XML element becomes a row. The columns are:

| Column | Content |
|--------|---------|
| `id` | Auto-assigned integer row identifier |
| `tag` | Element tag name |
| `text` | Text content of the element, if any |
| `parent_id` | Row identifier of the parent element (`NULL` for the root) |
| `attrs_*` | One column per attribute, prefixed with `attrs_` |

Querying is the same as for any SQL-backed source:

```python
execute_query("config_xml", "SELECT tag, text FROM data WHERE attrs_enabled = 'true'")
```

### INI table layout

Each key-value pair becomes a row:

| Column | Content |
|--------|---------|
| `section` | Section name from the INI file |
| `key` | Key within that section |
| `value` | Value as a string |

```python
execute_query("app_ini", "SELECT value FROM data WHERE section = 'database' AND key = 'host'")
```

### Available tools for XML and INI

Because these formats map to SQL tables, all standard database tools apply:

- `describe_database(name)` — list tables and row counts
- `describe_table(name, table)` — column names and types
- `execute_query(name, sql)` — run any read query
- `find_table(name, keyword)` — search table names by keyword

The tree tools (`get_node`, `get_value`, etc.) are not available for XML or INI connections.
