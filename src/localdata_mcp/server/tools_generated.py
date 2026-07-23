"""MACHINE-WRITTEN by localdata_mcp.nexus.contract.generators.wrapper — DO NOT EDIT.

FastMCP registration wrappers for every registered ToolSpec
(ARCHITECTURE.md 6.1 artifacts 1+2). Regenerate via
`python -m localdata_mcp.nexus.contract.generate`; hand edits fail CI
through nexus/contract/check_drift.py.
"""

from __future__ import annotations

from typing import Any

from fastmcp import FastMCP

from localdata_mcp.nexus.contract.registry import default_registry
from localdata_mcp.nexus.contract.spec_modules import load_spec_modules
from localdata_mcp.nexus.response.shaping import shaped_call


def register_tools(app: FastMCP) -> None:
    """Register every generated tool wrapper on `app`."""
    load_spec_modules()
    registry = default_registry()

    _impl_ping = registry.lookup("ping").func

    def ping() -> Any:
        """
        Report server liveness with a constant probe response.

        Input shape: NONE (chain endpoint — composes with nothing).
        Output shape: SCALAR.
        Streaming-capable: no.
        """
        arguments: dict[str, Any] = {}
        return shaped_call("ping", _impl_ping, arguments)

    app.tool(ping)

    _impl_probe_table = registry.lookup("probe_table").func

    def probe_table(rows: int) -> Any:
        """
        Produce a small numbered table of squares for pipeline probing.

        Input shape: NONE (chain endpoint — composes with nothing).
        Output shape: TABULAR.
        Streaming-capable: no.

        Args:
            rows: How many rows the probe table carries.
        """
        arguments: dict[str, Any] = {"rows": rows}
        return shaped_call("probe_table", _impl_probe_table, arguments)

    app.tool(probe_table)

    _impl_probe_vector = registry.lookup("probe_vector").func

    def probe_vector(length: int) -> Any:
        """
        Produce an ordered series of triangular numbers for probing.

        Input shape: NONE (chain endpoint — composes with nothing).
        Output shape: VECTOR.
        Streaming-capable: no.

        Args:
            length: How many entries the series carries.
        """
        arguments: dict[str, Any] = {"length": length}
        return shaped_call("probe_vector", _impl_probe_vector, arguments)

    app.tool(probe_vector)

    _impl_probe_matrix = registry.lookup("probe_matrix").func

    def probe_matrix(size: int) -> Any:
        """
        Produce an identity matrix of the requested size for probing.

        Input shape: NONE (chain endpoint — composes with nothing).
        Output shape: MATRIX.
        Streaming-capable: no.

        Args:
            size: Row and column count of the matrix.
        """
        arguments: dict[str, Any] = {"size": size}
        return shaped_call("probe_matrix", _impl_probe_matrix, arguments)

    app.tool(probe_matrix)

    _impl_probe_model = registry.lookup("probe_model").func

    def probe_model(points: int) -> Any:
        """
        Fit a line to a tiny generated sample and report coefficients.

        Input shape: NONE (chain endpoint — composes with nothing).
        Output shape: FITTED_MODEL.
        Streaming-capable: no.

        Args:
            points: Sample size drawn from y = 2n + 1.
        """
        arguments: dict[str, Any] = {"points": points}
        return shaped_call("probe_model", _impl_probe_model, arguments)

    app.tool(probe_model)

    _impl_probe_graph = registry.lookup("probe_graph").func

    def probe_graph(nodes: int) -> Any:
        """
        Produce a path graph with the requested node count for probing.

        Input shape: NONE (chain endpoint — composes with nothing).
        Output shape: GRAPH.
        Streaming-capable: no.

        Args:
            nodes: How many nodes the path graph carries.
        """
        arguments: dict[str, Any] = {"nodes": nodes}
        return shaped_call("probe_graph", _impl_probe_graph, arguments)

    app.tool(probe_graph)

    _impl_probe_geo = registry.lookup("probe_geo").func

    def probe_geo(points: int) -> Any:
        """
        Produce evenly spaced points along the equator for probing.

        Input shape: NONE (chain endpoint — composes with nothing).
        Output shape: GEO.
        Streaming-capable: no.

        Args:
            points: How many geometry-bearing rows to emit.
        """
        arguments: dict[str, Any] = {"points": points}
        return shaped_call("probe_geo", _impl_probe_geo, arguments)

    app.tool(probe_geo)

    _impl_probe_chart = registry.lookup("probe_chart").func

    def probe_chart(points: int) -> Any:
        """
        Build a line-chart specification over computed square values.

        Input shape: NONE (chain endpoint — composes with nothing).
        Output shape: CHART_SPEC.
        Streaming-capable: no.

        Args:
            points: How many x/y pairs the chart spec plots.
        """
        arguments: dict[str, Any] = {"points": points}
        return shaped_call("probe_chart", _impl_probe_chart, arguments)

    app.tool(probe_chart)

    _impl_probe_sink = registry.lookup("probe_sink").func

    def probe_sink(text: str) -> Any:
        """
        Measure a text payload and report its size as a terminal result.

        Input shape: TABULAR.
        Output shape: NONE (chain endpoint — composes with nothing).
        Streaming-capable: no.

        Args:
            text: Payload whose size the sink reports.
        """
        arguments: dict[str, Any] = {"text": text}
        return shaped_call("probe_sink", _impl_probe_sink, arguments)

    app.tool(probe_sink)

    _impl_list_endpoints = registry.lookup("list_endpoints").func

    def list_endpoints() -> Any:
        """
        Enumerate every operator-declared endpoint (SQL, key-value, and graph/tree alike) with its backend kind, posture, and health.

        Input shape: NONE (chain endpoint — composes with nothing).
        Output shape: TABULAR.
        Streaming-capable: no.
        Domain: ingest.
        """
        arguments: dict[str, Any] = {}
        return shaped_call("list_endpoints", _impl_list_endpoints, arguments)

    app.tool(list_endpoints)

    _impl_fetch_chunk = registry.lookup("fetch_chunk").func

    def fetch_chunk(stream_id: str) -> Any:
        """
        Retrieve the next servable chunk of a streamed result (cursor semantics: a served chunk leaves the buffer). Once the source is exhausted the answer reports the final total and the stream closes.

        Input shape: NONE (chain endpoint — composes with nothing).
        Output shape: TABULAR.
        Streaming-capable: yes.
        Domain: ingest.

        Args:
            stream_id: The stream reference a large result returned.
        """
        arguments: dict[str, Any] = {"stream_id": stream_id}
        return shaped_call("fetch_chunk", _impl_fetch_chunk, arguments)

    app.tool(fetch_chunk)

    _impl_close_stream = registry.lookup("close_stream").func

    def close_stream(stream_id: str) -> Any:
        """
        Release a streamed result ahead of the idle TTL, returning its buffer memory (and any pinned connection) immediately. Idempotent.

        Input shape: NONE (chain endpoint — composes with nothing).
        Output shape: SCALAR.
        Streaming-capable: no.
        Domain: ingest.

        Args:
            stream_id: The stream reference to release.
        """
        arguments: dict[str, Any] = {"stream_id": stream_id}
        return shaped_call("close_stream", _impl_close_stream, arguments)

    app.tool(close_stream)

    _impl_query = registry.lookup("query").func

    def query(endpoint: str, sql: str) -> Any:
        """
        Run a read-only SQL statement against a declared endpoint and return the rows (guarded: allow-list validated, any posture).

        Input shape: NONE (chain endpoint — composes with nothing).
        Output shape: TABULAR.
        Streaming-capable: yes.
        Domain: ingest.

        Args:
            endpoint: The operator-declared endpoint name.
            sql: One read-only SQL statement.
        """
        arguments: dict[str, Any] = {"endpoint": endpoint, "sql": sql}
        return shaped_call("query", _impl_query, arguments)

    app.tool(query)

    _impl_write_query = registry.lookup("write_query").func

    def write_query(endpoint: str, sql: str) -> Any:
        """
        Run a mutating SQL statement (INSERT/UPDATE/DELETE or a write-side local-file construct) against a declared read-write endpoint (guarded: posture enforced, allow-list validated).

        Input shape: NONE (chain endpoint — composes with nothing).
        Output shape: TABULAR.
        Streaming-capable: no.
        Domain: ingest.

        Args:
            endpoint: The operator-declared endpoint name.
            sql: One mutating SQL statement.
        """
        arguments: dict[str, Any] = {"endpoint": endpoint, "sql": sql}
        return shaped_call("write_query", _impl_write_query, arguments)

    app.tool(write_query)

    _impl_read_file = registry.lookup("read_file").func

    def read_file(path: str, format: str | None = None) -> Any:
        """
        Read a local data file (14 core formats: CSV, TSV, JSON, YAML, TOML, INI, XML, Excel, ODS, Numbers, Parquet, Feather, Arrow, HDF5) inside the operator's allowed paths.

        Input shape: NONE (chain endpoint — composes with nothing).
        Output shape: TABULAR.
        Streaming-capable: yes.
        Domain: ingest.

        Args:
            path: The file path (must lie inside allowed_paths).
            format (optional): The format name, or "auto" to infer from the suffix.
        """
        arguments: dict[str, Any] = {"path": path}
        if format is not None:
            arguments["format"] = format
        return shaped_call("read_file", _impl_read_file, arguments)

    app.tool(read_file)

    _impl_query_file = registry.lookup("query_file").func

    def query_file(path: str, sql: str) -> Any:
        """
        Run a read-only SQL statement over a local SQLite or DuckDB file (ad-hoc, contained, read-only unless the operator grants otherwise; results are read whole under the memory budget).

        Input shape: NONE (chain endpoint — composes with nothing).
        Output shape: TABULAR.
        Streaming-capable: yes.
        Domain: ingest.

        Args:
            path: The database file path (inside allowed_paths).
            sql: One SQL statement.
        """
        arguments: dict[str, Any] = {"path": path, "sql": sql}
        return shaped_call("query_file", _impl_query_file, arguments)

    app.tool(query_file)

    _impl_get_value = registry.lookup("get_value").func

    def get_value(endpoint: str, path: str, key: str) -> Any:
        """
        Get one property value from a node of a declared kv, tree, or graph store endpoint (path addresses the node; node_id for graph stores).

        Input shape: NONE (chain endpoint — composes with nothing).
        Output shape: SCALAR.
        Streaming-capable: no.
        Domain: ingest.

        Args:
            endpoint: The operator-declared store endpoint name.
            path: The node's dot-path (or graph node_id).
            key: The property key.
        """
        arguments: dict[str, Any] = {"endpoint": endpoint, "path": path, "key": key}
        return shaped_call("get_value", _impl_get_value, arguments)

    app.tool(get_value)

    _impl_set_value = registry.lookup("set_value").func

    def set_value(endpoint: str, path: str, key: str, value: str, value_type: str | None = None) -> Any:
        """
        Set (upsert) one property on a node of a declared read-write store endpoint, auto-creating the node; string values infer their type unless value_type names one.

        Input shape: NONE (chain endpoint — composes with nothing).
        Output shape: SCALAR.
        Streaming-capable: no.
        Domain: ingest.

        Args:
            endpoint: The operator-declared store endpoint name.
            path: The node's dot-path (or graph node_id).
            key: The property key.
            value: The value to store.
            value_type (optional): Optional explicit type: string, integer, float, boolean, array, null, or datetime.
        """
        arguments: dict[str, Any] = {"endpoint": endpoint, "path": path, "key": key, "value": value}
        if value_type is not None:
            arguments["value_type"] = value_type
        return shaped_call("set_value", _impl_set_value, arguments)

    app.tool(set_value)

    _impl_delete_key = registry.lookup("delete_key").func

    def delete_key(endpoint: str, path: str, key: str) -> Any:
        """
        Delete one property from a node of a declared read-write store endpoint.

        Input shape: NONE (chain endpoint — composes with nothing).
        Output shape: SCALAR.
        Streaming-capable: no.
        Domain: ingest.

        Args:
            endpoint: The operator-declared store endpoint name.
            path: The node's dot-path (or graph node_id).
            key: The property key.
        """
        arguments: dict[str, Any] = {"endpoint": endpoint, "path": path, "key": key}
        return shaped_call("delete_key", _impl_delete_key, arguments)

    app.tool(delete_key)

    _impl_list_keys = registry.lookup("list_keys").func

    def list_keys(endpoint: str, path: str, offset: int | None = None, limit: int | None = None) -> Any:
        """
        List a node's properties (key, value, value_type) from a declared kv, tree, or graph store endpoint, key-ordered.

        Input shape: NONE (chain endpoint — composes with nothing).
        Output shape: TABULAR.
        Streaming-capable: no.
        Domain: ingest.

        Args:
            endpoint: The operator-declared store endpoint name.
            path: The node's dot-path (or graph node_id).
            offset (optional): Pagination offset (default 0).
            limit (optional): Optional page size; omitted serves all rows.
        """
        arguments: dict[str, Any] = {"endpoint": endpoint, "path": path}
        if offset is not None:
            arguments["offset"] = offset
        if limit is not None:
            arguments["limit"] = limit
        return shaped_call("list_keys", _impl_list_keys, arguments)

    app.tool(list_keys)

    _impl_get_node = registry.lookup("get_node").func

    def get_node(endpoint: str, path: str | None = None) -> Any:
        """
        Get node details from a declared tree or graph store endpoint (counts and addressing; properties via list_keys); omit path for the store-level summary.

        Input shape: NONE (chain endpoint — composes with nothing).
        Output shape: SCALAR.
        Streaming-capable: no.
        Domain: ingest.

        Args:
            endpoint: The operator-declared store endpoint name.
            path (optional): The node's dot-path (or graph node_id); omit for a summary.
        """
        arguments: dict[str, Any] = {"endpoint": endpoint}
        if path is not None:
            arguments["path"] = path
        return shaped_call("get_node", _impl_get_node, arguments)

    app.tool(get_node)

    _impl_set_node = registry.lookup("set_node").func

    def set_node(endpoint: str, path: str, label: str | None = None) -> Any:
        """
        Create a node on a declared read-write store endpoint: tree kinds create the path (and missing ancestors), graph kinds upsert the node with an optional label.

        Input shape: NONE (chain endpoint — composes with nothing).
        Output shape: SCALAR.
        Streaming-capable: no.
        Domain: ingest.

        Args:
            endpoint: The operator-declared store endpoint name.
            path: The node's dot-path (or graph node_id).
            label (optional): Optional label (graph stores only).
        """
        arguments: dict[str, Any] = {"endpoint": endpoint, "path": path}
        if label is not None:
            arguments["label"] = label
        return shaped_call("set_node", _impl_set_node, arguments)

    app.tool(set_node)

    _impl_delete_node = registry.lookup("delete_node").func

    def delete_node(endpoint: str, path: str) -> Any:
        """
        Delete a node from a declared read-write store endpoint: tree kinds delete the whole subtree (properties cascade), graph kinds cascade the node's edges and properties.

        Input shape: NONE (chain endpoint — composes with nothing).
        Output shape: SCALAR.
        Streaming-capable: no.
        Domain: ingest.

        Args:
            endpoint: The operator-declared store endpoint name.
            path: The node's dot-path (or graph node_id).
        """
        arguments: dict[str, Any] = {"endpoint": endpoint, "path": path}
        return shaped_call("delete_node", _impl_delete_node, arguments)

    app.tool(delete_node)

    _impl_get_children = registry.lookup("get_children").func

    def get_children(endpoint: str, path: str | None = None, offset: int | None = None, limit: int | None = None) -> Any:
        """
        List direct children of a tree-store node (root nodes when path is omitted), name-ordered with counts.

        Input shape: NONE (chain endpoint — composes with nothing).
        Output shape: TABULAR.
        Streaming-capable: no.
        Domain: ingest.

        Args:
            endpoint: The operator-declared store endpoint name.
            path (optional): The parent's dot-path; omit for root nodes.
            offset (optional): Pagination offset (default 0).
            limit (optional): Optional page size; omitted serves all rows.
        """
        arguments: dict[str, Any] = {"endpoint": endpoint}
        if path is not None:
            arguments["path"] = path
        if offset is not None:
            arguments["offset"] = offset
        if limit is not None:
            arguments["limit"] = limit
        return shaped_call("get_children", _impl_get_children, arguments)

    app.tool(get_children)

    _impl_move_node = registry.lookup("move_node").func

    def move_node(endpoint: str, path: str, new_parent: str | None = None) -> Any:
        """
        Move a tree-store node and its whole subtree under a new parent (or to root level when new_parent is omitted).

        Input shape: NONE (chain endpoint — composes with nothing).
        Output shape: SCALAR.
        Streaming-capable: no.
        Domain: ingest.

        Args:
            endpoint: The operator-declared store endpoint name.
            path: The node's dot-path.
            new_parent (optional): Target parent path; omit for root.
        """
        arguments: dict[str, Any] = {"endpoint": endpoint, "path": path}
        if new_parent is not None:
            arguments["new_parent"] = new_parent
        return shaped_call("move_node", _impl_move_node, arguments)

    app.tool(move_node)

    _impl_get_neighbors = registry.lookup("get_neighbors").func

    def get_neighbors(endpoint: str, node_id: str, direction: str | None = None, offset: int | None = None, limit: int | None = None) -> Any:
        """
        List a graph node's neighbors with edge label/weight and direction ('in', 'out', or 'both').

        Input shape: NONE (chain endpoint — composes with nothing).
        Output shape: TABULAR.
        Streaming-capable: no.
        Domain: ingest.

        Args:
            endpoint: The operator-declared graph endpoint name.
            node_id: The node whose neighbors to list.
            direction (optional): 'in', 'out', or 'both' (default).
            offset (optional): Pagination offset (default 0).
            limit (optional): Optional page size; omitted serves all rows.
        """
        arguments: dict[str, Any] = {"endpoint": endpoint, "node_id": node_id}
        if direction is not None:
            arguments["direction"] = direction
        if offset is not None:
            arguments["offset"] = offset
        if limit is not None:
            arguments["limit"] = limit
        return shaped_call("get_neighbors", _impl_get_neighbors, arguments)

    app.tool(get_neighbors)

    _impl_get_edges = registry.lookup("get_edges").func

    def get_edges(endpoint: str, node_id: str | None = None, offset: int | None = None, limit: int | None = None) -> Any:
        """
        List a graph store's edges (source, target, label, weight), optionally filtered to those touching one node.

        Input shape: NONE (chain endpoint — composes with nothing).
        Output shape: TABULAR.
        Streaming-capable: no.
        Domain: ingest.

        Args:
            endpoint: The operator-declared graph endpoint name.
            node_id (optional): Optional node filter.
            offset (optional): Pagination offset (default 0).
            limit (optional): Optional page size; omitted serves all rows.
        """
        arguments: dict[str, Any] = {"endpoint": endpoint}
        if node_id is not None:
            arguments["node_id"] = node_id
        if offset is not None:
            arguments["offset"] = offset
        if limit is not None:
            arguments["limit"] = limit
        return shaped_call("get_edges", _impl_get_edges, arguments)

    app.tool(get_edges)

    _impl_add_edge = registry.lookup("add_edge").func

    def add_edge(endpoint: str, source: str, target: str, label: str | None = None, weight: float | None = None) -> Any:
        """
        Add (or re-weight) a directed edge on a declared read-write graph endpoint, auto-creating missing nodes; returns the harvested integrity warnings (self-loop, duplicates, contradictory reverse edge).

        Input shape: NONE (chain endpoint — composes with nothing).
        Output shape: SCALAR.
        Streaming-capable: no.
        Domain: ingest.

        Args:
            endpoint: The operator-declared graph endpoint name.
            source: The edge's source node id.
            target: The edge's target node id.
            label (optional): Optional edge label.
            weight (optional): Optional edge weight.
        """
        arguments: dict[str, Any] = {"endpoint": endpoint, "source": source, "target": target}
        if label is not None:
            arguments["label"] = label
        if weight is not None:
            arguments["weight"] = weight
        return shaped_call("add_edge", _impl_add_edge, arguments)

    app.tool(add_edge)

    _impl_remove_edge = registry.lookup("remove_edge").func

    def remove_edge(endpoint: str, source: str, target: str, label: str | None = None) -> Any:
        """
        Remove a directed edge (and its properties) from a declared read-write graph endpoint; warns when a node becomes an orphan.

        Input shape: NONE (chain endpoint — composes with nothing).
        Output shape: SCALAR.
        Streaming-capable: no.
        Domain: ingest.

        Args:
            endpoint: The operator-declared graph endpoint name.
            source: The edge's source node id.
            target: The edge's target node id.
            label (optional): Optional edge label (NULL-labeled when omitted).
        """
        arguments: dict[str, Any] = {"endpoint": endpoint, "source": source, "target": target}
        if label is not None:
            arguments["label"] = label
        return shaped_call("remove_edge", _impl_remove_edge, arguments)

    app.tool(remove_edge)

    _impl_find_path = registry.lookup("find_path").func

    def find_path(endpoint: str, source: str, target: str, algorithm: str | None = None) -> Any:
        """
        Find path(s) between two graph nodes: the shortest path, or all simple paths (bounded enumeration).

        Input shape: NONE (chain endpoint — composes with nothing).
        Output shape: SCALAR.
        Streaming-capable: no.
        Domain: ingest.

        Args:
            endpoint: The operator-declared graph endpoint name.
            source: The start node id.
            target: The end node id.
            algorithm (optional): 'shortest' (default) or 'all'.
        """
        arguments: dict[str, Any] = {"endpoint": endpoint, "source": source, "target": target}
        if algorithm is not None:
            arguments["algorithm"] = algorithm
        return shaped_call("find_path", _impl_find_path, arguments)

    app.tool(find_path)

    _impl_get_graph_stats = registry.lookup("get_graph_stats").func

    def get_graph_stats(endpoint: str) -> Any:
        """
        Summary statistics for a declared graph endpoint: node, edge, and property counts plus density.

        Input shape: NONE (chain endpoint — composes with nothing).
        Output shape: SCALAR.
        Streaming-capable: no.
        Domain: ingest.

        Args:
            endpoint: The operator-declared graph endpoint name.
        """
        arguments: dict[str, Any] = {"endpoint": endpoint}
        return shaped_call("get_graph_stats", _impl_get_graph_stats, arguments)

    app.tool(get_graph_stats)

    _impl_describe_database = registry.lookup("describe_database").func

    def describe_database(endpoint: str) -> Any:
        """
        Describe a declared endpoint's schema: SQL kinds return the table catalog (columns, keys, row counts), kv/tree stores their key-space shape, graph stores their node/edge shape, rdf stores their triple shape.

        Input shape: NONE (chain endpoint — composes with nothing).
        Output shape: SCALAR.
        Streaming-capable: no.
        Domain: explore.

        Args:
            endpoint: The operator-declared endpoint name.
        """
        arguments: dict[str, Any] = {"endpoint": endpoint}
        return shaped_call("describe_database", _impl_describe_database, arguments)

    app.tool(describe_database)

    _impl_describe_table = registry.lookup("describe_table").func

    def describe_table(endpoint: str, table: str) -> Any:
        """
        Describe one table of a declared SQL-kind endpoint: columns with types and nullability, primary key, row count.

        Input shape: NONE (chain endpoint — composes with nothing).
        Output shape: SCALAR.
        Streaming-capable: no.
        Domain: explore.

        Args:
            endpoint: The operator-declared endpoint name.
            table: The table name (as the catalog lists it).
        """
        arguments: dict[str, Any] = {"endpoint": endpoint, "table": table}
        return shaped_call("describe_table", _impl_describe_table, arguments)

    app.tool(describe_table)

    _impl_find_table = registry.lookup("find_table").func

    def find_table(endpoint: str, name_pattern: str) -> Any:
        """
        Find tables on a declared SQL-kind endpoint whose names match a glob pattern (e.g. 'sales_*').

        Input shape: NONE (chain endpoint — composes with nothing).
        Output shape: SCALAR.
        Streaming-capable: no.
        Domain: explore.

        Args:
            endpoint: The operator-declared endpoint name.
            name_pattern: A glob pattern matched against table names.
        """
        arguments: dict[str, Any] = {"endpoint": endpoint, "name_pattern": name_pattern}
        return shaped_call("find_table", _impl_find_table, arguments)

    app.tool(find_table)

    _impl_profile_data = registry.lookup("profile_data").func

    def profile_data(endpoint: str | None = None, path: str | None = None, table: str | None = None, query: str | None = None) -> Any:
        """
        Profile a tabular source's data quality: per-column null counts, inferred types, numeric ranges, and cardinality. Address with exactly one of endpoint= or path=; endpoint sources take exactly one of table= or query=.

        Input shape: NONE (chain endpoint — composes with nothing).
        Output shape: SCALAR.
        Streaming-capable: no.
        Domain: explore.

        Args:
            endpoint (optional): The operator-declared endpoint name (exactly one of endpoint/path).
            path (optional): A local file inside allowed_paths (exactly one of endpoint/path).
            table (optional): A table on the endpoint (exactly one of table/query for endpoint sources).
            query (optional): One read-only SQL statement (endpoint sources and local database files).
        """
        arguments: dict[str, Any] = {}
        if endpoint is not None:
            arguments["endpoint"] = endpoint
        if path is not None:
            arguments["path"] = path
        if table is not None:
            arguments["table"] = table
        if query is not None:
            arguments["query"] = query
        return shaped_call("profile_data", _impl_profile_data, arguments)

    app.tool(profile_data)

    _impl_search_data = registry.lookup("search_data").func

    def search_data(query: str, endpoint: str | None = None, path: str | None = None, target: str | None = None, columns: str | None = None, case_sensitive: bool | None = None) -> Any:
        """
        Regex-search a tabular source's cell values. Address with exactly one of endpoint= or path=; target= is the table or SQL statement to search (omit for a document/table file); query= is the search pattern.

        Input shape: NONE (chain endpoint — composes with nothing).
        Output shape: SCALAR.
        Streaming-capable: no.
        Domain: explore.

        Args:
            endpoint (optional): The operator-declared endpoint name (exactly one of endpoint/path).
            path (optional): A local file inside allowed_paths (exactly one of endpoint/path).
            target (optional): What to search: a table name or SQL statement on an endpoint; a SQL statement on a database file; omit for a document/table format file.
            query: The search pattern (a regular expression).
            columns (optional): Comma-separated column names to search (omitted = all).
            case_sensitive (optional): Case-sensitive matching (default true).
        """
        arguments: dict[str, Any] = {"query": query}
        if endpoint is not None:
            arguments["endpoint"] = endpoint
        if path is not None:
            arguments["path"] = path
        if target is not None:
            arguments["target"] = target
        if columns is not None:
            arguments["columns"] = columns
        if case_sensitive is not None:
            arguments["case_sensitive"] = case_sensitive
        return shaped_call("search_data", _impl_search_data, arguments)

    app.tool(search_data)

    _impl_map_categories = registry.lookup("map_categories").func

    def map_categories(column: str, endpoint: str | None = None, path: str | None = None, table: str | None = None, query: str | None = None) -> Any:
        """
        Map one column's categorical values: distinct values with frequencies and a suggested encoding (label vs one-hot) — a report only, nothing is transformed or persisted. Address with exactly one of endpoint= or path=.

        Input shape: NONE (chain endpoint — composes with nothing).
        Output shape: SCALAR.
        Streaming-capable: no.
        Domain: explore.

        Args:
            endpoint (optional): The operator-declared endpoint name (exactly one of endpoint/path).
            path (optional): A local file inside allowed_paths (exactly one of endpoint/path).
            table (optional): A table on the endpoint (exactly one of table/query for endpoint sources).
            query (optional): One read-only SQL statement (endpoint sources and local database files).
            column: The column whose categories to map.
        """
        arguments: dict[str, Any] = {"column": column}
        if endpoint is not None:
            arguments["endpoint"] = endpoint
        if path is not None:
            arguments["path"] = path
        if table is not None:
            arguments["table"] = table
        if query is not None:
            arguments["query"] = query
        return shaped_call("map_categories", _impl_map_categories, arguments)

    app.tool(map_categories)

    _impl_analyze_hypothesis_test = registry.lookup("analyze_hypothesis_test").func

    def analyze_hypothesis_test(endpoint: str | None = None, path: str | None = None, table: str | None = None, query: str | None = None, test_type: str | None = None, column: str | None = None, second_column: str | None = None, group_column: str | None = None, popmean: float | None = None, alpha: float | None = None, alternative: str | None = None) -> Any:
        """
        Run a hypothesis test on an addressed tabular source. test_type auto (default) selects from the supplied columns: group_column= compares two groups, second_column= correlates two columns, column= alone tests normality. Explicit types: ttest_1samp, ttest_ind, ttest_rel, mann_whitney, wilcoxon, chi2, normality, correlation.

        Input shape: TABULAR.
        Output shape: SCALAR.
        Streaming-capable: no.
        Domain: process.

        Args:
            endpoint (optional): The operator-declared endpoint name (exactly one of endpoint/path).
            path (optional): A local file inside allowed_paths (exactly one of endpoint/path).
            table (optional): A table on the endpoint (exactly one of table/query for endpoint sources).
            query (optional): One read-only SQL statement (endpoint sources and local database files).
            test_type (optional): Test to run (default auto — selected from the supplied columns).
            column (optional): The primary value column.
            second_column (optional): Second column for paired/correlation/chi2 tests.
            group_column (optional): Column defining the two groups for two-sample tests.
            popmean (optional): Population mean for ttest_1samp (implementation default 0.0).
            alpha (optional): Significance level for the verdict (implementation default 0.05).
            alternative (optional): Alternative hypothesis: two-sided (default), greater, or less.
        """
        arguments: dict[str, Any] = {}
        if endpoint is not None:
            arguments["endpoint"] = endpoint
        if path is not None:
            arguments["path"] = path
        if table is not None:
            arguments["table"] = table
        if query is not None:
            arguments["query"] = query
        if test_type is not None:
            arguments["test_type"] = test_type
        if column is not None:
            arguments["column"] = column
        if second_column is not None:
            arguments["second_column"] = second_column
        if group_column is not None:
            arguments["group_column"] = group_column
        if popmean is not None:
            arguments["popmean"] = popmean
        if alpha is not None:
            arguments["alpha"] = alpha
        if alternative is not None:
            arguments["alternative"] = alternative
        return shaped_call("analyze_hypothesis_test", _impl_analyze_hypothesis_test, arguments)

    app.tool(analyze_hypothesis_test)

    _impl_analyze_anova = registry.lookup("analyze_anova").func

    def analyze_anova(dependent_var: str, group_var: str, endpoint: str | None = None, path: str | None = None, table: str | None = None, query: str | None = None, alpha: float | None = None) -> Any:
        """
        One-way ANOVA across every group of group_var on an addressed tabular source: F statistic, p-value, eta squared, per-group summary, and Tukey HSD post-hoc when significant.

        Input shape: TABULAR.
        Output shape: SCALAR.
        Streaming-capable: no.
        Domain: process.

        Args:
            endpoint (optional): The operator-declared endpoint name (exactly one of endpoint/path).
            path (optional): A local file inside allowed_paths (exactly one of endpoint/path).
            table (optional): A table on the endpoint (exactly one of table/query for endpoint sources).
            query (optional): One read-only SQL statement (endpoint sources and local database files).
            dependent_var: The numeric outcome column.
            group_var: The column defining the groups.
            alpha (optional): Significance level for the verdict (implementation default 0.05).
        """
        arguments: dict[str, Any] = {"dependent_var": dependent_var, "group_var": group_var}
        if endpoint is not None:
            arguments["endpoint"] = endpoint
        if path is not None:
            arguments["path"] = path
        if table is not None:
            arguments["table"] = table
        if query is not None:
            arguments["query"] = query
        if alpha is not None:
            arguments["alpha"] = alpha
        return shaped_call("analyze_anova", _impl_analyze_anova, arguments)

    app.tool(analyze_anova)

    _impl_analyze_effect_sizes = registry.lookup("analyze_effect_sizes").func

    def analyze_effect_sizes(column: str, group_column: str, endpoint: str | None = None, path: str | None = None, table: str | None = None, query: str | None = None) -> Any:
        """
        Effect sizes for a grouping on an addressed tabular source: Cohen's d, Hedges' g, Glass's delta, and Cliff's delta for two groups; eta and omega squared for three or more.

        Input shape: TABULAR.
        Output shape: SCALAR.
        Streaming-capable: no.
        Domain: process.

        Args:
            endpoint (optional): The operator-declared endpoint name (exactly one of endpoint/path).
            path (optional): A local file inside allowed_paths (exactly one of endpoint/path).
            table (optional): A table on the endpoint (exactly one of table/query for endpoint sources).
            query (optional): One read-only SQL statement (endpoint sources and local database files).
            column: The numeric value column.
            group_column: The column defining the groups.
        """
        arguments: dict[str, Any] = {"column": column, "group_column": group_column}
        if endpoint is not None:
            arguments["endpoint"] = endpoint
        if path is not None:
            arguments["path"] = path
        if table is not None:
            arguments["table"] = table
        if query is not None:
            arguments["query"] = query
        return shaped_call("analyze_effect_sizes", _impl_analyze_effect_sizes, arguments)

    app.tool(analyze_effect_sizes)

    _impl_analyze_ab_test = registry.lookup("analyze_ab_test").func

    def analyze_ab_test(metric_column: str, variant_column: str, endpoint: str | None = None, path: str | None = None, table: str | None = None, query: str | None = None, test_type: str | None = None, alpha: float | None = None, alternative: str | None = None) -> Any:
        """
        A/B test between the exactly-two variants of variant_column on an addressed tabular source. test_type auto (default) picks the two-proportion z-test for a binary metric, Welch's t-test otherwise; mann_whitney on request. Names the winner and lift.

        Input shape: TABULAR.
        Output shape: SCALAR.
        Streaming-capable: no.
        Domain: process.

        Args:
            endpoint (optional): The operator-declared endpoint name (exactly one of endpoint/path).
            path (optional): A local file inside allowed_paths (exactly one of endpoint/path).
            table (optional): A table on the endpoint (exactly one of table/query for endpoint sources).
            query (optional): One read-only SQL statement (endpoint sources and local database files).
            metric_column: The outcome metric column.
            variant_column: The column assigning the two variants.
            test_type (optional): auto (default), proportion, t_test, or mann_whitney.
            alpha (optional): Significance level for the verdict (implementation default 0.05).
            alternative (optional): Alternative hypothesis: two-sided (default), greater, or less.
        """
        arguments: dict[str, Any] = {"metric_column": metric_column, "variant_column": variant_column}
        if endpoint is not None:
            arguments["endpoint"] = endpoint
        if path is not None:
            arguments["path"] = path
        if table is not None:
            arguments["table"] = table
        if query is not None:
            arguments["query"] = query
        if test_type is not None:
            arguments["test_type"] = test_type
        if alpha is not None:
            arguments["alpha"] = alpha
        if alternative is not None:
            arguments["alternative"] = alternative
        return shaped_call("analyze_ab_test", _impl_analyze_ab_test, arguments)

    app.tool(analyze_ab_test)
