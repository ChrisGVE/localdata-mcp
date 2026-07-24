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
    """Register every served (production) tool wrapper on `app`."""
    load_spec_modules()
    registry = default_registry()

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

        Input shape: TABULAR.
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

        Input shape: TABULAR.
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

        Input shape: TABULAR.
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
        Domain: statistical_analysis.

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
            alpha (optional): Significance level for the verdict (default: the configured process value).
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
        Domain: statistical_analysis.

        Args:
            endpoint (optional): The operator-declared endpoint name (exactly one of endpoint/path).
            path (optional): A local file inside allowed_paths (exactly one of endpoint/path).
            table (optional): A table on the endpoint (exactly one of table/query for endpoint sources).
            query (optional): One read-only SQL statement (endpoint sources and local database files).
            dependent_var: The numeric outcome column.
            group_var: The column defining the groups.
            alpha (optional): Significance level for the verdict (default: the configured process value).
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
        Domain: statistical_analysis.

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
        Domain: statistical_analysis.

        Args:
            endpoint (optional): The operator-declared endpoint name (exactly one of endpoint/path).
            path (optional): A local file inside allowed_paths (exactly one of endpoint/path).
            table (optional): A table on the endpoint (exactly one of table/query for endpoint sources).
            query (optional): One read-only SQL statement (endpoint sources and local database files).
            metric_column: The outcome metric column.
            variant_column: The column assigning the two variants.
            test_type (optional): auto (default), proportion, t_test, or mann_whitney.
            alpha (optional): Significance level for the verdict (default: the configured process value).
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

    _impl_analyze_regression = registry.lookup("analyze_regression").func

    def analyze_regression(target_column: str, endpoint: str | None = None, path: str | None = None, table: str | None = None, query: str | None = None, feature_columns: list | None = None, model_type: str | None = None, regularization: str | None = None, degree: int | None = None, algorithm_params: dict | None = None) -> Any:
        """
        Fit a regression model on an addressed tabular source: model_type linear (default), ridge, lasso, elastic_net, logistic, or polynomial (regularization l1/l2/elastic_net maps onto the penalised estimators). Reports coefficients, fit metrics, and design-matrix health (rank, condition number); algorithm_params passes tuning straight to the estimator.

        Input shape: TABULAR.
        Output shape: FITTED_MODEL.
        Streaming-capable: no.
        Domain: regression_modeling.

        Args:
            endpoint (optional): The operator-declared endpoint name (exactly one of endpoint/path).
            path (optional): A local file inside allowed_paths (exactly one of endpoint/path).
            table (optional): A table on the endpoint (exactly one of table/query for endpoint sources).
            query (optional): One read-only SQL statement (endpoint sources and local database files).
            target_column: The numeric outcome column to fit.
            feature_columns (optional): Feature columns (default: every other numeric column).
            model_type (optional): linear (default), ridge, lasso, elastic_net, logistic, polynomial.
            regularization (optional): Penalty spelling l1, l2, or elastic_net — maps onto the estimator.
            degree (optional): Polynomial expansion degree (implementation default 2).
            algorithm_params (optional): Estimator constructor parameters, passed through verbatim (FR-306).
        """
        arguments: dict[str, Any] = {"target_column": target_column}
        if endpoint is not None:
            arguments["endpoint"] = endpoint
        if path is not None:
            arguments["path"] = path
        if table is not None:
            arguments["table"] = table
        if query is not None:
            arguments["query"] = query
        if feature_columns is not None:
            arguments["feature_columns"] = feature_columns
        if model_type is not None:
            arguments["model_type"] = model_type
        if regularization is not None:
            arguments["regularization"] = regularization
        if degree is not None:
            arguments["degree"] = degree
        if algorithm_params is not None:
            arguments["algorithm_params"] = algorithm_params
        return shaped_call("analyze_regression", _impl_analyze_regression, arguments)

    app.tool(analyze_regression)

    _impl_evaluate_model_performance = registry.lookup("evaluate_model_performance").func

    def evaluate_model_performance(target_column: str, prediction_column: str, endpoint: str | None = None, path: str | None = None, table: str | None = None, query: str | None = None, model_type: str | None = None) -> Any:
        """
        Score stored predictions against actuals on an addressed tabular source: regression metrics (r2, mse, rmse, mae, residual summary) for a numeric pair, weighted classification metrics (accuracy, precision, recall, f1) on request.

        Input shape: TABULAR.
        Output shape: SCALAR.
        Streaming-capable: no.
        Domain: regression_modeling.

        Args:
            endpoint (optional): The operator-declared endpoint name (exactly one of endpoint/path).
            path (optional): A local file inside allowed_paths (exactly one of endpoint/path).
            table (optional): A table on the endpoint (exactly one of table/query for endpoint sources).
            query (optional): One read-only SQL statement (endpoint sources and local database files).
            target_column: The actual-values column.
            prediction_column: The predicted-values column.
            model_type (optional): regression (default) or classification.
        """
        arguments: dict[str, Any] = {"target_column": target_column, "prediction_column": prediction_column}
        if endpoint is not None:
            arguments["endpoint"] = endpoint
        if path is not None:
            arguments["path"] = path
        if table is not None:
            arguments["table"] = table
        if query is not None:
            arguments["query"] = query
        if model_type is not None:
            arguments["model_type"] = model_type
        return shaped_call("evaluate_model_performance", _impl_evaluate_model_performance, arguments)

    app.tool(evaluate_model_performance)

    _impl_analyze_clusters = registry.lookup("analyze_clusters").func

    def analyze_clusters(endpoint: str | None = None, path: str | None = None, table: str | None = None, query: str | None = None, columns: list | None = None, method: str | None = None, n_clusters: int | None = None, seed: int | None = None, algorithm_params: dict | None = None) -> Any:
        """
        Cluster an addressed tabular source: method kmeans (default), hierarchical, dbscan, gmm, or spectral. Without n_clusters= a silhouette sweep picks k. Reports labels, cluster sizes, and silhouette score; seed= pins stochastic initialization.

        Input shape: TABULAR.
        Output shape: SCALAR.
        Streaming-capable: no.
        Domain: pattern_recognition.

        Args:
            endpoint (optional): The operator-declared endpoint name (exactly one of endpoint/path).
            path (optional): A local file inside allowed_paths (exactly one of endpoint/path).
            table (optional): A table on the endpoint (exactly one of table/query for endpoint sources).
            query (optional): One read-only SQL statement (endpoint sources and local database files).
            columns (optional): Columns to analyze (default: every numeric column).
            method (optional): kmeans (default), hierarchical, dbscan, gmm, spectral.
            n_clusters (optional): Cluster count (default: silhouette sweep over 2..8).
            seed (optional): Random seed pinning stochastic steps (default: library behavior).
            algorithm_params (optional): Estimator constructor parameters, passed through verbatim.
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
        if columns is not None:
            arguments["columns"] = columns
        if method is not None:
            arguments["method"] = method
        if n_clusters is not None:
            arguments["n_clusters"] = n_clusters
        if seed is not None:
            arguments["seed"] = seed
        if algorithm_params is not None:
            arguments["algorithm_params"] = algorithm_params
        return shaped_call("analyze_clusters", _impl_analyze_clusters, arguments)

    app.tool(analyze_clusters)

    _impl_assign_clusters = registry.lookup("assign_clusters").func

    def assign_clusters(endpoint: str | None = None, path: str | None = None, table: str | None = None, query: str | None = None, columns: list | None = None, method: str | None = None, n_clusters: int | None = None, seed: int | None = None, algorithm_params: dict | None = None) -> Any:
        """
        Cluster an addressed tabular source and return the clustered rows tagged with an integer cluster label — the composable (TABULAR) counterpart to analyze_clusters' verdict, so a clustering result feeds a downstream stage (e.g. a chart coloured by cluster). method kmeans (default), hierarchical, dbscan, gmm, spectral; without n_clusters a silhouette sweep picks k; seed pins stochastic initialization.

        Input shape: TABULAR.
        Output shape: TABULAR.
        Streaming-capable: no.
        Domain: pattern_recognition.

        Args:
            endpoint (optional): The operator-declared endpoint name (exactly one of endpoint/path).
            path (optional): A local file inside allowed_paths (exactly one of endpoint/path).
            table (optional): A table on the endpoint (exactly one of table/query for endpoint sources).
            query (optional): One read-only SQL statement (endpoint sources and local database files).
            columns (optional): Columns to analyze (default: every numeric column).
            method (optional): kmeans (default), hierarchical, dbscan, gmm, spectral.
            n_clusters (optional): Cluster count (default: silhouette sweep over 2..8).
            seed (optional): Random seed pinning stochastic steps (default: library behavior).
            algorithm_params (optional): Estimator constructor parameters, passed through verbatim.
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
        if columns is not None:
            arguments["columns"] = columns
        if method is not None:
            arguments["method"] = method
        if n_clusters is not None:
            arguments["n_clusters"] = n_clusters
        if seed is not None:
            arguments["seed"] = seed
        if algorithm_params is not None:
            arguments["algorithm_params"] = algorithm_params
        return shaped_call("assign_clusters", _impl_assign_clusters, arguments)

    app.tool(assign_clusters)

    _impl_detect_anomalies = registry.lookup("detect_anomalies").func

    def detect_anomalies(endpoint: str | None = None, path: str | None = None, table: str | None = None, query: str | None = None, columns: list | None = None, method: str | None = None, contamination: float | None = None, seed: int | None = None, algorithm_params: dict | None = None) -> Any:
        """
        Find anomalous rows in an addressed tabular source: method isolation_forest (default), lof, or zscore (three-sigma rule). Reports anomaly indices, share, and a score summary.

        Input shape: TABULAR.
        Output shape: SCALAR.
        Streaming-capable: no.
        Domain: pattern_recognition.

        Args:
            endpoint (optional): The operator-declared endpoint name (exactly one of endpoint/path).
            path (optional): A local file inside allowed_paths (exactly one of endpoint/path).
            table (optional): A table on the endpoint (exactly one of table/query for endpoint sources).
            query (optional): One read-only SQL statement (endpoint sources and local database files).
            columns (optional): Columns to analyze (default: every numeric column).
            method (optional): isolation_forest (default), lof, or zscore.
            contamination (optional): Expected anomaly share (implementation default 0.1).
            seed (optional): Random seed pinning stochastic steps (default: library behavior).
            algorithm_params (optional): Estimator constructor parameters, passed through verbatim.
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
        if columns is not None:
            arguments["columns"] = columns
        if method is not None:
            arguments["method"] = method
        if contamination is not None:
            arguments["contamination"] = contamination
        if seed is not None:
            arguments["seed"] = seed
        if algorithm_params is not None:
            arguments["algorithm_params"] = algorithm_params
        return shaped_call("detect_anomalies", _impl_detect_anomalies, arguments)

    app.tool(detect_anomalies)

    _impl_reduce_dimensions = registry.lookup("reduce_dimensions").func

    def reduce_dimensions(endpoint: str | None = None, path: str | None = None, table: str | None = None, query: str | None = None, columns: list | None = None, method: str | None = None, n_components: int | None = None, seed: int | None = None, algorithm_params: dict | None = None) -> Any:
        """
        Embed an addressed tabular source into fewer dimensions: method pca (default, always reports explained_variance_ratio) or tsne (reports trustworthiness against the original data).

        Input shape: TABULAR.
        Output shape: MATRIX.
        Streaming-capable: no.
        Domain: pattern_recognition.

        Args:
            endpoint (optional): The operator-declared endpoint name (exactly one of endpoint/path).
            path (optional): A local file inside allowed_paths (exactly one of endpoint/path).
            table (optional): A table on the endpoint (exactly one of table/query for endpoint sources).
            query (optional): One read-only SQL statement (endpoint sources and local database files).
            columns (optional): Columns to analyze (default: every numeric column).
            method (optional): pca (default) or tsne.
            n_components (optional): Target dimensionality (implementation default 2).
            seed (optional): Random seed pinning stochastic steps (default: library behavior).
            algorithm_params (optional): Estimator constructor parameters, passed through verbatim.
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
        if columns is not None:
            arguments["columns"] = columns
        if method is not None:
            arguments["method"] = method
        if n_components is not None:
            arguments["n_components"] = n_components
        if seed is not None:
            arguments["seed"] = seed
        if algorithm_params is not None:
            arguments["algorithm_params"] = algorithm_params
        return shaped_call("reduce_dimensions", _impl_reduce_dimensions, arguments)

    app.tool(reduce_dimensions)

    _impl_transform_data = registry.lookup("transform_data").func

    def transform_data(column: str, find: str, replace: str, endpoint: str | None = None, path: str | None = None, table: str | None = None, query: str | None = None, case_sensitive: bool | None = None) -> Any:
        """
        Regex find/replace over one column of an addressed tabular source (pattern crosses the hardened safety screen). Returns the rewritten relation plus a change summary — composable into downstream stages.

        Input shape: TABULAR.
        Output shape: TABULAR.
        Streaming-capable: no.
        Domain: pattern_recognition.

        Args:
            endpoint (optional): The operator-declared endpoint name (exactly one of endpoint/path).
            path (optional): A local file inside allowed_paths (exactly one of endpoint/path).
            table (optional): A table on the endpoint (exactly one of table/query for endpoint sources).
            query (optional): One read-only SQL statement (endpoint sources and local database files).
            column: The column to rewrite.
            find: The regex pattern to find (safety-screened).
            replace: The replacement text (backrefs allowed).
            case_sensitive (optional): Match case-sensitively (implementation default true).
        """
        arguments: dict[str, Any] = {"column": column, "find": find, "replace": replace}
        if endpoint is not None:
            arguments["endpoint"] = endpoint
        if path is not None:
            arguments["path"] = path
        if table is not None:
            arguments["table"] = table
        if query is not None:
            arguments["query"] = query
        if case_sensitive is not None:
            arguments["case_sensitive"] = case_sensitive
        return shaped_call("transform_data", _impl_transform_data, arguments)

    app.tool(transform_data)

    _impl_analyze_time_series = registry.lookup("analyze_time_series").func

    def analyze_time_series(date_column: str, value_column: str, endpoint: str | None = None, path: str | None = None, table: str | None = None, query: str | None = None, frequency: str | None = None) -> Any:
        """
        Analyze a time series on an addressed tabular source: trend direction and slope, ADF stationarity, autocorrelation with significant lags, and calendar seasonality strength.

        Input shape: TABULAR.
        Output shape: SCALAR.
        Streaming-capable: no.
        Domain: time_series.

        Args:
            endpoint (optional): The operator-declared endpoint name (exactly one of endpoint/path).
            path (optional): A local file inside allowed_paths (exactly one of endpoint/path).
            table (optional): A table on the endpoint (exactly one of table/query for endpoint sources).
            query (optional): One read-only SQL statement (endpoint sources and local database files).
            date_column: The timestamp column.
            value_column: The numeric value column.
            frequency (optional): Pandas frequency alias to align the series on (e.g. 'D', 'MS').
        """
        arguments: dict[str, Any] = {"date_column": date_column, "value_column": value_column}
        if endpoint is not None:
            arguments["endpoint"] = endpoint
        if path is not None:
            arguments["path"] = path
        if table is not None:
            arguments["table"] = table
        if query is not None:
            arguments["query"] = query
        if frequency is not None:
            arguments["frequency"] = frequency
        return shaped_call("analyze_time_series", _impl_analyze_time_series, arguments)

    app.tool(analyze_time_series)

    _impl_forecast_time_series = registry.lookup("forecast_time_series").func

    def forecast_time_series(date_column: str, value_column: str, endpoint: str | None = None, path: str | None = None, table: str | None = None, query: str | None = None, horizon: int | None = None, method: str | None = None, order: list | None = None, seasonal_order: list | None = None) -> Any:
        """
        Forecast a time series on an addressed tabular source: method arima (default, order= [p,d,q]), sarima (seasonal_order= [P,D,Q,s], defaulted from the calendar frequency), auto_arima (AIC grid search), or ets. Returns the forecast with confidence intervals and the solver's convergence verdict.

        Input shape: TABULAR.
        Output shape: VECTOR.
        Streaming-capable: no.
        Domain: time_series.

        Args:
            endpoint (optional): The operator-declared endpoint name (exactly one of endpoint/path).
            path (optional): A local file inside allowed_paths (exactly one of endpoint/path).
            table (optional): A table on the endpoint (exactly one of table/query for endpoint sources).
            query (optional): One read-only SQL statement (endpoint sources and local database files).
            date_column: The timestamp column.
            value_column: The numeric value column.
            horizon (optional): Steps ahead to forecast (implementation default 10).
            method (optional): arima (default), sarima, auto_arima, or ets.
            order (optional): ARIMA order [p, d, q] (default [1,1,1]).
            seasonal_order (optional): Seasonal order [P, D, Q, s] for sarima.
        """
        arguments: dict[str, Any] = {"date_column": date_column, "value_column": value_column}
        if endpoint is not None:
            arguments["endpoint"] = endpoint
        if path is not None:
            arguments["path"] = path
        if table is not None:
            arguments["table"] = table
        if query is not None:
            arguments["query"] = query
        if horizon is not None:
            arguments["horizon"] = horizon
        if method is not None:
            arguments["method"] = method
        if order is not None:
            arguments["order"] = order
        if seasonal_order is not None:
            arguments["seasonal_order"] = seasonal_order
        return shaped_call("forecast_time_series", _impl_forecast_time_series, arguments)

    app.tool(forecast_time_series)

    _impl_generate_sample = registry.lookup("generate_sample").func

    def generate_sample(endpoint: str | None = None, path: str | None = None, table: str | None = None, query: str | None = None, sampling_method: str | None = None, sample_size: float | None = None, stratify_column: str | None = None, cluster_column: str | None = None, weight_column: str | None = None, seed: int | None = None) -> Any:
        """
        Draw a sample from an addressed tabular source: sampling_method simple_random (default), stratified, systematic, cluster, or weighted. sample_size: integer = row count, fraction = share (default 0.1). Returns the drawn relation plus the design summary.

        Input shape: TABULAR.
        Output shape: TABULAR.
        Streaming-capable: no.
        Domain: sampling_estimation.

        Args:
            endpoint (optional): The operator-declared endpoint name (exactly one of endpoint/path).
            path (optional): A local file inside allowed_paths (exactly one of endpoint/path).
            table (optional): A table on the endpoint (exactly one of table/query for endpoint sources).
            query (optional): One read-only SQL statement (endpoint sources and local database files).
            sampling_method (optional): simple_random (default), stratified, systematic, cluster, weighted.
            sample_size (optional): Integer row count, or fractional share (implementation default 0.1).
            stratify_column (optional): Stratum column (stratified).
            cluster_column (optional): Cluster column (cluster).
            weight_column (optional): Weight column (weighted).
            seed (optional): Random seed pinning every draw (default: fresh entropy).
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
        if sampling_method is not None:
            arguments["sampling_method"] = sampling_method
        if sample_size is not None:
            arguments["sample_size"] = sample_size
        if stratify_column is not None:
            arguments["stratify_column"] = stratify_column
        if cluster_column is not None:
            arguments["cluster_column"] = cluster_column
        if weight_column is not None:
            arguments["weight_column"] = weight_column
        if seed is not None:
            arguments["seed"] = seed
        return shaped_call("generate_sample", _impl_generate_sample, arguments)

    app.tool(generate_sample)

    _impl_bootstrap_statistic = registry.lookup("bootstrap_statistic").func

    def bootstrap_statistic(column: str, endpoint: str | None = None, path: str | None = None, table: str | None = None, query: str | None = None, statistic: str | None = None, resamples: int | None = None, confidence_level: float | None = None, seed: int | None = None) -> Any:
        """
        Percentile-bootstrap a statistic (mean, median, std, var) of one column on an addressed tabular source: estimate, confidence interval, and standard error. resamples defaults to the operator-configured count (S8 row 30).

        Input shape: TABULAR.
        Output shape: SCALAR.
        Streaming-capable: no.
        Domain: sampling_estimation.

        Args:
            endpoint (optional): The operator-declared endpoint name (exactly one of endpoint/path).
            path (optional): A local file inside allowed_paths (exactly one of endpoint/path).
            table (optional): A table on the endpoint (exactly one of table/query for endpoint sources).
            query (optional): One read-only SQL statement (endpoint sources and local database files).
            column: The numeric column to bootstrap.
            statistic (optional): mean (default), median, std, or var.
            resamples (optional): Bootstrap resamples (default: the configured S8 row-30 count).
            confidence_level (optional): Interval coverage (default: the configured process value).
            seed (optional): Random seed pinning every draw (default: fresh entropy).
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
        if statistic is not None:
            arguments["statistic"] = statistic
        if resamples is not None:
            arguments["resamples"] = resamples
        if confidence_level is not None:
            arguments["confidence_level"] = confidence_level
        if seed is not None:
            arguments["seed"] = seed
        return shaped_call("bootstrap_statistic", _impl_bootstrap_statistic, arguments)

    app.tool(bootstrap_statistic)

    _impl_monte_carlo_simulate = registry.lookup("monte_carlo_simulate").func

    def monte_carlo_simulate(column: str, endpoint: str | None = None, path: str | None = None, table: str | None = None, query: str | None = None, simulation_type: str | None = None, iterations: int | None = None, bounds: list | None = None, seed: int | None = None) -> Any:
        """
        Monte Carlo over one column of an addressed tabular source: simulation_type uncertainty (default — resampled distribution of the mean) or integration (probability mass inside bounds=[lower, upper] under the fitted normal). iterations defaults to the operator-configured count (S8 row 31).

        Input shape: TABULAR.
        Output shape: SCALAR.
        Streaming-capable: no.
        Domain: sampling_estimation.

        Args:
            endpoint (optional): The operator-declared endpoint name (exactly one of endpoint/path).
            path (optional): A local file inside allowed_paths (exactly one of endpoint/path).
            table (optional): A table on the endpoint (exactly one of table/query for endpoint sources).
            query (optional): One read-only SQL statement (endpoint sources and local database files).
            column: The numeric column to simulate over.
            simulation_type (optional): uncertainty (default) or integration.
            iterations (optional): Simulation draws (default: the configured S8 row-31 count).
            bounds (optional): [lower, upper] integration bounds (integration only).
            seed (optional): Random seed pinning every draw (default: fresh entropy).
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
        if simulation_type is not None:
            arguments["simulation_type"] = simulation_type
        if iterations is not None:
            arguments["iterations"] = iterations
        if bounds is not None:
            arguments["bounds"] = bounds
        if seed is not None:
            arguments["seed"] = seed
        return shaped_call("monte_carlo_simulate", _impl_monte_carlo_simulate, arguments)

    app.tool(monte_carlo_simulate)

    _impl_bayesian_estimate = registry.lookup("bayesian_estimate").func

    def bayesian_estimate(column: str, endpoint: str | None = None, path: str | None = None, table: str | None = None, query: str | None = None, prior_distribution: str | None = None, credible_level: float | None = None) -> Any:
        """
        Conjugate-normal Bayesian posterior of one column's mean on an addressed tabular source (noninformative prior): posterior mean, scale, and the Student-t credible interval.

        Input shape: TABULAR.
        Output shape: SCALAR.
        Streaming-capable: no.
        Domain: sampling_estimation.

        Args:
            endpoint (optional): The operator-declared endpoint name (exactly one of endpoint/path).
            path (optional): A local file inside allowed_paths (exactly one of endpoint/path).
            table (optional): A table on the endpoint (exactly one of table/query for endpoint sources).
            query (optional): One read-only SQL statement (endpoint sources and local database files).
            column: The numeric column to estimate.
            prior_distribution (optional): Conjugate prior family (normal — the launch set).
            credible_level (optional): Interval coverage (default: the configured process value).
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
        if prior_distribution is not None:
            arguments["prior_distribution"] = prior_distribution
        if credible_level is not None:
            arguments["credible_level"] = credible_level
        return shaped_call("bayesian_estimate", _impl_bayesian_estimate, arguments)

    app.tool(bayesian_estimate)

    _impl_analyze_rfm = registry.lookup("analyze_rfm").func

    def analyze_rfm(customer_column: str, date_column: str, value_column: str, endpoint: str | None = None, path: str | None = None, table: str | None = None, query: str | None = None) -> Any:
        """
        RFM customer segmentation on an addressed tabular source: quintile recency/frequency/monetary scores and the named segment cascade (Champions ... Lost — every segment reachable). Returns per-customer scores and per-segment summaries.

        Input shape: TABULAR.
        Output shape: SCALAR.
        Streaming-capable: no.
        Domain: business_intelligence.

        Args:
            endpoint (optional): The operator-declared endpoint name (exactly one of endpoint/path).
            path (optional): A local file inside allowed_paths (exactly one of endpoint/path).
            table (optional): A table on the endpoint (exactly one of table/query for endpoint sources).
            query (optional): One read-only SQL statement (endpoint sources and local database files).
            customer_column: The customer identifier column.
            date_column: The transaction date column.
            value_column: The transaction amount column.
        """
        arguments: dict[str, Any] = {"customer_column": customer_column, "date_column": date_column, "value_column": value_column}
        if endpoint is not None:
            arguments["endpoint"] = endpoint
        if path is not None:
            arguments["path"] = path
        if table is not None:
            arguments["table"] = table
        if query is not None:
            arguments["query"] = query
        return shaped_call("analyze_rfm", _impl_analyze_rfm, arguments)

    app.tool(analyze_rfm)

    _impl_calculate_clv = registry.lookup("calculate_clv").func

    def calculate_clv(customer_column: str, date_column: str, value_column: str, endpoint: str | None = None, path: str | None = None, table: str | None = None, query: str | None = None, gross_margin: float | None = None) -> Any:
        """
        Historical customer lifetime value on an addressed tabular source: per-customer average order value x purchase frequency x gross_margin, annualized. The customer identifier column is whatever customer_column names.

        Input shape: TABULAR.
        Output shape: SCALAR.
        Streaming-capable: no.
        Domain: business_intelligence.

        Args:
            endpoint (optional): The operator-declared endpoint name (exactly one of endpoint/path).
            path (optional): A local file inside allowed_paths (exactly one of endpoint/path).
            table (optional): A table on the endpoint (exactly one of table/query for endpoint sources).
            query (optional): One read-only SQL statement (endpoint sources and local database files).
            customer_column: The customer identifier column.
            date_column: The transaction date column.
            value_column: The transaction amount column.
            gross_margin (optional): Gross margin share applied to revenue (implementation default 0.2).
        """
        arguments: dict[str, Any] = {"customer_column": customer_column, "date_column": date_column, "value_column": value_column}
        if endpoint is not None:
            arguments["endpoint"] = endpoint
        if path is not None:
            arguments["path"] = path
        if table is not None:
            arguments["table"] = table
        if query is not None:
            arguments["query"] = query
        if gross_margin is not None:
            arguments["gross_margin"] = gross_margin
        return shaped_call("calculate_clv", _impl_calculate_clv, arguments)

    app.tool(calculate_clv)

    _impl_analyze_network = registry.lookup("analyze_network").func

    def analyze_network(source_column: str, target_column: str, endpoint: str | None = None, path: str | None = None, table: str | None = None, query: str | None = None, weight_column: str | None = None, directed: bool | None = None, include_centrality: bool | None = None) -> Any:
        """
        Analyze a network stored as a tabular edge list (addressed source with source_column/target_column, optional weight_column, directed on request): density, connectivity, components, degree summary, clustering, and top centrality nodes.

        Input shape: TABULAR.
        Output shape: SCALAR.
        Streaming-capable: no.
        Domain: network_graph.

        Args:
            endpoint (optional): The operator-declared endpoint name (exactly one of endpoint/path).
            path (optional): A local file inside allowed_paths (exactly one of endpoint/path).
            table (optional): A table on the endpoint (exactly one of table/query for endpoint sources).
            query (optional): One read-only SQL statement (endpoint sources and local database files).
            source_column: The edge-source node column.
            target_column: The edge-target node column.
            weight_column (optional): Optional edge-weight column.
            directed (optional): Treat edges as directed (implementation default false).
            include_centrality (optional): Compute centrality measures (implementation default true).
        """
        arguments: dict[str, Any] = {"source_column": source_column, "target_column": target_column}
        if endpoint is not None:
            arguments["endpoint"] = endpoint
        if path is not None:
            arguments["path"] = path
        if table is not None:
            arguments["table"] = table
        if query is not None:
            arguments["query"] = query
        if weight_column is not None:
            arguments["weight_column"] = weight_column
        if directed is not None:
            arguments["directed"] = directed
        if include_centrality is not None:
            arguments["include_centrality"] = include_centrality
        return shaped_call("analyze_network", _impl_analyze_network, arguments)

    app.tool(analyze_network)

    _impl_solve_linear_program = registry.lookup("solve_linear_program").func

    def solve_linear_program(objective_column: str, endpoint: str | None = None, path: str | None = None, table: str | None = None, query: str | None = None, constraint_columns: list | None = None, constraint_values: list | None = None, constraint_types: list | None = None, bounds: list | None = None, integer_variables: list | None = None) -> Any:
        """
        Minimise a linear objective on an addressed tabular source: rows are decision variables, objective_column the cost vector, each constraint column one constraint's coefficients with its constraint_values right-hand side and constraint_types (<=, >=, =). HiGHS solver; reports the solution, objective value, and solver status.

        Input shape: TABULAR.
        Output shape: SCALAR.
        Streaming-capable: no.
        Domain: optimization.

        Args:
            endpoint (optional): The operator-declared endpoint name (exactly one of endpoint/path).
            path (optional): A local file inside allowed_paths (exactly one of endpoint/path).
            table (optional): A table on the endpoint (exactly one of table/query for endpoint sources).
            query (optional): One read-only SQL statement (endpoint sources and local database files).
            objective_column: The cost-vector column.
            constraint_columns (optional): Constraint coefficient columns (one per constraint).
            constraint_values (optional): Right-hand sides, one per constraint column.
            constraint_types (optional): Per-constraint <= (default), >=, or =.
            bounds (optional): Per-variable [lower, upper] pairs (default: x >= 0).
            integer_variables (optional): Indices of variables constrained to integers.
        """
        arguments: dict[str, Any] = {"objective_column": objective_column}
        if endpoint is not None:
            arguments["endpoint"] = endpoint
        if path is not None:
            arguments["path"] = path
        if table is not None:
            arguments["table"] = table
        if query is not None:
            arguments["query"] = query
        if constraint_columns is not None:
            arguments["constraint_columns"] = constraint_columns
        if constraint_values is not None:
            arguments["constraint_values"] = constraint_values
        if constraint_types is not None:
            arguments["constraint_types"] = constraint_types
        if bounds is not None:
            arguments["bounds"] = bounds
        if integer_variables is not None:
            arguments["integer_variables"] = integer_variables
        return shaped_call("solve_linear_program", _impl_solve_linear_program, arguments)

    app.tool(solve_linear_program)

    _impl_optimize_constrained = registry.lookup("optimize_constrained").func

    def optimize_constrained(objective_expression: str, initial_guess_column: str, endpoint: str | None = None, path: str | None = None, table: str | None = None, query: str | None = None, constraint_expressions: list | None = None, constraint_types: list | None = None, method: str | None = None) -> Any:
        """
        Minimise a nonlinear objective expression over the decision vector x (e.g. '(x[0]-1)**2 + x[1]'), starting from initial_guess_column on an addressed tabular source. Expressions are evaluated by the deny-by-default numeric grammar — no host code can run. Optional constraint expressions (ineq: >= 0, or eq).

        Input shape: TABULAR.
        Output shape: SCALAR.
        Streaming-capable: no.
        Domain: optimization.

        Args:
            endpoint (optional): The operator-declared endpoint name (exactly one of endpoint/path).
            path (optional): A local file inside allowed_paths (exactly one of endpoint/path).
            table (optional): A table on the endpoint (exactly one of table/query for endpoint sources).
            query (optional): One read-only SQL statement (endpoint sources and local database files).
            objective_expression: Numeric expression over x, evaluated by the safe grammar.
            initial_guess_column: Column holding the starting decision vector.
            constraint_expressions (optional): Constraint expressions over x (safe grammar).
            constraint_types (optional): Per-constraint ineq (default, >= 0) or eq.
            method (optional): scipy minimize method (implementation default SLSQP).
        """
        arguments: dict[str, Any] = {"objective_expression": objective_expression, "initial_guess_column": initial_guess_column}
        if endpoint is not None:
            arguments["endpoint"] = endpoint
        if path is not None:
            arguments["path"] = path
        if table is not None:
            arguments["table"] = table
        if query is not None:
            arguments["query"] = query
        if constraint_expressions is not None:
            arguments["constraint_expressions"] = constraint_expressions
        if constraint_types is not None:
            arguments["constraint_types"] = constraint_types
        if method is not None:
            arguments["method"] = method
        return shaped_call("optimize_constrained", _impl_optimize_constrained, arguments)

    app.tool(optimize_constrained)

    _impl_solve_assignment_problem = registry.lookup("solve_assignment_problem").func

    def solve_assignment_problem(cost_columns: list, endpoint: str | None = None, path: str | None = None, table: str | None = None, query: str | None = None, agent_column: str | None = None) -> Any:
        """
        Optimal agent-task assignment (Hungarian algorithm) on an addressed tabular source: rows are agents, cost_columns the per-task cost columns; optional agent_column names the agents. Reports the assignment and total cost.

        Input shape: TABULAR.
        Output shape: SCALAR.
        Streaming-capable: no.
        Domain: optimization.

        Args:
            endpoint (optional): The operator-declared endpoint name (exactly one of endpoint/path).
            path (optional): A local file inside allowed_paths (exactly one of endpoint/path).
            table (optional): A table on the endpoint (exactly one of table/query for endpoint sources).
            query (optional): One read-only SQL statement (endpoint sources and local database files).
            cost_columns: Per-task cost columns.
            agent_column (optional): Column naming the agents (default: row index).
        """
        arguments: dict[str, Any] = {"cost_columns": cost_columns}
        if endpoint is not None:
            arguments["endpoint"] = endpoint
        if path is not None:
            arguments["path"] = path
        if table is not None:
            arguments["table"] = table
        if query is not None:
            arguments["query"] = query
        if agent_column is not None:
            arguments["agent_column"] = agent_column
        return shaped_call("solve_assignment_problem", _impl_solve_assignment_problem, arguments)

    app.tool(solve_assignment_problem)

    _impl_check_geospatial_capabilities = registry.lookup("check_geospatial_capabilities").func

    def check_geospatial_capabilities() -> Any:
        """
        Report which geospatial backend libraries are installed and what the geospatial extra enables — answers unconditionally.

        Input shape: NONE (chain endpoint — composes with nothing).
        Output shape: SCALAR.
        Streaming-capable: no.
        Domain: geospatial_analysis.
        """
        arguments: dict[str, Any] = {}
        return shaped_call("check_geospatial_capabilities", _impl_check_geospatial_capabilities, arguments)

    app.tool(check_geospatial_capabilities)

    _impl_analyze_spatial_autocorrelation = registry.lookup("analyze_spatial_autocorrelation").func

    def analyze_spatial_autocorrelation(value_column: str, endpoint: str | None = None, path: str | None = None, table: str | None = None, query: str | None = None, x_column: str | None = None, y_column: str | None = None, k_neighbors: int | None = None) -> Any:
        """
        Global Moran's I on an addressed point source: whether nearby locations hold similar value_column values, over a k-nearest-neighbour neighbourhood. Reports I, z-score, and p-value.

        Input shape: TABULAR.
        Output shape: SCALAR.
        Streaming-capable: no.
        Domain: geospatial_analysis.

        Args:
            endpoint (optional): The operator-declared endpoint name (exactly one of endpoint/path).
            path (optional): A local file inside allowed_paths (exactly one of endpoint/path).
            table (optional): A table on the endpoint (exactly one of table/query for endpoint sources).
            query (optional): One read-only SQL statement (endpoint sources and local database files).
            value_column: The measured value column.
            x_column (optional): The x/longitude column (default 'x').
            y_column (optional): The y/latitude column (default 'y').
            k_neighbors (optional): Neighbours per point (default: the configured process value).
        """
        arguments: dict[str, Any] = {"value_column": value_column}
        if endpoint is not None:
            arguments["endpoint"] = endpoint
        if path is not None:
            arguments["path"] = path
        if table is not None:
            arguments["table"] = table
        if query is not None:
            arguments["query"] = query
        if x_column is not None:
            arguments["x_column"] = x_column
        if y_column is not None:
            arguments["y_column"] = y_column
        if k_neighbors is not None:
            arguments["k_neighbors"] = k_neighbors
        return shaped_call("analyze_spatial_autocorrelation", _impl_analyze_spatial_autocorrelation, arguments)

    app.tool(analyze_spatial_autocorrelation)

    _impl_find_spatial_hotspots = registry.lookup("find_spatial_hotspots").func

    def find_spatial_hotspots(value_column: str, endpoint: str | None = None, path: str | None = None, table: str | None = None, query: str | None = None, x_column: str | None = None, y_column: str | None = None, significance_level: float | None = None) -> Any:
        """
        Getis-Ord Gi* hot- and cold-spots on an addressed point source: the statistically significant clusters of high or low value_column at the significance level.

        Input shape: TABULAR.
        Output shape: SCALAR.
        Streaming-capable: no.
        Domain: geospatial_analysis.

        Args:
            endpoint (optional): The operator-declared endpoint name (exactly one of endpoint/path).
            path (optional): A local file inside allowed_paths (exactly one of endpoint/path).
            table (optional): A table on the endpoint (exactly one of table/query for endpoint sources).
            query (optional): One read-only SQL statement (endpoint sources and local database files).
            value_column: The measured value column.
            x_column (optional): The x/longitude column (default 'x').
            y_column (optional): The y/latitude column (default 'y').
            significance_level (optional): Two-sided significance (default: the configured process value).
        """
        arguments: dict[str, Any] = {"value_column": value_column}
        if endpoint is not None:
            arguments["endpoint"] = endpoint
        if path is not None:
            arguments["path"] = path
        if table is not None:
            arguments["table"] = table
        if query is not None:
            arguments["query"] = query
        if x_column is not None:
            arguments["x_column"] = x_column
        if y_column is not None:
            arguments["y_column"] = y_column
        if significance_level is not None:
            arguments["significance_level"] = significance_level
        return shaped_call("find_spatial_hotspots", _impl_find_spatial_hotspots, arguments)

    app.tool(find_spatial_hotspots)

    _impl_calculate_spatial_distances = registry.lookup("calculate_spatial_distances").func

    def calculate_spatial_distances(endpoint: str | None = None, path: str | None = None, table: str | None = None, query: str | None = None, x_column: str | None = None, y_column: str | None = None) -> Any:
        """
        Summarize pairwise Euclidean distances between the points of an addressed source (min/max/mean/median), bounded by point count to keep the matrix legible.

        Input shape: TABULAR.
        Output shape: SCALAR.
        Streaming-capable: no.
        Domain: geospatial_analysis.

        Args:
            endpoint (optional): The operator-declared endpoint name (exactly one of endpoint/path).
            path (optional): A local file inside allowed_paths (exactly one of endpoint/path).
            table (optional): A table on the endpoint (exactly one of table/query for endpoint sources).
            query (optional): One read-only SQL statement (endpoint sources and local database files).
            x_column (optional): The x/longitude column (default 'x').
            y_column (optional): The y/latitude column (default 'y').
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
        if x_column is not None:
            arguments["x_column"] = x_column
        if y_column is not None:
            arguments["y_column"] = y_column
        return shaped_call("calculate_spatial_distances", _impl_calculate_spatial_distances, arguments)

    app.tool(calculate_spatial_distances)

    _impl_perform_spatial_join = registry.lookup("perform_spatial_join").func

    def perform_spatial_join(geometry_column: str, right_geometries: list, endpoint: str | None = None, path: str | None = None, table: str | None = None, query: str | None = None, predicate: str | None = None) -> Any:
        """
        Attach the addressed source's rows to the inline right_geometries they spatially relate to (predicate intersects (default), within, or contains). The addressed source's geometry_column holds WKT.

        Input shape: TABULAR.
        Output shape: SCALAR.
        Streaming-capable: no.
        Domain: geospatial_analysis.

        Args:
            endpoint (optional): The operator-declared endpoint name (exactly one of endpoint/path).
            path (optional): A local file inside allowed_paths (exactly one of endpoint/path).
            table (optional): A table on the endpoint (exactly one of table/query for endpoint sources).
            query (optional): One read-only SQL statement (endpoint sources and local database files).
            geometry_column: Column of WKT geometries in the addressed source.
            right_geometries: The second geometry set as WKT strings.
            predicate (optional): intersects (default), within, or contains.
        """
        arguments: dict[str, Any] = {"geometry_column": geometry_column, "right_geometries": right_geometries}
        if endpoint is not None:
            arguments["endpoint"] = endpoint
        if path is not None:
            arguments["path"] = path
        if table is not None:
            arguments["table"] = table
        if query is not None:
            arguments["query"] = query
        if predicate is not None:
            arguments["predicate"] = predicate
        return shaped_call("perform_spatial_join", _impl_perform_spatial_join, arguments)

    app.tool(perform_spatial_join)

    _impl_perform_spatial_overlay = registry.lookup("perform_spatial_overlay").func

    def perform_spatial_overlay(geometry_column: str, right_geometries: list, endpoint: str | None = None, path: str | None = None, table: str | None = None, query: str | None = None, operation: str | None = None) -> Any:
        """
        Set operation (intersection (default), union, difference, symmetric_difference) between the addressed source's WKT geometry_column and the inline right_geometries.

        Input shape: TABULAR.
        Output shape: GEO.
        Streaming-capable: no.
        Domain: geospatial_analysis.

        Args:
            endpoint (optional): The operator-declared endpoint name (exactly one of endpoint/path).
            path (optional): A local file inside allowed_paths (exactly one of endpoint/path).
            table (optional): A table on the endpoint (exactly one of table/query for endpoint sources).
            query (optional): One read-only SQL statement (endpoint sources and local database files).
            geometry_column: Column of WKT geometries in the addressed source.
            right_geometries: The second geometry set as WKT strings.
            operation (optional): intersection (default), union, difference, symmetric_difference.
        """
        arguments: dict[str, Any] = {"geometry_column": geometry_column, "right_geometries": right_geometries}
        if endpoint is not None:
            arguments["endpoint"] = endpoint
        if path is not None:
            arguments["path"] = path
        if table is not None:
            arguments["table"] = table
        if query is not None:
            arguments["query"] = query
        if operation is not None:
            arguments["operation"] = operation
        return shaped_call("perform_spatial_overlay", _impl_perform_spatial_overlay, arguments)

    app.tool(perform_spatial_overlay)

    _impl_aggregate_points_in_polygons = registry.lookup("aggregate_points_in_polygons").func

    def aggregate_points_in_polygons(value_column: str, polygons: list, endpoint: str | None = None, path: str | None = None, table: str | None = None, query: str | None = None, x_column: str | None = None, y_column: str | None = None, aggregations: list | None = None) -> Any:
        """
        Summarize an addressed point source's value_column inside each inline polygon (WKT list): mean/sum/count by default.

        Input shape: TABULAR.
        Output shape: SCALAR.
        Streaming-capable: no.
        Domain: geospatial_analysis.

        Args:
            endpoint (optional): The operator-declared endpoint name (exactly one of endpoint/path).
            path (optional): A local file inside allowed_paths (exactly one of endpoint/path).
            table (optional): A table on the endpoint (exactly one of table/query for endpoint sources).
            query (optional): One read-only SQL statement (endpoint sources and local database files).
            value_column: The point measurement column.
            polygons: Containing polygons as WKT strings.
            x_column (optional): The x column (default 'x').
            y_column (optional): The y column (default 'y').
            aggregations (optional): Aggregations to apply (default mean, sum, count).
        """
        arguments: dict[str, Any] = {"value_column": value_column, "polygons": polygons}
        if endpoint is not None:
            arguments["endpoint"] = endpoint
        if path is not None:
            arguments["path"] = path
        if table is not None:
            arguments["table"] = table
        if query is not None:
            arguments["query"] = query
        if x_column is not None:
            arguments["x_column"] = x_column
        if y_column is not None:
            arguments["y_column"] = y_column
        if aggregations is not None:
            arguments["aggregations"] = aggregations
        return shaped_call("aggregate_points_in_polygons", _impl_aggregate_points_in_polygons, arguments)

    app.tool(aggregate_points_in_polygons)

    _impl_optimize_route = registry.lookup("optimize_route").func

    def optimize_route(edges: list, waypoints: list, endpoint: str | None = None, path: str | None = None, table: str | None = None, query: str | None = None, node_id_column: str | None = None, x_column: str | None = None, y_column: str | None = None, return_to_start: bool | None = None) -> Any:
        """
        Order waypoints greedily and connect them by shortest path over a network whose nodes are the addressed source (id, x, y) and whose edges arrive inline. Reports the order, path, and total distance.

        Input shape: TABULAR.
        Output shape: SCALAR.
        Streaming-capable: no.
        Domain: geospatial_analysis.

        Args:
            endpoint (optional): The operator-declared endpoint name (exactly one of endpoint/path).
            path (optional): A local file inside allowed_paths (exactly one of endpoint/path).
            table (optional): A table on the endpoint (exactly one of table/query for endpoint sources).
            query (optional): One read-only SQL statement (endpoint sources and local database files).
            edges: Network edges as [source, target] or [source, target, weight].
            waypoints: Node ids to visit.
            node_id_column (optional): The node-id column (default 'id').
            x_column (optional): The x column (default 'x').
            y_column (optional): The y column (default 'y').
            return_to_start (optional): Close the loop back to the first waypoint (default false).
        """
        arguments: dict[str, Any] = {"edges": edges, "waypoints": waypoints}
        if endpoint is not None:
            arguments["endpoint"] = endpoint
        if path is not None:
            arguments["path"] = path
        if table is not None:
            arguments["table"] = table
        if query is not None:
            arguments["query"] = query
        if node_id_column is not None:
            arguments["node_id_column"] = node_id_column
        if x_column is not None:
            arguments["x_column"] = x_column
        if y_column is not None:
            arguments["y_column"] = y_column
        if return_to_start is not None:
            arguments["return_to_start"] = return_to_start
        return shaped_call("optimize_route", _impl_optimize_route, arguments)

    app.tool(optimize_route)

    _impl_analyze_accessibility = registry.lookup("analyze_accessibility").func

    def analyze_accessibility(edges: list, service_locations: list, demand_locations: list, endpoint: str | None = None, path: str | None = None, table: str | None = None, query: str | None = None, node_id_column: str | None = None, x_column: str | None = None, y_column: str | None = None, max_travel_time: float | None = None) -> Any:
        """
        Score demand nodes by travel time to their nearest service node over a network (addressed nodes, inline edges), optionally capped at max_travel_time.

        Input shape: TABULAR.
        Output shape: SCALAR.
        Streaming-capable: no.
        Domain: geospatial_analysis.

        Args:
            endpoint (optional): The operator-declared endpoint name (exactly one of endpoint/path).
            path (optional): A local file inside allowed_paths (exactly one of endpoint/path).
            table (optional): A table on the endpoint (exactly one of table/query for endpoint sources).
            query (optional): One read-only SQL statement (endpoint sources and local database files).
            edges: Network edges as [source, target] or [source, target, weight].
            service_locations: Service node ids.
            demand_locations: Demand node ids.
            node_id_column (optional): The node-id column (default 'id').
            x_column (optional): The x column (default 'x').
            y_column (optional): The y column (default 'y').
            max_travel_time (optional): Reachability cap (default: unbounded).
        """
        arguments: dict[str, Any] = {"edges": edges, "service_locations": service_locations, "demand_locations": demand_locations}
        if endpoint is not None:
            arguments["endpoint"] = endpoint
        if path is not None:
            arguments["path"] = path
        if table is not None:
            arguments["table"] = table
        if query is not None:
            arguments["query"] = query
        if node_id_column is not None:
            arguments["node_id_column"] = node_id_column
        if x_column is not None:
            arguments["x_column"] = x_column
        if y_column is not None:
            arguments["y_column"] = y_column
        if max_travel_time is not None:
            arguments["max_travel_time"] = max_travel_time
        return shaped_call("analyze_accessibility", _impl_analyze_accessibility, arguments)

    app.tool(analyze_accessibility)

    _impl_generate_service_isochrones = registry.lookup("generate_service_isochrones").func

    def generate_service_isochrones(edges: list, service_locations: list, time_bands: list, endpoint: str | None = None, path: str | None = None, table: str | None = None, query: str | None = None, node_id_column: str | None = None, x_column: str | None = None, y_column: str | None = None) -> Any:
        """
        The node set reachable from the service nodes within each travel-time band, with its convex-hull footprint, over a network (addressed nodes, inline edges).

        Input shape: TABULAR.
        Output shape: GEO.
        Streaming-capable: no.
        Domain: geospatial_analysis.

        Args:
            endpoint (optional): The operator-declared endpoint name (exactly one of endpoint/path).
            path (optional): A local file inside allowed_paths (exactly one of endpoint/path).
            table (optional): A table on the endpoint (exactly one of table/query for endpoint sources).
            query (optional): One read-only SQL statement (endpoint sources and local database files).
            edges: Network edges as [source, target] or [source, target, weight].
            service_locations: Service node ids.
            time_bands: Travel-time bands to map.
            node_id_column (optional): The node-id column (default 'id').
            x_column (optional): The x column (default 'x').
            y_column (optional): The y column (default 'y').
        """
        arguments: dict[str, Any] = {"edges": edges, "service_locations": service_locations, "time_bands": time_bands}
        if endpoint is not None:
            arguments["endpoint"] = endpoint
        if path is not None:
            arguments["path"] = path
        if table is not None:
            arguments["table"] = table
        if query is not None:
            arguments["query"] = query
        if node_id_column is not None:
            arguments["node_id_column"] = node_id_column
        if x_column is not None:
            arguments["x_column"] = x_column
        if y_column is not None:
            arguments["y_column"] = y_column
        return shaped_call("generate_service_isochrones", _impl_generate_service_isochrones, arguments)

    app.tool(generate_service_isochrones)

    _impl_prepare_missing_values = registry.lookup("prepare_missing_values").func

    def prepare_missing_values(endpoint: str | None = None, path: str | None = None, table: str | None = None, query: str | None = None, columns: list | None = None, missing_strategy: str | None = None, fill_value: str | None = None) -> Any:
        """
        Handle missing values on an addressed tabular source: missing_strategy drop (default — fabricates nothing), mean, median, mode, forward_fill, or constant (needs fill_value). Returns the cleaned relation; composes as a pipeline stage.

        Input shape: TABULAR.
        Output shape: TABULAR.
        Streaming-capable: no.
        Domain: preprocessing.

        Args:
            endpoint (optional): The operator-declared endpoint name (exactly one of endpoint/path).
            path (optional): A local file inside allowed_paths (exactly one of endpoint/path).
            table (optional): A table on the endpoint (exactly one of table/query for endpoint sources).
            query (optional): One read-only SQL statement (endpoint sources and local database files).
            columns (optional): Columns to clean (default: every column).
            missing_strategy (optional): drop (default), mean, median, mode, forward_fill, constant.
            fill_value (optional): Fill value for missing_strategy='constant'.
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
        if columns is not None:
            arguments["columns"] = columns
        if missing_strategy is not None:
            arguments["missing_strategy"] = missing_strategy
        if fill_value is not None:
            arguments["fill_value"] = fill_value
        return shaped_call("prepare_missing_values", _impl_prepare_missing_values, arguments)

    app.tool(prepare_missing_values)

    _impl_convert_types = registry.lookup("convert_types").func

    def convert_types(conversions: dict, endpoint: str | None = None, path: str | None = None, table: str | None = None, query: str | None = None) -> Any:
        """
        Coerce named columns of an addressed tabular source to a target type (numeric, integer, string, datetime, boolean) via the conversions map. Reports failed casts; composes as a pipeline stage.

        Input shape: TABULAR.
        Output shape: TABULAR.
        Streaming-capable: no.
        Domain: preprocessing.

        Args:
            endpoint (optional): The operator-declared endpoint name (exactly one of endpoint/path).
            path (optional): A local file inside allowed_paths (exactly one of endpoint/path).
            table (optional): A table on the endpoint (exactly one of table/query for endpoint sources).
            query (optional): One read-only SQL statement (endpoint sources and local database files).
            conversions: Map of column name to target type.
        """
        arguments: dict[str, Any] = {"conversions": conversions}
        if endpoint is not None:
            arguments["endpoint"] = endpoint
        if path is not None:
            arguments["path"] = path
        if table is not None:
            arguments["table"] = table
        if query is not None:
            arguments["query"] = query
        return shaped_call("convert_types", _impl_convert_types, arguments)

    app.tool(convert_types)

    _impl_compose_pipeline = registry.lookup("compose_pipeline").func

    def compose_pipeline(dag_spec: list) -> Any:
        """
        Compose registered tools into one validated pipeline: dag_spec is an ordered list of {stage, tool, params, depends_on} entries (linear chains with fan-out; a chain-initial stage addresses its own source, a dependent stage consumes its upstream stage's output). The whole chain is validated against the tool contracts and the type-shape adjacency table BEFORE anything runs — an incompatible chain is rejected naming the offending stage, with no partial run. Returns one envelope per terminal stage under a single provenance chain.

        Input shape: DYNAMIC (validated per submitted dag_spec).
        Output shape: DYNAMIC (validated per submitted dag_spec).
        Streaming-capable: no.
        Domain: composition.

        Args:
            dag_spec: Ordered stage entries: {stage: name, tool: registered tool, params: tool arguments, depends_on: [upstream stage]}.
        """
        arguments: dict[str, Any] = {"dag_spec": dag_spec}
        return shaped_call("compose_pipeline", _impl_compose_pipeline, arguments)

    app.tool(compose_pipeline)

    _impl_clean_then_profile = registry.lookup("clean_then_profile").func

    def clean_then_profile(endpoint: str | None = None, path: str | None = None, table: str | None = None, query: str | None = None, missing_strategy: str | None = None) -> Any:
        """
        Convenience pipeline: prepare_missing_values -> profile_data on one addressed source. missing_strategy defaults to drop (fabricates nothing); callable with a source alone. Equivalent to the explicit two-stage compose_pipeline.

        Input shape: DYNAMIC (validated per submitted dag_spec).
        Output shape: DYNAMIC (validated per submitted dag_spec).
        Streaming-capable: no.
        Domain: composition.

        Args:
            endpoint (optional): The operator-declared endpoint name (exactly one of endpoint/path).
            path (optional): A local file inside allowed_paths (exactly one of endpoint/path).
            table (optional): A table on the endpoint (exactly one of table/query for endpoint sources).
            query (optional): One read-only SQL statement (endpoint sources and local database files).
            missing_strategy (optional): drop (default), mean, median, mode, forward_fill, constant.
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
        if missing_strategy is not None:
            arguments["missing_strategy"] = missing_strategy
        return shaped_call("clean_then_profile", _impl_clean_then_profile, arguments)

    app.tool(clean_then_profile)

    _impl_clean_then_regress = registry.lookup("clean_then_regress").func

    def clean_then_regress(target: str, endpoint: str | None = None, path: str | None = None, table: str | None = None, query: str | None = None, missing_strategy: str | None = None) -> Any:
        """
        Convenience pipeline: prepare_missing_values -> analyze_regression on one addressed source, predicting target. missing_strategy defaults to drop. Equivalent to the explicit two-stage compose_pipeline.

        Input shape: DYNAMIC (validated per submitted dag_spec).
        Output shape: DYNAMIC (validated per submitted dag_spec).
        Streaming-capable: no.
        Domain: composition.

        Args:
            endpoint (optional): The operator-declared endpoint name (exactly one of endpoint/path).
            path (optional): A local file inside allowed_paths (exactly one of endpoint/path).
            table (optional): A table on the endpoint (exactly one of table/query for endpoint sources).
            query (optional): One read-only SQL statement (endpoint sources and local database files).
            target: The regression target column.
            missing_strategy (optional): drop (default), mean, median, mode, forward_fill, constant.
        """
        arguments: dict[str, Any] = {"target": target}
        if endpoint is not None:
            arguments["endpoint"] = endpoint
        if path is not None:
            arguments["path"] = path
        if table is not None:
            arguments["table"] = table
        if query is not None:
            arguments["query"] = query
        if missing_strategy is not None:
            arguments["missing_strategy"] = missing_strategy
        return shaped_call("clean_then_regress", _impl_clean_then_regress, arguments)

    app.tool(clean_then_regress)

    _impl_cluster_then_chart = registry.lookup("cluster_then_chart").func

    def cluster_then_chart(endpoint: str | None = None, path: str | None = None, table: str | None = None, query: str | None = None, k: int | None = None, seed: int | None = None, format: str | None = None) -> Any:
        """
        Convenience pipeline: assign_clusters -> render_chart(scatter_fit). Clusters an addressed source, then scatters its first two numeric feature columns coloured by cluster. k pins the cluster count (default: silhouette sweep); format is svg (default) or png. Equivalent to the explicit two-stage compose_pipeline.

        Input shape: DYNAMIC (validated per submitted dag_spec).
        Output shape: DYNAMIC (validated per submitted dag_spec).
        Streaming-capable: no.
        Domain: composition.

        Args:
            endpoint (optional): The operator-declared endpoint name (exactly one of endpoint/path).
            path (optional): A local file inside allowed_paths (exactly one of endpoint/path).
            table (optional): A table on the endpoint (exactly one of table/query for endpoint sources).
            query (optional): One read-only SQL statement (endpoint sources and local database files).
            k (optional): Cluster count (default: silhouette sweep over 2..8).
            seed (optional): Random seed pinning the clustering (default: library behavior) — set it for a reproducible chart.
            format (optional): Chart image format: svg (default) or png.
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
        if k is not None:
            arguments["k"] = k
        if seed is not None:
            arguments["seed"] = seed
        if format is not None:
            arguments["format"] = format
        return shaped_call("cluster_then_chart", _impl_cluster_then_chart, arguments)

    app.tool(cluster_then_chart)

    _impl_render_chart = registry.lookup("render_chart").func

    def render_chart(kind: str, endpoint: str | None = None, path: str | None = None, table: str | None = None, query: str | None = None, encoding: dict | None = None, format: str | None = None, title: str | None = None, palette: str | None = None, colors: list | None = None, style: dict | None = None) -> Any:
        """
        Render an addressed tabular source as a chart image. kind is one of histogram, heatmap, scatter_fit, line_timeseries, geo_map, network_layout (one per analysis domain). encoding maps the kind's visual channels to columns (e.g. {'x': col, 'y': col}; heatmap defaults to all numeric columns). format is svg (default, sanitized to an inert document) or png. Styling is progressive: palette names a qualitative preset (deep, muted, pastel, bright, dark, colorblind — the default), colors supplies a custom color cycle instead, and style tunes figure size, dpi, grid, and the sequential colormap. The artifact is returned inline in the envelope, extractable through the Output surface.

        Input shape: TABULAR.
        Output shape: NONE (chain endpoint — composes with nothing).
        Streaming-capable: no.
        Domain: visualize.

        Args:
            endpoint (optional): The operator-declared endpoint name (exactly one of endpoint/path).
            path (optional): A local file inside allowed_paths (exactly one of endpoint/path).
            table (optional): A table on the endpoint (exactly one of table/query for endpoint sources).
            query (optional): One read-only SQL statement (endpoint sources and local database files).
            kind: Chart kind: histogram, heatmap, scatter_fit, line_timeseries, geo_map, or network_layout.
            encoding (optional): Visual-channel to column map for the kind (e.g. {'x': <col>, 'y': <col>}); heatmap accepts {'columns': [<col>, …]} or omits it for all numeric columns.
            format (optional): Image format: svg (default, inert-sanitized) or png.
            title (optional): Chart title drawn above the plot.
            palette (optional): Qualitative palette preset for categorical marks: deep, muted, pastel, bright, dark, or colorblind (the configured default). Ignored when colors is supplied.
            colors (optional): Custom categorical color cycle (hex like '#1b9e77' or named matplotlib colors) — overrides palette when given.
            style (optional): Fine styling overrides: figure_width_inches, figure_height_inches, dpi, grid, despine, sequential_cmap (the continuous colormap), fit_color, edge_color.
        """
        arguments: dict[str, Any] = {"kind": kind}
        if endpoint is not None:
            arguments["endpoint"] = endpoint
        if path is not None:
            arguments["path"] = path
        if table is not None:
            arguments["table"] = table
        if query is not None:
            arguments["query"] = query
        if encoding is not None:
            arguments["encoding"] = encoding
        if format is not None:
            arguments["format"] = format
        if title is not None:
            arguments["title"] = title
        if palette is not None:
            arguments["palette"] = palette
        if colors is not None:
            arguments["colors"] = colors
        if style is not None:
            arguments["style"] = style
        return shaped_call("render_chart", _impl_render_chart, arguments)

    app.tool(render_chart)

    _impl_export_result = registry.lookup("export_result").func

    def export_result(format: str, path: str, source: object | None = None, stream_id: str | None = None, overwrite: bool | None = None) -> Any:
        """
        Write a result to a file in any supported format. format is one of csv, parquet, arrow, json, excel, markdown (tabular data), schema (a table mapping), graph or tree (a structure mapping), or svg/png (a rendered chart artifact). path is the destination file (inside allowed_paths). The source is exactly one of: source (inline data — a records list, a mapping, or a rendered chart envelope), stream_id (a buffered result drained to the file), or — as a composition terminal — the upstream stage's output, injected automatically when both are omitted. An existing target is refused unless overwrite=true. Round-trip fidelity is type-preserving for parquet/arrow, documented-lossy for markdown.

        Input shape: TABULAR.
        Output shape: NONE (chain endpoint — composes with nothing).
        Streaming-capable: no.
        Domain: output.

        Args:
            format: Output format: csv, parquet, arrow, json, excel, markdown, schema, graph, tree, svg, or png.
            path: Destination file path inside allowed_paths.
            source (optional): Inline data to export: a records list, a mapping (schema / graph / tree / key-value), or a rendered chart artifact envelope. Omit when exporting a stream_id or a composition leaf.
            stream_id (optional): A buffered result to drain to the file (cursor semantics — the stream is consumed). Omit when exporting inline source or a composition leaf.
            overwrite (optional): Replace an existing target file (NFR-115: destructive operations need the explicit disambiguator). Defaults to refusing an existing target.
        """
        arguments: dict[str, Any] = {"format": format, "path": path}
        if source is not None:
            arguments["source"] = source
        if stream_id is not None:
            arguments["stream_id"] = stream_id
        if overwrite is not None:
            arguments["overwrite"] = overwrite
        return shaped_call("export_result", _impl_export_result, arguments)

    app.tool(export_result)


def register_skeleton_tools(app: FastMCP) -> None:
    """Register the test_only walking-skeleton probe wrappers on `app` — a battery-local app only, never the served surface (CR-012)."""
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
