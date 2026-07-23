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
