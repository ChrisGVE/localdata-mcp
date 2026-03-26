# Directed Graphs

LocalData MCP parses graph files into a directed multigraph backed by SQLite. Nodes carry labels and typed metadata properties. Edges carry labels, optional weights, and metadata. The full graph is then available for path finding, structural analysis, and export.

## Supported formats

| Format | Extension | Notes |
|--------|-----------|-------|
| DOT (Graphviz) | `.dot` | Directed and undirected graphs |
| GML | `.gml` | Graph Modelling Language with nested attributes |
| GraphML | `.graphml` | XML-based with typed node/edge properties |
| Mermaid | `.mmd`, `.mermaid` | Flowchart syntax with subgraphs and shapes |

```python
connect_database("kg",   "graphml", "./knowledge_graph.graphml")
connect_database("flow", "dot",     "./pipeline.dot")
connect_database("arch", "mermaid", "./architecture.mmd")
```

## Graph tools

### Node operations

| Tool | Description |
|------|-------------|
| `set_node(name, node_id, label=)` | Create or update a node |
| `delete_node(name, node_id)` | Remove a node, cascading its edges and properties |
| `get_node(name, node_id)` | Inspect a node's label and properties |
| `get_neighbors(name, node_id, direction=)` | List adjacent nodes — `"in"`, `"out"`, or `"both"` |

### Edge operations

| Tool | Description |
|------|-------------|
| `add_edge(name, source, target, label=, weight=)` | Create a directed edge; endpoint nodes are created automatically if absent |
| `remove_edge(name, source, target, label=)` | Remove an edge |
| `get_edges(name, node_id=)` | List edges for a node |

### Property operations

Nodes carry typed key-value metadata. Types are inferred automatically from the value.

| Tool | Description |
|------|-------------|
| `get_value(name, node_id, key)` | Read a property |
| `set_value(name, node_id, key, value)` | Set a property |
| `delete_key(name, node_id, key)` | Remove a property |
| `list_keys(name, node_id)` | List all properties on a node |

### Analysis

| Tool | Description |
|------|-------------|
| `find_path(name, source, target, algorithm=)` | `"shortest"` (default) or `"all"` paths |
| `get_graph_stats(name)` | Node/edge counts, density, DAG check, connected components, and degree statistics |

### Export

```python
export_graph("kg", "graphml")                          # full metadata preserved
export_graph("kg", "dot")
export_graph("kg", "gml")
export_graph("kg", "mermaid")
export_graph("kg", "mermaid", node_id="root")          # subgraph around a node
```

All node and edge metadata properties are included in the export. Output is capped at 100 KB; larger graphs receive a truncation notice in the response.

## Validation warnings

Import and edit operations automatically check for structural and semantic issues. Warnings are returned in the `"warnings"` key of the response and never block the operation.

### Structural checks

| Code | Description |
|------|-------------|
| `self_loop` | Edge where source equals target (A→A) |
| `duplicate_edges` | Same source, target, and label appearing more than once |
| `orphan_nodes` | Nodes with no edges |
| `missing_edge_labels` | Edges without a relationship label |
| `cycle` | Circular paths (A→B→C→A), which break the DAG property |
| `disconnected_component` | Subgraphs unreachable from the main component |

### Semantic checks

| Code | Description |
|------|-------------|
| `contradictory_edges` | A→B and B→A with the same label |
| `conflicting_parallel_labels` | Multiple labeled edges between the same pair (e.g., both `"broader"` and `"related"` from A to B) |

### Property checks

| Code | Description |
|------|-------------|
| `duplicate_casing` | Node IDs differing only in case (`"Data Science"` vs `"Data science"`) |
| `missing_common_property` | Nodes missing a property that more than 50% of other nodes carry |
| `near_duplicate_names` | Node labels with 80% or greater string similarity |

### DAG-specific checks

| Code | Description |
|------|-------------|
| `redundant_transitive` | Edge A→C when A→B→C already exists, detected via transitive reduction |
| `diamond_ambiguity` | Node with multiple parents, which may indicate unintended polyhierarchy |

### When checks run

Full validation (all 13 checks) runs after every file import. Mutations trigger a targeted subset:

- `add_edge` — self-loop, duplicate edge, missing label, contradictory edges
- `remove_edge` — orphan node detection
- `set_node` — casing conflict detection

Cycle detection, transitive reduction, and near-duplicate name checks are skipped automatically for graphs with more than 10,000 nodes.

### Example warning response

```json
{
  "node_count": 42,
  "edge_count": 58,
  "warnings": [
    {
      "code": "orphan_nodes",
      "message": "3 orphan node(s): X, Y, Z",
      "details": {"nodes": ["X", "Y", "Z"]}
    },
    {
      "code": "cycle",
      "message": "Cycle detected: A → B → C → A",
      "details": {"nodes": ["A", "B", "C"]}
    }
  ]
}
```
