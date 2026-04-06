---
name: graph-explore
description: Explore a graph or network dataset — structure, statistics, paths, and visualization. Use when working with DOT, GML, GraphML, or Mermaid files.
allowed-tools: mcp__localdata__connect_database mcp__localdata__get_graph_stats mcp__localdata__get_edges mcp__localdata__get_neighbors mcp__localdata__get_node mcp__localdata__find_path mcp__localdata__export_graph
argument-hint: "<file-path>"
---

# Graph Explore

Connect to a graph file, analyze its structure and key nodes, find paths, and export visualizations.

## Steps

1. **Connect to the graph.** Call `connect_database` with the file path from `$ARGUMENTS`. The tool auto-detects graph formats including DOT, GML, GraphML, and Mermaid. Note the assigned database name.

2. **Get graph statistics.** Call `get_graph_stats` with the database name. Review: node count, edge count, density, whether the graph is directed or undirected, connected components count, and average degree. This gives an overview of the graph's scale and connectivity.

3. **Identify hub nodes.** From the stats, note nodes with the highest degree (most connections). Call `get_neighbors` for the top 3 highest-degree nodes to understand what they connect to. These hubs are often the most important entities in the network.

4. **Explore structure.** Call `get_edges` to retrieve a sample of edges. Look for patterns: are edges weighted? Do they have labels or types? Is the graph sparse or dense? Identify any isolated components or bridges.

5. **Inspect specific nodes.** For nodes of interest (hubs or user-specified), call `get_node` to retrieve node attributes and metadata. Note any labels, types, or properties that provide context.

6. **Find paths.** If there are at least two notable nodes, call `find_path` between them. Review the shortest path length and the intermediate nodes. This reveals how information or relationships flow through the network.

7. **Export a visualization.** Call `export_graph` with a suitable format (DOT for Graphviz rendering, or the original format for round-tripping). For large graphs, suggest filtering to a subgraph around nodes of interest before exporting.

8. **Summarize findings.** Present:
   - Graph type (directed/undirected, weighted/unweighted)
   - Scale: node and edge counts, density
   - Key structural features: hubs, communities, bridges
   - Notable paths or relationships discovered
   - Suggestions for further exploration or subgraph analysis
