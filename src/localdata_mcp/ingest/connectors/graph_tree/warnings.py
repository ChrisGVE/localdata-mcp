"""localdata_mcp/ingest/connectors/graph_tree/warnings.py — edge signals (E8.3).

The harvested graph-integrity warnings `main` attached to mutations
(graph_mutation_tools.py): self-loops, unlabeled edges, duplicate
parallel edges, contradictory reverse-labeled edges, orphaned nodes,
and node-id casing variants. Pure composition over graph_store's
guarded integrity queries — advisory signals in the tool result,
never refusals (the harvested semantics: the mutation succeeded, the
caller is told what looks off). Neighbors: graph_tools.py attaches
these to add_edge/remove_edge/set_node results.
"""

from __future__ import annotations

from typing import Any, Optional

from .. import graph_store


def edge_warnings(
    endpoint: str, source: str, target: str, label: Optional[str]
) -> list[dict[str, Any]]:
    """Targeted signals after adding an edge (harvested semantics)."""
    found: list[dict[str, Any]] = []
    if source == target:
        found.append({"code": "self_loop", "message": f"Self-loop on '{source}'"})
    if not label:
        found.append(
            {
                "code": "missing_edge_labels",
                "message": f"Edge {source}→{target} has no label",
            }
        )
    duplicates = graph_store.parallel_edge_count(endpoint, source, target, label)
    if duplicates > 1:
        found.append(
            {
                "code": "duplicate_edges",
                "message": f"Edge {source}→{target}"
                + (f" [{label}]" if label else "")
                + f" now exists {duplicates} times",
            }
        )
    if label and graph_store.reverse_labeled_edge_exists(
        endpoint, source, target, label
    ):
        found.append(
            {
                "code": "contradictory_edges",
                "message": f"Both {source}→{target} and "
                f"{target}→{source} labeled '{label}'",
            }
        )
    return found


def orphan_warnings(endpoint: str, *node_ids: str) -> list[dict[str, Any]]:
    """Which of the given nodes are now edgeless (harvested)."""
    found: list[dict[str, Any]] = []
    for node_id in node_ids:
        if not graph_store.node_exists(endpoint, node_id):
            continue
        if graph_store.edge_count(endpoint, node_id) == 0:
            found.append(
                {
                    "code": "orphan_nodes",
                    "message": f"Node '{node_id}' is now an orphan (no edges)",
                }
            )
    return found


def casing_warnings(endpoint: str, node_id: str) -> list[dict[str, Any]]:
    """Node-id casing conflicts (harvested)."""
    variants = graph_store.casing_variants(endpoint, node_id)
    if not variants:
        return []
    return [
        {
            "code": "duplicate_casing",
            "message": f"'{node_id}' has casing variants: {', '.join(variants)}",
        }
    ]
