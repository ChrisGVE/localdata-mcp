"""SQL-based graph validation checks and full validation orchestrator.

Detects structural, semantic, and property issues via efficient SQL
queries.  NetworkX-based checks live in :mod:`graph_validation_nx`.
"""

import difflib
from typing import Any, Dict, List, Tuple

from sqlalchemy import text

from .graph_manager import GraphStorageManager

MAX_VALIDATE_NODES = 10_000
_NEAR_DUP_THRESHOLD = 0.80


def _warn(code: str, message: str, **details: Any) -> Dict[str, Any]:
    """Build a warning dict."""
    w: Dict[str, Any] = {"code": code, "message": message}
    if details:
        w["details"] = details
    return w


# ---------------------------------------------------------------------------
# Structural checks
# ---------------------------------------------------------------------------


def check_self_loops(manager: GraphStorageManager) -> List[Dict[str, Any]]:
    """Detect edges where source equals target."""
    with manager.engine.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT source_id, target_id, label "
                "FROM graph_edges WHERE source_id = target_id"
            )
        ).fetchall()
    return [
        _warn(
            "self_loop",
            f"Self-loop on '{r[0]}'" + (f" (label: {r[2]})" if r[2] else ""),
            node=r[0],
            label=r[2],
        )
        for r in rows
    ]


def check_duplicate_edges(
    manager: GraphStorageManager,
) -> List[Dict[str, Any]]:
    """Detect duplicate edges with same source, target, and label."""
    with manager.engine.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT source_id, target_id, label, COUNT(*) AS cnt "
                "FROM graph_edges "
                "GROUP BY source_id, target_id, IFNULL(label, '') "
                "HAVING cnt > 1"
            )
        ).fetchall()
    return [
        _warn(
            "duplicate_edges",
            f"Duplicate edge: {r[0]}→{r[1]}"
            + (f" [{r[2]}]" if r[2] else "")
            + f" appears {r[3]} times",
            source=r[0],
            target=r[1],
            label=r[2],
            count=r[3],
        )
        for r in rows
    ]


def check_orphan_nodes(
    manager: GraphStorageManager,
) -> List[Dict[str, Any]]:
    """Detect nodes with no edges at all."""
    with manager.engine.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT n.node_id FROM graph_nodes n "
                "WHERE NOT EXISTS "
                "(SELECT 1 FROM graph_edges e WHERE e.source_id = n.node_id) "
                "AND NOT EXISTS "
                "(SELECT 1 FROM graph_edges e WHERE e.target_id = n.node_id)"
            )
        ).fetchall()
    if not rows:
        return []
    orphans = [r[0] for r in rows]
    sample = ", ".join(orphans[:10])
    extra = f" (and {len(orphans) - 10} more)" if len(orphans) > 10 else ""
    return [
        _warn(
            "orphan_nodes",
            f"{len(orphans)} orphan node(s): {sample}{extra}",
            nodes=orphans,
        )
    ]


def check_missing_edge_labels(
    manager: GraphStorageManager,
) -> List[Dict[str, Any]]:
    """Detect edges without a relationship label."""
    with manager.engine.connect() as conn:
        count = conn.execute(
            text("SELECT COUNT(*) FROM graph_edges WHERE label IS NULL OR label = ''")
        ).fetchone()[0]
        if count == 0:
            return []
        rows = conn.execute(
            text(
                "SELECT source_id, target_id FROM graph_edges "
                "WHERE label IS NULL OR label = '' LIMIT 5"
            )
        ).fetchall()
    examples = [f"{r[0]}→{r[1]}" for r in rows]
    extra = f" (and {count - 5} more)" if count > 5 else ""
    return [
        _warn(
            "missing_edge_labels",
            f"{count} edge(s) without label: {', '.join(examples)}{extra}",
            count=count,
            examples=examples,
        )
    ]


# ---------------------------------------------------------------------------
# Semantic / label checks
# ---------------------------------------------------------------------------


def check_contradictory_edges(
    manager: GraphStorageManager,
) -> List[Dict[str, Any]]:
    """Detect A→B[label] and B→A[label] simultaneously."""
    with manager.engine.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT e1.source_id, e1.target_id, e1.label "
                "FROM graph_edges e1 JOIN graph_edges e2 "
                "ON e1.source_id = e2.target_id "
                "AND e1.target_id = e2.source_id "
                "AND e1.label = e2.label "
                "WHERE e1.label IS NOT NULL "
                "AND e1.source_id < e1.target_id"
            )
        ).fetchall()
    return [
        _warn(
            "contradictory_edges",
            f"Contradictory: {r[0]}↔{r[1]} both labeled '{r[2]}'",
            source=r[0],
            target=r[1],
            label=r[2],
        )
        for r in rows
    ]


def check_conflicting_parallel_labels(
    manager: GraphStorageManager,
) -> List[Dict[str, Any]]:
    """Detect parallel edges with different labels on same source→target."""
    with manager.engine.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT source_id, target_id, "
                "GROUP_CONCAT(label, ', ') AS labels, COUNT(*) AS cnt "
                "FROM graph_edges "
                "WHERE label IS NOT NULL AND label != '' "
                "GROUP BY source_id, target_id HAVING cnt > 1"
            )
        ).fetchall()
    return [
        _warn(
            "conflicting_parallel_labels",
            f"Multiple labels on {r[0]}→{r[1]}: [{r[2]}]",
            source=r[0],
            target=r[1],
            labels=r[2].split(", ") if r[2] else [],
        )
        for r in rows
    ]


# ---------------------------------------------------------------------------
# Property checks
# ---------------------------------------------------------------------------


def check_duplicate_casing(
    manager: GraphStorageManager,
) -> List[Dict[str, Any]]:
    """Detect node IDs differing only in casing."""
    with manager.engine.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT LOWER(node_id) AS lid, "
                "GROUP_CONCAT(node_id, ', ') AS ids, COUNT(*) AS cnt "
                "FROM graph_nodes GROUP BY lid HAVING cnt > 1"
            )
        ).fetchall()
    return [
        _warn(
            "duplicate_casing",
            f"Node IDs differ only in casing: {r[1]}",
            variants=r[1].split(", "),
        )
        for r in rows
    ]


def _common_property_keys(engine, node_count: int) -> Dict[str, int]:
    """Return property keys present on >50% of nodes."""
    threshold = node_count * 0.5
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT key, COUNT(DISTINCT owner_id) AS cnt "
                "FROM graph_properties WHERE owner_type = 'node' "
                "GROUP BY key HAVING cnt > :th"
            ),
            {"th": threshold},
        ).fetchall()
    return {r[0]: r[1] for r in rows}


def check_missing_common_properties(
    manager: GraphStorageManager,
) -> List[Dict[str, Any]]:
    """Detect nodes missing properties that most (>50%) nodes have."""
    node_count = manager.get_node_count()
    if node_count == 0:
        return []
    common = _common_property_keys(manager.engine, node_count)
    if not common:
        return []
    warnings: List[Dict[str, Any]] = []
    with manager.engine.connect() as conn:
        for key, present in common.items():
            rows = conn.execute(
                text(
                    "SELECT n.node_id FROM graph_nodes n "
                    "WHERE NOT EXISTS (SELECT 1 FROM graph_properties p "
                    "WHERE p.owner_type = 'node' "
                    "AND p.owner_id = n.node_id AND p.key = :k) LIMIT 5"
                ),
                {"k": key},
            ).fetchall()
            missing = node_count - present
            if missing > 0:
                nodes = [r[0] for r in rows]
                extra = f" (and {missing - 5} more)" if missing > 5 else ""
                warnings.append(
                    _warn(
                        "missing_common_property",
                        f"{missing} node(s) missing '{key}': {', '.join(nodes)}{extra}",
                        property_key=key,
                        missing_count=missing,
                        examples=nodes,
                    )
                )
    return warnings


def _collect_labeled_nodes(
    manager: GraphStorageManager,
) -> List[Tuple[str, str]]:
    """Collect (node_id, label) for all nodes that have a label."""
    result: List[Tuple[str, str]] = []
    offset = 0
    while True:
        batch = manager.list_nodes(offset=offset, limit=500)
        for n in batch:
            if n.label:
                result.append((n.node_id, n.label))
        if len(batch) < 500:
            break
        offset += 500
    return result


def check_near_duplicate_names(
    manager: GraphStorageManager,
) -> List[Dict[str, Any]]:
    """Detect nodes with very similar labels (>= 80% similarity)."""
    labeled = _collect_labeled_nodes(manager)
    if len(labeled) > MAX_VALIDATE_NODES:
        return []
    warnings: List[Dict[str, Any]] = []
    for i, (id_a, lbl_a) in enumerate(labeled):
        for id_b, lbl_b in labeled[i + 1 :]:
            ratio = difflib.SequenceMatcher(
                None,
                lbl_a.lower(),
                lbl_b.lower(),
            ).ratio()
            if ratio >= _NEAR_DUP_THRESHOLD:
                warnings.append(
                    _warn(
                        "near_duplicate_names",
                        f"Similar labels: '{lbl_a}' ({id_a}) ↔ "
                        f"'{lbl_b}' ({id_b}) — {ratio:.0%}",
                        node_a=id_a,
                        label_a=lbl_a,
                        node_b=id_b,
                        label_b=lbl_b,
                        similarity=round(ratio, 2),
                    )
                )
                if len(warnings) >= 20:
                    return warnings
    return warnings


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def validate_graph(
    manager: GraphStorageManager,
    *,
    include_expensive: bool = True,
) -> List[Dict[str, Any]]:
    """Run all validation checks and return warnings.

    Cheap SQL checks always run.  Expensive NetworkX and O(n²) checks
    are skipped when *include_expensive* is False or the graph exceeds
    :data:`MAX_VALIDATE_NODES`.
    """
    from .graph_algorithms import _storage_to_networkx
    from .graph_validation_nx import (
        check_cycles,
        check_diamond_ambiguity,
        check_disconnected_components,
        check_redundant_transitive,
    )

    warnings: List[Dict[str, Any]] = []
    warnings.extend(check_self_loops(manager))
    warnings.extend(check_duplicate_edges(manager))
    warnings.extend(check_orphan_nodes(manager))
    warnings.extend(check_missing_edge_labels(manager))
    warnings.extend(check_duplicate_casing(manager))
    warnings.extend(check_contradictory_edges(manager))
    warnings.extend(check_conflicting_parallel_labels(manager))

    node_count = manager.get_node_count()
    if not include_expensive or node_count > MAX_VALIDATE_NODES:
        if node_count > MAX_VALIDATE_NODES:
            warnings.append(
                _warn(
                    "validation_limited",
                    f"Skipped expensive checks: {node_count} nodes "
                    f"(limit: {MAX_VALIDATE_NODES})",
                )
            )
        return warnings

    G = _storage_to_networkx(manager)
    warnings.extend(check_cycles(G))
    warnings.extend(check_disconnected_components(G))
    warnings.extend(check_redundant_transitive(G))
    warnings.extend(check_diamond_ambiguity(G))
    warnings.extend(check_missing_common_properties(manager))
    warnings.extend(check_near_duplicate_names(manager))
    return warnings
