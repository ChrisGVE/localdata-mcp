"""NetworkX-based graph validation checks.

Detects cycles, disconnected components, redundant transitive edges,
and diamond ambiguity patterns.  All functions accept a pre-built
:class:`~networkx.MultiDiGraph` to avoid redundant graph construction.
"""

from typing import Any, Dict, List

import networkx as nx


def _warn(code: str, message: str, **details: Any) -> Dict[str, Any]:
    """Build a warning dict."""
    w: Dict[str, Any] = {"code": code, "message": message}
    if details:
        w["details"] = details
    return w


# ---------------------------------------------------------------------------
# Structural
# ---------------------------------------------------------------------------


def check_cycles(G: nx.MultiDiGraph) -> List[Dict[str, Any]]:
    """Detect cycles in the graph (max 10 reported)."""
    if nx.is_directed_acyclic_graph(G):
        return []
    warnings: List[Dict[str, Any]] = []
    for cycle in nx.simple_cycles(G):
        path = " → ".join(str(n) for n in cycle) + f" → {cycle[0]}"
        warnings.append(
            _warn(
                "cycle",
                f"Cycle detected: {path}",
                nodes=[str(n) for n in cycle],
            )
        )
        if len(warnings) >= 10:
            warnings.append(
                _warn(
                    "cycle",
                    "Additional cycles not shown (limit: 10)",
                )
            )
            break
    return warnings


def check_disconnected_components(
    G: nx.MultiDiGraph,
) -> List[Dict[str, Any]]:
    """Detect disconnected subgraphs beyond the largest component."""
    components = list(nx.weakly_connected_components(G))
    if len(components) <= 1:
        return []
    warnings: List[Dict[str, Any]] = []
    ranked = sorted(components, key=len, reverse=True)
    for i, comp in enumerate(ranked[1:], 2):
        sample = sorted(comp)[:5]
        extra = f" (and {len(comp) - 5} more)" if len(comp) > 5 else ""
        warnings.append(
            _warn(
                "disconnected_component",
                f"Disconnected component {i}/{len(components)}: "
                f"{len(comp)} node(s) — {', '.join(sample)}{extra}",
                component_size=len(comp),
                sample_nodes=sample,
            )
        )
        if len(warnings) >= 10:
            break
    return warnings


# ---------------------------------------------------------------------------
# DAG-specific
# ---------------------------------------------------------------------------


def check_redundant_transitive(
    G: nx.MultiDiGraph,
) -> List[Dict[str, Any]]:
    """Detect edges made redundant by transitive paths (DAGs only)."""
    simple = nx.DiGraph(G)
    if not nx.is_directed_acyclic_graph(simple):
        return []
    try:
        reduced = nx.transitive_reduction(simple)
    except nx.NetworkXError:
        return []
    redundant = set(simple.edges()) - set(reduced.edges())
    if not redundant:
        return []
    warnings: List[Dict[str, Any]] = []
    for u, v in sorted(redundant)[:10]:
        warnings.append(
            _warn(
                "redundant_transitive",
                f"Potentially redundant edge {u}→{v} (indirect path exists)",
                source=str(u),
                target=str(v),
            )
        )
    if len(redundant) > 10:
        warnings.append(
            _warn(
                "redundant_transitive",
                f"{len(redundant) - 10} additional redundant edge(s) not shown",
            )
        )
    return warnings


def check_diamond_ambiguity(
    G: nx.MultiDiGraph,
) -> List[Dict[str, Any]]:
    """Detect nodes with multiple parents (potential polyhierarchy)."""
    simple = nx.DiGraph(G)
    warnings: List[Dict[str, Any]] = []
    for node in sorted(simple.nodes()):
        parents = sorted(simple.predecessors(node))
        if len(parents) <= 1:
            continue
        sample = parents[:5]
        extra = f" (and {len(parents) - 5} more)" if len(parents) > 5 else ""
        warnings.append(
            _warn(
                "diamond_ambiguity",
                f"Node '{node}' has {len(parents)} parents: "
                f"{', '.join(sample)}{extra}"
                " — verify polyhierarchy is intentional",
                node=str(node),
                parents=parents,
            )
        )
        if len(warnings) >= 20:
            break
    return warnings
