"""Parser for Mermaid flowchart/graph syntax into NetworkX DiGraph.

Handles node shapes, edge styles/labels, subgraphs, comments, and
chained edges.  The resulting DiGraph is suitable for ingestion via
:func:`localdata_mcp.graph_parsers._networkx_to_storage`.
"""

import os
import re
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx

from localdata_mcp.graph_manager import GraphStorageManager

# -- Shape patterns (order matters: longer delimiters first) ------------------

_SHAPE_PATTERNS: List[Tuple[str, str, str]] = [
    # (open_re, close_re, shape_name)
    (r"\(\(", r"\)\)", "circle"),
    (r"\(\[", r"\]\)", "stadium"),
    (r"\[\[", r"\]\]", "subroutine"),
    (r"\[\(", r"\)\]", "database"),
    (r">", r"\]", "asymmetric"),
    (r"\(", r"\)", "rounded"),
    (r"\{", r"\}", "diamond"),
    (r"\[", r"\]", "rectangle"),
]

# -- Edge patterns ------------------------------------------------------------

# Each tuple: (regex for the arrow, style name, is_directed)
_EDGE_DEFS: List[Tuple[str, str, bool]] = [
    (r"={2,}>", "thick", True),
    (r"-\.+->", "dotted", True),
    (r"-{2,}>", "solid", True),
    (r"-{3,}", "solid", False),
]

# Pre-compiled header pattern
_HEADER_RE = re.compile(
    r"^\s*(?:graph|flowchart)\s*(TD|TB|LR|RL|BT)?\s*$", re.IGNORECASE
)


class MermaidFlowchartParser:
    """Parse Mermaid flowchart/graph syntax into a NetworkX DiGraph."""

    def parse(self, text: str) -> nx.DiGraph:
        """Parse Mermaid text and return a NetworkX directed graph.

        Args:
            text: Raw Mermaid flowchart source.

        Returns:
            A NetworkX DiGraph with node/edge attributes populated.

        Raises:
            ValueError: If no valid ``graph`` or ``flowchart`` header is found.
        """
        lines = text.splitlines()
        G = nx.DiGraph()
        direction = "TD"
        header_found = False
        subgraph_stack: List[str] = []

        for raw_line in lines:
            line = raw_line.split("%%")[0].strip().rstrip(";")
            if not line:
                continue

            # Header detection
            m = _HEADER_RE.match(line)
            if m:
                header_found = True
                direction = (m.group(1) or "TD").upper()
                continue

            if not header_found:
                continue

            # Subgraph handling
            if line.lower().startswith("subgraph"):
                title = line[len("subgraph") :].strip()
                subgraph_stack.append(title)
                continue
            if line.lower() == "end":
                if subgraph_stack:
                    subgraph_stack.pop()
                continue

            # Try to parse edges; fall back to standalone node declaration
            if not self._parse_edge_line(line, G, subgraph_stack):
                self._parse_standalone_node(line, G, subgraph_stack)

        if not header_found:
            raise ValueError(
                "No valid Mermaid flowchart header found "
                "(expected 'graph' or 'flowchart')"
            )

        G.graph["direction"] = direction
        return G

    # -- internal helpers -----------------------------------------------------

    def _ensure_node(
        self,
        G: nx.DiGraph,
        node_id: str,
        subgraph_stack: List[str],
        label: Optional[str] = None,
        shape: Optional[str] = None,
    ) -> None:
        """Add or update a node in *G*.

        When *shape* is None (bare node reference), the default ``rectangle``
        is applied only for nodes that have not been seen before.  This
        prevents a bare reference like ``C`` from overwriting an earlier
        explicit shape declaration such as ``C(Cache Layer)``.
        """
        attrs: Dict[str, Any] = {}
        if label is not None:
            attrs["label"] = label
        if shape is not None:
            attrs["shape"] = shape
        if subgraph_stack:
            attrs["subgraph"] = subgraph_stack[-1]

        if node_id in G:
            for k, v in attrs.items():
                G.nodes[node_id][k] = v
        else:
            # First encounter: default shape to rectangle if not explicit
            if "shape" not in attrs:
                attrs["shape"] = "rectangle"
            G.add_node(node_id, **attrs)

    def _extract_node_decl(
        self, token: str
    ) -> Tuple[str, Optional[str], Optional[str]]:
        """Parse a token like ``A[Label]`` into (id, label, shape).

        Returns:
            Tuple of (node_id, label_or_None, shape_or_None).
            Shape is None for bare node identifiers (no brackets).
        """
        for open_re, close_re, shape in _SHAPE_PATTERNS:
            pattern = re.compile(
                r"^([A-Za-z_]\w*)\s*" + open_re + r"(.+?)" + close_re + r"$"
            )
            m = pattern.match(token)
            if m:
                return m.group(1), m.group(2).strip(), shape
        # Bare node id — no explicit shape
        bare = re.match(r"^([A-Za-z_]\w*)$", token)
        if bare:
            return bare.group(1), None, None
        return token.strip(), None, None

    def _find_edge_arrow(
        self, line: str, start: int
    ) -> Optional[Tuple[int, int, str, bool, Optional[str]]]:
        """Scan *line* from *start* for the next edge arrow.

        Returns:
            (arrow_start, arrow_end, style, is_directed, pipe_label) or None.
        """
        # Check for pipe-label after arrow: -->|label|
        for arrow_re, style, directed in _EDGE_DEFS:
            pat = re.compile(arrow_re + r"\|([^|]*)\|")
            m = pat.search(line, start)
            if m:
                return m.start(), m.end(), style, directed, m.group(1).strip()

        # Check for pre-label: -- label -->
        pre_label = re.compile(r"--\s+(.+?)\s+(-->)")
        m = pre_label.search(line, start)
        if m:
            return m.start(), m.end(), "solid", True, m.group(1).strip()

        # Plain arrows
        for arrow_re, style, directed in _EDGE_DEFS:
            pat = re.compile(arrow_re)
            m = pat.search(line, start)
            if m:
                return m.start(), m.end(), style, directed, None

        return None

    def _parse_edge_line(
        self,
        line: str,
        G: nx.DiGraph,
        subgraph_stack: List[str],
    ) -> bool:
        """Try to parse *line* as one or more edges.  Returns True on success."""
        result = self._find_edge_arrow(line, 0)
        if result is None:
            return False

        pos = 0
        prev_node: Optional[str] = None

        while result is not None:
            arrow_start, arrow_end, style, directed, label = result

            left_token = line[pos:arrow_start].strip()
            if left_token:
                nid, nlabel, nshape = self._extract_node_decl(left_token)
                self._ensure_node(G, nid, subgraph_stack, nlabel, nshape)
                prev_node = nid

            # Find the right-hand node
            next_arrow = self._find_edge_arrow(line, arrow_end)
            if next_arrow is not None:
                right_token = line[arrow_end : next_arrow[0]].strip()
            else:
                right_token = line[arrow_end:].strip()

            if right_token:
                rnid, rlabel, rshape = self._extract_node_decl(right_token)
                self._ensure_node(G, rnid, subgraph_stack, rlabel, rshape)
            else:
                rnid = None

            if prev_node and rnid:
                edge_attrs: Dict[str, Any] = {"style": style}
                if label:
                    edge_attrs["label"] = label
                G.add_edge(prev_node, rnid, **edge_attrs)
                if not directed:
                    G.add_edge(rnid, prev_node, **edge_attrs)
                prev_node = rnid

            pos = arrow_end
            result = next_arrow

        return True

    def _parse_standalone_node(
        self,
        line: str,
        G: nx.DiGraph,
        subgraph_stack: List[str],
    ) -> None:
        """Parse a line that contains only a node declaration (no edge)."""
        token = line.strip()
        if not token:
            return
        nid, nlabel, nshape = self._extract_node_decl(token)
        self._ensure_node(G, nid, subgraph_stack, nlabel, nshape)


def parse_mermaid_to_graph(
    file_path: str,
    manager: GraphStorageManager,
) -> Dict[str, Any]:
    """Parse a Mermaid flowchart file and populate graph storage.

    Args:
        file_path: Path to a ``.mmd`` or ``.mermaid`` file.
        manager: The graph storage manager to populate.

    Returns:
        Summary statistics dict from :meth:`manager.get_graph_stats`.

    Raises:
        ValueError: If the file is missing or has no valid header.
    """
    if not os.path.isfile(file_path):
        raise ValueError(f"File not found: {file_path}")
    with open(file_path, "r", encoding="utf-8") as fh:
        text = fh.read()
    parser = MermaidFlowchartParser()
    G = parser.parse(text)
    # Deferred import to avoid circular dependency with graph_parsers
    from localdata_mcp.graph_parsers import _networkx_to_storage

    _networkx_to_storage(G, manager)
    return manager.get_graph_stats()
