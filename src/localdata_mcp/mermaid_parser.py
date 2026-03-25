"""Parser for Mermaid flowchart/graph syntax into NetworkX DiGraph.

Handles node shapes, edge styles/labels, subgraphs, comments, and
chained edges.  The resulting DiGraph is suitable for ingestion via
:func:`localdata_mcp.graph_parsers._networkx_to_storage`.
"""

import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx

from localdata_mcp.graph_manager import GraphStorageManager
from localdata_mcp.mermaid_export import export_mermaid  # noqa: F401

logger = logging.getLogger(__name__)

_SHAPE_PATTERNS: List[Tuple[str, str, str]] = [
    (r"\(\(", r"\)\)", "circle"),
    (r"\(\[", r"\]\)", "stadium"),
    (r"\[\[", r"\]\]", "subroutine"),
    (r"\[\(", r"\)\]", "database"),
    (r">", r"\]", "asymmetric"),
    (r"\(", r"\)", "rounded"),
    (r"\{", r"\}", "diamond"),
    (r"\[", r"\]", "rectangle"),
]

_EDGE_DEFS: List[Tuple[str, str, bool]] = [
    (r"={2,}>", "thick", True),
    (r"-\.+->", "dotted", True),
    (r"-{2,}>", "solid", True),
    (r"-{3,}", "solid", False),
]

_HEADER_RE = re.compile(
    r"^\s*(?:graph|flowchart)\s*(TD|TB|LR|RL|BT)?\s*$", re.IGNORECASE
)


class _ParseState:
    """Mutable state bag carried through the parse loop."""

    __slots__ = ("G", "direction", "header_found", "subgraph_stack")

    def __init__(self) -> None:
        self.G: nx.DiGraph = nx.DiGraph()
        self.direction: str = "TD"
        self.header_found: bool = False
        self.subgraph_stack: List[str] = []


class MermaidFlowchartParser:
    """Parse Mermaid flowchart/graph syntax into a NetworkX DiGraph."""

    def parse(self, text: str) -> nx.DiGraph:
        """Parse Mermaid text and return a NetworkX directed graph."""
        state = _ParseState()
        for raw_line in text.splitlines():
            self._process_line(raw_line, state)
        if not state.header_found:
            logger.warning("No valid Mermaid flowchart header found")
            raise ValueError(
                "No valid Mermaid flowchart header found "
                "(expected 'graph' or 'flowchart')"
            )
        state.G.graph["direction"] = state.direction
        return state.G

    def _process_line(self, raw_line: str, state: _ParseState) -> None:
        """Process a single raw line and update *state* in place."""
        line = raw_line.split("%%")[0].strip().rstrip(";")
        if not line:
            return
        m = _HEADER_RE.match(line)
        if m:
            state.header_found = True
            state.direction = (m.group(1) or "TD").upper()
            return
        if not state.header_found:
            return
        if line.lower().startswith("subgraph"):
            state.subgraph_stack.append(line[len("subgraph") :].strip())
            return
        if line.lower() == "end":
            if state.subgraph_stack:
                state.subgraph_stack.pop()
            return
        if not self._parse_edge_line(line, state.G, state.subgraph_stack):
            self._parse_standalone_node(line, state.G, state.subgraph_stack)

    def _ensure_node(
        self,
        G: nx.DiGraph,
        node_id: str,
        subgraph_stack: List[str],
        label: Optional[str] = None,
        shape: Optional[str] = None,
    ) -> None:
        """Add or update a node in *G*."""
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
            if "shape" not in attrs:
                attrs["shape"] = "rectangle"
            G.add_node(node_id, **attrs)

    def _extract_node_decl(
        self, token: str
    ) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """Parse ``A[Label]`` into (id, label, shape). Returns (None,None,None) on failure."""
        for open_re, close_re, shape in _SHAPE_PATTERNS:
            pat = re.compile(
                r"^([A-Za-z0-9_]\w*)\s*" + open_re + r"(.+?)" + close_re + r"$"
            )
            m = pat.match(token)
            if m:
                return m.group(1), m.group(2).strip(), shape
        bare = re.match(r"^([A-Za-z0-9_]\w*)$", token)
        if bare:
            return bare.group(1), None, None
        logger.warning("Unparseable node token: %r", token)
        return None, None, None

    def _scan_edge_operator(
        self, line: str, start: int
    ) -> Optional[Tuple[int, int, str, bool, Optional[str]]]:
        """Find the next edge arrow in *line* from *start*."""
        for arrow_re, style, directed in _EDGE_DEFS:
            pat = re.compile(arrow_re + r"\|([^|]*)\|")
            m = pat.search(line, start)
            if m:
                return m.start(), m.end(), style, directed, m.group(1).strip()
        pre_label = re.compile(r"--\s+(.+?)\s+(-->)")
        m = pre_label.search(line, start)
        if m:
            return m.start(), m.end(), "solid", True, m.group(1).strip()
        for arrow_re, style, directed in _EDGE_DEFS:
            m = re.compile(arrow_re).search(line, start)
            if m:
                return m.start(), m.end(), style, directed, None
        return None

    def _parse_edge_line(
        self,
        line: str,
        G: nx.DiGraph,
        sg: List[str],
    ) -> bool:
        """Try to parse *line* as one or more chained edges."""
        result = self._scan_edge_operator(line, 0)
        if result is None:
            return False
        pos = 0
        prev: Optional[str] = None
        while result is not None:
            arrow_start, arrow_end, style, directed, label = result
            prev = self._resolve_left(line, pos, arrow_start, G, sg, prev)
            next_arrow = self._scan_edge_operator(line, arrow_end)
            rnid = self._resolve_right(line, arrow_end, G, sg, next_arrow)
            if prev and rnid:
                self._add_edge(G, prev, rnid, style, directed, label)
                prev = rnid
            pos = arrow_end
            result = next_arrow
        return True

    def _add_edge(
        self,
        G: nx.DiGraph,
        src: str,
        tgt: str,
        style: str,
        directed: bool,
        label: Optional[str],
    ) -> None:
        """Add one (or two, if undirected) edges to *G*."""
        attrs: Dict[str, Any] = {"style": style, "directed": directed}
        if label:
            attrs["label"] = label
        G.add_edge(src, tgt, **attrs)
        if not directed:
            G.add_edge(tgt, src, **attrs)

    def _resolve_left(
        self,
        line: str,
        pos: int,
        arrow_start: int,
        G: nx.DiGraph,
        sg: List[str],
        prev: Optional[str],
    ) -> Optional[str]:
        """Extract and register the left-hand node of an edge."""
        token = line[pos:arrow_start].strip()
        if token:
            nid, nlabel, nshape = self._extract_node_decl(token)
            if nid is not None:
                self._ensure_node(G, nid, sg, nlabel, nshape)
                return nid
        return prev

    def _resolve_right(
        self,
        line: str,
        arrow_end: int,
        G: nx.DiGraph,
        sg: List[str],
        next_arrow: Optional[Tuple[int, int, str, bool, Optional[str]]],
    ) -> Optional[str]:
        """Extract and register the right-hand node of an edge."""
        end = next_arrow[0] if next_arrow is not None else len(line)
        token = line[arrow_end:end].strip()
        if token:
            nid, nlabel, nshape = self._extract_node_decl(token)
            if nid is not None:
                self._ensure_node(G, nid, sg, nlabel, nshape)
                return nid
        return None

    def _parse_standalone_node(
        self,
        line: str,
        G: nx.DiGraph,
        sg: List[str],
    ) -> None:
        """Parse a line containing only a node declaration."""
        token = line.strip()
        if not token:
            return
        nid, nlabel, nshape = self._extract_node_decl(token)
        if nid is None:
            logger.warning("Skipping unparseable line: %r", line)
            return
        self._ensure_node(G, nid, sg, nlabel, nshape)


def parse_mermaid_to_graph(
    file_path: str,
    manager: GraphStorageManager,
) -> Dict[str, Any]:
    """Parse a Mermaid flowchart file and populate graph storage."""
    if not os.path.isfile(file_path):
        raise ValueError(f"File not found: {file_path}")
    try:
        with open(file_path, "r", encoding="utf-8") as fh:
            text = fh.read()
    except UnicodeDecodeError as e:
        raise ValueError(f"Failed to read Mermaid file {file_path}: {e}") from e
    G = MermaidFlowchartParser().parse(text)
    from localdata_mcp.graph_parsers import _networkx_to_storage

    _networkx_to_storage(G, manager)
    return manager.get_graph_stats()
