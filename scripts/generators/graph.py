"""Generate graph test fixtures (DOT, GML, GraphML, Mermaid)."""

from __future__ import annotations

import os

from generators._common import sub_dir, write_text

_DOT = """\
digraph test {
    rankdir=LR;
    A [label="Start"];
    B [label="Process"];
    C [label="Decision"];
    D [label="Output"];
    E [label="End"];
    A -> B;
    B -> C;
    C -> D [label="yes"];
    C -> B [label="no"];
    D -> E;
}
"""

_GML = """\
graph [
  directed 1
  node [ id 0 label "Start" ]
  node [ id 1 label "Process" ]
  node [ id 2 label "Decision" ]
  node [ id 3 label "Output" ]
  node [ id 4 label "End" ]
  edge [ source 0 target 1 ]
  edge [ source 1 target 2 ]
  edge [ source 2 target 3 label "yes" ]
  edge [ source 2 target 1 label "no" ]
  edge [ source 3 target 4 ]
]
"""

_GRAPHML = """\
<?xml version="1.0" encoding="UTF-8"?>
<graphml xmlns="http://graphml.graphstruct.org/xmlns">
  <key id="label" for="node" attr.name="label" attr.type="string"/>
  <key id="elabel" for="edge" attr.name="label" attr.type="string"/>
  <graph id="G" edgedefault="directed">
    <node id="A"><data key="label">Start</data></node>
    <node id="B"><data key="label">Process</data></node>
    <node id="C"><data key="label">Decision</data></node>
    <node id="D"><data key="label">Output</data></node>
    <node id="E"><data key="label">End</data></node>
    <edge source="A" target="B"/>
    <edge source="B" target="C"/>
    <edge source="C" target="D"><data key="elabel">yes</data></edge>
    <edge source="C" target="B"><data key="elabel">no</data></edge>
    <edge source="D" target="E"/>
  </graph>
</graphml>
"""

_MERMAID = """\
graph LR
    A[Start] --> B[Process]
    B --> C{Decision}
    C -->|yes| D[Output]
    C -->|no| B
    D --> E[End]
"""


def generate_graph(output_dir: str) -> list[str]:
    """Generate DOT, GML, GraphML, and Mermaid files in graph/ subdirectory."""
    d = sub_dir(output_dir, "graph")
    templates = [
        ("test.dot", _DOT),
        ("test.gml", _GML),
        ("test.graphml", _GRAPHML),
        ("test.mmd", _MERMAID),
    ]
    return [write_text(os.path.join(d, name), content) for name, content in templates]
