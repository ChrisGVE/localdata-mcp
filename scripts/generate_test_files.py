#!/usr/bin/env python3
"""Pre-generate test fixture files for all supported formats."""

import argparse
import os
import sys

import pandas as pd
import numpy as np

SIZE_ROWS = {"small": 1_000, "medium": 10_000, "large": 100_000}


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def generate_tabular(output_dir: str, n_rows: int) -> list[str]:
    """Generate CSV, TSV, and JSON tabular files."""
    d = os.path.join(output_dir, "tabular")
    ensure_dir(d)

    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "id": range(1, n_rows + 1),
            "name": [f"item_{i}" for i in range(1, n_rows + 1)],
            "value": rng.normal(100.0, 25.0, n_rows).round(2),
            "category": rng.choice(["A", "B", "C", "D"], n_rows),
            "flag": rng.choice([True, False], n_rows),
        }
    )

    files = []
    csv_path = os.path.join(d, "standard.csv")
    df.to_csv(csv_path, index=False)
    files.append(csv_path)

    tsv_path = os.path.join(d, "standard.tsv")
    df.to_csv(tsv_path, index=False, sep="\t")
    files.append(tsv_path)

    json_path = os.path.join(d, "standard.json")
    df.to_json(json_path, orient="records", indent=2)
    files.append(json_path)

    return files


def generate_tree(output_dir: str) -> list[str]:
    """Generate YAML and TOML config-like files."""
    d = os.path.join(output_dir, "tree")
    ensure_dir(d)

    yaml_content = """\
app:
  name: test-application
  version: 1.2.3
  debug: false

database:
  host: localhost
  port: 5432
  name: testdb
  pool:
    min: 2
    max: 10
    timeout: 30

logging:
  level: info
  format: json
  outputs:
    - type: file
      path: /var/log/app.log
    - type: stdout

features:
  cache_enabled: true
  rate_limit: 100
  allowed_origins:
    - "https://example.com"
    - "https://api.example.com"
"""

    toml_content = """\
[app]
name = "test-application"
version = "1.2.3"
debug = false

[database]
host = "localhost"
port = 5432
name = "testdb"

[database.pool]
min = 2
max = 10
timeout = 30

[logging]
level = "info"
format = "json"

[[logging.outputs]]
type = "file"
path = "/var/log/app.log"

[[logging.outputs]]
type = "stdout"

[features]
cache_enabled = true
rate_limit = 100
allowed_origins = ["https://example.com", "https://api.example.com"]
"""

    files = []
    yaml_path = os.path.join(d, "config.yaml")
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
    files.append(yaml_path)

    toml_path = os.path.join(d, "config.toml")
    with open(toml_path, "w") as f:
        f.write(toml_content)
    files.append(toml_path)

    return files


def generate_graph(output_dir: str) -> list[str]:
    """Generate DOT, GML, GraphML, and Mermaid graph files."""
    d = os.path.join(output_dir, "graph")
    ensure_dir(d)

    dot_content = """\
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

    gml_content = """\
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

    graphml_content = """\
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

    mmd_content = """\
graph LR
    A[Start] --> B[Process]
    B --> C{Decision}
    C -->|yes| D[Output]
    C -->|no| B
    D --> E[End]
"""

    files = []
    for name, content in [
        ("test.dot", dot_content),
        ("test.gml", gml_content),
        ("test.graphml", graphml_content),
        ("test.mmd", mmd_content),
    ]:
        path = os.path.join(d, name)
        with open(path, "w") as f:
            f.write(content)
        files.append(path)

    return files


def generate_edge_cases(output_dir: str, n_cols: int = 50) -> list[str]:
    """Generate edge-case CSV files."""
    d = os.path.join(output_dir, "edge_cases")
    ensure_dir(d)

    files = []

    # empty.csv - headers only
    empty_path = os.path.join(d, "empty.csv")
    with open(empty_path, "w") as f:
        f.write("id,name,value\n")
    files.append(empty_path)

    # unicode.csv
    unicode_path = os.path.join(d, "unicode.csv")
    rows = [
        "id,name,city,note",
        '1,"Rene Descartes","Touraine","cogito ergo sum"',
        '2,"Sophia Mueller","Munchen","Uberraschung"',
        '3,"Takashi Yamamoto","Tokyo","konnichiwa"',
        '4,"Emir Ozdemir","Istanbul","merhaba dunya"',
        '5,"Olga Ivanova","Moskva","privet mir"',
        '6,"Wei Zhang","Beijing","ni hao shijie"',
        '7,"Aine Ni Bhriain","Baile Atha Cliath","dia duit"',
        '8,"Bjorn Johansson","Goteborg","hej varlden"',
        '9,"Carlos Munoz","Espana","hola mundo"',
        '10,"Hans Gruber","Osterreich","servus"',
    ]
    with open(unicode_path, "w", encoding="utf-8") as f:
        f.write("\n".join(rows) + "\n")
    files.append(unicode_path)

    # wide.csv - 50 columns
    rng = np.random.default_rng(99)
    wide_df = pd.DataFrame(
        {f"col_{i:03d}": rng.normal(0, 1, 100).round(4) for i in range(n_cols)}
    )
    wide_path = os.path.join(d, "wide.csv")
    wide_df.to_csv(wide_path, index=False)
    files.append(wide_path)

    return files


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate test fixture files for all supported formats."
    )
    parser.add_argument(
        "--output-dir",
        default="tests/fixtures",
        help="Output directory (default: tests/fixtures)",
    )
    parser.add_argument(
        "--size",
        choices=["small", "medium", "large"],
        default="small",
        help="Row count: small=1000, medium=10000, large=100000 (default: small)",
    )
    args = parser.parse_args()

    n_rows = SIZE_ROWS[args.size]
    out = args.output_dir

    print(f"Generating test fixtures: size={args.size} ({n_rows} rows), output={out}")

    all_files: list[str] = []

    print("  [1/4] Tabular files (csv, tsv, json)...")
    all_files.extend(generate_tabular(out, n_rows))

    print("  [2/4] Tree files (yaml, toml)...")
    all_files.extend(generate_tree(out))

    print("  [3/4] Graph files (dot, gml, graphml, mmd)...")
    all_files.extend(generate_graph(out))

    print("  [4/4] Edge-case files (empty, unicode, wide csv)...")
    all_files.extend(generate_edge_cases(out))

    print(f"\nDone. Generated {len(all_files)} files:")
    for fp in all_files:
        size = os.path.getsize(fp)
        unit = "B"
        if size >= 1024:
            size /= 1024
            unit = "KB"
            if size >= 1024:
                size /= 1024
                unit = "MB"
        print(f"  {fp} ({size:.1f} {unit})")


if __name__ == "__main__":
    main()
