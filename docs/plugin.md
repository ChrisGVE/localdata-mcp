# Claude Code plugin

The LocalData MCP repository is also a Claude Code plugin. Its manifest,
`.claude-plugin/plugin.json`, registers the `localdata` MCP server with the
command `uvx localdata-mcp` and ships 18 skills and 11 agents that drive the
tools documented in the [tools reference](tools-reference.md).

Skills and agents call the MCP server; they add no analytical capability of
their own. Everything they do is reachable by calling the tools directly.

## Skills

Skills live under `skills/<group>/<skill-name>/SKILL.md`. Each declares an
`allowed-tools` list restricting it to the tools it needs, and an
`argument-hint` describing what to pass.

### `exploration/`

| Skill | Purpose |
| --- | --- |
| `explore-data` | Connect to a source, profile schema and quality, recommend which analyses fit |
| `data-quality` | Assess completeness, consistency, validity, and uniqueness before analysis |
| `find-reference-data` | Identify and prepare external reference data — benchmarks, demographics, economic indicators, geographic context |

### `statistical/`

| Skill | Purpose |
| --- | --- |
| `hypothesis-test` | Check assumptions, select the test, report the result in plain language |
| `ab-test` | Analyze an experiment and return a ship / iterate / no-ship recommendation |
| `analyze-correlations` | Find strong relationships between variables and suggest regression models |
| `sampling-estimation` | Design a sampling strategy and estimate parameters with confidence intervals |

### `modeling/`

| Skill | Purpose |
| --- | --- |
| `regression` | Fit and evaluate models predicting a target variable |
| `cluster-analysis` | Discover groupings, with cluster-count evaluation |
| `anomaly-detection` | Flag outliers with isolation forest or local outlier factor |
| `dimensionality-reduction` | Compress high-dimensional data with PCA, t-SNE, or UMAP |
| `forecast` | Analyze a time series and forecast forward with confidence intervals |
| `geospatial` | Distances, geographic clusters, accessibility |
| `optimization` | Resource allocation, scheduling, and process optimization under constraints |

### `graph-data/`

| Skill | Purpose |
| --- | --- |
| `graph-data-explore` | Walk a DOT, GML, GraphML, or Mermaid graph — nodes, edges, statistics, paths, export |

### `workflow/`

| Skill | Purpose |
| --- | --- |
| `data-pipeline` | End-to-end run: connect, profile, analyze, report |
| `research-pipeline` | Hypotheses, power analysis, assumption checks, reproducible reporting |
| `process-control` | Control charts, process stability and capability, out-of-control conditions |

## Agents

Agents live in `agents/`, one Markdown file per agent. All run on Sonnet; the
turn budget in the table is the agent's `maxTurns` and bounds how long an
analysis may run before it must report back.

| Agent | Turns | Scope |
| --- | --- | --- |
| `data-explorer` | 15 | Profiles an unfamiliar dataset: schema, quality, patterns, summary report |
| `data-scientist` | 30 | Composes multi-step pipelines across domains when the right approach is unclear |
| `statistical-analyst` | 20 | Hypothesis tests, ANOVA, effect sizes, sampling design, bootstrap estimation, non-parametric tests |
| `ml-analyst` | 25 | Clustering, anomaly detection, dimensionality reduction, regression modeling |
| `forecaster` | 20 | Decomposition, stationarity testing, model selection, ensemble forecasts with uncertainty |
| `bi-analyst` | 20 | A/B testing, cohort analysis, CLV, attribution, funnels; translates statistics into business recommendations |
| `graph-data-analyst` | 15 | Centrality, community detection, path finding, visualization export |
| `geospatial-analyst` | 20 | Coordinate systems, spatial relationships, distances, clustering, interpolation, accessibility |
| `operations-analyst` | 20 | Statistical process control, optimization, capacity planning, efficiency analysis |
| `research-analyst` | 25 | Methodological rigor: assumption documentation, power analysis, reproducible reporting |
| `data-researcher` | 20 | Finds, downloads, and prepares public reference datasets to enrich user data |

## Adding a skill or agent

Place a new skill in the domain directory it belongs to — `skills/exploration/`,
`skills/statistical/`, `skills/modeling/`, `skills/graph-data/`, or
`skills/workflow/` — as `<skill-name>/SKILL.md`. The directory name is the skill
name and must match the `name` field in the file's frontmatter. Agents are a
single `agents/<agent-name>.md`, with the same name-matching rule.
