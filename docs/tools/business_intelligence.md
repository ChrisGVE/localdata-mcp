<!-- MACHINE-WRITTEN by localdata_mcp.nexus.contract.generators.docs — DO NOT EDIT; regenerate via `python -m localdata_mcp.nexus.contract.generate` -->

# Tools — business_intelligence

| Tool | Summary | Input shape | Output shape | Streaming | Params |
|---|---|---|---|---|---|
| `analyze_rfm` | RFM customer segmentation on an addressed tabular source: quintile recency/frequency/monetary scores and the named segment cascade (Champions ... Lost — every segment reachable). Returns per-customer scores and per-segment summaries. | TABULAR | SCALAR | no | `endpoint?`, `path?`, `table?`, `query?`, `customer_column`, `date_column`, `value_column` |
| `calculate_clv` | Historical customer lifetime value on an addressed tabular source: per-customer average order value x purchase frequency x gross_margin, annualized. The customer identifier column is whatever customer_column names. | TABULAR | SCALAR | no | `endpoint?`, `path?`, `table?`, `query?`, `customer_column`, `date_column`, `value_column`, `gross_margin?` |
