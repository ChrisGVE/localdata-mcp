# Advanced Examples

Multi-tool workflows: getting a large result through a small context, taking a
question from a raw file to a defensible answer, and composing analytical tools
across domains.

Every workflow on this page is executed by `tests/test_advanced_examples_e2e.py`
against generated fixtures. If a tool's parameters change, those tests fail and
this page gets fixed — the examples cannot quietly stop working.

## How to read these

Calls are written the way an agent issues them:

```python
analyze_regression("sales", "SELECT * FROM orders", target_column="revenue")
```

Two conventions run through everything:

- **The first two arguments are almost always a connection name and a SQL query.**
  The query selects the data. There is no separate load step and no data-frame
  parameter; column arguments name columns in that query's result set.
- **File connections answer SQL, and their table is called `data_table`.** A CSV,
  Excel sheet or Parquet file is read into SQLite on connect, so
  `SELECT * FROM data_table` is how you address it.

## Reading a result larger than your context

The problem: 50,000 rows will not fit in a context window, and pulling them
anyway wastes the budget you need for reasoning.

**Look before you leap.** `analyze_query_preview` runs the estimator without
executing:

```python
analyze_query_preview("warehouse", "SELECT * FROM events WHERE day = '2026-07-01'")
```

```json
{
  "query_info": {"query_hash": "0be8e5dd", "complexity_score": 1},
  "estimates": {"rows": 14400, "memory_mb": 5.6, "tokens": 288000}
}
```

288,000 tokens is the number that matters. Now you know to page rather than fetch.

**Execute, then page.** `execute_query` returns the first chunk plus a `query_id`
that names the buffered result:

```python
execute_query("warehouse", "SELECT * FROM events WHERE day = '2026-07-01'")
```

```json
{
  "metadata": {
    "total_rows": 50000,
    "showing_rows": "1-2000",
    "query_id": "warehouse_1784578488_95f7",
    "chunked": true,
    "buffer_complete": true
  },
  "data": []
}
```

From there, two ways to page, and they index differently:

```python
# By row: start_row is 1-based, chunk_size is a string or "all".
next_chunk("warehouse_1784578488_95f7", 2001, "500")

# By chunk: chunk ids are 0-based, sized by the server's own chunk plan.
request_data_chunk("warehouse_1784578488_95f7", 12)

# Several chunks at once, comma-separated in one string.
request_multiple_chunks("warehouse_1784578488_95f7", "0,2,7")
```

Use `next_chunk` when you are walking the result in order and want to choose the
window. Use `request_data_chunk` when you want the server's chunking — the chunk
size comes from a token estimate, so a chunk is sized to be readable.

**Ask what the result is like before reading it.** `get_query_metadata` and
`get_data_quality_report` describe the buffered result without spending context
on its rows:

```python
get_data_quality_report("warehouse_1784578488_95f7")
```

```json
{
  "overall_quality": "excellent",
  "quality_score": 1.0,
  "dimensions": {"completeness": 1.0, "consistency": 1.0, "validity": 1.0},
  "issues": [],
  "statistical_summary": {"null_percentage": 0.0, "duplicate_percentage": 0.0}
}
```

Checking this first is worth the one call: an analysis run on a column that is
40% null will return a number, and the number will be worthless.

**Release the buffer** when you are done, rather than waiting for it to expire:

```python
clear_streaming_buffer("warehouse_1784578488_95f7")
```

### When the server refuses

If a result exceeds both the RAM and the disk budget, `execute_query` does not
run it. It returns a `requires_refinement` response with suggestions — add a
`LIMIT`, a `WHERE`, an aggregation. This is deliberate: a refusal you can act on
beats a result that exhausts the machine. Narrow the query and try again, or use
`analyze_query_preview` to size the narrowed version first.

## From a raw file to a defensible answer

An experiment lands as a CSV, with a `group` column, a numeric `value`, and a
`converted` flag. The question is whether the groups actually differ.

**Connect and look at the shape.** `connect_database` reports the tables and
columns it built, so a separate describe call is usually unnecessary:

```python
connect_database("experiment", "csv", "./data/experiment.csv")
```

**Test whether the difference is real:**

```python
analyze_hypothesis_test(
    "experiment",
    'SELECT value, "group" AS grp FROM data_table',
    column="value",
    group_column="grp",
)
```

The response carries several tests with their interpretations — normality checks
alongside the group comparison — so you can see whether the test that answers
your question was appropriate for the data.

**Then ask how big the difference is**, which is the question a p-value does not
answer:

```python
analyze_effect_sizes(
    "experiment",
    'SELECT value, "group" AS grp FROM data_table',
    column="value",
    group_column="grp",
)
```

```json
{
  "effect_sizes": {
    "cohens_d_value_by_grp": {
      "cohens_d": -1.051,
      "effect_description": "large",
      "group1_mean": 9.93,
      "group2_mean": 12.01,
      "group1_size": 60, "group2_size": 60
    }
  }
}
```

Significant and large are different claims. Reporting both is the difference
between "the groups differ" and "the groups differ by about one pooled standard
deviation".

**For a conversion rate rather than a continuous measure**, use the A/B tool,
which does the proportion test and the power calculation together:

```python
analyze_ab_test(
    "experiment",
    'SELECT "group" AS grp, converted FROM data_table',
    variant_column="grp",
    metric_column="converted",
)
```

```json
{
  "test_name": "Proportion A/B Test (A vs B)",
  "p_value": 2.62e-05,
  "effect_size": 0.788,
  "power": 0.988,
  "conclusion": "Result is statistically significant (p=0.0000)."
}
```

The `power` figure is the one to read before acting on a null result: a
non-significant test at low power means you did not look hard enough, not that
there is nothing there.

**More than two groups** is ANOVA rather than a repeated pairwise test:

```python
analyze_anova(
    "trials",
    "SELECT * FROM data_table",
    dependent_var="response",
    group_var="treatment",
)
```

## Segmenting customers, then acting on the segments

RFM scores every customer on recency, frequency and monetary value from a
transaction log:

```python
connect_database("sales", "csv", "./data/transactions.csv")

analyze_rfm(
    "sales",
    "SELECT * FROM data_table",
    customer_column="customer_id",
    date_column="order_date",
    value_column="amount",
)
```

```json
{
  "rfm_scores": [
    {"customer_id": "C001", "recency": 141, "frequency": 6,
     "monetary": 763.3, "R": 2, "F": 3, "M": 3, "RFM_Score": "233"},
    {"customer_id": "C003", "recency": 26, "frequency": 8,
     "monetary": 745.93, "R": 4, "F": 4, "M": 3, "RFM_Score": "443"}
  ]
}
```

C001 and C003 have spent about the same amount. The R score separates them: one
bought three weeks ago, the other four months ago. That is the actionable
distinction, and it is invisible in the monetary total alone.

To find groupings you did not specify in advance, cluster instead:

```python
analyze_clusters(
    "sales",
    "SELECT recency_days, order_count, total_spend FROM customer_features",
    n_clusters=4,
)
```

Leave `n_clusters` out and the tool chooses. Supply it when the number of
segments is a business decision rather than a statistical one.

## Time series: describe, then project

Always describe before forecasting — the description tells you whether the
forecast is worth having:

```python
connect_database("metrics", "csv", "./data/monthly.csv")

analyze_time_series(
    "metrics",
    "SELECT * FROM data_table",
    date_column="date",
    value_column="value",
)
```

```json
{
  "series_info": {"length": 96, "frequency": "MS", "missing_values": 0},
  "trend_analysis": {
    "linear_trend": {"slope": 0.816, "r_squared": 0.874, "p_value": 5.3e-43}
  }
}
```

96 points, no gaps, a strong trend. Now the forecast means something:

```python
forecast_time_series(
    "metrics",
    "SELECT * FROM data_table",
    date_column="date",
    value_column="value",
    horizon=6,
    method="arima",
)
```

```json
{
  "model_type": "ARIMA",
  "metadata": {"auto_arima": true},
  "forecast_values": [179.97, 185.77, 189.94, 191.37, 189.68, 185.31],
  "forecast_index": ["2030-01-01", "2030-02-01"]
}
```

Had `missing_values` been large or `length` short, the honest move is to report
that the series does not support a forecast rather than to project from it.

## Modelling and checking the model

Fit, then evaluate against held-out predictions — the second step is the one that
tells you whether to trust the first:

```python
analyze_regression(
    "housing",
    "SELECT * FROM data_table",
    target_column="target",
    feature_columns=["x1", "x2", "x3"],
)

evaluate_model_performance(
    "housing",
    "SELECT * FROM data_table",
    target_column="target",
    prediction_column="predicted",
)
```

```json
{
  "metrics": {"r2": 0.9797, "rmse": 0.5905, "mae": 0.4776,
              "mean_residual": -0.081}
}
```

Read `mean_residual` alongside `r2`: a residual mean far from zero means the model
is biased, which a high `r2` will happily hide.

**Find the points the model should not be judged on:**

```python
detect_anomalies(
    "housing",
    "SELECT x1, x2, x3 FROM data_table",
    method="isolation_forest",
    contamination=0.1,
)
```

**Reduce the feature space** when there are too many correlated columns to
reason about:

```python
reduce_dimensions(
    "housing",
    "SELECT x1, x2, x3 FROM data_table",
    method="pca",
    n_components=2,
)
```

## Estimating without assuming a distribution

When the sample is small or plainly not normal, bootstrap the statistic instead
of trusting a parametric interval:

```python
bootstrap_statistic(
    "experiment",
    "SELECT value FROM data_table",
    column="value",
    statistic="mean",
    n_bootstrap=1000,
    confidence_level=0.95,
)
```

```json
{
  "bootstrap_results": [{
    "statistic_name": "mean_value",
    "original_statistic": 10.969,
    "bias_estimate": 0.0129,
    "standard_error": 0.2008
  }]
}
```

To work on a manageable slice of a large table while keeping it representative,
sample first and keep the strata intact:

```python
generate_sample(
    "warehouse",
    "SELECT * FROM events",
    sampling_method="stratified",
    sample_size=0.05,
    stratify_column="region",
)
```

`sample_size` is a row count when it is 1 or more, and a fraction when it is
below 1.

## Spatial analysis

Test for clustering before hunting for clusters — if the values are spatially
random, any "hotspot" you find is noise:

```python
connect_database("sensors", "sqlite", "./data/spatial.sqlite")

analyze_spatial_autocorrelation(
    "sensors",
    "SELECT x, y, value FROM readings",
    value_column="value",
    method="moran",
)
```

```json
{
  "statistic": "morans_i", "value": 0.998, "z_score": 18.90,
  "p_value": 0.0, "is_significant": true,
  "interpretation": "Significant positive spatial autocorrelation (clustering)"
}
```

Clustering confirmed, so locating it is meaningful:

```python
find_spatial_hotspots(
    "sensors",
    "SELECT x, y, value FROM readings",
    value_column="value",
    significance_level=0.05,
)
```

```json
{
  "method": "getis_ord_gi_star", "n_points": 60,
  "n_hotspots": 20, "n_coldspots": 20,
  "points": [{"x": -1.79, "y": 0.54, "gi_star_z_score": -3.94,
              "is_hotspot": false, "is_coldspot": true}]
}
```

Each point carries its own z-score and p-value, so you can tighten the threshold
without recomputing.

Geospatial tools take three input shapes, and the parameters follow that split:
**points** use `x_column`/`y_column`, **geometries** use a WKT text column, and
**networks** take a nodes query plus an edges query. Routing and accessibility
tools are network-shaped:

```python
optimize_route(
    "city",
    nodes_query="SELECT id, x, y FROM net_nodes",
    edges_query="SELECT source, target, weight FROM net_edges",
    waypoints=[1, 5, 9],
)
```

## Finding your way around unfamiliar data

```python
# Which connection has a table by this name?
find_table("customers")            # -> ["warehouse", "staging"]

# Regex search across a query's result set.
search_data("warehouse", "SELECT * FROM data_table",
            pattern="^SKU-99", columns="product_code")

# Bulk find-and-replace, previewed as a sample rather than applied blindly.
transform_data("warehouse", "SELECT * FROM data_table",
               column="region", find="EMEA", replace="Europe")

# Machine-readable schema for generating code or validating input.
export_schema("warehouse", format="json_schema")
```

## When a workflow goes wrong

The audit log is the fastest way to see what the server actually received:

```python
get_error_log()
```

```json
{
  "entries": [{
    "database": "warehouse",
    "query": "SELECT nope FROM data_table",
    "status": "error",
    "duration_ms": 1009.36,
    "error_type": "OperationalError"
  }]
}
```

```python
# Everything recent, including successes, with timings.
get_query_log(database="warehouse", since_minutes=30)

# Is memory the reason things are slow or being refused?
get_streaming_status()
```

## Composition: what actually chains

Results feed each other well when you drive the chaining. The pattern that works
is: run a tool, read a value out of its result, put that value into the next
query.

```python
# 1. Segment.
rfm = analyze_rfm("sales", "SELECT * FROM data_table",
                  customer_column="customer_id",
                  date_column="order_date", value_column="amount")

# 2. Take the champions from that result, and query only them.
analyze_time_series(
    "sales",
    "SELECT order_date, amount FROM data_table "
    "WHERE customer_id IN ('C003', 'C017', 'C042')",
    date_column="order_date",
    value_column="amount",
)
```

Two limits are worth stating plainly, because they are not obvious from the
tool descriptions:

- **Results do not carry composition metadata.** Nothing in a result tells the
  next tool what upstream analysis produced it, so the chaining above is yours to
  do, not the server's. Tracked as issue #26.
- **Analytical tools do not go through the SELECT-only gate.** That check guards
  `execute_query` and `analyze_query_preview`. An analytical tool passes its
  query straight through, so treat the query you hand one as you would any other
  statement against that database. Tracked as issue #25.

**Cross-connection joins are not supported.** Each query runs against one
connection. To combine sources, query each one separately and join the results in
your own reasoning — or load both into the same connection first.

## See also

- [Tools reference](tools-reference.md) — every parameter of all 71 tools
- [Architecture](architecture/index.md) — why the memory model refuses queries
- [Analytical domains](domains/index.md) — what each domain covers
- [Configuration](configuration.md) — memory budgets, paths, limits
