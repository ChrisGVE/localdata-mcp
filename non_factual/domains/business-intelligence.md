# Business Intelligence Domain

## Overview

The business intelligence domain provides customer analytics, experiment analysis, and marketing
measurement tools. It is designed for workflows that start with transactional or event data stored
in a connected database and need to produce actionable business metrics.

Use this domain when you need to:

- Segment customers by engagement and value (RFM)
- Measure the statistical significance of a product or marketing experiment (A/B testing)

Cohort retention, lifetime value, marketing attribution and funnel analysis are implemented in the
domain package but reachable only from Python — no MCP tool exposes them.

All tools accept a SQL query, execute it against the named connection, and return a
JSON-serializable result. Row loading is capped at 500,000 rows
(`MAX_ANALYSIS_ROWS`, `datascience_tools.py:22`). No MCP tool exposes a way to
raise that cap — narrow the query instead.

---

## Available Analyses

| Analysis | Description |
|---|---|
| RFM segmentation | Scores customers on Recency, Frequency, and Monetary value; assigns segments |
| A/B test analysis | Proportion or mean comparison with p-value, effect size, CI, and power |
| Cohort analysis | Retention tables by acquisition cohort across daily/weekly/monthly periods |
| Customer lifetime value | Historical or predictive CLV per customer with segment breakdown |
| Marketing attribution | First-touch, last-touch, linear, time-decay, or position-based channel credits |
| Funnel analysis | Step-by-step conversion rates, drop-off rates, and bottleneck identification |
| Enhanced A/B test | BI test combined with statistical domain hypothesis testing for additional rigor |
| Power analysis | Required sample sizes for target power at given effect sizes and significance levels |

---

## MCP Tool Reference

The domain is reached through two MCP tools. Like every analytical tool except the four optimization tools,
each takes the name of a live connection and a SQL query — there is no
data-frame parameter and no separate load step, and column parameters name
columns in the query's result set. The functions listed under *Available
Analyses* above are the internal implementation; most have no MCP tool at all,
and where a name is shared its Python signature differs from the tool's.

Full parameter tables live in the
[tools reference](../tools-reference.md#data-science-12-tools).

### `analyze_rfm`

Perform RFM customer segmentation on transaction data.

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `connection_name` | str | required | Name of an active database connection |
| `query` | str | required | SQL query returning transaction rows |
| `customer_column` | str | required | Column identifying each customer |
| `date_column` | str | required | Column with transaction dates |
| `value_column` | str | required | Column with transaction monetary amounts |

**Return format**

```text
{
  "rfm_scores": [
    {"customer_id": "C001", "recency": 141, "frequency": 6, "monetary": 763.3,
     "R": 2, "F": 3, "M": 3, "RFM_Score": "233"}
  ],
  "segments": [
    {"customer_id": "C001", ..., "RFM_Score": "233", "Segment": "Loyal Customers"}
  ],
  "segment_summary": [
    {"Segment": "About to Sleep", "customer_count": 1, "percentage": 2.5,
     "recency_mean": 150.0, "recency_median": 150.0,
     "frequency_mean": 5.0, "frequency_median": 5.0,
     "monetary_mean": 341.39, "monetary_median": 341.39, "monetary_sum": 341.39}
  ],
  "quartile_boundaries": {
    "recency": [32.75, 82.5, 152.75],
    "frequency": [3.0, 5.0, 6.0],
    "monetary": [356.05, 522.59, 765.8]
  }
}
```

---

### `analyze_ab_test`

Analyze A/B test results from experiment data.

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `connection_name` | str | required | Name of an active database connection |
| `query` | str | required | SQL query returning experiment rows |
| `variant_column` | str | required | Column identifying control vs treatment |
| `metric_column` | str | required | Column with the outcome metric |
| `alpha` | float | `0.05` | Significance level |

**Return format**

```json
{
  "test_name": "Proportion A/B Test (control vs treatment)",
  "test_statistic": -2.14,
  "p_value": 0.032,
  "confidence_interval": [-0.045, -0.002],
  "effect_size": 0.18,
  "power": 0.81,
  "conclusion": "Result is statistically significant (p=0.0322). Effect direction: negative.",
  "sample_sizes": {"control": 5000, "treatment": 4980},
  "conversion_rates": {"control": 0.124, "treatment": 0.101}
}
```

---

## Method Details

### RFM Segmentation

RFM scores each customer on three dimensions calculated from transaction history:

- **Recency** — days since last purchase relative to `analysis_date` (or today); lower is better
- **Frequency** — total number of transactions
- **Monetary** — total spend

Each dimension is ranked into quartiles (1–4, higher is better for all three). The quartile
boundaries are returned in `quartile_boundaries`, so you can see exactly where each cut fell and
reproduce the scoring by hand. Note that this does **not** make a run repeatable on a later day:
recency is measured against today (see *Key parameters*), so the same query re-run next week
shifts every recency value and can move customers between segments.

The three scores are concatenated, not summed: `RFM_Score` is a string like `"433"`, meaning R=4,
F=3, M=3. A dimension in which every customer ties — an order log where all of them placed the same
number of orders — carries no ordering, so every customer scores the neutral 2 on it and is
separated by the other two.

Customers are assigned to named segments by a fixed priority cascade over the three 1–4 scores,
tested in order and stopping at the first match. Champions require R≥4, F≥4 and M≥3. At Risk reads
like a catch-all from its rule alone, but by the time the cascade reaches it every customer scoring
F≥2 or R≥3 is already claimed, so it captures exactly the infrequent, lapsed buyers — F=1 and R≤2 —
not high-value ones. Two named segments never appear on the 1–4 scale: **Lost**, whose rule fires
only when frequency or monetary scores 0, which the scoring never produces; and **Need Attention**,
whose F≥3, M≥3 customers are always caught one step earlier by Loyal Customers. Both would surface
only if the scoring range changed.

**When to use:** Initial customer health audit, churn prevention targeting, email list tiering,
loyalty programme design.

**Key parameters:**

- The MCP tool takes exactly `connection_name`, `query`, `customer_column`, `date_column` and
  `value_column`. The Python function beneath it also accepts an `analysis_date`, but **the tool
  exposes no way to set it**, so recency is always measured against today. Two runs of the same
  query on different days therefore produce different recency scores — pin the reference date in
  the query itself (compute a day count against a literal date) if you need reproducibility.

**Interpretation:** Customers in the Champions segment are high-value and recent — they should be
rewarded and surveyed. At Risk customers scored lowest on frequency and are not recent — infrequent,
lapsing buyers rather than former high-spenders — so they warrant win-back rather than premium
retention effort. Compare `segment_summary` counts across time periods to track portfolio health.

---

### A/B Test Analysis

The `ABTestAnalyzer` performs frequentist hypothesis testing for two-group experiments.

**Test types:**

| `test_type` | Statistical method | Use when |
|---|---|---|
| `proportion` | Z-test for proportions (`proportions_ztest`) | Binary outcomes: conversion, click, sign-up |
| `mean` | Welch's t-test (`ttest_ind`) | Continuous outcomes: revenue, session duration |
| `conversion` | Chi-squared contingency test | Funnel step conversion with count data |

**Effect size:**

- Proportion tests report Cohen's h (`2 * (arcsin(sqrt(p2)) - arcsin(sqrt(p1)))`)
- Mean tests report Cohen's d

**Power:** The result includes observed statistical power so you can determine whether a
non-significant result is due to a real null effect or insufficient sample.

**When to use:** Any time you have a randomised control group and a treatment group and need to
decide whether observed differences are attributable to the intervention.

**Interpretation:**

- `p_value < alpha` → reject null hypothesis; the difference is unlikely due to chance alone
- Check `effect_size` to assess practical significance, not just statistical significance
- If `power < 0.8` and p_value > alpha, collect more data before concluding no effect

**Bayesian alternative:** Use `enhanced_ab_test` (direct Python API) which calls the statistical
domain's `run_hypothesis_test` alongside the BI analysis for a Bayesian perspective.

---

### Cohort Analysis

Groups customers by their acquisition period (first transaction) and tracks how many remain active
in each subsequent period.

Output includes:

- `cohort_table` — cross-tabulation of cohort × period with active customer counts
- `retention_rates` — same table expressed as percentages of the original cohort size
- `period_summary` — average retention rate per period lag across all cohorts

**Period types:** `daily`, `weekly`, `monthly` (default), `quarterly`

**When to use:** Product retention reporting, subscription churn measurement, understanding how
product changes affect long-term engagement across cohorts.

---

### Customer Lifetime Value (CLV)

Calculates per-customer CLV and aggregates by segment.

**Methods:**

| `method` | Description |
|---|---|
| `historical` | Average order value × purchase frequency × average customer age |
| Predictive methods | Available via direct `CLVCalculator` API |

Output includes `clv_distribution` (percentile breakdown) and `segment_clv` for cross-segment
comparison.

**When to use:** Acquisition budget allocation, customer tier design, payback period modelling.

---

### Marketing Attribution

Assigns conversion credit across the marketing touchpoints in a customer's journey.

**Models:**

| `model` | Credit allocation |
|---|---|
| `first_touch` | 100% to first channel |
| `last_touch` | 100% to last channel (default) |
| `linear` | Equal credit to all channels |
| `time_decay` | Exponentially more credit to channels closer to conversion |
| `position_based` | 40% first, 40% last, 20% split across middle |

Output includes `attribution_weights`, `channel_attribution` summary, and `conversion_paths`
showing the most common multi-touch sequences.

**When to use:** Marketing mix optimisation, budget reallocation decisions, understanding the
roles of awareness vs. conversion channels.

---

### Funnel Analysis

Measures conversion through an ordered sequence of steps, identifying where users drop off.

Input data should have one boolean column per funnel step (True = completed). Pass `steps` to
specify the column order; if omitted, columns are used in their natural order.

Output includes `conversion_rates` (cumulative and step-to-step), `drop_off_rates`, detected
`bottlenecks`, and `optimization_recommendations`.

**When to use:** Onboarding analysis, checkout flow optimisation, feature adoption tracking.

---

## Composition

The business intelligence domain is designed to chain with other domains:

| After BI analysis | Chain to | Purpose |
|---|---|---|
| `analyze_rfm` | Regression/Modeling | Predict CLV from RFM features |
| `analyze_rfm` | Statistical Analysis | Test whether segment differences are significant |
| `analyze_ab_test` | Statistical Analysis (`enhanced_ab_test`) | Bayesian layer on frequentist results |
| Cohort retention table | Time Series Analysis | Forecast future retention curves |
| Funnel step counts | Statistical Analysis | Test funnel step improvement significance |

Results do not carry composition metadata -- nothing in one names the analysis that produced it, so chaining is the caller's work (issue #26). Read a value out of one result and narrow the next query with it.

---

## Examples

### Segment an e-commerce customer base

```python
analyze_rfm(
    "ecommerce_db",
    "SELECT customer_id, order_date, order_total FROM orders WHERE order_date >= '2023-01-01'",
    customer_column="customer_id",
    date_column="order_date",
    value_column="order_total",
)
```

The returned JSON carries `segment_summary` — count and average metrics per
segment — and `rfm_scores`, the per-customer scores. The same call as an MCP
request:

```json
{
  "tool": "analyze_rfm",
  "arguments": {
    "connection_name": "ecommerce_db",
    "query": "SELECT customer_id, order_date, order_total FROM orders",
    "customer_column": "customer_id",
    "date_column": "order_date",
    "value_column": "order_total"
  }
}
```

### Evaluate a checkout redesign experiment

```json
{
  "tool": "analyze_ab_test",
  "arguments": {
    "connection_name": "analytics_db",
    "query": "SELECT variant, converted FROM checkout_experiment WHERE experiment_id = 42",
    "variant_column": "variant",
    "metric_column": "converted",
    "alpha": 0.05
  }
}
```

### Multi-step workflow: RFM then CLV

This one is the Python domain API, not the MCP surface. `analyze_rfm` here is
`localdata_mcp.domains.business_intelligence.analyze_rfm`, which takes a
DataFrame and an `amount_column`; the MCP tool of the same name takes a
connection and a query and calls its parameter `value_column`. `calculate_clv`
has no MCP tool at all.

```python
import pandas as pd
from localdata_mcp.domains.business_intelligence import analyze_rfm, calculate_clv

# df has columns customer_id, date, amount; date must be datetime, not string
df["date"] = pd.to_datetime(df["date"])

# Step 1: Segment customers
rfm = analyze_rfm(df, customer_column="customer_id", date_column="date", amount_column="amount")

# Step 2: Calculate CLV and compare across RFM segments
clv = calculate_clv(df, customer_column="customer_id", date_column="date", amount_column="amount")

# Step 3: Join and analyse — Champions should have highest CLV
merged = pd.DataFrame(rfm.rfm_scores).merge(
    pd.DataFrame(clv.clv_scores), on="customer_id"
)
```

Two constraints this example depends on:

- `calculate_clv` accepts `customer_column` but does not honour it — it groups on
  a column literally named `customer_id` and raises `KeyError: 'customer_id'`
  otherwise. Rename the column before calling it.
- `analyze_rfm` keeps whatever the customer column was called, so the merge key
  above is `customer_id` only because the input column is. A dimension with no
  spread — every customer placing the same number of orders, say — has no
  ordering to express, so each customer receives the same neutral score on that
  dimension and is separated by the other two.
