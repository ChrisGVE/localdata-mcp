# Business Intelligence Domain

## Overview

The business intelligence domain provides customer analytics, experiment analysis, and marketing
measurement tools. It is designed for workflows that start with transactional or event data stored
in a connected database and need to produce actionable business metrics.

Use this domain when you need to:

- Segment customers by engagement and value (RFM)
- Measure the statistical significance of a product or marketing experiment (A/B testing)
- Track how cohorts of customers retain over time
- Estimate the lifetime value of customer segments
- Attribute revenue across marketing touchpoints
- Identify bottlenecks in a multi-step user funnel

All tools accept a SQL query, execute it against the named connection, and return a
JSON-serializable result. Row loading is capped at 500,000 rows by default; pass `max_rows`
to override.

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
    {"customer_id": "...", "recency": 5, "frequency": 3, "monetary": 4,
     "rfm_score": 12, "segment": "Champions"}
  ],
  "segments": [...],
  "segment_summary": [
    {"segment": "Champions", "count": 120, "avg_recency": 4.8, ...}
  ],
  "quartile_boundaries": {
    "recency": [7.0, 30.0, 90.0],
    "frequency": [1.0, 3.0, 8.0],
    "monetary": [50.0, 200.0, 800.0]
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

Each dimension is ranked into quartiles (1–5, higher is better for all three). The quartile
boundaries are stored in the result so segments are reproducible across runs.

Customers are assigned to named segments based on combined score patterns. Common segments include
Champions (5,5,5), At Risk (2–3 on recency with high F/M), and Lost (low on all three).

**When to use:** Initial customer health audit, churn prevention targeting, email list tiering,
loyalty programme design.

**Key parameters:**

- `analysis_date` — pass an explicit ISO date string to fix the reference date for recency; omit
  to use today
- `customer_column`, `date_column`, `amount_column` — column name mappings on the underlying
  `analyze_rfm` function (mapped from `customer_column`/`date_column`/`value_column` at the tool
  level)

**Interpretation:** Customers in the Champions segment are high-value and recent — they should be
rewarded and surveyed. At Risk customers had strong past behaviour but are lapsing — they warrant
intervention. Compare `segment_summary` counts across time periods to track portfolio health.

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

Every result includes enough metadata for downstream tools to understand data provenance.

---

## Examples

### Segment an e-commerce customer base

```python
result = analyze_rfm(
    connection_name="ecommerce_db",
    query="SELECT customer_id, order_date, order_total FROM orders WHERE order_date >= '2023-01-01'",
    customer_column="customer_id",
    date_column="order_date",
    value_column="order_total",
)
# result.segment_summary shows count and avg metrics per segment
# result.rfm_scores provides per-customer scores for downstream modelling
```

Via MCP tool call:

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

```python
# Step 1: Segment customers
rfm = analyze_rfm(data=df, customer_column="cid", date_column="dt", amount_column="amt")

# Step 2: Calculate CLV and compare across RFM segments
clv = calculate_clv(data=df, customer_column="cid", date_column="dt", amount_column="amt")

# Step 3: Join and analyse — Champions should have highest CLV
import pandas as pd
merged = pd.DataFrame(rfm.rfm_scores).merge(
    pd.DataFrame(clv.clv_scores), on="customer_id"
)
```
