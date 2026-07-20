# Sampling & Estimation Domain

## Overview

The sampling and estimation domain provides methods for drawing representative samples from data,
quantifying uncertainty around statistics, and performing probabilistic inference. It covers
classical sampling theory, bootstrap resampling, Monte Carlo simulation, and Bayesian estimation.

Use this domain when you need to:

- Draw a representative subset from a large dataset for faster downstream analysis
- Estimate confidence intervals for a statistic when distributional assumptions are uncertain
- Simulate outcomes or propagate uncertainty through a model using Monte Carlo methods
- Update prior beliefs with observed data and obtain posterior credible intervals

The transformers underneath are sklearn-compatible (`BaseEstimator`,
`TransformerMixin`), but an MCP client does not reach them: it calls the four
tools below, each of which takes a connection name and a SQL query.

---

## Available Analyses

Every value in this table was checked against the running server. A token not
listed here raises rather than falling back to a default.

| Analysis | Call | Description |
|---|---|---|
| Simple random sampling | `generate_sample` with `sampling_method="simple_random"` | Uniform random selection |
| Stratified sampling | `generate_sample` with `sampling_method="stratified"` | Proportional allocation across strata; needs `stratify_column` |
| Systematic sampling | `generate_sample` with `sampling_method="systematic"` | Regular interval selection with a random start |
| Cluster sampling | `generate_sample` with `sampling_method="cluster"` | Select random clusters, take all members |
| Percentile bootstrap CI | `bootstrap_statistic` | Distribution-free confidence interval for `mean`, `median`, `std` or `var` |
| Monte Carlo integration | `monte_carlo_simulate` with `simulation_type="integration"` | Numerical integration by random sampling |
| Uncertainty propagation | `monte_carlo_simulate` with `simulation_type="uncertainty"` | Forward propagation of input uncertainty |
| Importance sampling | `monte_carlo_simulate` with `simulation_type="importance"` | Variance reduction for rare events |
| MCMC | `monte_carlo_simulate` with `simulation_type="mcmc"` | Markov chain Monte Carlo sampling |
| Posterior estimation | `bayesian_estimate` with `estimation_type="posterior"` | Bayesian parameter estimation |
| Credible intervals | `bayesian_estimate` with `estimation_type="credible_interval"` | Equal-tailed interval at `confidence_level` |
| Model comparison | `bayesian_estimate` with `estimation_type="model_comparison"` | Compare candidate models |

The bootstrap reports a percentile interval. There is no BCa, basic or
studentised variant, and no parameter selects one.

---

## MCP Tool Reference

The domain is reached through four MCP tools. Like every other analytical tool,
each takes the name of a live connection and a SQL query — there is no
data-frame parameter and no separate load step, and column parameters name
columns in the query's result set. The classes listed under *Available Analyses*
above are the internal implementation those tools call; they are not reachable
from an MCP client.

Full parameter tables live in the
[tools reference](../tools-reference.md#sampling--estimation-4-tools).

### `generate_sample`

Draw a sample from a dataset using a chosen sampling method.

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `connection_name` | str | required | Name of the connected database |
| `query` | str | required | SQL query returning the population to sample from |
| `sampling_method` | str | `"simple_random"` | `simple_random`, `stratified`, `systematic`, or `cluster` |
| `sample_size` | float | `0.1` | Row count when 1 or greater, fraction of the population when below 1 |
| `columns` | list[str] | None | Restrict the sample to these columns (default: all) |
| `stratify_column` | str | `""` | Column defining strata. Required for `stratified`, which raises without it |

**Return format**

```text
{
  "sample_data": [
    {"col_a": 1.2, "col_b": "foo"},
    ...
  ],
  "sampling_results": {
    "sampling_method": "stratified",
    "sample_size": 500,
    "population_size": 5000,
    "sampling_params": {"stratify_column": "region"},
    "quality_metrics": {
      "representativeness_score": 0.97,
      "mean_absolute_difference": 0.03,
      "std_ratio_mean": 0.99
    },
    "strata_info": {
      "North": {"population_size": 1500, "sample_size": 150, "proportion_in_population": 0.30},
      "South": {"population_size": 3500, "sample_size": 350, "proportion_in_population": 0.70}
    }
  }
}
```

---

### `bootstrap_statistic`

Estimate confidence intervals for a statistic via bootstrap resampling.

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `connection_name` | str | required | Name of the connected database |
| `query` | str | required | SQL query returning the data to resample |
| `column` | str | `""` | Numeric column to bootstrap |
| `statistic` | str | `"mean"` | `mean`, `median`, `std`, or `var`. Any other name raises |
| `n_bootstrap` | int | `1000` | Number of bootstrap resamples |
| `confidence_level` | float | `0.95` | Confidence level, e.g. 0.95 for a 95% interval |

**Return format**

```json
{
  "bootstrap_results": [
    {
      "statistic_name": "mean_value",
      "original_statistic": 10.97,
      "bootstrap_method": "percentile",
      "n_bootstrap": 1000,
      "bias_estimate": 0.013,
      "bias_corrected_estimate": 10.956,
      "variance_estimate": 0.040,
      "standard_error": 0.201,
      "confidence_intervals": {"percentile": [10.67, 11.37]},
      "bootstrap_params": {}
    }
  ],
  "n_bootstrap": 1000,
  "confidence_level": 0.95,
  "method": "percentile"
}
```

---

### `monte_carlo_simulate`

Run a Monte Carlo simulation or numerical integration.

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `connection_name` | str | required | Name of the connected database |
| `query` | str | required | SQL query supplying the simulation parameters |
| `simulation_type` | str | `"integration"` | `integration`, `uncertainty`, `importance`, or `mcmc`. Any other value raises |
| `n_simulations` | int | `10000` | Number of simulation draws |
| `columns` | list[str] | None | Restrict the input to these columns (default: all) |

**Return format**

```json
{
  "monte_carlo_results": [
    {
      "simulation_type": "integration",
      "n_simulations": 10000,
      "estimated_value": 0.9187,
      "confidence_interval": [0.8083, 1.0291],
      "standard_error": 0.0563,
      "convergence_diagnostic": {
        "batch_variance": 0.0371,
        "relative_std_error": 0.0613
      },
      "simulation_params": {"random_state": null},
      "integration_bounds": [-3, 3]
    }
  ],
  "simulation_type": "integration",
  "n_simulations": 10000,
  "confidence_level": 0.95
}
```

---

### `bayesian_estimate`

Perform Bayesian parameter estimation with credible intervals.

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `connection_name` | str | required | Name of the connected database |
| `query` | str | required | SQL query returning the observations |
| `column` | str | `""` | Numeric column to estimate from |
| `estimation_type` | str | `"posterior"` | `posterior`, `credible_interval`, or `model_comparison`. Any other value raises |
| `prior_distribution` | str | `"normal"` | `normal`, `beta`, `gamma`, or `uniform` |
| `confidence_level` | float | `0.95` | Credible interval level |

**Return format**

```json
{
  "bayesian_results": [
    {
      "parameter_name": "mean_value",
      "estimation_method": "normal_conjugate",
      "posterior_mean": 10.964,
      "posterior_median": 10.963,
      "credible_intervals": {"95%": [10.568, 11.361]},
      "prior_info": {
        "distribution": "normal",
        "prior_mu": 0.0,
        "prior_sigma2": 100.0,
        "posterior_mu": 10.964,
        "posterior_sigma2": 0.0409
      }
    }
  ],
  "estimation_type": "posterior",
  "prior_distribution": "normal",
  "prior_params": {},
  "confidence_level": 0.95
}
```

The credible interval is keyed by its level, and there is no separate
highest-density entry: `credible_intervals["95%"]` is the equal-tailed interval
at `confidence_level`. A normal prior takes the conjugate path, which is what
`estimation_method` reports.

---

## Method Details

### Sampling Methods

#### Simple Random Sampling

Selects rows uniformly at random. The default and simplest method. Use when the population is
homogeneous or when no auxiliary information is available to guide allocation.

Each row appears at most once. The tool exposes no with-replacement option; use
`bootstrap_statistic` when you need resampling with replacement.

#### Stratified Sampling

Divides the population into non-overlapping strata defined by `stratify_column`, then samples
from each stratum in proportion to its share of the population. This guarantees representation
of all groups and typically reduces variance compared to simple random sampling.

Output includes `strata_info` showing the population size, sample size, and proportions for each
stratum. The `representativeness_score` (0–1, higher is better) compares stratum means between
the sample and population.

**When to use:** Surveys with demographic subgroups, A/B test allocation, analysis where rare
categories must appear in sufficient numbers.

#### Cluster Sampling

Selects random clusters, then includes all (or a sample of) members from those clusters. If no
`cluster_column` is provided, clusters are created automatically using K-means on numeric columns,
with the number of clusters set to `sqrt(sample_size)`.

More efficient than stratified sampling when travel cost or data collection cost is grouped
geographically or organisationally. Variance is higher than SRS for the same total sample size.

**When to use:** Geographic surveys, school studies (sample schools, then survey all students in
selected schools), log analysis where records cluster by session.

#### Systematic Sampling

Selects every k-th element after a random starting position, where k = population_size /
sample_size. Provides even coverage over an ordered list.

The result includes `sampling_interval` and `starting_point` in `sampling_params`.

**When to use:** Quality control sampling on ordered production lines, time-series subsampling,
sorted database tables where a uniform spread is needed.

There is no weighted sampling tool. To oversample rare events, express the
weighting in the SQL query itself -- filter, or `ORDER BY` a computed weight --
and sample the result.

---

### Bootstrap Resampling

Bootstrap methods estimate the sampling distribution of a statistic by resampling with
replacement from the observed data. No parametric distributional assumptions are required.

**n_bootstrap recommendations:**

- 1000 for exploratory work and interval width estimation
- 5000-10000 when the interval itself is the deliverable
- 10000+ for tail probabilities and when the statistic has high variability

**Interval method:** the percentile interval, always. There is no parameter that
selects BCa, basic or studentised intervals, and no code path that produces one.

**Bias correction:** When `bias_estimate` is non-negligible relative to `standard_error`, use
`bias_corrected_estimate` as the point estimate instead of `original_statistic`.

**Statistic:** one of `mean`, `median`, `std`, `var`. Any other name raises
`Unknown statistic function`, including `sum`. A custom callable cannot cross the
MCP boundary; compute the quantity in SQL and bootstrap the resulting column.

---

### Monte Carlo Simulation

Monte Carlo methods approximate quantities by averaging over random draws. The key result fields:

- `estimated_value` — the Monte Carlo estimate of the target quantity
- `standard_error` — uncertainty of the estimate (decreases as 1/sqrt(n_simulations))
- `confidence_interval` — normal approximation CI around the estimate
- `convergence_diagnostic.relative_error` — SE / estimated_value; below 0.01 indicates good
  convergence

**Simulation types:**

| Type | Description |
|---|---|
| `integration` | Estimate the integral of a function over a domain by uniform random sampling |
| `simulation` | Forward propagation: draw uncertain inputs, compute output distribution |
| `importance` | Reduce variance for rare-event probabilities by sampling from a proposal distribution |

**n_simulations guidance:** Start with 1000 to verify setup, then increase to 10,000–100,000
for stable estimates. Check `convergence_diagnostic.relative_error < 0.01` for 1% accuracy.

---

### Bayesian Estimation

Bayesian estimation combines a prior belief about a parameter with observed data to produce a
posterior distribution.

**Prior distributions:**

| `prior_distribution` | Parameters | Typical use |
|---|---|---|
| `normal` | `loc`, `scale` | Continuous unbounded parameters (mean, regression coefficients) |
| `beta` | `alpha`, `beta` | Probabilities and proportions (0–1 range) |
| `gamma` | `alpha`, `beta` | Positive-valued parameters (rates, variances) |
| `uniform` | `low`, `high` | Completely uninformative over a bounded range |

**Credible intervals vs. confidence intervals:**

A 95% credible interval `[a, b]` means there is a 95% posterior probability that the true
parameter lies in `[a, b]`. This is the intuitive interpretation often (incorrectly) attributed
to frequentist confidence intervals.

Two credible interval types are reported:

- `equal_tailed` — 2.5th to 97.5th percentile of the posterior
- The interval is equal-tailed and keyed by its level, e.g. `credible_intervals["95%"]`. No highest-density interval is computed.
  mass; preferred for skewed posteriors

**Bayes factor:** When available, summarises the evidence ratio between hypotheses. BF > 10
is considered strong evidence; BF > 100 is decisive.

**MCMC diagnostics:**

- `r_hat` — Gelman-Rubin convergence statistic; values < 1.01 indicate convergence
- `ess` — effective sample size; below 400 suggests the chain needs more iterations

---

## Composition

| After sampling/estimation | Chain to | Purpose |
|---|---|---|
| `generate_sample` result | Any domain | All downstream analyses on the sample instead of full data |
| `bootstrap_statistic` CIs | Business Intelligence | Uncertainty-aware reporting of KPIs |
| `bootstrap_statistic` CIs | Statistical Analysis | Non-parametric comparison of two statistics |
| `monte_carlo_simulate` | Regression/Modeling | Uncertainty propagation through a fitted model |
| `bayesian_estimate` posterior | Statistical Analysis | Posterior predictive checks |
| Stratified sample | Regression/Modeling | Balanced training sets for model fitting |

---

## Examples

Each call names a live connection and a SQL query. The query selects the
population; there is no data-frame parameter and no separate load step.

### Draw a stratified sample for a survey

```python
generate_sample(
    "crm",
    "SELECT customer_id, region, spend FROM customers",
    sampling_method="stratified",
    sample_size=1000,
    stratify_column="region",
)
```

`sample_size` is a row count at 1 or above and a fraction below it, so `0.2`
would draw a fifth of the population instead. `stratified` raises without
`stratify_column`.

### Bootstrap a median without assuming a distribution

```python
bootstrap_statistic(
    "sales",
    "SELECT revenue FROM orders WHERE year = 2026",
    column="revenue",
    statistic="median",
    n_bootstrap=5000,
    confidence_level=0.95,
)
```

`statistic` accepts `mean`, `median`, `std` and `var`. There is no hook for a
custom statistic and no interval-method parameter: the result reports the
percentile interval along with the bias estimate and standard error.

### Propagate uncertainty through a Monte Carlo simulation

```python
monte_carlo_simulate(
    "model",
    "SELECT rate, volume, margin FROM parameters",
    simulation_type="uncertainty",
    n_simulations=50000,
)
```

`simulation_type` accepts `integration`, `uncertainty`, `importance` and `mcmc`.
Any other value raises rather than falling back to a default.

### Estimate a conversion rate with a Beta prior

```python
bayesian_estimate(
    "experiment",
    "SELECT converted FROM trials",
    column="converted",
    estimation_type="posterior",
    prior_distribution="beta",
    confidence_level=0.95,
)
```

`estimation_type` accepts `posterior`, `credible_interval` and
`model_comparison`.

### Sample first, then estimate on the sample

The two steps do not chain automatically — a result carries no handle the next
tool consumes. Narrow the second query to the population the first described:

```python
generate_sample(
    "warehouse",
    "SELECT order_value, product_category FROM orders",
    sampling_method="stratified",
    sample_size=0.2,
    stratify_column="product_category",
)

bootstrap_statistic(
    "warehouse",
    "SELECT order_value FROM orders WHERE product_category = 'hardware'",
    column="order_value",
    statistic="mean",
    n_bootstrap=2000,
)
```
