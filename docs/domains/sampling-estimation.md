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

All transformers are sklearn-compatible (`BaseEstimator`, `TransformerMixin`). High-level
functions accept a DataFrame or a file path and return JSON-serializable dictionaries.

---

## Available Analyses

| Analysis | Function | Description |
|---|---|---|
| Simple random sampling | `generate_sample` with `method="simple_random"` | Uniform random selection without replacement |
| Stratified sampling | `generate_sample` with `method="stratified"` | Proportional allocation across strata |
| Cluster sampling | `generate_sample` with `method="cluster"` | Select random clusters, take all members |
| Systematic sampling | `generate_sample` with `method="systematic"` | Regular interval selection with random start |
| Weighted sampling | `generate_sample` with `method="weighted"` | Probability-proportional-to-size sampling |
| Percentile bootstrap CI | `bootstrap_statistic` with `method="percentile"` | Distribution-free confidence intervals |
| BCa bootstrap CI | `bootstrap_statistic` with `method="bca"` | Bias-corrected and accelerated intervals |
| Basic bootstrap | `bootstrap_statistic` with `method="basic"` | Pivotal confidence intervals |
| Studentised bootstrap | `bootstrap_statistic` with `method="studentized"` | Bootstrap-t intervals |
| Monte Carlo integration | `monte_carlo_simulate` with `type="integration"` | Numerical integration by random sampling |
| Monte Carlo simulation | `monte_carlo_simulate` with `type="simulation"` | Forward uncertainty propagation |
| Importance sampling | `monte_carlo_simulate` with `type="importance_sampling"` | Variance reduction for rare events |
| Posterior estimation | `bayesian_estimate` with `type="posterior"` | Bayesian parameter estimation |
| Bayesian updating | `bayesian_estimate` with `type="updating"` | Sequential belief update |
| Credible intervals | `bayesian_estimate` | Highest density interval (HDI) or equal-tailed CI |

---

## MCP Tool Reference

### `generate_sample`

Draw a sample from a dataset using a chosen sampling method.

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `data` | DataFrame or str | required | Input DataFrame or path to CSV/JSON file |
| `sampling_method` | str | `"simple_random"` | Sampling method (see table above) |
| `sample_size` | int or float | `0.1` | Absolute count (int) or fraction of population (float 0–1) |
| `random_state` | int | None | Seed for reproducibility |
| `stratify_column` | str | None | Column to stratify by (required for `stratified`) |
| `cluster_column` | str | None | Column with cluster labels (optional for `cluster`) |
| `weights_column` | str | None | Column with sampling weights (required for `weighted`) |
| `replacement` | bool | `False` | Sample with replacement |

**Return format**

```json
{
  "sample_data": [
    {"col_a": 1.2, "col_b": "foo"},
    ...
  ],
  "sampling_results": {
    "sampling_method": "stratified",
    "sample_size": 500,
    "population_size": 5000,
    "sampling_params": {"stratify_column": "region", "replacement": false},
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
| `data` | DataFrame or str | required | Input DataFrame or file path |
| `statistic_func` | callable or str | `"mean"` | Statistic to bootstrap; string names: `mean`, `median`, `std`, `var`, `sum` |
| `n_bootstrap` | int | `1000` | Number of bootstrap resamples |
| `confidence_level` | float | `0.95` | Confidence level (e.g., 0.95 for 95% CI) |
| `method` | str | `"percentile"` | Interval method: `percentile`, `bca`, `basic`, `studentized` |
| `random_state` | int | None | Seed for reproducibility |

**Return format**

```json
{
  "statistic_name": "mean",
  "original_statistic": 42.7,
  "bootstrap_method": "percentile",
  "n_bootstrap": 1000,
  "bias_estimate": 0.03,
  "bias_corrected_estimate": 42.67,
  "variance_estimate": 1.24,
  "standard_error": 1.11,
  "confidence_intervals": {
    "percentile": [40.5, 44.9],
    "bca": [40.3, 44.7]
  },
  "convergence_info": {"bootstrap_se_stability": 0.02}
}
```

---

### `monte_carlo_simulate`

Run a Monte Carlo simulation or numerical integration.

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `data` | DataFrame or str | required | Input data for simulation parameters |
| `simulation_type` | str | `"integration"` | Type: `integration`, `simulation`, `importance_sampling` |
| `n_simulations` | int | `10000` | Number of simulation draws |
| `random_state` | int | None | Seed for reproducibility |

**Return format**

```json
{
  "simulation_type": "simulation",
  "n_simulations": 10000,
  "estimated_value": 18.4,
  "confidence_interval": [17.1, 19.7],
  "standard_error": 0.66,
  "convergence_diagnostic": {
    "relative_error": 0.004,
    "effective_sample_size": 9800
  },
  "simulation_params": {...}
}
```

---

### `bayesian_estimate`

Perform Bayesian parameter estimation with credible intervals.

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `data` | DataFrame or str | required | Input DataFrame or file path |
| `estimation_type` | str | `"posterior"` | Type: `posterior`, `updating` |
| `prior_distribution` | str | `"normal"` | Prior: `normal`, `beta`, `gamma`, `uniform` |
| `confidence_level` | float | `0.95` | Credible interval level |
| `random_state` | int | None | Seed for reproducibility |

**Return format**

```json
{
  "parameter_name": "mu",
  "estimation_method": "posterior",
  "posterior_mean": 5.23,
  "posterior_mode": 5.19,
  "posterior_median": 5.21,
  "credible_intervals": {
    "equal_tailed": [4.81, 5.65],
    "hdi": [4.79, 5.62]
  },
  "prior_info": {"distribution": "normal", "params": {"loc": 0, "scale": 10}},
  "bayes_factor": 12.4,
  "mcmc_diagnostics": {"r_hat": 1.002, "ess": 3200}
}
```

---

## Method Details

### Sampling Methods

#### Simple Random Sampling

Selects rows uniformly at random. The default and simplest method. Use when the population is
homogeneous or when no auxiliary information is available to guide allocation.

With `replacement=False` (default), each row appears at most once. With `replacement=True`,
the same row can appear multiple times (needed for bootstrap-style samples).

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

#### Weighted Sampling

Samples rows with probability proportional to values in `weights_column`. Weights are normalised
to sum to 1 internally. Use with `replacement=True` for importance sampling applications.

**When to use:** Oversampling rare events, inverse-probability-of-treatment weighting (IPTW),
upweighting recent records.

---

### Bootstrap Resampling

Bootstrap methods estimate the sampling distribution of a statistic by resampling with
replacement from the observed data. No parametric distributional assumptions are required.

**n_bootstrap recommendations:**

- 1000 for exploratory work and interval width estimation
- 5000–10000 for stable BCa intervals or tail probabilities
- 10000+ for p-values and when the statistic has high variability

**CI methods comparison:**

| Method | When to prefer |
|---|---|
| `percentile` | Symmetric distributions; large samples; quick results |
| `bca` | Default recommendation; corrects for bias and skewness automatically |
| `basic` | Alternative when distribution is approximately symmetric |
| `studentized` | When studentisation (dividing by bootstrap SE) is feasible; more accurate for small samples |

**Bias correction:** When `bias_estimate` is non-negligible relative to `standard_error`, use
`bias_corrected_estimate` as the point estimate instead of `original_statistic`.

**Statistic functions:** Pass a string name (`mean`, `median`, `std`, `var`, `sum`) or a Python
callable `f(x) -> float` that operates on a 1D NumPy array.

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
| `importance_sampling` | Reduce variance for rare-event probabilities by sampling from a proposal distribution |

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
- `hdi` — Highest Density Interval; the narrowest interval containing the specified probability
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

### Draw a stratified sample for a survey

```python
result = generate_sample(
    data=customer_df,
    sampling_method="stratified",
    sample_size=1000,
    stratify_column="region",
    random_state=42,
)
sample = pd.DataFrame(result["sample_data"])
print(result["sampling_results"]["strata_info"])
```

### Bootstrap a median with BCa intervals

```python
result = bootstrap_statistic(
    data=revenue_df,
    statistic_func="median",
    n_bootstrap=5000,
    confidence_level=0.95,
    method="bca",
    random_state=0,
)
print(f"Median: {result['original_statistic']:.2f}")
print(f"95% BCa CI: {result['confidence_intervals']['bca']}")
```

### Custom statistic: interquartile range

```python
import numpy as np

result = bootstrap_statistic(
    data=df,
    statistic_func=lambda x: np.percentile(x, 75) - np.percentile(x, 25),
    n_bootstrap=2000,
    confidence_level=0.90,
)
```

### Monte Carlo uncertainty propagation

```python
result = monte_carlo_simulate(
    data=model_params_df,
    simulation_type="simulation",
    n_simulations=50000,
    random_state=1,
)
print(f"Expected output: {result['estimated_value']:.3f} ± {result['standard_error']:.3f}")
print(f"90% CI: {result['confidence_interval']}")
```

### Bayesian estimation of a conversion rate

```python
# Prior: Beta(2, 20) — weak prior of ~9% conversion
result = bayesian_estimate(
    data=experiment_df,
    estimation_type="posterior",
    prior_distribution="beta",
    confidence_level=0.95,
)
print(f"Posterior mean: {result['posterior_mean']:.3f}")
print(f"95% HDI: {result['credible_intervals']['hdi']}")
```

### Full workflow: sample then analyse

```python
# 1. Draw a stratified 20% sample
sample_result = generate_sample(
    data=large_df,
    sampling_method="stratified",
    sample_size=0.2,
    stratify_column="product_category",
    random_state=7,
)
sample_df = pd.DataFrame(sample_result["sample_data"])

# 2. Bootstrap the mean order value on the sample
ci_result = bootstrap_statistic(
    data=sample_df[["order_value"]],
    statistic_func="mean",
    n_bootstrap=2000,
    method="bca",
)
print(f"Mean order value: {ci_result['original_statistic']:.2f}")
print(f"95% CI: {ci_result['confidence_intervals']['bca']}")
```
