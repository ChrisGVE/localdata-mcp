# Statistical Analysis Domain

## Overview

The statistical analysis domain provides hypothesis testing, ANOVA, non-parametric tests, and experimental design tools. Use it when you need to determine whether observed differences between groups are statistically significant, quantify effect sizes, or design experiments with adequate statistical power.

**When to use this domain:**

- Comparing means or distributions between two or more groups
- Testing associations between categorical variables
- Checking whether data meets normality assumptions before other analyses
- Quantifying the practical magnitude of a difference (effect size)
- Estimating required sample sizes for planned studies

**Source:** `src/localdata_mcp/domains/statistical_analysis/`

---

## Available Analyses

| Method | Class / Function | Description |
|---|---|---|
| One-sample t-test | `HypothesisTestingTransformer` | Test whether a sample mean differs from a known value |
| Independent t-test | `HypothesisTestingTransformer` | Compare means from two independent groups |
| Paired t-test | `HypothesisTestingTransformer` | Compare means from two related measurements |
| Chi-square test | `HypothesisTestingTransformer` | Test independence between two categorical variables |
| Normality tests | `HypothesisTestingTransformer` | Shapiro-Wilk and Kolmogorov-Smirnov normality checks |
| Pearson / Spearman correlation | `HypothesisTestingTransformer` | Test linear and rank correlations between numeric variables |
| One-way ANOVA | `ANOVAAnalysisTransformer` | Compare means across three or more groups |
| Two-way ANOVA | `ANOVAAnalysisTransformer` | Test main effects and interactions of two factors |
| Tukey HSD post-hoc | `ANOVAAnalysisTransformer` | Pairwise comparisons after significant ANOVA |
| Bonferroni post-hoc | `ANOVAAnalysisTransformer` | Conservative pairwise corrections |
| Mann-Whitney U | `NonParametricTestTransformer` | Non-parametric two-group comparison |
| Wilcoxon signed-rank | `NonParametricTestTransformer` | Non-parametric paired comparison |
| Kruskal-Wallis H | `NonParametricTestTransformer` | Non-parametric multi-group comparison |
| Friedman test | `NonParametricTestTransformer` | Non-parametric repeated-measures test |
| Cohen's d | `ExperimentalDesignTransformer` | Standardized mean difference effect size |
| Eta-squared / Omega-squared | `ANOVAAnalysisTransformer` | ANOVA effect size measures |
| Cramer's V | `HypothesisTestingTransformer` | Effect size for chi-square associations |
| Confidence intervals | `ExperimentalDesignTransformer` | Interval estimates for means and correlations |
| Power analysis | `ExperimentalDesignTransformer` | Required sample size for a given power level |

---

## MCP Tool Reference

The domain exposes three MCP tools via `src/localdata_mcp/datascience_tools.py`.

### `tool_hypothesis_test`

Run hypothesis tests on data retrieved from a SQL query.

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `engine` | `Engine` | required | SQLAlchemy engine from an active connection |
| `query` | `str` | required | SQL query returning the data to analyse |
| `test_type` | `str` | `"auto"` | Test to run: `"auto"`, `"ttest_1samp"`, `"ttest_ind"`, `"ttest_rel"`, `"chi2"`, `"normality"`, `"correlation"` |
| `column` | `str` | `None` | Target numeric column for focused testing |
| `group_column` | `str` | `None` | Column defining groups for two-sample tests |
| `alpha` | `float` | `0.05` | Significance level |
| `alternative` | `str` | `"two-sided"` | Direction: `"two-sided"`, `"less"`, `"greater"` |
| `max_rows` | `int` | `None` | Row cap (default 500,000) |

**Returns:** `dict` with keys:

- `test_results` — list of test result objects, each with `test_name`, `statistic`, `p_value`, `degrees_of_freedom`, `effect_size`, `interpretation`
- `assumptions_checked` — dict of assumption check results
- `effect_sizes` — dict of calculated effect sizes
- `alpha_level` — significance level used
- `correction_applied` — multiple comparison correction method if any

---

### `tool_anova_analysis`

Perform one-way or two-way ANOVA with post-hoc tests.

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `engine` | `Engine` | required | SQLAlchemy engine |
| `query` | `str` | required | SQL query returning the data |
| `dependent_var` | `str` | required | Numeric outcome column |
| `group_var` | `str` | required | Categorical grouping column |
| `alpha` | `float` | `0.05` | Significance level |
| `max_rows` | `int` | `None` | Row cap |

Underlying `ANOVAAnalysisTransformer` also accepts:

| Parameter | Type | Default | Description |
|---|---|---|---|
| `anova_type` | `str` | `"one_way"` | `"one_way"`, `"two_way"`, `"auto"` |
| `post_hoc` | `str` | `"tukey"` | `"tukey"`, `"bonferroni"`, `"scheffe"`, `None` |
| `effect_size` | `str` | `"eta_squared"` | `"eta_squared"`, `"partial_eta_squared"`, `"omega_squared"` |
| `check_assumptions` | `bool` | `True` | Run Shapiro-Wilk and Levene's tests before ANOVA |

**Returns:** `dict` with keys:

- `anova_results` — F-statistic, p-value, degrees of freedom, group means and sizes, interpretation
- `post_hoc_results` — pairwise comparisons with adjusted p-values and confidence intervals
- `effect_sizes` — eta-squared and omega-squared per factor
- `assumptions_checked` — normality and homoscedasticity check results

---

### `tool_effect_sizes`

Calculate standardized effect sizes (Cohen's d, Cramer's V, correlation r).

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `engine` | `Engine` | required | SQLAlchemy engine |
| `query` | `str` | required | SQL query |
| `column` | `str` | required | Numeric column to analyse |
| `group_column` | `str` | required | Categorical grouping column |
| `max_rows` | `int` | `None` | Row cap |

**Returns:** `dict` with keys:

- `effect_sizes` — Cohen's d (two-group) or Cramer's V (categorical), with `effect_description` (`negligible`, `small`, `medium`, `large`)
- `confidence_intervals` — interval estimates for means and correlations
- `power_analysis` — power curve across sample sizes for the observed effect
- `sample_sizes` — required n per group for small/medium/large effects

---

## Method Details

### T-tests

**One-sample t-test** (`ttest_1samp`): Tests whether the mean of a single sample differs from a hypothesised population mean (default 0). Requires at least 3 observations.

**Independent t-test** (`ttest_ind`): Compares means from two separate groups. Requires a numeric column and a binary categorical grouping column. The `equal_var` parameter (default `True`) switches between Student's t and Welch's correction.

**Paired t-test** (`ttest_rel`): Compares two numeric columns measured on the same subjects. Cohen's d is computed from the paired differences.

**Effect size interpretation for Cohen's d:**

| Range | Label |
|---|---|
| < 0.2 | negligible |
| 0.2 – 0.5 | small |
| 0.5 – 0.8 | medium |
| ≥ 0.8 | large |

---

### Chi-square Test

Tests whether two categorical variables are independent. A contingency table is constructed automatically. Cramer's V is reported as the effect size.

**Effect size interpretation for Cramer's V (2×2 table):**

| Range | Label |
|---|---|
| < 0.1 | negligible |
| 0.1 – 0.3 | small |
| 0.3 – 0.5 | medium |
| ≥ 0.5 | large |

---

### Normality Tests

Two tests run in parallel:

- **Shapiro-Wilk** — preferred for n ≤ 5,000; sensitive to small departures from normality in large samples
- **Kolmogorov-Smirnov** — used for all sample sizes; slightly less powerful than Shapiro-Wilk for small samples

If p > alpha, data is treated as approximately normal.

---

### ANOVA

**One-way ANOVA**: Tests whether at least one group mean differs from the others. Uses `scipy.stats.f_oneway`. Assumptions are checked automatically (normality via Shapiro-Wilk per group; homoscedasticity via Levene's test).

Post-hoc tests run only when ANOVA is significant and there are more than two groups:
- **Tukey HSD** — controls familywise error rate; appropriate when group sizes are roughly equal
- **Bonferroni** — more conservative; better when only a few specific comparisons are planned
- **Scheffé** — most conservative; appropriate for all possible contrasts

**Two-way ANOVA**: Uses `statsmodels` OLS with interaction term. Reports eta-squared and partial eta-squared per factor.

**Effect size interpretation for eta-squared:**

| Range | Label |
|---|---|
| < 0.01 | negligible |
| 0.01 – 0.06 | small |
| 0.06 – 0.14 | medium |
| ≥ 0.14 | large |

---

### Non-Parametric Tests

Use these when normality assumptions are violated or data is ordinal.

**Mann-Whitney U**: Non-parametric alternative to the independent t-test. Effect size is rank-biserial correlation r.

**Wilcoxon signed-rank**: Non-parametric alternative to the paired t-test. Requires at least 6 paired observations.

**Kruskal-Wallis H**: Non-parametric alternative to one-way ANOVA. Effect size is an eta-squared analogue.

**Friedman test**: Non-parametric repeated-measures test across three or more conditions. Effect size is Kendall's W.

---

### Confidence Intervals

Computed using t-distribution critical values for means and Fisher's z-transformation for correlations. Default confidence level is 95%.

---

### Power Analysis

Power curves are calculated for sample sizes from 10 to 500. Required sample size is solved analytically for the desired power (default 0.80) at alpha = 0.05. Supports t-test, ANOVA, and correlation test types.

---

## Composition

After running statistical analysis, consider chaining:

| Next step | Purpose |
|---|---|
| `regression_modeling` | Model the relationship quantified by a significant correlation or group difference |
| `pattern_recognition` (clustering) | Explore whether statistically different groups correspond to natural data clusters |
| `business_intelligence` (A/B test) | Frame a group comparison as a controlled experiment with business metrics |
| `sampling_estimation` (bootstrap) | Obtain distribution-free confidence intervals when normality is violated |

The `test_results` list and `effect_sizes` dict from this domain pass naturally into regression feature selection and experimental design planning.

---

## Examples

### Basic hypothesis test on sales data

```python
result = tool_hypothesis_test(
    engine=engine,
    query="SELECT revenue, region FROM sales WHERE year = 2024",
    test_type="ttest_ind",
    column="revenue",
    group_column="region",
    alpha=0.05,
)

# Inspect the first test result
first = result["test_results"][0]
print(first["test_name"])        # "Independent t-test (revenue by region)"
print(first["p_value"])          # e.g. 0.0023
print(first["effect_size"])      # Cohen's d
print(first["interpretation"])   # "Significant difference between groups (medium effect)"
```

### ANOVA across multiple product categories

```python
result = tool_anova_analysis(
    engine=engine,
    query="SELECT satisfaction_score, product_category FROM survey",
    dependent_var="satisfaction_score",
    group_var="product_category",
    alpha=0.05,
)

# Check overall significance
for key, anova in result["anova_results"].items():
    print(f"{key}: F={anova['f_statistic']:.3f}, p={anova['p_value']:.4f}")

# Inspect post-hoc comparisons
for key, post_hoc in result["post_hoc_results"].items():
    for comp in post_hoc["comparisons"]:
        if comp["significant"]:
            print(f"{comp['group1']} vs {comp['group2']}: p={comp['p_value']:.4f}")
```

### Effect size calculation before running a study

```python
# Step 1: estimate effect size from pilot data
effect_result = tool_effect_sizes(
    engine=engine,
    query="SELECT conversion, variant FROM pilot_experiment",
    column="conversion",
    group_column="variant",
)

# Step 2: use power analysis to determine required sample size
power = effect_result["power_analysis"]["ttest"]
print(f"Required n per group: {power['required_sample_size']}")
print(power["interpretation"])
```

### Multi-step workflow: normality check then appropriate test

```python
# 1. Check normality first
normality = tool_hypothesis_test(
    engine=engine,
    query="SELECT response_time FROM api_logs",
    test_type="normality",
)

is_normal = all(
    r["p_value"] > 0.05
    for r in normality["test_results"]
    if "Shapiro" in r["test_name"]
)

# 2. Choose parametric or non-parametric test accordingly
query = "SELECT response_time, server_zone FROM api_logs"
if is_normal:
    result = tool_hypothesis_test(
        engine=engine, query=query,
        test_type="ttest_ind",
        column="response_time",
        group_column="server_zone",
    )
else:
    # Use NonParametricTestTransformer directly
    from localdata_mcp.domains.statistical_analysis import NonParametricTestTransformer
    import pandas as pd
    df = pd.read_sql(query, engine)
    transformer = NonParametricTestTransformer(test_type="mann_whitney", alpha=0.05)
    transformer.fit(df)
    result = transformer.transform(df).iloc[0].to_dict()
```
