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
| Bonferroni post-hoc | `ANOVAAnalysisTransformer` | Named but not implemented; no MCP tool selects it |
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

The domain is reached through three MCP tools. Like every analytical tool except the four optimization tools,
each takes the name of a live connection and a SQL query — there is no
data-frame parameter and no separate load step, and column parameters name
columns in the query's result set. The classes listed under *Available Analyses*
above are the internal implementation those tools call; they are not reachable
from an MCP client.

Full parameter tables for all three live in the
[tools reference](../tools-reference.md#data-science-12-tools). This page covers
what each tool is for and when to reach for it.

### `analyze_hypothesis_test`

Answers "is this difference real, or is it noise?" Set `test_type` to
`ttest_1samp`, `ttest_ind`, `ttest_rel`, `chi2`, `normality` or `correlation`;
the default `auto` picks from the shape of the data. `column` names the numeric
column under test and `group_column` the grouping column for two-sample tests;
`alpha` (default 0.05) and `alternative` (`two-sided`, `less`, `greater`)
control the decision rule. Returns a `test_results` list, one entry per test,
each with its statistic, p-value and an interpretation string.

Called with `test_type="auto"` and no `column`, it reports normality and
correlation checks across every numeric column — a cheap first look before
committing to a specific test.

### `analyze_anova`

Answers "do these three or more groups differ?" — the case a t-test cannot
handle without inflating the false-positive rate. `dependent_var` is the numeric
outcome, `group_var` the categorical factor. Returns the F-statistic, its
p-value, per-group means and an effect size. A significant F says at least one
group differs, not which one.

### `analyze_effect_sizes`

Answers "does the difference matter?" — the question significance leaves open, since
a large enough sample makes a trivial gap significant. Takes `column` and
`group_column` and nothing else; the measure follows from the data (Cohen's d
for two numeric groups, Cramer's V for categorical association). Returns the
value with a plain-language magnitude label.

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

Post-hoc comparisons run only when the ANOVA is significant and there are more than two groups, and the test is always **Tukey HSD**, which controls the familywise error rate and suits roughly equal group sizes.

`analyze_anova` exposes no `post_hoc` parameter, so the choice is not yours to make from an MCP client. The underlying transformer names `bonferroni` and `scheffe` as alternatives but implements neither — selecting one returns silently with no comparisons at all — so treat Tukey as the only post-hoc this domain performs.

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

Each step is a separate call. `test_results` and the effect-size values come back
as JSON to the caller, not as a handle another tool can consume — deciding what
the next call should be, and with which columns, is the caller's work.

---

## Examples

Every example below is an MCP tool call, the way an agent would issue it.

### Does revenue differ between two regions?

```python
analyze_hypothesis_test(
    "sales", "SELECT revenue, region FROM orders WHERE year = 2024",
    test_type="ttest_ind", column="revenue", group_column="region",
)
```

The first entry of `test_results` carries the t-statistic, its p-value and an
interpretation. A p-value below `alpha` says the two regions differ; it says
nothing about by how much — follow with `analyze_effect_sizes`.

### Do satisfaction scores differ across product categories?

```python
analyze_anova(
    "survey", "SELECT satisfaction_score, product_category FROM responses",
    dependent_var="satisfaction_score", group_var="product_category",
)
```

Three or more groups need ANOVA rather than repeated t-tests, which would
inflate the false-positive rate with every extra pair compared.

### Is the pilot effect large enough to be worth a full study?

```python
analyze_effect_sizes(
    "pilot", "SELECT conversion, variant FROM data_table",
    column="conversion", group_column="variant",
)
```

`pilot` here is a CSV connection, so its single table is `data_table`. A
`negligible` or `small` label on a significant result usually means the sample
was large, not that the treatment worked.

### Check normality before choosing a test

```python
analyze_hypothesis_test(
    "logs", "SELECT response_time FROM api_calls", test_type="normality",
)
```

Shapiro-Wilk and Kolmogorov-Smirnov both run. If either rejects normality, a
t-test on that column rests on an assumption the data does not meet. The
non-parametric transformers listed under *Available Analyses* have no MCP tool
of their own in this release, but two of them are reached automatically: with
`test_type="auto"` and a `group_column`, non-normal data selects **Mann-Whitney
U** for two groups (reported with a rank-biserial effect size) and
**Kruskal-Wallis H** for three or more (with epsilon squared). You get the
distribution-free test by letting the tool choose, not by naming it. Wilcoxon
signed-rank and Friedman are not reachable at all, and neither is a paired
design. The rank correlation reported by `test_type="correlation"` is
distribution-free too.

### Is response time associated with payload size?

```python
analyze_hypothesis_test(
    "logs", "SELECT response_time, payload_bytes FROM api_calls",
    test_type="correlation",
)
```

Both Pearson and Spearman coefficients are reported. A large gap between them
points at a monotone but non-linear relationship, which a linear model in
`analyze_regression` will fit badly.
