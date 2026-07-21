---
name: hypothesis-test
description: Run a hypothesis test with assumption checks, test selection, and plain-language interpretation. Use when comparing groups or testing a specific statistical claim.
allowed-tools: mcp__localdata__describe_database mcp__localdata__execute_query mcp__localdata__analyze_hypothesis_test mcp__localdata__analyze_anova mcp__localdata__analyze_effect_sizes mcp__localdata__get_data_quality_report
argument-hint: "<database-name>"
---

# Hypothesis Test

Select and run the appropriate statistical test based on data characteristics, with full assumption checking and effect size reporting.

## Steps

1. **Explore the data.** Call `describe_database` with the database name from `$ARGUMENTS`. Identify the columns to compare and the grouping variable. Call `get_data_quality_report` to check for missing values that could bias results.

2. **Extract and inspect.** Call `execute_query` to pull the relevant columns. Determine:
   - Number of groups (2 vs 3+)
   - Sample size per group
   - Whether observations are paired or independent
   - Whether the outcome is continuous or categorical

3. **Check assumptions.** Call `execute_query` to compute summary statistics per group (mean, median, sd, skewness). Assess:
   - Normality: skewness beyond +/-1 or small samples (n < 30) suggest non-parametric tests
   - Variance homogeneity: ratio of largest to smallest group SD above 2 suggests unequal variances
   - Sample balance: highly unequal group sizes affect test power

4. **Select and run the test.** Based on the assessment:
   Call `analyze_hypothesis_test` with `test_type="auto"` (the default) and a `group_column`, and it selects the test from the data's own assumptions. There is no `test_type` value naming a specific test, so do not pass one:
   - 2 groups, normal: Welch's or Student's t-test, chosen on Levene's test
   - 2 groups, non-normal: Mann-Whitney U with a rank-biserial effect size
   - 3+ groups, normal and homoscedastic: one-way ANOVA with eta squared
   - 3+ groups otherwise: Kruskal-Wallis H with epsilon squared
   - Paired designs are not supported -- there is no paired t-test and no Wilcoxon signed-rank on this surface
   - For a full ANOVA table with Tukey post-hoc comparisons, call `analyze_anova` instead (it takes no `post_hoc` parameter; Tukey is the only one implemented)

5. **Compute effect sizes.** Call `analyze_effect_sizes` with the same data. Report Cohen's d (two groups), eta-squared (ANOVA), or Cramer's V (chi-squared). Classify as small, medium, or large.

6. **Interpret results.** Present:
   - Hypotheses stated in plain language
   - Test selected and why (citing assumption check results)
   - Test statistic, degrees of freedom, p-value
   - Effect size with confidence interval
   - One-sentence conclusion: what this means for the question at hand

7. **Flag caveats.** Report any assumption violations, small sample warnings, or multiple comparison adjustments. If the result is statistically significant but the effect is trivial, say so explicitly.
