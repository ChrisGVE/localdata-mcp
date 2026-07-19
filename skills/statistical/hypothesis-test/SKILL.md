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
   - 2 groups, normal, equal variance: call `analyze_hypothesis_test` with independent t-test
   - 2 groups, normal, unequal variance: call `analyze_hypothesis_test` with Welch's t-test
   - 2 groups, non-normal: call `analyze_hypothesis_test` with Mann-Whitney U
   - 2 groups, paired: call `analyze_hypothesis_test` with paired t-test or Wilcoxon signed-rank
   - 3+ groups: call `analyze_anova` with appropriate post-hoc tests
   - Categorical outcome: call `analyze_hypothesis_test` with chi-squared test

5. **Compute effect sizes.** Call `analyze_effect_sizes` with the same data. Report Cohen's d (two groups), eta-squared (ANOVA), or Cramer's V (chi-squared). Classify as small, medium, or large.

6. **Interpret results.** Present:
   - Hypotheses stated in plain language
   - Test selected and why (citing assumption check results)
   - Test statistic, degrees of freedom, p-value
   - Effect size with confidence interval
   - One-sentence conclusion: what this means for the question at hand

7. **Flag caveats.** Report any assumption violations, small sample warnings, or multiple comparison adjustments. If the result is statistically significant but the effect is trivial, say so explicitly.
