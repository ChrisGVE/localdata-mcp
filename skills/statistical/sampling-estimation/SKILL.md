---
name: sampling-estimation
description: Design sampling strategies and compute estimates with confidence intervals using bootstrap, Bayesian, or classical methods. Use when estimating population parameters from sample data.
allowed-tools: mcp__localdata__describe_database mcp__localdata__execute_query mcp__localdata__analyze_hypothesis_test mcp__localdata__analyze_effect_sizes mcp__localdata__get_data_quality_report
argument-hint: "<database-name>"
---

# Sampling and Estimation

Design a sampling strategy and compute estimates with proper uncertainty quantification.

## Steps

1. **Understand the estimation goal.** Identify what parameter needs to be estimated (mean, proportion, difference, ratio) and the target population from the user's question.

2. **Profile the data.** Call `describe_database` and `get_data_quality_report` with the database name from `$ARGUMENTS`. Assess available sample size, data completeness, and any stratification variables present in the data.

3. **Assess sample representativeness.** Call `execute_query` to examine the distribution of key demographic or stratification variables. Determine whether the sample is a plausible representation of the target population. Flag potential selection biases.

4. **Compute point estimates.** Call `execute_query` to calculate the sample statistic of interest (mean, proportion, median, etc.) along with summary statistics (n, SD, IQR) needed for confidence interval construction.

5. **Construct confidence intervals.** Based on the data characteristics:
   - Large sample, normal: use classical parametric intervals
   - Small sample or skewed: describe bootstrap approach (resample with replacement, compute statistic on each resample, use percentile method for CI)
   - Proportion near 0 or 1: use Wilson or Clopper-Pearson interval rather than Wald

6. **Compute required sample size.** If the user needs to plan future data collection, calculate the sample size needed for a target margin of error. Report assumptions about expected variability and confidence level.

7. **Report estimates.** Present:
   - Point estimate with units
   - Confidence interval (95% default, note level)
   - Margin of error
   - Sample size and effective sample size (accounting for missing data)
   - Method used for interval construction

8. **Discuss limitations.** Address sampling bias, non-response, measurement error, and any extrapolation concerns. Distinguish between the precision of the estimate (narrow CI) and its accuracy (freedom from bias).
