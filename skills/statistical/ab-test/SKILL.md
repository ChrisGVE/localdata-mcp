---
name: ab-test
description: Analyze an A/B test experiment and provide a clear ship/iterate/no-ship recommendation. Use when evaluating experiment results.
allowed-tools: mcp__localdata__describe_database mcp__localdata__execute_query mcp__localdata__analyze_ab_test mcp__localdata__analyze_effect_sizes
argument-hint: "<database-name>"
---

# A/B Test Analysis

Analyze experiment results and deliver a clear ship, iterate, or no-ship recommendation.

## Steps

1. **Explore experiment data.** Call `describe_database` with the database name from `$ARGUMENTS`. Identify the table containing experiment results. Look for columns indicating: group assignment (treatment/control), the primary metric, and any segmentation variables.

2. **Inspect the data.** Call `execute_query` to sample rows from the experiment table. Confirm the group labels (e.g., "control" vs "treatment", "A" vs "B"). Note the metric column name and type. Check for any timestamp or user ID columns.

3. **Check group balance.** Call `execute_query` with a query that counts observations per group and computes mean/stddev of the primary metric per group. Verify that group sizes are roughly balanced (within 10% of each other). Flag any imbalance as a potential validity concern.

4. **Run the A/B test.** Call `analyze_ab_test` with the database name, metric column, and group column. Review the test results: test statistic, p-value, confidence interval for the difference, and the detected difference between groups.

5. **Calculate effect sizes.** Call `analyze_effect_sizes` with the same data. Get Cohen's d or the appropriate effect size measure. Classify the effect:
   - Small: d around 0.2
   - Medium: d around 0.5
   - Large: d above 0.8

6. **Assess statistical significance.** Determine if the p-value is below the significance threshold (typically 0.05). Note the confidence interval for the treatment effect. A significant result with a confidence interval that excludes zero provides strong evidence.

7. **Assess practical significance.** Compare the effect size against business-relevant thresholds. A statistically significant result with a tiny effect size may not justify shipping. Conversely, a marginally insignificant result with a meaningful effect size may warrant further testing.

8. **Make a recommendation.** Apply this decision framework:
   - **Ship**: p-value below 0.05 AND effect size is practically meaningful AND confidence interval is entirely positive (or negative, depending on desired direction)
   - **Iterate**: effect is in the right direction but either not significant (need more data) or effect size is borderline (need refinement)
   - **No-ship**: effect is zero, negative, or opposite to the desired direction with sufficient statistical power

9. **Present results.** Provide:
   - Group sizes and balance assessment
   - Primary metric: mean per group and absolute/relative difference
   - Statistical test result: p-value and confidence interval
   - Effect size and practical interpretation
   - Clear recommendation with reasoning

10. **Recommend next steps.** Based on the recommendation, suggest: shipping the change, running a follow-up experiment with modifications, increasing sample size, or testing on specific segments using `/localdata-mcp:analyze-correlations` to find heterogeneous treatment effects.
