---
name: analyze-correlations
description: Run correlation analysis on connected data, identify strong relationships, and suggest regression models. Use when exploring relationships between variables.
allowed-tools: mcp__localdata__describe_database mcp__localdata__execute_query mcp__localdata__analyze_hypothesis_test mcp__localdata__analyze_effect_sizes
argument-hint: "<database-name>"
---

# Analyze Correlations

Discover and quantify relationships between variables in a connected dataset.

## Steps

1. **Get the schema.** Call `describe_database` with the database name from `$ARGUMENTS`. Identify all numeric columns across tables. Note column names and which table each belongs to.

2. **Extract numeric data.** Call `execute_query` to select all numeric columns from the primary table. If data spans multiple tables, join on shared keys. Limit to 50,000 rows if the dataset is large.

3. **Test pairwise correlations.** For each meaningful pair of numeric columns, call `analyze_hypothesis_test` with test type "correlation". Use Pearson for normally distributed data, Spearman for skewed or ordinal data. Record the correlation coefficient and p-value for each pair.

4. **Filter significant results.** Retain pairs where p-value is below 0.05. Sort by absolute correlation strength. Classify relationships:
   - Strong: absolute r above 0.7
   - Moderate: absolute r between 0.4 and 0.7
   - Weak: absolute r between 0.2 and 0.4

5. **Measure effect sizes.** For the top 5 strongest correlations, call `analyze_effect_sizes` to quantify practical significance. Compare statistical significance (p-value) against practical significance (effect size). Flag cases where a correlation is statistically significant but practically negligible.

6. **Check for confounders.** Look for pairs of strong correlations that share a common variable. Note potential confounding relationships where A correlates with B and A correlates with C.

7. **Present results.** Provide a ranked table of correlations with columns: variable pair, correlation coefficient, p-value, effect size, and interpretation. Group by strength category.

8. **Recommend next steps.** For the strongest relationships:
   - Suggest `/localdata-mcp:regression` with the best predictor-outcome pairs
   - Flag multicollinearity risks if predictors are highly correlated with each other
   - Recommend further investigation for surprising or counterintuitive correlations
