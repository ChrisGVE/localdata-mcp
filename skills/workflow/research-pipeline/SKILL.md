---
name: research-pipeline
description: Run a structured research analysis — hypotheses, power analysis, assumption checks, primary analysis, and reproducible reporting. Use when results must meet academic or regulatory standards.
allowed-tools: mcp__localdata__connect_database mcp__localdata__describe_database mcp__localdata__execute_query mcp__localdata__get_data_quality_report mcp__localdata__analyze_hypothesis_test mcp__localdata__analyze_anova mcp__localdata__analyze_effect_sizes mcp__localdata__analyze_regression mcp__localdata__evaluate_model_performance mcp__localdata__reduce_dimensions
argument-hint: "<database-name>"
---

# Research Pipeline

Execute a rigorous research analysis workflow with pre-specified hypotheses, assumption verification, and publication-quality reporting.

## Steps

1. **Define hypotheses.** Before touching data, articulate the null and alternative hypotheses in precise terms. State the primary outcome variable, the predictor or grouping variable, and the expected direction of the effect.

2. **Assess data fitness.** Call `describe_database` and `get_data_quality_report` with the database name from `$ARGUMENTS`. Evaluate:
   - Sample size relative to power requirements
   - Missing data patterns and their potential impact
   - Measurement quality (appropriate scales, reasonable ranges)

3. **Power analysis.** Call `execute_query` to compute the observed effect variability. Estimate whether the available sample provides at least 80% power to detect the expected effect size at alpha = 0.05. If underpowered, report the minimum sample needed.

4. **Verify assumptions.** For the planned statistical test:
   - Normality: compute skewness and kurtosis via `execute_query`; run Shapiro-Wilk via `analyze_hypothesis_test` if supported
   - Homoscedasticity: compare group variances via `execute_query`
   - Independence: assess based on study design (not testable from data alone, but note clustering)
   - Multicollinearity (for regression): compute VIF or correlation matrix via `execute_query`

5. **Run primary analysis.** Execute the pre-specified test:
   - Group comparisons: `analyze_hypothesis_test` or `analyze_anova`
   - Regression: `analyze_regression` with `evaluate_model_performance`
   Let assumption checks guide method selection (parametric vs. non-parametric).

6. **Compute effect sizes.** Call `analyze_effect_sizes`. Report with confidence intervals. Classify magnitude using standard conventions (Cohen's d: small=0.2, medium=0.5, large=0.8).

7. **Sensitivity analysis.** Re-run the primary analysis under alternative conditions:
   - Excluding outliers (identified via IQR or z-score)
   - Using the alternative test (parametric if non-parametric was primary, or vice versa)
   - Adjusting for covariates if available
   Report whether conclusions are robust to these changes.

8. **Report in academic format.** Present:
   - **Research Question**: formal hypotheses
   - **Method**: design, variables, test selection with assumption justification
   - **Power**: achieved power or minimum detectable effect
   - **Results**: full statistical reporting (test statistic, df, p, effect size, CI)
   - **Sensitivity**: robustness across alternative analyses
   - **Limitations**: threats to validity, generalizability bounds
   - **Reproducibility**: exact parameters and filters for replication
