<!-- MACHINE-WRITTEN by localdata_mcp.nexus.contract.generators.docs — DO NOT EDIT; regenerate via `python -m localdata_mcp.nexus.contract.generate` -->

# Tools — process

| Tool | Summary | Input shape | Output shape | Streaming | Params |
|---|---|---|---|---|---|
| `analyze_hypothesis_test` | Run a hypothesis test on an addressed tabular source. test_type auto (default) selects from the supplied columns: group_column= compares two groups, second_column= correlates two columns, column= alone tests normality. Explicit types: ttest_1samp, ttest_ind, ttest_rel, mann_whitney, wilcoxon, chi2, normality, correlation. | TABULAR | SCALAR | no | `endpoint?`, `path?`, `table?`, `query?`, `test_type?`, `column?`, `second_column?`, `group_column?`, `popmean?`, `alpha?`, `alternative?` |
| `analyze_anova` | One-way ANOVA across every group of group_var on an addressed tabular source: F statistic, p-value, eta squared, per-group summary, and Tukey HSD post-hoc when significant. | TABULAR | SCALAR | no | `endpoint?`, `path?`, `table?`, `query?`, `dependent_var`, `group_var`, `alpha?` |
| `analyze_effect_sizes` | Effect sizes for a grouping on an addressed tabular source: Cohen's d, Hedges' g, Glass's delta, and Cliff's delta for two groups; eta and omega squared for three or more. | TABULAR | SCALAR | no | `endpoint?`, `path?`, `table?`, `query?`, `column`, `group_column` |
| `analyze_ab_test` | A/B test between the exactly-two variants of variant_column on an addressed tabular source. test_type auto (default) picks the two-proportion z-test for a binary metric, Welch's t-test otherwise; mann_whitney on request. Names the winner and lift. | TABULAR | SCALAR | no | `endpoint?`, `path?`, `table?`, `query?`, `metric_column`, `variant_column`, `test_type?`, `alpha?`, `alternative?` |
