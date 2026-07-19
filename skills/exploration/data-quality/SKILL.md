---
name: data-quality
description: Run a thorough data quality assessment — completeness, consistency, validity, and uniqueness. Use when you need to understand data fitness before analysis.
allowed-tools: mcp__localdata__connect_database mcp__localdata__describe_database mcp__localdata__execute_query mcp__localdata__get_data_quality_report mcp__localdata__describe_table
argument-hint: "<database-name or file-path>"
---

# Data Quality Assessment

Perform a comprehensive data quality audit covering completeness, consistency, validity, and uniqueness.

## Steps

1. **Connect if needed.** If `$ARGUMENTS` is a file path, call `connect_database` to load it. If it is a database name, proceed directly. Call `describe_database` to list all tables and row counts.

2. **Profile each table.** For each table (or the primary tables if many), call `describe_table` to get column types, nullability, and cardinality. Call `get_data_quality_report` for detailed quality metrics.

3. **Assess completeness.** Call `execute_query` to compute null percentages per column. Classify:
   - Complete (< 1% null): no action needed
   - Minor gaps (1-10% null): note but likely manageable
   - Significant gaps (10-30% null): flag for imputation or exclusion decisions
   - Severe gaps (> 30% null): column may be unusable without careful treatment

4. **Check uniqueness.** For each column, call `execute_query` to compare distinct count against total count. Identify:
   - Candidate keys (100% unique)
   - High-cardinality categoricals (many unique values but not keys)
   - Suspicious duplicates (IDs that should be unique but are not)

5. **Validate value ranges.** Call `execute_query` to compute min, max, mean, and percentiles for numeric columns. Flag:
   - Impossible values (negative ages, future dates in historical data, percentages > 100)
   - Extreme outliers (values beyond 3 IQR from quartiles)
   - Suspicious constants (columns with a single value)

6. **Check consistency.** Look for:
   - Mixed data types within columns (numbers stored as strings)
   - Inconsistent formats (date formats, case sensitivity, encoding)
   - Referential integrity (foreign keys with no matching parent)
   - Contradictory records (same entity with conflicting attribute values)

7. **Report findings.** Present a structured quality scorecard:
   - Overall data health score (percentage of columns with no issues)
   - Per-column quality summary: completeness, uniqueness, validity
   - Ranked list of issues from most to least severe
   - Impact assessment: which issues would affect which types of analysis
   - Remediation suggestions for each issue category

8. **Recommend next steps.** Based on quality, suggest:
   - Ready for analysis: proceed to `/localdata-mcp:explore-data` or domain-specific skills
   - Needs cleaning: specific columns to impute, filter, or transform before analysis
   - Needs investigation: anomalies that require domain expert input
