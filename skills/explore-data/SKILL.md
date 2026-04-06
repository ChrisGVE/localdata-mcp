---
name: explore-data
description: Connect to a data source, profile its schema and quality, and recommend appropriate analyses. Use when starting work with a new dataset.
allowed-tools: mcp__localdata__connect_database mcp__localdata__describe_database mcp__localdata__execute_query mcp__localdata__get_data_quality_report mcp__localdata__describe_table
argument-hint: "<file-path or connection-string>"
---

# Explore Data

Connect to a dataset, profile its structure and quality, and recommend what analyses to run next.

## Steps

1. **Connect to the data source.** Call `connect_database` with the path or connection string from `$ARGUMENTS`. Note the assigned database name in the response.

2. **Describe the schema.** Call `describe_database` with the database name. Record the list of tables, column names, and column types. Identify which columns are numeric, categorical, datetime, and text.

3. **Sample rows from each table.** For each table (or the first 3 if many), call `execute_query` with `SELECT * FROM <table> LIMIT 10`. Inspect the returned rows to understand value ranges, formats, and potential join keys.

4. **Describe key tables.** For the most important tables (largest or most-referenced), call `describe_table` to get detailed column statistics including cardinality, null counts, and value distributions.

5. **Run a quality report.** Call `get_data_quality_report` with the database name. Review completeness, uniqueness, and consistency scores. Flag columns with high null rates (above 20%) or low cardinality that may need attention.

6. **Summarize findings.** Present a structured summary:
   - Number of tables, total rows, and columns
   - Data types breakdown (numeric, categorical, datetime, text)
   - Quality issues found (nulls, duplicates, inconsistencies)
   - Key relationships between tables (shared column names)

7. **Recommend next analyses.** Based on data characteristics, suggest specific next steps:
   - Numeric pairs with potential relationships: suggest `/localdata-mcp:analyze-correlations`
   - Datetime column with a metric: suggest `/localdata-mcp:forecast`
   - Many numeric features: suggest `/localdata-mcp:cluster-analysis`
   - Target variable present: suggest `/localdata-mcp:regression`
   - Treatment/control groups: suggest `/localdata-mcp:ab-test`
   - Graph or network file: suggest `/localdata-mcp:graph-explore`
