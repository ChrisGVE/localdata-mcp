---
name: find-reference-data
description: Identify and prepare external reference datasets to enrich analysis — benchmarks, demographics, economic indicators, or geographic context. Use when analysis needs external context.
allowed-tools: mcp__localdata__connect_database mcp__localdata__describe_database mcp__localdata__execute_query mcp__localdata__get_data_quality_report mcp__localdata__describe_table mcp__localdata__list_databases
argument-hint: "<database-name> <what-context-is-needed>"
---

# Find Reference Data

Locate, download, and prepare public reference datasets to provide context for analysis.

## Steps

1. **Understand the enrichment need.** From `$ARGUMENTS`, identify the user's database and what external context is needed (demographics, benchmarks, economic indicators, geographic boundaries, etc.). If unclear, describe what types of reference data would be most useful given the data at hand.

2. **Assess the user's data.** Call `describe_database` with the user's database name. Identify the join keys available: geographic codes (ZIP, FIPS, country), time periods (years, months), industry codes (SIC, NAICS), or entity identifiers.

3. **Identify candidate sources.** Based on the need and available join keys, determine the best public data sources:
   - Population/demographics: Census Bureau, Eurostat, UN Population Division
   - Economic indicators: FRED, World Bank, OECD, BLS
   - Geographic boundaries: Census TIGER/Line, Natural Earth
   - Industry benchmarks: BLS industry data, SEC EDGAR
   - Health/scientific: WHO, CDC, public research repositories
   - General-purpose: data.gov, Kaggle Datasets, UCI ML Repository

4. **Download and connect.** Locate a direct download URL for the most suitable dataset (prefer CSV or Parquet). Download it and call `connect_database` to load it. If the primary source is unavailable, try mirror sites or alternative sources.

5. **Validate the reference data.** Call `describe_database` and `get_data_quality_report` on the loaded reference data. Verify:
   - Expected columns and types are present
   - Value ranges are reasonable
   - Time period and geographic coverage overlap with the user's data
   - Join key format matches the user's data

6. **Test the join.** Call `execute_query` to check how many records from the user's data would match the reference data on the proposed join key. Report the match rate. If low, investigate key format mismatches or coverage gaps.

7. **Document provenance.** For each dataset, record: source name, URL, access date, data vintage/release date, license or terms, and any transformations applied.

8. **Present results.** Provide:
   - What reference data was found and loaded
   - Source and provenance details
   - Join strategy and expected match rate
   - Any caveats (temporal gaps, geographic approximations, granularity mismatches)
   - The database name under which the reference data is now available for analysis
