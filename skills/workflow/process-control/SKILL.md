---
name: process-control
description: Build control charts, assess process stability and capability, and identify out-of-control conditions. Use for quality control and process monitoring.
allowed-tools: mcp__localdata__describe_database mcp__localdata__execute_query mcp__localdata__get_data_quality_report mcp__localdata__analyze_hypothesis_test mcp__localdata__analyze_effect_sizes mcp__localdata__detect_anomalies
argument-hint: "<database-name> <measurement-column>"
---

# Process Control

Assess process stability using statistical process control methods and determine process capability.

## Steps

1. **Parse arguments.** Extract the database name and measurement column from `$ARGUMENTS`. The measurement column is the quality characteristic being monitored.

2. **Extract process data.** Call `describe_database` to understand the data structure. Call `execute_query` to pull the measurement column along with any timestamp, batch, or subgroup identifier. Sort by time order.

3. **Compute control chart statistics.** Call `execute_query` to calculate:
   - Overall mean (center line)
   - Within-subgroup variation (R-bar or S-bar) for control limits
   - Upper and lower control limits (mean +/- 3 sigma)
   - Subgroup means and ranges if subgroups exist

4. **Test for stability.** Examine the data against Western Electric rules:
   - Points beyond control limits
   - 7+ consecutive points on one side of the center line (run)
   - 6+ consecutive points trending in one direction
   - 2 of 3 consecutive points beyond 2-sigma
   Report any violations with their location in the time series.

5. **Detect anomalies.** Call `detect_anomalies` on the measurement series as a complementary check. Compare algorithm-detected anomalies with control chart violations.

6. **Assess process capability.** If specification limits are available:
   - Calculate Cp (potential capability): specification width / process width
   - Calculate Cpk (actual capability): minimum distance to spec limit / half process width
   - Estimate percentage out of specification
   - Cp >= 1.33 and Cpk >= 1.0 are typical minimums

7. **Test for shifts.** If the process appears unstable, call `analyze_hypothesis_test` to compare measurements before and after suspected change points. Call `analyze_effect_sizes` to quantify the magnitude of any shift.

8. **Present results.** Provide:
   - Control chart summary: center line, control limits, violations found
   - Stability assessment: in control or out of control, with evidence
   - Capability indices (Cp, Cpk) if specifications were provided
   - Identified assignable causes or patterns
   - Recommendations: investigate specific violations, adjust process, or continue monitoring
