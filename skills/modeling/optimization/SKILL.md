---
name: optimization
description: Solve resource allocation, scheduling, and process optimization problems. Use when finding the best solution under constraints.
allowed-tools: mcp__localdata__describe_database mcp__localdata__execute_query mcp__localdata__get_data_quality_report mcp__localdata__describe_table mcp__localdata__analyze_regression
argument-hint: "<database-name>"
---

# Optimization

Formulate and solve optimization problems from data — resource allocation, scheduling, cost minimization, or process tuning.

## Steps

1. **Understand the objective.** From the user's question, identify what is being optimized (minimize cost, maximize throughput, best allocation) and what the constraints are (budget limits, capacity, time windows, quality thresholds).

2. **Extract problem data.** Call `describe_database` with the database name from `$ARGUMENTS`. Identify tables containing:
   - Decision variables (what can be changed)
   - Objective coefficients (costs, profits, rates)
   - Constraint parameters (capacities, limits, requirements)

3. **Profile the data.** Call `execute_query` to pull the relevant values. Verify completeness with `get_data_quality_report`. Missing constraint data makes optimization unreliable.

4. **Formulate the problem.** Translate the data into an optimization formulation:
   - Objective function (linear or nonlinear)
   - Decision variables and their bounds
   - Constraints (equality and inequality)
   - Report the formulation clearly before solving

5. **Solve.** Apply the appropriate optimization approach (available when optimization domain tools are exposed):
   - Linear programming for linear objectives and constraints
   - Constrained optimization for nonlinear problems
   - Assignment problems for matching tasks to resources
   - Network optimization for flow and routing

6. **Analyze the solution.** Examine:
   - Optimal objective value
   - Decision variable values at the optimum
   - Which constraints are binding (at their limit) vs. slack
   - Sensitivity: how much would the objective change if a constraint were relaxed?

7. **Validate against reality.** Call `execute_query` to compare the optimal solution against historical performance. Is the improvement realistic? Are there practical constraints the model does not capture?

8. **Present results.** Provide:
   - Problem formulation summary
   - Optimal solution with all variable values
   - Objective value and improvement over baseline
   - Binding constraints and sensitivity analysis
   - Implementation recommendations and caveats
