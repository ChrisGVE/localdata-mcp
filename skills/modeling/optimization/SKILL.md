---
name: optimization
description: Solve resource allocation, scheduling, and process optimization problems. Use when finding the best solution under constraints.
allowed-tools: mcp__localdata__describe_database mcp__localdata__execute_query mcp__localdata__get_data_quality_report mcp__localdata__describe_table mcp__localdata__analyze_regression mcp__localdata__solve_linear_program mcp__localdata__optimize_constrained mcp__localdata__solve_assignment_problem mcp__localdata__analyze_network
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

5. **Solve.** Call the tool that matches the formulation. All four take a `table_name` rather than a query — the solver reads the whole table, since it needs the full constraint set — and every column they compute on must be numeric:
   - `solve_linear_program` for a linear objective under linear constraints. Pass `integer_variables` for a mixed-integer problem
   - `optimize_constrained` for a nonlinear objective. `method` is `SLSQP` or `COBYLA`
   - `solve_assignment_problem` for matching agents to tasks at minimum cost. `agent_id_column` and `task_id_column` may be text
   - `analyze_network` for flow and routing. Node identifiers must be numeric, so map them to integers first and map them back for reporting

6. **Analyze the solution.** Examine:
   - Optimal objective value
   - Decision variable values at the optimum
   - Which constraints are binding (at their limit) vs. slack, from `binding_constraints` and `constraint_slack` on a linear program, or the `active` flag in `constraint_analysis` on a constrained solve
   - Sensitivity is **not** computed: `objective_sensitivity` and `rhs_sensitivity` are always empty, `shadow_prices` is always empty, `dual_values` is populated only from equality constraints, and `lagrange_multipliers` is always null. To learn what relaxing a constraint is worth, re-solve with the constraint changed and compare the objective

7. **Validate against reality.** Call `execute_query` to compare the optimal solution against historical performance. Is the improvement realistic? Are there practical constraints the model does not capture?

8. **Present results.** Provide:
   - Problem formulation summary
   - Optimal solution with all variable values
   - Objective value and improvement over baseline
   - Binding constraints and sensitivity analysis
   - Implementation recommendations and caveats
