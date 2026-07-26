<!-- MACHINE-WRITTEN by localdata_mcp.nexus.contract.generators.docs — DO NOT EDIT; regenerate via `python -m localdata_mcp.nexus.contract.generate` -->

# Tools — optimization

| Tool | Summary | Input shape | Output shape | Streaming | Params |
|---|---|---|---|---|---|
| `solve_linear_program` | Minimise a linear objective on an addressed tabular source: rows are decision variables, objective_column the cost vector, each constraint column one constraint's coefficients with its constraint_values right-hand side and constraint_types (<=, >=, =). HiGHS solver; reports the solution, objective value, and solver status. | TABULAR | SCALAR | no | `endpoint?`, `path?`, `table?`, `query?`, `objective_column`, `constraint_columns?`, `constraint_values?`, `constraint_types?`, `bounds?`, `integer_variables?` |
| `optimize_constrained` | Minimise a nonlinear objective expression over the decision vector x (e.g. '(x[0]-1)**2 + x[1]'), starting from initial_guess_column on an addressed tabular source. Expressions are evaluated by the deny-by-default numeric grammar — no host code can run. Optional constraint expressions (ineq: >= 0, or eq). | TABULAR | SCALAR | no | `endpoint?`, `path?`, `table?`, `query?`, `objective_expression`, `initial_guess_column`, `constraint_expressions?`, `constraint_types?`, `method?` |
| `solve_assignment_problem` | Optimal agent-task assignment (Hungarian algorithm) on an addressed tabular source: rows are agents, cost_columns the per-task cost columns; optional agent_column names the agents. Reports the assignment and total cost. | TABULAR | SCALAR | no | `endpoint?`, `path?`, `table?`, `query?`, `cost_columns`, `agent_column?` |
