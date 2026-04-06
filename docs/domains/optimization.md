# Optimization Domain

## Overview

The optimization domain covers linear programming, constrained nonlinear optimization, network
flow problems, and combinatorial assignment problems. It uses scipy.optimize as the primary
solver backend and networkx for graph-based problems.

Use this domain when you need to:

- Allocate limited resources to maximise or minimise an objective (linear programming)
- Find optimal parameters for a nonlinear objective subject to equality or inequality constraints
- Assign agents to tasks at minimum total cost (assignment / matching problems)
- Analyse network topology, find shortest paths, compute maximum flow, or build a minimum
  spanning tree

All tool functions read problem data from a connected database table, solve the problem, and
return a JSON-serializable result dict.

---

## Available Analyses

| Analysis | Function | Description |
|---|---|---|
| Linear programming | `solve_linear_program` | Minimise linear objective subject to linear constraints |
| Integer programming | `solve_linear_program` | LP with integer variable constraints (MIP) |
| Constrained optimisation | `optimize_constrained` | Nonlinear objective with equality/inequality constraints |
| Multi-objective optimisation | `optimize_constrained` | Pareto-front exploration (via direct API) |
| Assignment problem | `solve_assignment_problem` | Hungarian algorithm for minimum-cost matching |
| Network analysis | `analyze_network` | Shortest paths, MST, max flow, TSP heuristic, centrality |

---

## MCP Tool Reference

### `solve_linear_program`

Solve a linear programming problem from database data.

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `connection_name` | str | required | Name of an active database connection |
| `table_name` | str | required | Table containing the objective and constraint data |
| `objective_column` | str | required | Column with objective function coefficients (c vector) |
| `constraint_columns` | list[str] | None | Columns defining the constraint matrix rows |
| `constraint_values` | list[float] | None | Right-hand side values for each constraint |
| `constraint_types` | list[str] | None | Constraint directions: `"<="`, `">="`, `"="` |
| `bounds` | list[tuple] | None | Variable bounds as `[(lb, ub), ...]`; use `None` for unbounded |
| `method` | str | `"highs"` | Solver: `highs`, `highs-ds`, `highs-ipm`, `interior-point`, `revised simplex` |
| `integer_variables` | list[int] | None | Indices of variables that must be integers (MIP) |

**Return format**

```text
{
  "success": true,
  "optimal_value": 1240.5,
  "optimal_solution": [3.0, 5.0, 0.0, 2.5],
  "message": "Optimization terminated successfully.",
  "method": "highs",
  "execution_time": 0.012,
  "iterations": 8,
  "function_evaluations": 8,
  "is_integer_solution": false,
  "sensitivity_analysis": {...},
  "dual_values": [0.0, 4.2, 0.0],
  "constraint_slack": [0.0, 0.0, 5.5]
}
```

---

### `optimize_constrained`

Solve a nonlinear constrained optimisation problem.

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `connection_name` | str | required | Name of an active database connection |
| `table_name` | str | required | Table providing data for objective and constraints |
| `objective_function` | str | required | Python expression for the objective (uses column names and `x`) |
| `initial_guess_column` | str | required | Column with initial variable values (x0) |
| `constraint_functions` | list[str] | None | Python expressions for constraint functions |
| `constraint_types` | list[str] | None | Constraint types: `"eq"` (equality) or `"ineq"` (inequality ≥ 0) |
| `bounds_columns` | list[str] | None | Two columns providing lower and upper bounds `[lb_col, ub_col]` |
| `method` | str | `"SLSQP"` | scipy.optimize method: `SLSQP`, `COBYLA`, `trust-constr` |
| `multi_objective` | bool | `False` | Enable multi-objective mode |

**Return format**

```text
{
  "success": true,
  "optimal_value": 87.3,
  "optimal_solution": [1.5, 2.0, 0.8],
  "message": "Optimization terminated successfully.",
  "method": "SLSQP",
  "execution_time": 0.045,
  "iterations": 23,
  "function_evaluations": 156,
  "constraint_analysis": [...],
  "constraint_violations": [0.0, 0.0],
  "lagrange_multipliers": [3.1, 0.0]
}
```

---

### `analyze_network`

Perform comprehensive network analysis on graph edge data.

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `connection_name` | str | required | Name of an active database connection |
| `table_name` | str | required | Table with edge list data |
| `source_column` | str | required | Column with source node identifiers |
| `target_column` | str | required | Column with target node identifiers |
| `weight_column` | str | None | Column with edge weights (omit for unweighted) |
| `directed` | bool | `False` | Whether the graph is directed |
| `include_centrality` | bool | `True` | Compute centrality measures for all nodes |
| `algorithms` | list[str] | None | Restrict to specific algorithms (e.g., `["shortest_path", "mst"]`) |

**Return format**

```text
{
  "success": true,
  "graph_properties": {
    "n_nodes": 42,
    "n_edges": 118,
    "is_connected": true,
    "density": 0.137,
    "average_degree": 5.6
  },
  "shortest_paths": {"A": {"B": {"distance": 4.2, "path": ["A", "C", "B"]}}},
  "minimum_spanning_tree": {"edges": [...], "total_weight": 28.3},
  "max_flow": {"value": 15.0, "details": {...}},
  "tsp_solution": {"tour": [...], "total_distance": 94.1},
  "centrality_measures": {
    "degree": {"node1": 0.4, ...},
    "betweenness": {...},
    "closeness": {...}
  },
  "execution_time": 0.21,
  "method": "networkx"
}
```

---

### `solve_assignment_problem`

Solve a minimum-cost assignment (matching) problem.

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `connection_name` | str | required | Name of an active database connection |
| `table_name` | str | required | Table with cost matrix data (one row per agent) |
| `cost_matrix_columns` | list[str] | required | Columns representing task costs (one column per task) |
| `agent_id_column` | str | None | Column with agent identifiers (uses row index if omitted) |
| `task_id_column` | str | None | Column with task identifiers (uses column names if omitted) |
| `method` | str | `"hungarian"` | Algorithm: `"hungarian"` (scipy linear_sum_assignment) |
| `maximize` | bool | `False` | Maximise total value instead of minimising cost |
| `allow_partial` | bool | `False` | Allow some agents/tasks to go unassigned |

**Return format**

```json
{
  "success": true,
  "total_cost": 142.0,
  "assignment_pairs": [
    {"agent_id": "worker_1", "task_id": "task_B", "cost": 25.0},
    {"agent_id": "worker_2", "task_id": "task_A", "cost": 42.0}
  ],
  "assignment_matrix": [[0, 1, 0], [1, 0, 0]],
  "is_perfect_matching": true,
  "unassigned_agents": [],
  "unassigned_tasks": [],
  "method": "hungarian",
  "execution_time": 0.003
}
```

---

## Method Details

### Linear Programming

The problem form is:

```
Minimise:   c^T * x
Subject to: A_ub * x <= b_ub
            A_eq * x  = b_eq
            lb <= x <= ub
```

The objective column provides the `c` vector. The constraint matrix is built by reading the
`constraint_columns` and transposing, so each column represents the coefficients for one
constraint row. Constraint directions `">="` are converted to `"<="` form by negation.

**Solver methods:**

| Method | Notes |
|---|---|
| `highs` | Default; HiGHS solver, fast and robust for large problems |
| `highs-ds` | HiGHS dual simplex |
| `highs-ipm` | HiGHS interior-point method |
| `interior-point` | scipy legacy interior-point |
| `revised simplex` | scipy revised simplex; requires bounded, full-rank problems |

**Integer programming:** Pass `integer_variables` as a list of column indices. The solver uses
branch-and-bound. Results include `is_integer_solution` and `integrality_gap`.

**Sensitivity analysis:** Available in `sensitivity_analysis` for problems solved with HiGHS.
Reports reduced costs, ranging for objective coefficients, and constraint right-hand-side ranges.

**Dual values:** Shadow prices for each active constraint — the marginal value of relaxing a
constraint by one unit.

**When to use:**
- Production planning (maximise output subject to resource limits)
- Portfolio allocation (maximise return subject to budget and risk constraints)
- Diet/blending problems (minimise cost subject to nutritional requirements)
- Scheduling (minimise makespan or cost subject to capacity)

---

### Constrained Optimisation

For nonlinear objectives or constraints, `optimize_constrained` uses `scipy.optimize.minimize`
with the SLSQP method (Sequential Least Squares Programming) by default.

**Objective and constraint expressions** are Python strings evaluated with the row data from the
table. Column values are available by column name; `x` refers to the current solution vector.

Example:
```python
objective_function = "np.sum((x - targets)**2)"
constraint_functions = ["np.sum(x) - budget"]  # equality: sum(x) == budget
constraint_types = ["eq"]
```

**Methods comparison:**

| Method | Best for |
|---|---|
| `SLSQP` | Smooth objectives with equality and inequality constraints (default) |
| `COBYLA` | Derivative-free; inequality constraints only; noisy objectives |
| `trust-constr` | Robust for large-scale problems; supports equality and inequality |

**Lagrange multipliers** in the result show the sensitivity of the optimal objective to each
constraint — a large multiplier indicates the constraint is binding and relaxing it would
significantly improve the objective.

---

### Assignment Problems

The assignment problem matches N agents to M tasks (or jobs to machines, workers to shifts) to
minimise total cost. The `solve_assignment_problem` tool uses the Hungarian algorithm
(`scipy.optimize.linear_sum_assignment`), which runs in O(N³) time.

The cost matrix is read from the database: one row per agent, one column per task, with each
cell containing the cost of assigning that agent to that task.

**Rectangular matrices:** When M ≠ N, the solver handles the imbalance. With `allow_partial=True`,
agents or tasks may be left unassigned; unassigned items appear in `unassigned_agents` and
`unassigned_tasks`.

**Maximisation:** Set `maximize=True` to convert to a profit-maximisation problem (internally
negates the cost matrix).

**When to use:**
- Shift scheduling (workers to time slots with efficiency scores)
- Vehicle routing (vehicles to depots/zones with travel costs)
- Task allocation in project management

---

### Network Optimisation

NetworkX must be installed. Check `NETWORKX_AVAILABLE` before calling.

**Supported algorithms:**

| Algorithm | What it computes | Typical use |
|---|---|---|
| All-pairs shortest path | Distance and path between every node pair | Travel time matrices |
| Max flow | Maximum flow through a directed network from source to sink | Logistics capacity |
| Minimum spanning tree | Lowest-weight subgraph connecting all nodes | Infrastructure layout |
| TSP heuristic | Near-optimal tour visiting every node once | Delivery route optimisation |
| Centrality measures | Degree, betweenness, closeness, eigenvector | Network importance ranking |

Pass `algorithms` to run only a subset; omitting it runs all applicable algorithms, which may be
slow on large graphs (>1000 nodes).

**Graph properties** in the result include node count, edge count, density, connectivity status,
and average degree.

---

## Composition

| After optimisation | Chain to | Purpose |
|---|---|---|
| LP solution vector | Statistical Analysis | Sensitivity analysis on optimal decision variables |
| Assignment result | Business Intelligence | Analyse cost distribution across agent segments |
| Network centrality | Pattern Recognition | Cluster nodes by centrality profile |
| Network shortest paths | Geospatial | Spatial routing when edges have geographic coordinates |

---

## Examples

### Minimise production costs with resource constraints

Suppose a table `production_plan` has columns `profit` (objective), `labor_hours` and
`material_kg` (constraints), with a total labour budget of 120 hours and materials of 500 kg.

```json
{
  "tool": "solve_linear_program",
  "arguments": {
    "connection_name": "ops_db",
    "table_name": "production_plan",
    "objective_column": "profit",
    "constraint_columns": ["labor_hours", "material_kg"],
    "constraint_values": [120.0, 500.0],
    "constraint_types": ["<=", "<="],
    "bounds": [[0, null], [0, null]],
    "method": "highs"
  }
}
```

### Solve a worker-to-task assignment

```json
{
  "tool": "solve_assignment_problem",
  "arguments": {
    "connection_name": "hr_db",
    "table_name": "worker_costs",
    "cost_matrix_columns": ["task_a_cost", "task_b_cost", "task_c_cost"],
    "agent_id_column": "worker_id",
    "method": "hungarian",
    "maximize": false
  }
}
```

### Analyse a supply chain network

```json
{
  "tool": "analyze_network",
  "arguments": {
    "connection_name": "supply_db",
    "table_name": "routes",
    "source_column": "origin_hub",
    "target_column": "destination_hub",
    "weight_column": "transit_days",
    "directed": true,
    "include_centrality": true
  }
}
```

### Portfolio optimisation via constrained solver

```python
# Table: portfolio_data with columns: expected_return, risk, current_weight
result = optimize_constrained(
    connection_name="finance_db",
    table_name="portfolio_data",
    objective_function="-np.dot(x, expected_return)",  # maximise return
    initial_guess_column="current_weight",
    constraint_functions=["np.sum(x) - 1.0"],          # weights sum to 1
    constraint_types=["eq"],
    bounds_columns=["lb", "ub"],                        # 0 <= x_i <= 0.4
    method="SLSQP",
)
print("Optimal weights:", result["optimal_solution"])
print("Expected return:", -result["optimal_value"])
```
