"""localdata_mcp/process/domains/optimization — FR-301's optimization family.

Three tools carried by name from `main` (tools.py declares them):
`solve_linear_program`, `optimize_constrained` (FR-305: every
caller-supplied objective/constraint string is evaluated exclusively
by NX-6's deny-by-default expression service, through the guard
seam — the `eval()` RCE class this replaces is #42), and
`solve_assignment_problem`.
"""
