"""localdata_mcp/process/domains/sampling_estimation — FR-301's sampling family.

Four tools carried by name from `main` (tools.py declares them):
`generate_sample` (five sampling designs), `bootstrap_statistic`
(percentile CIs, S8 row-30 default resamples), `monte_carlo_simulate`
(row-31 default iterations), `bayesian_estimate` (conjugate-normal
posterior). Every stochastic step is seed-pinnable (S3.3).
"""
