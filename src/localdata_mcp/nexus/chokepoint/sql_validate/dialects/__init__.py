"""localdata_mcp/nexus/chokepoint/sql_validate/dialects — data fragments.

One module per dialect, each exporting exactly one `POLICY`
(DialectPolicy) — pure data, no control flow (§7/§9: adding a dialect
is a one-file data edit). policy.py aggregates these into the one
mapping; nothing else imports them.
"""
