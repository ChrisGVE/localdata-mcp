"""localdata_mcp/output/ — the Export/Output capability (E13).

Feature E-1 (FR-901/902/903): the single `export_result` tool that
writes any v3 result to a file in any FR-902 format. The tool is a thin
domain surface over the NX-8 export nexus (nexus/export/) — it resolves
the export source, then hands the payload to `export.interface`, which
owns the one renderer roster and the three ordered write guards
(containment, overwrite, atomic rename). Homed as its own capability
package alongside visualize/ so the FR-901 "one export surface"
invariant is a package boundary, not a convention: no other tool writes
files.
"""
