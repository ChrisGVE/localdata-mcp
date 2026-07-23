"""localdata_mcp/explore/search.py — X-3 search (E9.3).

FR-203's `search_data` (carried from `main`, re-addressed): regex
search over any exactly-one-source address. The PRD signature's
`target` is WHAT to search — the second addressing slot (a table name
or SQL statement for endpoint sources, a SQL statement for local
database files, omitted for document/table formats); `query` is the
SEARCH expression (the regex), matching a search surface's natural
reading. Patterns cross the safe_regex screen (length/group/
nested-quantifier caps) and the whole scan runs under the wall-clock
backstop; matches are capped by a named constant with an explicit
`truncated` flag (harvested semantics). Neighbors: addressing.py
resolves the source; safe_regex.py screens.
"""

from __future__ import annotations

from typing import Any

from localdata_mcp.nexus.contract.spec import Param, TypeShape, tool_spec

from ..ingest.refusals import missing_entity_refusal
from .addressing import resolve_frame
from .safe_regex import UnsafePatternError, scan_with_timeout, validated_pattern

# Match-listing cap (non-config legibility bound; `truncated` states it).
_MATCH_CAP = 128


@tool_spec(
    name="search_data",
    summary=(
        "Regex-search a tabular source's cell values. Address with "
        "exactly one of endpoint= or path=; target= is the table or "
        "SQL statement to search (omit for a document/table file); "
        "query= is the search pattern."
    ),
    params=(
        Param(
            "endpoint",
            str,
            "The operator-declared endpoint name (exactly one of endpoint/path).",
            required=False,
        ),
        Param(
            "path",
            str,
            "A local file inside allowed_paths (exactly one of endpoint/path).",
            required=False,
        ),
        Param(
            "target",
            str,
            "What to search: a table name or SQL statement on an "
            "endpoint; a SQL statement on a database file; omit for a "
            "document/table format file.",
            required=False,
        ),
        Param("query", str, "The search pattern (a regular expression)."),
        Param(
            "columns",
            str,
            "Comma-separated column names to search (omitted = all).",
            required=False,
        ),
        Param(
            "case_sensitive",
            bool,
            "Case-sensitive matching (default true).",
            required=False,
        ),
    ),
    input_shape=TypeShape.TABULAR,
    output_shape=TypeShape.SCALAR,
    domain="explore",
)
def search_data(
    endpoint: str | None = None,
    path: str | None = None,
    target: str | None = None,
    query: str = "",
    columns: str | None = None,
    case_sensitive: bool = True,
) -> Any:
    try:
        compiled = validated_pattern(query, case_sensitive)
    except UnsafePatternError as unsafe:
        raise missing_entity_refusal(
            f"Search pattern refused: {unsafe}",
            "Simplify the pattern: shorter, fewer groups, no nested quantifiers.",
        ) from unsafe
    table, sql = _target_slots(endpoint, target)
    frame, source = resolve_frame(endpoint, path, table, sql)
    wanted = (
        [name.strip() for name in columns.split(",") if name.strip()]
        if columns
        else [str(name) for name in frame.columns]
    )
    unknown = [name for name in wanted if name not in frame.columns]
    if unknown:
        raise missing_entity_refusal(
            f"Column(s) not in the source: {unknown}",
            "Call profile_data on the same source to list its columns.",
        )

    def scan() -> tuple[list[dict[str, Any]], bool]:
        found: list[dict[str, Any]] = []
        for row_index, row in enumerate(frame.itertuples(index=False)):
            values = dict(zip(frame.columns, row))
            for column in wanted:
                value = values[column]
                text_value = "" if value is None else str(value)
                for match in compiled.finditer(text_value):
                    found.append(
                        {
                            "row_index": row_index,
                            "column": column,
                            "value": text_value,
                            "match": match.group(),
                            "start": match.start(),
                            "end": match.end(),
                        }
                    )
                    if len(found) >= _MATCH_CAP:
                        return found, True
        return found, False

    matches, truncated = scan_with_timeout(scan)
    return {
        "source": source,
        "pattern": query,
        "matches": matches,
        "total_matches": len(matches),
        "rows_searched": int(len(frame)),
        "truncated": truncated,
    }


def _target_slots(
    endpoint: str | None, target: str | None
) -> tuple[str | None, str | None]:
    """The addressing second slot from `target`: a bare identifier is
    a table name, anything with whitespace is a SQL statement — for
    ENDPOINT sources only (a path's target is always SQL when
    present). Deterministic on shape, refused on genuine ambiguity by
    the catalog-membership check inside read_table."""
    if target is None:
        return None, None
    if endpoint is not None and " " not in target.strip():
        return target.strip(), None
    return None, target
