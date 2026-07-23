"""localdata_mcp/process/domains/pattern_recognition/transform.py — FR-301.

`transform_data`'s computation, carried by name from `main`'s
`regex_tools.transform_data`: a regex find/replace over one column of
the addressed data. The pattern crosses the SAME hardened screen the
explore search uses (explore/safe_regex.py — deny-by-default on
backtracking shapes, scan under the wall-clock backstop), and the
refusal is the structured NX-3 shape, never a bare error dict. The
result is TABULAR — the rewritten relation plus a transformation
summary — so it can feed a downstream stage (E11). Neighbors:
tools.py declares the ToolSpec.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from localdata_mcp.explore.safe_regex import (
    UnsafePatternError,
    scan_with_timeout,
    validated_pattern,
)

from ..support import invalid_source_refusal, require_columns

# Change-sample cap: enough to show the rewrite's shape, small enough
# to stay legible (main sampled 5).
_SAMPLE_LIMIT = 5


def transform_column(
    frame: pd.DataFrame,
    column: str,
    find: str,
    replace: str,
    case_sensitive: bool = True,
) -> dict[str, Any]:
    """The rewritten relation plus a find/replace summary."""
    require_columns(frame, column)
    try:
        compiled = validated_pattern(find, case_sensitive)
    except UnsafePatternError as refused:
        raise invalid_source_refusal(f"find= pattern refused: {refused}") from None

    def rewrite() -> tuple[pd.DataFrame, int, list[dict[str, str]]]:
        rewritten = frame.copy()
        changed = 0
        sample: list[dict[str, str]] = []
        for index, value in rewritten[column].items():
            if value is None or (isinstance(value, float) and pd.isna(value)):
                continue
            original = str(value)
            replaced = compiled.sub(replace, original)
            if replaced != original:
                changed += 1
                if len(sample) < _SAMPLE_LIMIT:
                    sample.append({"original": original, "transformed": replaced})
                rewritten.loc[index, column] = replaced
        return rewritten, changed, sample

    rewritten, changed, sample = scan_with_timeout(rewrite)
    # Missing values leave as None, not NaN: NaN is not JSON, and a
    # NaN from an untouched column must not trip the numeric sentinel.
    cleaned = rewritten.astype(object).where(pd.notna(rewritten), None)
    return {
        "columns": [str(name) for name in cleaned.columns],
        "rows": cleaned.to_numpy().tolist(),
        "total_rows": int(len(cleaned)),
        "transformed_count": changed,
        "sample": sample,
        "column": column,
        "pattern": find,
    }
