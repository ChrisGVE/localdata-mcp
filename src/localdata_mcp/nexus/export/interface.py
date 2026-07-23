"""localdata_mcp/nexus/export/interface.py — NX-8's one renderer seam (E7.3).

The single export path (§8 NX-8, FR-402/902): every file an v3 tool
produces — extraction or visualization alike — renders through one of
the registered per-format renderers (renderers/, one module per
format, consolidating `main`'s five overlapping export modules) and
reaches disk ONLY through `export_to_file`, which enforces, in order:

1. **Containment** — the caller passes NX-6's path-containment service
   (`Chokepoint.contain_path`, the injected dependency — NX-8 never
   imports chokepoint internals) and no write happens without its
   verdict; the empty `allowed_paths` default is fail-closed (§8 NX-8
   Must-not).
2. **NFR-115 overwrite guard** — an existing target is refused unless
   the caller passed the explicit `overwrite=True` disambiguator;
   refusal, never default-to-proceed.
3. **NFR-111 atomic write** — bytes land in a same-directory temp file
   first and `os.replace` onto the target, so a failure at any point
   leaves the target fully written or untouched, never partial.

Shared here because more than one renderer needs them: tabular payload
normalization (`as_dataframe` — the guard's `Result`, a DataFrame, or
records) and the CWE-1236 formula-injection neutralizer the
CSV/Excel/Markdown renderers apply. Neighbors: renderers/ register
through the declared roster; guard.py supplies the containment
callable; tools call `render`/`export_to_file` and nothing deeper.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

import pandas as pd


class ExportError(ValueError):
    """A renderer refused its payload — wrong shape for the format."""


class UnknownFormatError(LookupError):
    """No renderer is registered under the requested format name."""


class OverwriteRefusedError(PermissionError):
    """NFR-115: the target exists and the caller did not pass the
    explicit `overwrite=True` disambiguator — refused, not defaulted."""


class Renderer(Protocol):
    """What the roster assumes about one format module: a name and a
    payload-to-bytes function — nothing more."""

    FORMAT: str
    render: Callable[[Any], bytes]


# `ContainService` is the shape of NX-6's path-containment seam
# (Chokepoint.contain_path): candidate + mode -> canonical real path,
# raising PathRefusedError otherwise. Injected, never imported.
ContainService = Callable[..., Path]


def renderer_for(format_name: str) -> Renderer:
    """The registered renderer, from the declared roster."""
    from .renderers import RENDERERS

    try:
        return RENDERERS[format_name]
    except KeyError:
        raise UnknownFormatError(
            f"no renderer registered for format {format_name!r} — "
            f"declared formats: {sorted(RENDERERS)}"
        ) from None


def supported_formats() -> tuple[str, ...]:
    """The registered format names, sorted — the FR-902 roster as
    declared data, for the export tool's up-front validation and its
    caller-facing format list."""
    from .renderers import RENDERERS

    return tuple(sorted(RENDERERS))


def render(payload: Any, format_name: str) -> bytes:
    """`payload` as `format_name` bytes (§6.2's `NX8.render`)."""
    return renderer_for(format_name).render(payload)


def export_to_file(
    payload: Any,
    format_name: str,
    target: str | Path,
    *,
    contain: ContainService,
    overwrite: bool = False,
) -> Path:
    """Render and write, under the three ordered guards above.

    Returns the canonical real path written. The temp file lives in
    the target's own directory so `os.replace` is a same-filesystem
    atomic rename, and it is removed on any failure.
    """
    real = contain(target, mode="write")
    if real.exists() and not overwrite:
        raise OverwriteRefusedError(
            f"target {str(real)!r} exists — pass overwrite=True to replace "
            "it (NFR-115: destructive operations need the explicit "
            "disambiguator, never a default)"
        )
    payload_bytes = render(payload, format_name)
    write_atomic(payload_bytes, real)
    return real


def write_atomic(payload: bytes, real_target: Path) -> None:
    """NFR-111's temp-file-rename semantics: fully written or
    unmodified, never a partial artifact."""
    real_target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(
        dir=real_target.parent, prefix=f".{real_target.name}."
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
        os.replace(temp_name, real_target)
    except BaseException:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


# -- shared payload helpers ------------------------------------------


def as_dataframe(payload: Any) -> pd.DataFrame:
    """The tabular payloads every tabular renderer accepts: a
    DataFrame, the guard's `Result` shape (columns/rows), or a
    records sequence — anything else is a shape refusal."""
    if isinstance(payload, pd.DataFrame):
        return payload
    columns = getattr(payload, "columns", None)
    rows = getattr(payload, "rows", None)
    if columns is not None and rows is not None:
        return pd.DataFrame(list(rows), columns=list(columns))
    if isinstance(payload, Sequence) and all(
        isinstance(entry, Mapping) for entry in payload
    ):
        return pd.DataFrame(list(payload))
    raise ExportError(
        f"payload of type {type(payload).__name__} is not tabular — "
        "expected a DataFrame, a columns/rows result, or a records list"
    )


# CWE-1236: a cell whose text starts with one of these is interpreted
# as a formula by spreadsheet software; the defense is the standard
# single-quote prefix, applied to string cells only (a numeric -5 is a
# number, not an injection vector).
_FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")


def neutralize_formula_cell(value: Any) -> Any:
    """One cell, neutralized when it is a formula-shaped string."""
    if isinstance(value, str) and value.startswith(_FORMULA_PREFIXES):
        return "'" + value
    return value


def neutralize_formulas(frame: pd.DataFrame) -> pd.DataFrame:
    """A copy of `frame` with every formula-shaped string cell
    neutralized — applied by the CSV/Excel/Markdown renderers."""
    return frame.map(neutralize_formula_cell)
