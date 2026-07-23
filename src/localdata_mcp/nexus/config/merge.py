"""localdata_mcp/nexus/config/merge.py — the two-tier layer merge.

Consumes the pin/derive metadata declared on the ConfigModel fields
(never a parallel list — ARCHITECTURE.md section 5): ordinary fields
merge last-wins across the trust order, pin-eligible fields first-wins
by trust with lower-trust shadowing recorded as typed refusals, and
introduction-gated declarations route through merge_gated.py. Unknown
fields and type mismatches raise; trust refusals are recorded on the
result so an untrusted layer cannot break startup. Neighbors:
loaders.py assembles the LayerSources; provenance.py defines the
records built here.
"""

from __future__ import annotations

from dataclasses import dataclass, fields as dataclass_fields, replace
from typing import Any, Mapping, Sequence

from .errors import (
    ConfigurationError,
    InvalidValueError,
    PinShadowingError,
    TypeMismatchError,
    UnknownFieldError,
)
from .fields import DERIVED, META_DERIVE, META_INTRODUCTION_GATED
from .merge_gated import merge_endpoints, merge_gated_paths
from .models import (
    ConfigModel,
    field_type,
    is_pin_eligible,
    iter_config_fields,
    section_class,
    section_names,
)
from .provenance import Contribution, FieldProvenance, LayerSource, Provenance

_MISSING: Any = object()

_Contribs = Sequence[tuple[LayerSource, Any]]
_FieldMerge = tuple[Any, FieldProvenance, list[ConfigurationError]]


@dataclass(frozen=True)
class ConfigLoadResult:
    """A loaded model plus the story of how it was assembled."""

    model: ConfigModel
    provenance: Provenance
    refusals: tuple[ConfigurationError, ...]


def merge_sources(sources: Sequence[LayerSource]) -> ConfigLoadResult:
    """Merge trust-layered sources into the one ConfigModel."""
    ordered = _application_sequence(sources)
    for source in ordered:
        _validate_shape(source)
    section_values: dict[str, dict[str, Any]] = {name: {} for name in section_names()}
    entries: dict[str, FieldProvenance] = {}
    refusals: list[ConfigurationError] = []
    for section, fld in iter_config_fields():
        contribs = _contributions(ordered, section, fld.name)
        value, entry, field_refusals = _merge_field(section, fld, contribs)
        refusals.extend(field_refusals)
        entries[entry.field_path] = entry
        if value is not _MISSING:
            section_values[section][fld.name] = value
    endpoints, endpoint_entries, endpoint_refusals = merge_endpoints(ordered)
    refusals.extend(endpoint_refusals)
    entries.update(endpoint_entries)
    model = _build_model(section_values, endpoints)
    return ConfigLoadResult(
        model, Provenance(_finalize_defaults(entries, model)), tuple(refusals)
    )


def _application_sequence(
    sources: Sequence[LayerSource],
) -> list[LayerSource]:
    """Highest trust first; intra-layer order breaks ties (user file
    before env). Input order never matters."""
    return sorted(sources, key=lambda s: (-int(s.layer), s.order, s.name))


def _validate_shape(source: LayerSource) -> None:
    """Refuse unknown sections and fields — fatal, whatever the layer."""
    known_sections = set(section_names()) | {"endpoints"}
    for section_name, table in source.values.items():
        if section_name not in known_sections:
            raise UnknownFieldError(
                f"{source.name} names unknown section {section_name!r}",
                field_path=section_name,
                source=source.name,
            )
        if not isinstance(table, Mapping):
            raise TypeMismatchError(
                f"section {section_name!r} must be a table",
                field_path=section_name,
                source=source.name,
            )
        if section_name == "endpoints":
            continue
        declared = {f.name for f in dataclass_fields(section_class(section_name))}
        for field_name in table:
            if field_name not in declared:
                raise UnknownFieldError(
                    f"{source.name} names unknown field {section_name}.{field_name}",
                    field_path=f"{section_name}.{field_name}",
                    source=source.name,
                )


def _contributions(
    ordered: Sequence[LayerSource], section: str, field_name: str
) -> list[tuple[LayerSource, Any]]:
    """Every source's coerced value for one field, in trust order."""
    out: list[tuple[LayerSource, Any]] = []
    for src in ordered:
        table = src.values.get(section)
        if table is not None and field_name in table:
            out.append((src, _coerce(table[field_name], section, field_name, src)))
    return out


def _coerce(value: Any, section: str, field_name: str, src: LayerSource) -> Any:
    """Read one raw value as the field's declared type (fatal on
    mismatch); the derive sentinel is never a legal operator value."""
    declared = field_type(section, field_name)
    path = f"{section}.{field_name}"
    coerced = _typed_value(value, declared)
    if coerced is _MISSING:
        raise TypeMismatchError(
            f"{path} expects {declared}, {src.name} supplied {value!r}",
            field_path=path,
            source=src.name,
            attempted_value=value,
        )
    fld = next(
        f for f in dataclass_fields(section_class(section)) if f.name == field_name
    )
    if fld.metadata.get(META_DERIVE) is not None and coerced == DERIVED:
        raise InvalidValueError(
            f"{path} = {value!r} collides with the derive sentinel; "
            "omit the field to use the declared derivation",
            field_path=path,
            source=src.name,
            attempted_value=value,
        )
    return coerced


def _typed_value(value: Any, declared: Any) -> Any:
    """The value as `declared`, or _MISSING when it cannot be read so.
    A bool declaration reads only a bool; conversely a TOML bool never
    satisfies int/float/str (TOML distinguishes them, and bool is an int
    subclass in Python)."""
    if declared is bool:
        return value if isinstance(value, bool) else _MISSING
    if isinstance(value, bool):
        return _MISSING  # a bool must not be read as int/float/str
    if declared is int:
        return value if isinstance(value, int) else _MISSING
    if declared is float:
        return float(value) if isinstance(value, (int, float)) else _MISSING
    if declared is str:
        return value if isinstance(value, str) else _MISSING
    # The model's only collection type: tuple[str, ...] path lists.
    if isinstance(value, (list, tuple)) and all(
        isinstance(item, str) for item in value
    ):
        return tuple(value)
    return _MISSING


def _merge_field(section: str, fld: Any, contribs: _Contribs) -> _FieldMerge:
    """Route one field to its merge rule per the declared metadata."""
    path = f"{section}.{fld.name}"
    if not contribs:
        return _MISSING, FieldProvenance(path, _MISSING, None), []
    if fld.metadata.get(META_INTRODUCTION_GATED):
        return merge_gated_paths(path, list(contribs), fld.default)
    if is_pin_eligible(section, fld):
        return _merge_pinned(path, contribs)
    return _merge_last_wins(path, contribs)


def _merge_pinned(path: str, contribs: _Contribs) -> _FieldMerge:
    """First-wins by trust; lower-trust attempts are recorded refusals."""
    win_src, win_val = contribs[0]
    contributions = [Contribution(win_src.name, win_src.layer, win_val, "won")]
    refusals: list[ConfigurationError] = []
    for src, val in contribs[1:]:
        if src.layer < win_src.layer:
            refusals.append(
                PinShadowingError(
                    f"{src.name} cannot shadow {path} pinned by {win_src.name}",
                    field_path=path,
                    source=src.name,
                    attempted_value=val,
                )
            )
            disposition = "pin_refused"
        else:
            disposition = "overridden"
        contributions.append(
            Contribution(src.name, src.layer, val, disposition)  # type: ignore[arg-type]
        )
    entry = FieldProvenance(path, win_val, win_src.name, tuple(contributions))
    return win_val, entry, refusals


def _merge_last_wins(path: str, contribs: _Contribs) -> _FieldMerge:
    """Ordinary cumulative merge: the lowest-trust setter wins."""
    win_src, win_val = contribs[-1]
    contributions = tuple(
        Contribution(src.name, src.layer, val, "overridden")
        for src, val in contribs[:-1]
    ) + (Contribution(win_src.name, win_src.layer, win_val, "won"),)
    entry = FieldProvenance(path, win_val, win_src.name, contributions)
    return win_val, entry, []


def _build_model(
    section_values: Mapping[str, Mapping[str, Any]],
    endpoints: Mapping[str, Any],
) -> ConfigModel:
    sections = {
        name: section_class(name)(**values) for name, values in section_values.items()
    }
    return ConfigModel(endpoints=endpoints, **sections)


def _finalize_defaults(
    entries: Mapping[str, FieldProvenance], model: ConfigModel
) -> dict[str, FieldProvenance]:
    """Fill default-valued entries from the built model, so derived
    defaults report their resolved value."""
    final: dict[str, FieldProvenance] = {}
    for path, entry in entries.items():
        if entry.value is _MISSING and not path.startswith("endpoints."):
            section, field_name = path.split(".", 1)
            resolved = getattr(getattr(model, section), field_name)
            final[path] = replace(entry, value=resolved)
        else:
            final[path] = entry
    return final
