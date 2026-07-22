"""localdata_mcp/nexus/config/merge_gated.py — the introduction rule.

Merge semantics for security-relevant declarations (ARCHITECTURE.md
section 5): allowed_paths entries and endpoint declarations are
introducible only at operator-trust layers (system/user, env included);
a project layer may narrow — drop a path, downgrade a posture — but
never mint. Refusals are recorded, never raised: the untrusted layer
that attempted them must not be able to break startup. Neighbor:
merge.py routes gated fields and the endpoints table here and merges
everything else.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping, Sequence

from .endpoints import EndpointDeclaration, endpoint_from_raw
from .errors import (
    ConfigurationError,
    IntroductionRefusedError,
    PinShadowingError,
    TypeMismatchError,
)
from .provenance import (
    OPERATOR_LAYERS,
    Contribution,
    FieldProvenance,
    LayerSource,
)

PathContribs = Sequence[tuple[LayerSource, tuple[str, ...]]]
_Declared = dict[str, tuple[EndpointDeclaration, LayerSource]]
_Contribs = dict[str, list[Contribution]]


def merge_gated_paths(
    path: str, contribs: PathContribs, default: tuple[str, ...]
) -> tuple[tuple[str, ...], FieldProvenance, list[ConfigurationError]]:
    """Merge an introduction-gated path list (security.allowed_paths)."""
    operator = [c for c in contribs if c[0].layer in OPERATOR_LAYERS]
    lower = [c for c in contribs if c[0].layer not in OPERATOR_LAYERS]
    contributions: list[Contribution] = []
    refusals: list[ConfigurationError] = []
    effective, winner = _paths_operator_phase(
        path, operator, default, contributions, refusals
    )
    effective, winner = _paths_narrowing_phase(
        path, lower, effective, winner, contributions, refusals
    )
    entry = FieldProvenance(path, effective, winner, tuple(contributions))
    return effective, entry, refusals


def _paths_operator_phase(
    path: str,
    operator: PathContribs,
    default: tuple[str, ...],
    contributions: list[Contribution],
    refusals: list[ConfigurationError],
) -> tuple[tuple[str, ...], str | None]:
    """Operator layers introduce; the highest-trust one pins the value."""
    if not operator:
        return default, None
    win_src, win_val = operator[0]
    contributions.append(Contribution(win_src.name, win_src.layer, win_val, "won"))
    for src, val in operator[1:]:
        if src.layer < win_src.layer:
            refusals.append(
                PinShadowingError(
                    f"{src.name} cannot shadow {path} pinned by {win_src.name}",
                    field_path=path,
                    source=src.name,
                    attempted_value=val,
                )
            )
            contributions.append(Contribution(src.name, src.layer, val, "pin_refused"))
        else:
            contributions.append(Contribution(src.name, src.layer, val, "overridden"))
    return win_val, win_src.name


def _paths_narrowing_phase(
    path: str,
    lower: PathContribs,
    effective: tuple[str, ...],
    winner: str | None,
    contributions: list[Contribution],
    refusals: list[ConfigurationError],
) -> tuple[tuple[str, ...], str | None]:
    """Non-operator layers may drop entries, never add them."""
    for src, val in lower:
        minted = tuple(p for p in val if p not in effective)
        kept = tuple(p for p in effective if p in val)
        if minted:
            refusals.append(
                IntroductionRefusedError(
                    f"{src.name} may not introduce {path} entries "
                    f"{minted} (operator-trust layers only)",
                    field_path=path,
                    source=src.name,
                    attempted_value=val,
                )
            )
        if kept != effective:
            effective, winner = kept, src.name
            disposition = "narrowed"
        else:
            disposition = "introduction_refused" if minted else "narrowed"
        contributions.append(
            Contribution(src.name, src.layer, val, disposition)  # type: ignore[arg-type]
        )
    return effective, winner


def merge_endpoints(
    sources: Sequence[LayerSource],
) -> tuple[
    dict[str, EndpointDeclaration],
    dict[str, FieldProvenance],
    list[ConfigurationError],
]:
    """Merge endpoint declarations across trust-ordered sources."""
    declared: _Declared = {}
    contributions: _Contribs = {}
    refusals: list[ConfigurationError] = []
    for src in sources:
        for name, table in _endpoints_table(src).items():
            contributions.setdefault(name, [])
            if src.layer in OPERATOR_LAYERS:
                _operator_endpoint(name, table, src, declared, contributions, refusals)
            else:
                _project_endpoint(name, table, src, declared, contributions, refusals)
    entries = {
        f"endpoints.{name}": FieldProvenance(
            field_path=f"endpoints.{name}",
            value=declared[name][0] if name in declared else None,
            winning_source=declared[name][1].name if name in declared else None,
            contributions=tuple(contribs),
        )
        for name, contribs in contributions.items()
    }
    endpoints = {name: decl for name, (decl, _) in declared.items()}
    return endpoints, entries, refusals


def _endpoints_table(src: LayerSource) -> Mapping[str, Mapping[str, Any]]:
    """The source's endpoints table, shape-checked (fatal on mismatch)."""
    raw = src.values.get("endpoints")
    if raw is None:
        return {}
    if not isinstance(raw, Mapping) or not all(
        isinstance(entry, Mapping) for entry in raw.values()
    ):
        raise TypeMismatchError(
            "endpoints must be a table of endpoint tables",
            field_path="endpoints",
            source=src.name,
        )
    return raw


def _operator_endpoint(
    name: str,
    table: Mapping[str, Any],
    src: LayerSource,
    declared: _Declared,
    contributions: _Contribs,
    refusals: list[ConfigurationError],
) -> None:
    """First operator declaration of a name wins; lower trust refused."""
    declaration = endpoint_from_raw(name, table, source=src.name)
    if name not in declared:
        declared[name] = (declaration, src)
        contributions[name].append(
            Contribution(src.name, src.layer, declaration, "won")
        )
        return
    _, win_src = declared[name]
    if src.layer < win_src.layer:
        refusals.append(
            PinShadowingError(
                f"{src.name} cannot shadow endpoint {name!r} declared "
                f"by {win_src.name}",
                field_path=f"endpoints.{name}",
                source=src.name,
                attempted_value=dict(table),
            )
        )
        disposition = "pin_refused"
    else:
        disposition = "overridden"
    contributions[name].append(
        Contribution(src.name, src.layer, dict(table), disposition)  # type: ignore[arg-type]
    )


def _project_endpoint(
    name: str,
    table: Mapping[str, Any],
    src: LayerSource,
    declared: _Declared,
    contributions: _Contribs,
    refusals: list[ConfigurationError],
) -> None:
    """A project layer may only downgrade a declared posture."""
    if name not in declared:
        refusals.append(
            IntroductionRefusedError(
                f"{src.name} may not mint endpoint {name!r} "
                "(operator-trust layers only)",
                field_path=f"endpoints.{name}",
                source=src.name,
                attempted_value=dict(table),
            )
        )
        contributions[name].append(
            Contribution(src.name, src.layer, dict(table), "introduction_refused")
        )
        return
    declaration, win_src = declared[name]
    if dict(table) == {"posture": "read_only"}:
        declared[name] = (replace(declaration, posture="read_only"), win_src)
        contributions[name].append(
            Contribution(src.name, src.layer, dict(table), "narrowed")
        )
        return
    refusals.append(
        PinShadowingError(
            f"{src.name} may only narrow endpoint {name!r} to "
            "posture = read_only; anything else shadows the operator "
            "declaration",
            field_path=f"endpoints.{name}",
            source=src.name,
            attempted_value=dict(table),
        )
    )
    contributions[name].append(
        Contribution(src.name, src.layer, dict(table), "pin_refused")
    )
