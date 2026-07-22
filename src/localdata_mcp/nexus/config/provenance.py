"""localdata_mcp/nexus/config/provenance.py — layers and per-field provenance.

The trust order (system > user > project, ARCHITECTURE.md section 5)
and the per-field record of who won and who lost: the winning
(value, source) plus every losing contribution as (layer, value)
entries — what makes the shadowed half of the startup pinned/shadowed
report (E2.6) derivable. Neighbors: merge.py builds these records while
merging LayerSources; loaders.py assembles the sources from files and
the environment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Iterator, Literal, Mapping


class Layer(IntEnum):
    """Trust rank; higher value = higher trust (operator-controlled)."""

    PROJECT = 0
    USER = 1
    SYSTEM = 2


OPERATOR_LAYERS = (Layer.SYSTEM, Layer.USER)


@dataclass(frozen=True)
class LayerSource:
    """One contribution source: a config file or the environment.

    `order` breaks application ties within a trust layer: the user file
    applies before env (both USER), so env wins last-wins fields while
    the file wins first-wins (pinned) ones.
    """

    name: str
    layer: Layer
    order: int = 0
    values: Mapping[str, Any] = field(default_factory=dict)


Disposition = Literal[
    "won",  # this contribution is the effective value
    "overridden",  # lost an ordinary last-wins merge (or a same-trust pin)
    "narrowed",  # project-layer narrowing accepted (paths/posture)
    "pin_refused",  # lower-trust attempt to shadow a pinned value
    "introduction_refused",  # non-operator layer minting a declaration
]


@dataclass(frozen=True)
class Contribution:
    """One source's attempt at one field, and how it fared."""

    source: str
    layer: Layer
    value: Any
    disposition: Disposition


@dataclass(frozen=True)
class FieldProvenance:
    """The story of one field: effective value, winner, every attempt."""

    field_path: str
    value: Any
    winning_source: str | None  # None = the declared default won
    contributions: tuple[Contribution, ...] = ()


class Provenance(Mapping[str, FieldProvenance]):
    """Queryable per-field provenance for a loaded ConfigModel."""

    def __init__(self, entries: Mapping[str, FieldProvenance]) -> None:
        self._entries = dict(entries)

    def __getitem__(self, field_path: str) -> FieldProvenance:
        return self._entries[field_path]

    def __iter__(self) -> Iterator[str]:
        return iter(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def winner(self, field_path: str) -> tuple[Any, str]:
        """The effective (value, source); source "default" when no
        layer contributed."""
        entry = self._entries[field_path]
        return entry.value, entry.winning_source or "default"

    def shadowed(self, field_path: str) -> tuple[Contribution, ...]:
        """Every contribution that did not become the effective value."""
        entry = self._entries[field_path]
        return tuple(c for c in entry.contributions if c.disposition != "won")
