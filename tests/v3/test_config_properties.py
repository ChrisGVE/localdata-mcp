"""tests/v3/test_config_properties.py — E1.5 property suites.

Hypothesis-driven invariants over the NX-2 merge and env derivation:
merging is deterministic and idempotent under source duplication, the
pin and last-wins winners are invariant under any permutation of the
source list, and every env encoding round-trips to its typed value.
"""

from __future__ import annotations

import os

from hypothesis import given
from hypothesis import strategies as st

from localdata_mcp.nexus.config.env_derive import env_overrides, env_var_name
from localdata_mcp.nexus.config.merge import merge_sources
from localdata_mcp.nexus.config.models import field_type, iter_config_fields
from localdata_mcp.nexus.config.provenance import Layer, LayerSource

# One ordinary and one pinned field exercise both merge rules.
ORDINARY = ("query", "default_chunk_size")
PINNED = ("resources", "query_timeout_seconds")

_SOURCE_SPECS = (
    ("system-file", Layer.SYSTEM, 0),
    ("user-file", Layer.USER, 0),
    ("env", Layer.USER, 1),
    ("project-file", Layer.PROJECT, 0),
)

_values = st.integers(min_value=1, max_value=10**9)


@st.composite
def source_lists(draw: st.DrawFn) -> list[LayerSource]:
    """A random subset of the four sources, each setting a random
    subset of {ordinary, pinned} fields to random values."""
    sources: list[LayerSource] = []
    for name, layer, order in _SOURCE_SPECS:
        if not draw(st.booleans()):
            continue
        values: dict[str, dict[str, int]] = {}
        if draw(st.booleans()):
            values.setdefault(ORDINARY[0], {})[ORDINARY[1]] = draw(_values)
        if draw(st.booleans()):
            values.setdefault(PINNED[0], {})[PINNED[1]] = draw(_values)
        sources.append(LayerSource(name, layer, order, values))
    return sources


class TestMergeInvariants:
    @given(sources=source_lists())
    def test_merge_is_deterministic(self, sources: list[LayerSource]) -> None:
        assert merge_sources(sources).model == merge_sources(sources).model

    @given(sources=source_lists())
    def test_duplicating_every_source_changes_nothing(
        self, sources: list[LayerSource]
    ) -> None:
        assert merge_sources(sources).model == merge_sources([*sources, *sources]).model

    @given(sources=source_lists(), data=st.data())
    def test_winners_are_invariant_under_permutation(
        self, sources: list[LayerSource], data: st.DataObject
    ) -> None:
        permuted = data.draw(st.permutations(sources))
        assert merge_sources(sources).model == merge_sources(permuted).model

    @given(sources=source_lists())
    def test_pinned_winner_is_the_highest_trust_setter(
        self, sources: list[LayerSource]
    ) -> None:
        setters = [s for s in sources if PINNED[1] in s.values.get(PINNED[0], {})]
        if not setters:
            return
        expected_source = sorted(setters, key=lambda s: (-int(s.layer), s.order))[0]
        result = merge_sources(sources)
        assert (
            result.model.resources.query_timeout_seconds
            == expected_source.values[PINNED[0]][PINNED[1]]
        )

    @given(sources=source_lists())
    def test_ordinary_winner_is_the_last_applied_setter(
        self, sources: list[LayerSource]
    ) -> None:
        setters = [s for s in sources if ORDINARY[1] in s.values.get(ORDINARY[0], {})]
        if not setters:
            return
        expected_source = sorted(setters, key=lambda s: (-int(s.layer), s.order))[-1]
        result = merge_sources(sources)
        assert (
            result.model.query.default_chunk_size
            == expected_source.values[ORDINARY[0]][ORDINARY[1]]
        )


_FIELD_PATHS = [(section, fld.name) for section, fld in iter_config_fields()]

_path_text = st.text(
    alphabet=st.characters(
        codec="ascii", categories=("L", "N"), include_characters="/_-."
    ),
    min_size=1,
    max_size=30,
)


class TestEnvRoundTrip:
    @given(field_path=st.sampled_from(_FIELD_PATHS), data=st.data())
    def test_every_field_round_trips_through_its_env_name(
        self, field_path: tuple[str, str], data: st.DataObject
    ) -> None:
        section, field_name = field_path
        declared = field_type(section, field_name)
        if declared is bool:
            value = data.draw(st.booleans())
            raw = "true" if value else "false"
        elif declared is int:
            value = data.draw(_values)
            raw = str(value)
        elif declared is float:
            value = data.draw(st.floats(allow_nan=False, allow_infinity=False))
            raw = repr(value)  # repr round-trips floats exactly
        elif declared is str:
            value = data.draw(_path_text)  # a plain scalar string field
            raw = value
        else:  # tuple[str, ...] path lists
            value = tuple(data.draw(st.lists(_path_text, min_size=1, max_size=4)))
            raw = os.pathsep.join(value)
        overrides = env_overrides({env_var_name(section, field_name): raw})
        assert overrides == {section: {field_name: value}}
