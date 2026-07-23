"""tests/v3/test_persistence_lifecycle.py — the §5 transition set, exactly.

The machine is complete and closed: every ARCHITECTURE §5 edge walks,
every other pair refuses with LifecycleError. Exhaustive over the
4×4 pair space so an accidentally-added edge fails the test, not just
an accidentally-removed one.
"""

from __future__ import annotations

import pytest

from localdata_mcp.nexus.persistence.lifecycle import (
    LEGAL_TRANSITIONS,
    LifecycleError,
    LifecycleState,
    checked_transition,
)

ALL_PAIRS = [(a, b) for a in LifecycleState for b in LifecycleState]


class TestTransitionSet:
    def test_the_declared_set_is_exactly_the_section_5_edges(self) -> None:
        expected = {
            (LifecycleState.HEALTHY, LifecycleState.FAULTED),
            (LifecycleState.FAULTED, LifecycleState.RESETTING),
            (LifecycleState.RESETTING, LifecycleState.HEALTHY),
            (LifecycleState.RESETTING, LifecycleState.CLOSED),
            (LifecycleState.HEALTHY, LifecycleState.CLOSED),
            (LifecycleState.FAULTED, LifecycleState.CLOSED),
        }
        assert LEGAL_TRANSITIONS == frozenset(expected)

    @pytest.mark.parametrize("current,target", ALL_PAIRS)
    def test_every_pair_walks_or_refuses_per_the_set(
        self, current: LifecycleState, target: LifecycleState
    ) -> None:
        if (current, target) in LEGAL_TRANSITIONS:
            assert checked_transition(current, target) is target
        else:
            with pytest.raises(LifecycleError):
                checked_transition(current, target)

    def test_closed_is_terminal(self) -> None:
        for target in LifecycleState:
            with pytest.raises(LifecycleError):
                checked_transition(LifecycleState.CLOSED, target)

    def test_the_fault_walk_composes(self) -> None:
        """healthy → faulted → resetting → healthy, the NFR-112 cycle."""
        state = LifecycleState.HEALTHY
        for target in (
            LifecycleState.FAULTED,
            LifecycleState.RESETTING,
            LifecycleState.HEALTHY,
        ):
            state = checked_transition(state, target)
        assert state is LifecycleState.HEALTHY
