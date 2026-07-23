"""tests/v3/test_resource_bounds.py — E6.5 NFR-105 fail-safe bounds.

The exit-gate assertions for resource_bounds.py: the fail-open defect
class is CLOSED (an internal error inside a check refuses, never
passes), dynamic analytical admission checks live headroom — not the
bare row cap — naming current residency in the refusal (S8 row 13),
the aggregate ledger holds the ceiling jointly across registries (§5
bound 2), and the wired staging branch enforces both disk rows (S8
rows 5-6) fail-safe.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from localdata_mcp.nexus.chokepoint.resource_bounds import (
    ResourceBounds,
    ResourceRefusedError,
)
from localdata_mcp.nexus.config.models import (
    ConfigModel,
    QueryConfig,
    ResourcesConfig,
)

# A deliberately tiny model: ceiling 1 MiB-ish so tests never allocate
# meaningfully; values chosen to be provably non-S8 (none appears as a
# ConfigModel default).
_CEILING = 1_000_000
_SPILL_MAX = 2_000_000
_MIN_FREE = 500_000


def small_config() -> ConfigModel:
    return ConfigModel(
        resources=ResourcesConfig(
            memory_ceiling_bytes=_CEILING,
            max_spill_bytes=_SPILL_MAX,
            min_free_disk_bytes=_MIN_FREE,
        )
    )


@pytest.fixture()
def bounds() -> ResourceBounds:
    return ResourceBounds(small_config())


class TestAggregateLedger:
    def test_charges_sum_across_registries(self, bounds: ResourceBounds) -> None:
        bounds.charge("a", 300_000)
        bounds.charge("b", 200_000)
        assert bounds.live_residency() == 500_000

    def test_recharge_replaces_not_accumulates(self, bounds: ResourceBounds) -> None:
        bounds.charge("a", 300_000)
        bounds.charge("a", 100_000)
        assert bounds.live_residency() == 100_000

    def test_joint_over_ceiling_refused_and_rolled_back(
        self, bounds: ResourceBounds
    ) -> None:
        """§5 bound 2: two registries individually under the ceiling
        cannot jointly exceed it — and the refused charge leaves the
        ledger untouched."""
        bounds.charge("a", 700_000)
        with pytest.raises(ResourceRefusedError) as refusal:
            bounds.charge("b", 400_000)
        assert refusal.value.resource_class == "memory"
        assert bounds.live_residency() == 700_000

    def test_refused_recharge_restores_previous_value(
        self, bounds: ResourceBounds
    ) -> None:
        bounds.charge("a", 100_000)
        with pytest.raises(ResourceRefusedError):
            bounds.charge("a", _CEILING + 1)
        assert bounds.live_residency() == 100_000

    def test_negative_charge_refused(self, bounds: ResourceBounds) -> None:
        with pytest.raises(ResourceRefusedError):
            bounds.charge("a", -1)

    def test_release_is_idempotent(self, bounds: ResourceBounds) -> None:
        bounds.charge("a", 100_000)
        bounds.release("a")
        bounds.release("a")
        bounds.release("never-registered")
        assert bounds.live_residency() == 0


class TestDynamicAnalyticalAdmission:
    def test_admits_under_cap_and_headroom(self, bounds: ResourceBounds) -> None:
        bounds.admit_analysis(rows=50, per_row_bytes=100.0)

    def test_row_cap_refusal_names_the_cap(self) -> None:
        """The upper cap is `query.max_analysis_rows`, derived from the
        ceiling (S8 row 13) — a request over it is refused by count
        alone, before any byte math."""
        config = small_config()
        cap = config.query.max_analysis_rows
        bounds = ResourceBounds(config)
        with pytest.raises(ResourceRefusedError) as refusal:
            bounds.admit_analysis(rows=cap + 1, per_row_bytes=1.0)
        assert str(cap) in str(refusal.value)
        assert refusal.value.resource_class == "memory"

    def test_admission_is_dynamic_not_bare_cap(self, bounds: ResourceBounds) -> None:
        """S8 row 13's operative clause: rows UNDER the cap still refuse
        when rows × per-row estimate exceeds ceiling − live residency,
        and the refusal names current residency."""
        bounds.charge("stream-1", 900_000)
        with pytest.raises(ResourceRefusedError) as refusal:
            bounds.admit_analysis(rows=100, per_row_bytes=2_000.0)
        assert "900000" in str(refusal.value)
        assert refusal.value.resource_class == "memory"

    def test_same_request_admits_once_residency_released(
        self, bounds: ResourceBounds
    ) -> None:
        bounds.charge("stream-1", 900_000)
        bounds.release("stream-1")
        bounds.admit_analysis(rows=100, per_row_bytes=2_000.0)

    def test_nonpositive_per_row_estimate_refused(self, bounds: ResourceBounds) -> None:
        """A zero or negative per-row estimate is not a measurement —
        trusting it would admit anything (fail-safe, NFR-105)."""
        with pytest.raises(ResourceRefusedError):
            bounds.admit_analysis(rows=10, per_row_bytes=0.0)
        with pytest.raises(ResourceRefusedError):
            bounds.admit_analysis(rows=10, per_row_bytes=-1.0)

    def test_negative_rows_refused(self, bounds: ResourceBounds) -> None:
        with pytest.raises(ResourceRefusedError):
            bounds.admit_analysis(rows=-1, per_row_bytes=1.0)


class TestLoadAdmission:
    def test_admits_within_headroom(self, bounds: ResourceBounds) -> None:
        bounds.admit_load(estimated_bytes=500_000)

    def test_refuses_over_headroom_before_the_load(
        self, bounds: ResourceBounds
    ) -> None:
        """The decompression-bomb closure point: the refusal is an
        upfront admission verdict on the ESTIMATE — nothing was read."""
        bounds.charge("stream-1", 600_000)
        with pytest.raises(ResourceRefusedError) as refusal:
            bounds.admit_load(estimated_bytes=500_000)
        assert "600000" in str(refusal.value)

    def test_negative_estimate_refused(self, bounds: ResourceBounds) -> None:
        with pytest.raises(ResourceRefusedError):
            bounds.admit_load(estimated_bytes=-1)


class TestFailOpenDefectClosed:
    """The E6.5 reason-to-exist: `main` answered an internal failure in
    its own bound-checking with safe defaults (streaming/memory.py:50-60).
    Here the same class of failure REFUSES (§4c)."""

    def test_internal_error_in_admission_refuses(
        self, bounds: ResourceBounds, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def broken(self: ResourceBounds) -> int:
            raise OSError("simulated introspection failure")

        monkeypatch.setattr(ResourceBounds, "live_residency", broken)
        with pytest.raises(ResourceRefusedError) as refusal:
            bounds.admit_analysis(rows=1, per_row_bytes=1.0)
        assert refusal.value.resource_class == "internal"
        assert "fail-safe" in str(refusal.value)

    def test_internal_error_in_load_admission_refuses(
        self, bounds: ResourceBounds, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            ResourceBounds,
            "live_residency",
            lambda self: (_ for _ in ()).throw(RuntimeError("boom")),
        )
        with pytest.raises(ResourceRefusedError):
            bounds.admit_load(estimated_bytes=1)


class TestStagingBranchWired:
    def test_spill_within_bounds_admits_and_accounts(
        self, bounds: ResourceBounds, tmp_path: Path
    ) -> None:
        bounds.admit_spill("reg-1", 100_000, tmp_path)
        bounds.admit_spill("reg-1", 100_000, tmp_path)
        assert bounds.live_spill() == 200_000

    def test_aggregate_spill_over_max_refused(
        self, bounds: ResourceBounds, tmp_path: Path
    ) -> None:
        bounds.admit_spill("reg-1", 1_500_000, tmp_path)
        with pytest.raises(ResourceRefusedError) as refusal:
            bounds.admit_spill("reg-2", 600_000, tmp_path)
        assert refusal.value.resource_class == "disk"
        # The refused admission is not accounted.
        assert bounds.live_spill() == 1_500_000

    def test_free_disk_floor_refuses(
        self,
        bounds: ResourceBounds,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """S8 row 6: a write that would leave less than the floor free
        is refused before the OS is starved."""
        import shutil as _shutil

        class Usage:
            free = _MIN_FREE + 50_000

        monkeypatch.setattr(_shutil, "disk_usage", lambda path: Usage())
        with pytest.raises(ResourceRefusedError) as refusal:
            bounds.admit_spill("reg-1", 100_000, tmp_path)
        assert refusal.value.resource_class == "disk"

    def test_failed_disk_probe_refuses_fail_safe(
        self,
        bounds: ResourceBounds,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import shutil as _shutil

        def broken(path: object) -> object:
            raise OSError("probe failed")

        monkeypatch.setattr(_shutil, "disk_usage", broken)
        with pytest.raises(ResourceRefusedError) as refusal:
            bounds.admit_spill("reg-1", 1, tmp_path)
        assert refusal.value.resource_class == "internal"

    def test_negative_spill_refused(
        self, bounds: ResourceBounds, tmp_path: Path
    ) -> None:
        with pytest.raises(ResourceRefusedError):
            bounds.admit_spill("reg-1", -1, tmp_path)

    def test_release_spill_is_idempotent(
        self, bounds: ResourceBounds, tmp_path: Path
    ) -> None:
        bounds.admit_spill("reg-1", 100_000, tmp_path)
        bounds.release_spill("reg-1")
        bounds.release_spill("reg-1")
        assert bounds.live_spill() == 0
