"""tests/v3/test_chunk_registry.py — E6.6 §5 streaming-buffer rules.

The exit-gate assertions for chunk_registry.py: cursor semantics
(served chunk evicted, re-request refused), K/B look-ahead bound with
backpressure, T10's structural closure (advertised count derived from
the live buffer, total only as exhausted metadata), idle-TTL eviction
returning connection + budget under the retrieval lock, per-row
extrapolated accounting on the aggregate ledger, and the row-24
per-endpoint admission cap naming close_stream.
"""

from __future__ import annotations

from typing import Iterator

import pandas as pd
import pytest

from localdata_mcp.nexus.chokepoint.chunk_registry import (
    ChunkAlreadyServedError,
    ChunkNotServableError,
    ChunkRegistry,
    StreamAdmissionRefusedError,
    StreamExpiredError,
)
from localdata_mcp.nexus.chokepoint.resource_bounds import ResourceBounds
from localdata_mcp.nexus.config.models import (
    ConfigModel,
    QueryConfig,
    ResourcesConfig,
)

_TTL = 60
_K = 2
_B = 1_000_000
_CAP = 2


def registry_config(ceiling: int = 100_000_000, max_bytes: int = _B) -> ConfigModel:
    return ConfigModel(
        resources=ResourcesConfig(memory_ceiling_bytes=ceiling),
        query=QueryConfig(
            chunk_buffer_max_chunks=_K,
            chunk_buffer_max_bytes=max_bytes,
            stream_idle_ttl_seconds=_TTL,
            max_concurrent_streams_per_endpoint=_CAP,
        ),
    )


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class CloseProbe:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self) -> None:
        self.calls += 1


def chunks(count: int, rows: int = 5) -> Iterator[pd.DataFrame]:
    for index in range(count):
        yield pd.DataFrame({"n": range(index * rows, index * rows + rows)})


@pytest.fixture()
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture()
def bounds() -> ResourceBounds:
    return ResourceBounds(registry_config())


@pytest.fixture()
def registry(clock: FakeClock, bounds: ResourceBounds) -> ChunkRegistry:
    return ChunkRegistry(registry_config(), bounds, clock=clock)


class TestAdmission:
    def test_row_24_cap_refusal_names_close_stream(
        self, registry: ChunkRegistry
    ) -> None:
        for index in range(_CAP):
            registry.open_stream(
                f"s{index}", "ep", chunks(3), "streaming", CloseProbe()
            )
        with pytest.raises(StreamAdmissionRefusedError) as refusal:
            registry.open_stream("one-more", "ep", chunks(3), "streaming", CloseProbe())
        assert "close_stream" in str(refusal.value)

    def test_cap_is_per_endpoint(self, registry: ChunkRegistry) -> None:
        for index in range(_CAP):
            registry.open_stream(
                f"s{index}", "ep-a", chunks(3), "streaming", CloseProbe()
            )
        registry.open_stream("other", "ep-b", chunks(3), "streaming", CloseProbe())
        assert registry.live_stream_count("ep-b") == 1

    def test_closing_frees_a_slot(self, registry: ChunkRegistry) -> None:
        for index in range(_CAP):
            registry.open_stream(
                f"s{index}", "ep", chunks(3), "streaming", CloseProbe()
            )
        registry.close_stream("s0")
        registry.open_stream("s-new", "ep", chunks(3), "streaming", CloseProbe())

    def test_duplicate_stream_id_refused(self, registry: ChunkRegistry) -> None:
        registry.open_stream("dup", "ep", chunks(3), "streaming", CloseProbe())
        with pytest.raises(StreamAdmissionRefusedError):
            registry.open_stream("dup", "ep", chunks(3), "streaming", CloseProbe())


class TestCursorSemantics:
    def test_served_chunk_is_evicted_and_rerequest_refused(
        self, registry: ChunkRegistry
    ) -> None:
        registry.open_stream("s", "ep", chunks(4), "streaming", CloseProbe())
        first = registry.request_chunk("s", 0)
        assert list(first["n"]) == [0, 1, 2, 3, 4]
        with pytest.raises(ChunkAlreadyServedError) as refusal:
            registry.request_chunk("s", 0)
        assert "re-issue" in str(refusal.value)

    def test_sequential_consumption_serves_every_chunk(
        self, registry: ChunkRegistry
    ) -> None:
        registry.open_stream("s", "ep", chunks(5), "streaming", CloseProbe())
        for index in range(5):
            chunk = registry.request_chunk("s", index)
            assert chunk["n"].iloc[0] == index * 5

    def test_beyond_end_names_the_total(self, registry: ChunkRegistry) -> None:
        registry.open_stream("s", "ep", chunks(2), "streaming", CloseProbe())
        registry.request_chunk("s", 0)
        registry.request_chunk("s", 1)
        with pytest.raises(ChunkNotServableError) as refusal:
            registry.request_chunk("s", 2)
        assert "2 chunks" in str(refusal.value)


class TestLookAheadBound:
    def test_buffer_never_exceeds_k(self, registry: ChunkRegistry) -> None:
        registry.open_stream("s", "ep", chunks(10), "streaming", CloseProbe())
        assert registry.advertised_count("s") <= _K

    def test_request_past_bound_refused(self, registry: ChunkRegistry) -> None:
        """Skipping ahead cannot inflate the buffer past K — the bound
        is a true residency cap, not a look-ahead suggestion."""
        registry.open_stream("s", "ep", chunks(10), "streaming", CloseProbe())
        with pytest.raises(ChunkNotServableError) as refusal:
            registry.request_chunk("s", 5)
        assert "look-ahead" in str(refusal.value)

    def test_byte_bound_paces_the_buffer(self, clock: FakeClock) -> None:
        """With B below one chunk's size the buffer holds at most the
        one chunk that crossed the bound — backpressure by bytes."""
        config = registry_config(max_bytes=1)
        bounds = ResourceBounds(config)
        registry = ChunkRegistry(config, bounds, clock=clock)
        registry.open_stream("s", "ep", chunks(4), "streaming", CloseProbe())
        assert registry.advertised_count("s") == 1
        registry.request_chunk("s", 0)
        assert registry.advertised_count("s") == 1


class TestT10Advertisement:
    def test_advertised_is_the_live_buffer_count(self, registry: ChunkRegistry) -> None:
        registry.open_stream("s", "ep", chunks(3), "streaming", CloseProbe())
        # Open primes the look-ahead to K: chunks 0 and 1 are buffered.
        assert registry.advertised_count("s") == _K
        # Serving chunk 0 evicts it and tops back up with chunk 2 — the
        # advertised number always equals the buffer's own contents.
        registry.request_chunk("s", 0)
        assert registry.advertised_count("s") == _K
        status = registry.stream_status("s")
        assert status.advertised_chunks == registry.advertised_count("s")

    def test_total_reported_only_once_exhausted_and_as_metadata(
        self, registry: ChunkRegistry
    ) -> None:
        registry.open_stream("s", "ep", chunks(3), "streaming", CloseProbe())
        assert registry.stream_status("s").total_chunks is None
        for index in range(3):
            registry.request_chunk("s", index)
        status = registry.stream_status("s")
        assert status.exhausted
        assert status.total_chunks == 3
        assert status.advertised_chunks == 0


class TestIdleTtlEviction:
    def test_sweep_evicts_returns_connection_and_budget(
        self, clock: FakeClock, bounds: ResourceBounds
    ) -> None:
        registry = ChunkRegistry(registry_config(), bounds, clock=clock)
        probe = CloseProbe()
        registry.open_stream("s", "ep", chunks(4), "streaming", probe)
        assert bounds.live_residency() > 0
        clock.now = _TTL + 1
        assert registry.evict_idle() == ("s",)
        assert probe.calls == 1
        assert bounds.live_residency() == 0

    def test_expired_stream_refused_structurally_on_touch(
        self, clock: FakeClock, bounds: ResourceBounds
    ) -> None:
        """Lazy expiry: even without a sweep, a touch past the TTL
        answers expired-and-non-resumable."""
        registry = ChunkRegistry(registry_config(), bounds, clock=clock)
        probe = CloseProbe()
        registry.open_stream("s", "ep", chunks(4), "streaming", probe)
        clock.now = _TTL + 1
        with pytest.raises(StreamExpiredError) as refusal:
            registry.request_chunk("s", 0)
        assert "non-resumable" in str(refusal.value)
        assert probe.calls == 1

    def test_activity_refreshes_the_ttl(
        self, clock: FakeClock, registry: ChunkRegistry
    ) -> None:
        registry.open_stream("s", "ep", chunks(6), "streaming", CloseProbe())
        clock.now = _TTL - 1
        registry.request_chunk("s", 0)
        clock.now = (_TTL - 1) * 2
        registry.request_chunk("s", 1)  # still live — access refreshed

    def test_unknown_stream_is_expired_vocabulary(
        self, registry: ChunkRegistry
    ) -> None:
        with pytest.raises(StreamExpiredError):
            registry.request_chunk("never-opened", 0)


class TestAccounting:
    def test_first_chunk_measured_then_extrapolated(
        self, bounds: ResourceBounds, clock: FakeClock
    ) -> None:
        registry = ChunkRegistry(registry_config(), bounds, clock=clock)
        registry.open_stream("s", "ep", chunks(4, rows=10), "streaming", CloseProbe())
        # Two identically-shaped chunks buffered: the extrapolated
        # second attribution equals the measured first (same rows,
        # same dtypes), so residency is exactly twice one chunk.
        residency = bounds.live_residency()
        assert residency > 0
        assert residency % 2 == 0

    def test_serving_returns_bytes_to_the_ledger(
        self, bounds: ResourceBounds, clock: FakeClock
    ) -> None:
        registry = ChunkRegistry(registry_config(), bounds, clock=clock)
        registry.open_stream("s", "ep", chunks(2, rows=10), "streaming", CloseProbe())
        full = bounds.live_residency()
        registry.request_chunk("s", 0)
        registry.request_chunk("s", 1)
        assert bounds.live_residency() < full
        registry.close_stream("s")
        assert bounds.live_residency() == 0

    def test_aggregate_ceiling_pauses_topup_but_serves_buffered(
        self, clock: FakeClock
    ) -> None:
        """§5 bound 2 as backpressure: a ceiling that fits one measured
        chunk but not two pauses the top-up at one buffered chunk, yet
        the buffered chunk stays servable and serving it resumes the
        reader."""
        one_chunk = int(next(chunks(1, rows=10)).memory_usage(deep=True).sum())
        config = registry_config(ceiling=one_chunk + one_chunk // 2)
        bounds = ResourceBounds(config)
        registry = ChunkRegistry(config, bounds, clock=clock)
        registry.open_stream("s", "ep", chunks(6, rows=10), "streaming", CloseProbe())
        # The pre-charged projection for a second chunk is refused, so
        # exactly one chunk is resident despite K allowing two.
        assert registry.advertised_count("s") == 1
        served = registry.request_chunk("s", 0)
        assert len(served) == 10
        # Consumption freed the headroom — the reader resumed.
        assert registry.advertised_count("s") == 1


class TestLoadThenServe:
    def test_whole_source_buffered_at_open(self, registry: ChunkRegistry) -> None:
        """FR-404: read whole then sliced — the look-ahead bound does
        not govern this kind (its gate was the upfront admit_load)."""
        registry.open_stream("s", "ep", chunks(6), "load_then_serve", CloseProbe())
        assert registry.advertised_count("s") == 6
        status = registry.stream_status("s")
        assert status.exhausted
        assert status.total_chunks == 6

    def test_out_of_order_service_within_buffer(self, registry: ChunkRegistry) -> None:
        registry.open_stream("s", "ep", chunks(4), "load_then_serve", CloseProbe())
        registry.request_chunk("s", 3)
        registry.request_chunk("s", 1)
        with pytest.raises(ChunkAlreadyServedError):
            registry.request_chunk("s", 3)


class TestClose:
    def test_close_is_idempotent_and_releases_once(
        self, registry: ChunkRegistry
    ) -> None:
        probe = CloseProbe()
        registry.open_stream("s", "ep", chunks(3), "streaming", probe)
        registry.close_stream("s")
        registry.close_stream("s")
        assert probe.calls == 1
        with pytest.raises(StreamExpiredError):
            registry.request_chunk("s", 0)
