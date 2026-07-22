"""localdata_mcp/nexus/config/models_testbench.py — the testbench section.

One section of the ONE NX-2 `ConfigModel` (PRD S8 preamble: "there is no
separate testbench config file or parallel loader") — split into its own
file only for size, beside models.py which composes it. Loaded by the
same loaders at user-layer rank (the fixture-path precedent,
ARCHITECTURE.md section 7.3). Every default cites its S8 row.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from .fields import cfg_field


@dataclass(frozen=True)
class TestbenchConfig:
    """`testbench.*` — battery tolerances, budgets, and retention (S8)."""

    SECURITY_CLASSED: ClassVar[bool] = False

    # S8 row 15a: deterministic quantities recomputed by the reference
    # implementation match to float precision; absorbs BLAS variation.
    tol_closed_form_rtol: float = cfg_field(1e-6, doc="S8 row 15a")
    # S8 row 15b: seeded iterative/optimizer fits legitimately vary at
    # small magnitudes across library versions.
    tol_iterative_rtol: float = cfg_field(1e-2, doc="S8 row 15b")
    # S8 row 15c: FR-202 floats are direct arithmetic on the fixture.
    tol_quality_rtol: float = cfg_field(1e-9, doc="S8 row 15c")
    # S8 row 15d: chunked-vs-materialized parity; only accumulation-order
    # float effects are legitimate (C-1's stable-combiner constraint).
    tol_stream_parity_rtol: float = cfg_field(1e-9, doc="S8 row 15d")
    # S8 row 15e: FR-502 perceptual threshold against pinned-container
    # goldens; structural SVG checks carry the exactness burden.
    png_ssim_threshold: float = cfg_field(0.95, doc="S8 row 15e")
    # S8 row 15f: trustworthiness floor for rotation/scale-indeterminate
    # embeddings, measured against the original high-dimensional data.
    embedding_trustworthiness_min: float = cfg_field(0.95, doc="S8 row 15f")
    # S8 row 20: chokepoint p99 latency bounds, ms (perf assertion).
    choke_latency_ms_miss: int = cfg_field(25, doc="S8 row 20 (cache miss)")
    choke_latency_ms_hit: int = cfg_field(1, doc="S8 row 20 (cache hit)")
    # S8 row 21: per-PR default; the nightly tier raises it to 1000 via
    # the derived env override (the row's second value — CI-tuned, S7.5).
    hypothesis_max_examples: int = cfg_field(200, doc="S8 row 21")
    # S8 row 22: unclosed-Figure leak guard — loop count and the resident
    # drift bound in MiB.
    leak_loop_iterations: int = cfg_field(500, doc="S8 row 22")
    leak_loop_max_drift_mib: int = cfg_field(50, doc="S8 row 22")
    # S8 row 25: per-chain p95 budgets by length class (perf assertion).
    pipeline_chain_p95_seconds: int = cfg_field(10, doc="S8 row 25 (length 2)")
    pipeline_chain_p95_seconds_long: int = cfg_field(15, doc="S8 row 25 (lengths 3-4)")
    # S8 row 27: non-gating stretch probe beyond the exhaustive envelope.
    stretch_max_length: int = cfg_field(6, doc="S8 row 27")
    stretch_sample_chains: int = cfg_field(200, doc="S8 row 27")
    # S8 row 28: restates NFR-201's warm-discovery bound so the perf
    # battery resolves it through config, not a test-code literal.
    discovery_p99_ms: int = cfg_field(100, doc="S8 row 28 (LOCKED NFR-201)")
    # S8 row 29: results-store retention per (battery_name, run_mode);
    # consumed by testbench/results_store/merge.py's post-merge prune
    # (owning story E0.4 — the parameter's home is here, E1.1).
    results_retention_runs: int = cfg_field(200, doc="S8 row 29")
