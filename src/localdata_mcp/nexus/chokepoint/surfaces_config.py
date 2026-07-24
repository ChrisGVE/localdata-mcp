"""localdata_mcp/nexus/chokepoint/surfaces_config.py — the config-value
seams (§6.2), the E11 composition accounting (§6.3), and FR-305's numeric
expression evaluation.

`_ConfigSeams` is where NX-2's operator-tunable numbers become the plain
seam values tool packages read — they never touch the ConfigModel
directly (§6.2). `process_defaults` / `visualize_defaults` /
`composition_limits` project the relevant S8 rows; `charge_composition`
/ `release_composition` are NFR-105's aggregate ledger for a running
pipeline's inter-stage data (the whole chain charges ONE entry against
the process-wide ceiling — §6.3). `evaluate_numeric_expression` is
FR-305's ONE evaluation surface (NX-6's asteval service, deny-by-default
symbol table) — E10's optimization tools reach expr_eval exclusively
through here. Chokepoint-internal by §6.2: composed into `Chokepoint`
(guard.py).
"""

from __future__ import annotations

from typing import Any, Mapping

from . import expr_eval
from .core import _GuardCore
from .types import CompositionLimits, ProcessDefaults, VisualizeDefaults


class _ConfigSeams(_GuardCore):
    """The config-value, composition-accounting, and expression seams."""

    def evaluate_numeric_expression(
        self, expression: str, columns: "Mapping[str, Any]"
    ) -> float:
        """FR-305's ONE evaluation surface for caller-supplied numeric
        expressions (NX-6's asteval service, deny-by-default symbol
        table): E10's optimization tools reach expr_eval exclusively
        through here — the module itself stays chokepoint-internal.
        Raises `ExpressionRefusedError` on any unsafe or non-numeric
        input."""
        return expr_eval.evaluate_numeric_expression(expression, columns)

    def process_defaults(self) -> "ProcessDefaults":
        """The S8 process-domain defaults (rows 30/31) as a plain
        value — the seam E10's stochastic tools read their
        operator-tunable counts through (tool packages never read
        NX-2 directly, section 6.2)."""
        return ProcessDefaults(
            bootstrap_resamples=self._config.process.bootstrap_default_resamples,
            monte_carlo_iterations=self._config.process.monte_carlo_default_iterations,
        )

    def visualize_defaults(self) -> "VisualizeDefaults":
        """The S8 `visualize.*` styling defaults as a plain value — the
        seam E12's render_chart reads its palette and figure defaults
        through (tool packages never read NX-2 directly, section 6.2)."""
        return VisualizeDefaults.from_config(self._config.visualize)

    # -- the composition seams (E11, section 6.3) ---------------------

    def composition_limits(self) -> "CompositionLimits":
        """The S8 composition bound (row 14) as a plain value — the
        seam the E11 engine reads `composition.max_pipeline_length`
        through (tool packages never read NX-2 directly)."""
        return CompositionLimits(
            max_pipeline_length=self._config.composition.max_pipeline_length,
        )

    def charge_composition(self, pipeline_id: str, resident_bytes: int) -> None:
        """NFR-105's aggregate accounting for a running pipeline's
        inter-stage data: the whole chain charges ONE ledger entry
        against the process-wide ceiling (section 6.3 — N stages
        cannot each sit under the single-operation bound while jointly
        exceeding it). Raises ResourceRefusedError over the ceiling."""
        self._bounds.charge(f"composition:{pipeline_id}", resident_bytes)

    def release_composition(self, pipeline_id: str) -> None:
        """Drop a pipeline's ledger entry (idempotent teardown)."""
        self._bounds.release(f"composition:{pipeline_id}")
