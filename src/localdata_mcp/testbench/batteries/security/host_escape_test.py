"""testbench/batteries/security/host_escape_test.py — FR-305 at the L3 seam.

The no-arbitrary-code-execution bar (T3/#42 — the live-reachable RCE via
`optimize_constrained`'s string-evaluated objective) proven through the
wire the agent uses. FR-305 widens the barred primitive class beyond
`eval`/`exec` to `compile`, `__import__`, attribute-walking to
`__subclasses__`, and `os.system`; the security battery runs that whole
host-escape payload set against *every* string-evaluated parameter of the
tool — the objective expression AND each constraint expression — and
asserts a structured FR-403 refusal from the deny-by-default numeric
grammar, with a filesystem sentinel proving the payload's side-effect
never fired. A legitimate numeric expression is the positive control that
the grammar still admits real optimization.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from . import _seam


@pytest.fixture()
def opt_bench(tmp_path: Path):
    """A booted guard contained to tmp, plus a one-row initial-guess
    fixture the optimizer consumes."""
    guess = tmp_path / "start.csv"
    pd.DataFrame({"start": [0.0]}).to_csv(guess, index=False)
    with _seam.booted(allowed_paths=(str(tmp_path),)):
        yield tmp_path, guess


def _payloads(sentinel: Path) -> tuple[tuple[str, str], ...]:
    """The FR-305 primitive class, each attempting a host reach; the
    write/`system` rows also try to create `sentinel`."""
    return (
        ("import_write", f"__import__('pathlib').Path('{sentinel}').write_text('x')"),
        ("os_system", f"__import__('os').system('touch {sentinel}')"),
        ("eval_call", "eval('1 + 1')"),
        ("exec_call", "exec('x = 1')"),
        ("compile_call", "compile('1', '<s>', 'eval')"),
        ("subclasses_walk", "().__class__.__base__.__subclasses__()"),
    )


@pytest.mark.parametrize(
    "label",
    [row[0] for row in _payloads(Path("/x"))],
)
def test_host_escape_in_objective_is_refused(opt_bench, label: str) -> None:
    tmp, guess = opt_bench
    sentinel = tmp / "pwned_obj.txt"
    expr = dict(_payloads(sentinel))[label]
    envelope = _seam.call_envelope(
        "optimize_constrained",
        {
            "path": str(guess),
            "objective_expression": expr,
            "initial_guess_column": "start",
        },
    )
    _seam.expect_refused(envelope)
    assert not sentinel.exists()


@pytest.mark.parametrize(
    "label",
    [row[0] for row in _payloads(Path("/x"))],
)
def test_host_escape_in_constraint_is_refused(opt_bench, label: str) -> None:
    tmp, guess = opt_bench
    sentinel = tmp / "pwned_con.txt"
    expr = dict(_payloads(sentinel))[label]
    envelope = _seam.call_envelope(
        "optimize_constrained",
        {
            "path": str(guess),
            "objective_expression": "x[0]**2",
            "initial_guess_column": "start",
            "constraint_expressions": [expr],
        },
    )
    _seam.expect_refused(envelope)
    assert not sentinel.exists()


def test_legitimate_numeric_objective_still_optimizes(opt_bench) -> None:
    """Positive control: the deny-by-default grammar admits a real
    numeric expression."""
    tmp, guess = opt_bench
    data = _seam.expect_ok(
        _seam.call_envelope(
            "optimize_constrained",
            {
                "path": str(guess),
                "objective_expression": "(x[0] - 1)**2",
                "initial_guess_column": "start",
            },
        )
    )
    assert data["converged"] is True
