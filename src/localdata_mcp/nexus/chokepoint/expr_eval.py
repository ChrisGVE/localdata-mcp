"""localdata_mcp/nexus/chokepoint/expr_eval.py — safe numeric eval (E6.7).

The NX-6 service that replaces the two live `eval()` RCE sites
(`domains/optimization/_tool_functions_lp.py:198,208` — T3/#42,
FR-305): `asteval` with a DENY-BY-DEFAULT symbol table per §7's
safe-AST detail. The interpreter is built `minimal=True` (every
statement-class feature off: if/while/for/try/functiondef/print/
import/f-strings/lambda/comprehensions), `use_numpy=False` (no numpy
namespace to prune), and with the attribute-access handler REMOVED
outright — asteval's history includes dunder-traversal bypasses, so
attribute access is structurally absent, not policed. The symbol table
starts empty and is bound only with the caller's numeric columns and
`NUMERIC_FUNCTION_WHITELIST` — the ONE named constant documentation
references, never restates. No attacker-controlled string ever enters
the namespace: non-numeric column values are refused before binding.
Neighbors: guard.py is the only caller surface; the FR-305 CI grep
gate keeps every other eval-class primitive out of the tree.
"""

from __future__ import annotations

import math
import statistics
from typing import Mapping, Sequence

from asteval import Interpreter

# The sole source of what an expression may call (§7 — sum/mean/abs/
# sqrt/exp/log plus the arithmetic staples min/max; arithmetic
# OPERATORS are AST-level and need no symbols). Additions are a
# security decision, not a convenience edit.
NUMERIC_FUNCTION_WHITELIST: Mapping[str, object] = {
    "sum": sum,
    "mean": statistics.fmean,
    "abs": abs,
    "min": min,
    "max": max,
    "sqrt": math.sqrt,
    "exp": math.exp,
    "log": math.log,
}

_MAX_EXPRESSION_LENGTH = 10_000  # refuse pathological inputs cheaply

_NumericValue = float | int | Sequence[float] | Sequence[int]


class ExpressionRefusedError(ValueError):
    """The expression (or its inputs) failed the deny-by-default gate —
    a structured refusal, never a partial evaluation."""


def evaluate_numeric_expression(
    expression: str, columns: Mapping[str, _NumericValue]
) -> float:
    """`expression` over `columns`, or an ExpressionRefusedError.

    Every disposition fails safe: oversized input, a non-numeric
    column value, a parse error, a forbidden construct, an unknown
    symbol, and a non-numeric result are all refusals.
    """
    if len(expression) > _MAX_EXPRESSION_LENGTH:
        raise ExpressionRefusedError(
            f"expression exceeds {_MAX_EXPRESSION_LENGTH} characters"
        )
    interpreter = _deny_by_default_interpreter(columns)
    try:
        result = interpreter.eval(expression, show_errors=False, raise_errors=True)
    except ExpressionRefusedError:
        raise
    except Exception as failure:
        raise ExpressionRefusedError(
            f"expression refused: {type(failure).__name__}: {failure}"
        ) from failure
    if isinstance(result, bool) or not isinstance(result, (int, float)):
        raise ExpressionRefusedError(
            f"expression must produce a number, got {type(result).__name__}"
        )
    return float(result)


def _deny_by_default_interpreter(
    columns: Mapping[str, _NumericValue],
) -> Interpreter:
    """An interpreter whose namespace holds ONLY the whitelist and the
    validated numeric columns (§7's empty-then-bind construction)."""
    symbols: dict[str, object] = dict(NUMERIC_FUNCTION_WHITELIST)
    for name, value in columns.items():
        symbols[name] = _validated_numeric(name, value)
    interpreter = Interpreter(
        symtable=symbols,
        use_numpy=False,
        minimal=True,
        builtins_readonly=True,
    )
    # The required explicit restriction (§7): attribute access is
    # structurally unsupported, closing the dunder-traversal class.
    del interpreter.node_handlers["attribute"]
    return interpreter


def _validated_numeric(name: str, value: _NumericValue) -> _NumericValue:
    """Numbers and sequences of numbers only — a string (or anything
    else) is refused so no attacker-controlled text enters the
    namespace, closing the format-string escape class by invariant."""
    if isinstance(value, bool):
        raise ExpressionRefusedError(f"column {name!r} is not numeric")
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if all(
            isinstance(item, (int, float)) and not isinstance(item, bool)
            for item in value
        ):
            return value
    raise ExpressionRefusedError(f"column {name!r} is not numeric")
