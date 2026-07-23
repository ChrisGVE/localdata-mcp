"""tests/v3/test_expr_eval.py — E6.7 deny-by-default evaluation.

Legitimate objective/constraint shapes evaluate; the FR-305
host-escape payload class is refused structurally: imports, attribute
traversal (dunder classes included), calls outside the one whitelist,
statements, comprehensions, f-strings, and string-valued columns all
land as ExpressionRefusedError — never a partial evaluation.
"""

from __future__ import annotations

import pytest

from localdata_mcp.nexus.chokepoint.expr_eval import (
    NUMERIC_FUNCTION_WHITELIST,
    ExpressionRefusedError,
    evaluate_numeric_expression,
)

COLUMNS = {"x": [1.0, 2.0, 3.0], "price": [10.0, 20.0, 30.0], "n": 3}


class TestLegitimateExpressions:
    def test_arithmetic_over_scalars(self) -> None:
        assert evaluate_numeric_expression("n * 2 + 1", COLUMNS) == 7.0

    def test_whitelisted_functions_over_columns(self) -> None:
        assert evaluate_numeric_expression("sum(price) / n", COLUMNS) == 20.0
        assert evaluate_numeric_expression("mean(x)", COLUMNS) == 2.0
        assert evaluate_numeric_expression("sqrt(abs(-16))", COLUMNS) == 4.0

    def test_objective_shape_from_the_lp_tools(self) -> None:
        # The optimize_constrained call pattern the service replaces.
        assert evaluate_numeric_expression(
            "max(x) - min(x) + log(exp(1))", COLUMNS
        ) == pytest.approx(3.0)

    def test_result_is_always_a_float(self) -> None:
        result = evaluate_numeric_expression("n", COLUMNS)
        assert isinstance(result, float) and result == 3.0


class TestHostEscapePayloads:
    """FR-305's payload class — every row refused."""

    @pytest.mark.parametrize(
        "payload",
        [
            "__import__('os').system('id')",
            "import os",
            "open('/etc/passwd').read()",
            "().__class__.__mro__[1].__subclasses__()",
            "n.__class__",
            "x.__len__()",
            "getattr(n, '__class__')",
            "eval('1+1')",
            "exec('pass')",
            "compile('1', '<s>', 'eval')",
            "[i for i in x]",
            "(lambda: 1)()",
            "f'{n.__class__}'",
            "print(n)",
            "globals()",
            "type(n)",
        ],
    )
    def test_payload_is_refused(self, payload: str) -> None:
        with pytest.raises(ExpressionRefusedError):
            evaluate_numeric_expression(payload, COLUMNS)

    def test_statements_are_refused(self) -> None:
        for payload in ("n = 99", "if n: 1", "while n: 1", "for i in x: 1"):
            with pytest.raises(ExpressionRefusedError):
                evaluate_numeric_expression(payload, COLUMNS)

    def test_oversized_expression_is_refused_cheaply(self) -> None:
        with pytest.raises(ExpressionRefusedError, match="exceeds"):
            evaluate_numeric_expression("1+" * 10_000 + "1", COLUMNS)


class TestNamespaceInvariant:
    """No attacker-controlled string ever enters the namespace (§7)."""

    def test_string_column_is_refused_before_binding(self) -> None:
        with pytest.raises(ExpressionRefusedError, match="not numeric"):
            evaluate_numeric_expression("1", {"note": "'; DROP TABLE t;--"})

    def test_bool_and_mixed_sequences_are_refused(self) -> None:
        with pytest.raises(ExpressionRefusedError):
            evaluate_numeric_expression("1", {"flag": True})
        with pytest.raises(ExpressionRefusedError):
            evaluate_numeric_expression("1", {"mixed": [1.0, "two"]})  # type: ignore[dict-item]

    def test_unknown_symbol_is_refused(self) -> None:
        with pytest.raises(ExpressionRefusedError):
            evaluate_numeric_expression("secret_function(1)", COLUMNS)

    def test_non_numeric_result_is_refused(self) -> None:
        with pytest.raises(ExpressionRefusedError, match="produce a number"):
            evaluate_numeric_expression("x", COLUMNS)  # a list, not a number
        with pytest.raises(ExpressionRefusedError, match="produce a number"):
            evaluate_numeric_expression("n == 3", COLUMNS)  # a bool


class TestTheOneWhitelist:
    def test_the_constant_is_the_sole_source(self) -> None:
        assert set(NUMERIC_FUNCTION_WHITELIST) == {
            "sum",
            "mean",
            "abs",
            "min",
            "max",
            "sqrt",
            "exp",
            "log",
        }
