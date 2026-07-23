"""localdata_mcp/process/domains/regression_modeling/estimators.py — FR-306.

The one estimator-construction site for the regression family — and
the FR-306 defect closure's home: `main`'s pipeline dropped
`**algorithm_params` on `clone()`, silently reverting tuned estimators
to defaults (#30). Here every caller-supplied algorithm parameter is
passed at construction, so sklearn's own `clone()` (which re-reads
`get_params()`) carries them by contract — E10.x1's regression test
asserts exactly that survival. The `regularization` spelling map is
`main`'s: the penalty name a caller reaches for, translated to the
estimator that actually applies it (the documented parameter must do
what it says). Neighbors: regression.py fits what this builds.
"""

from __future__ import annotations

from typing import Any

from ..support import invalid_source_refusal

MODEL_TYPES = ("linear", "ridge", "lasso", "elastic_net", "logistic", "polynomial")

# The penalty names a caller reaches for, mapped onto the estimators
# that actually apply them (main's translation, kept verbatim).
_REGULARIZATION_MODELS = {"l1": "lasso", "l2": "ridge", "elastic_net": "elastic_net"}


def resolve_model_type(model_type: str, regularization: str | None) -> str:
    """`main`'s rule: an explicit regularization= overrides a plain
    linear model_type; anything else must agree or is refused."""
    if model_type not in MODEL_TYPES:
        raise invalid_source_refusal(
            f"Unknown model_type {model_type!r} — one of {list(MODEL_TYPES)}."
        )
    if regularization is None:
        return model_type
    resolved = _REGULARIZATION_MODELS.get(regularization)
    if resolved is None:
        raise invalid_source_refusal(
            f"Unknown regularization {regularization!r} — one of "
            f"{list(_REGULARIZATION_MODELS)}."
        )
    if model_type not in ("linear", resolved):
        raise invalid_source_refusal(
            f"regularization={regularization!r} conflicts with "
            f"model_type={model_type!r} — drop one of the two."
        )
    return resolved


def build_estimator(model_type: str, algorithm_params: dict[str, Any] | None) -> Any:
    """The sklearn estimator with every caller parameter applied at
    construction (FR-306 — clone() re-reads constructor params)."""
    from sklearn.linear_model import (
        ElasticNet,
        Lasso,
        LinearRegression,
        LogisticRegression,
        Ridge,
    )

    constructors = {
        "linear": LinearRegression,
        "polynomial": LinearRegression,
        "ridge": Ridge,
        "lasso": Lasso,
        "elastic_net": ElasticNet,
        "logistic": LogisticRegression,
    }
    params = dict(algorithm_params or {})
    try:
        return constructors[model_type](**params)
    except TypeError as failure:
        raise invalid_source_refusal(
            f"algorithm_params not accepted by the {model_type!r} estimator: {failure}"
        ) from None
