"""tests/v3/test_process_closures.py — E10 defect-closure unit legs.

The two FR closures that are pure static/unit checks (the battery
covers the data-path closures):

- **FR-306 / E10.x1 — clone() parameter survival.** The regression
  family builds estimators with caller params at construction
  (estimators.py); sklearn's own `clone()` re-reads `get_params()`, so
  a tuned estimator must survive cloning inside a GridSearchCV-style
  flow. main dropped `**algorithm_params` on clone (#30); this asserts
  the property directly.
- **FR-311 / E10.x6 — abstract-hook completeness.** v3 deliberately
  dropped main's transformer/AnalysisPipelineBase hierarchy (DR GP2 —
  names, not class hierarchies), so no class should declare an
  abstract base it leaves unimplemented. This walks every class the
  process package defines and fails if any is left abstract (an
  unimplemented `@abstractmethod`), the #31 defect made structural.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
from abc import ABC

from sklearn.base import clone

import localdata_mcp.process as process_pkg
from localdata_mcp.process.domains.regression_modeling.estimators import build_estimator


class TestCloneParameterSurvival:
    """FR-306 / E10.x1."""

    def test_tuned_params_survive_clone(self) -> None:
        estimator = build_estimator("ridge", {"alpha": 0.5, "fit_intercept": False})
        cloned = clone(estimator)
        assert cloned.get_params()["alpha"] == 0.5
        assert cloned.get_params()["fit_intercept"] is False

    def test_clone_of_default_estimator_is_unpenalised(self) -> None:
        estimator = build_estimator("linear", None)
        assert clone(estimator).get_params() == estimator.get_params()


class TestNoUnimplementedAbstractHook:
    """FR-311 / E10.x6."""

    def _process_classes(self) -> list[type]:
        classes: list[type] = []
        for info in pkgutil.walk_packages(
            process_pkg.__path__, process_pkg.__name__ + "."
        ):
            module = importlib.import_module(info.name)
            for _name, obj in inspect.getmembers(module, inspect.isclass):
                if obj.__module__.startswith("localdata_mcp.process"):
                    classes.append(obj)
        return classes

    def test_every_concrete_class_implements_its_abstract_hooks(self) -> None:
        offenders = [
            f"{cls.__module__}.{cls.__qualname__}"
            for cls in self._process_classes()
            if issubclass(cls, ABC) and getattr(cls, "__abstractmethods__", frozenset())
        ]
        assert offenders == [], (
            f"process classes with unimplemented abstract hooks (FR-311): {offenders}"
        )
