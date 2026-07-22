"""tests/v3/test_pin_drift.py — tests for scripts/check_pin_drift.py.

The pin-drift gate asserts that the security-critical libraries carry
exact `==` pins in pyproject.toml and that uv.lock resolved exactly the
pinned versions. These tests drive the parsing and comparison logic on
synthetic documents; the real-files path is exercised against the repo's
own pyproject.toml/uv.lock as an end-to-end sanity check.
"""

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "check_pin_drift.py"


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("check_pin_drift", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def script() -> ModuleType:
    return _load_script()


class TestExactPinParsing:
    def test_exact_pin_is_parsed(self, script: ModuleType) -> None:
        assert script.parse_exact_pin("sqlglot==30.13.0", "sqlglot") == "30.13.0"

    def test_floor_constraint_is_refused(self, script: ModuleType) -> None:
        with pytest.raises(script.PinDriftError):
            script.parse_exact_pin("sqlglot>=30.0.0", "sqlglot")

    def test_unpinned_requirement_is_refused(self, script: ModuleType) -> None:
        with pytest.raises(script.PinDriftError):
            script.parse_exact_pin("asteval", "asteval")

    def test_missing_package_is_refused(self, script: ModuleType) -> None:
        with pytest.raises(script.PinDriftError):
            script.find_requirement(["pandas>=1.3.0"], "sqlglot")


class TestLockAgreement:
    LOCK_DOC = {
        "package": [
            {"name": "sqlglot", "version": "30.13.0"},
            {"name": "asteval", "version": "1.0.9"},
        ]
    }

    def test_locked_version_found(self, script: ModuleType) -> None:
        assert script.locked_version(self.LOCK_DOC, "sqlglot") == "30.13.0"

    def test_missing_lock_entry_is_refused(self, script: ModuleType) -> None:
        with pytest.raises(script.PinDriftError):
            script.locked_version(self.LOCK_DOC, "duckdb")

    def test_drift_between_pin_and_lock_is_refused(self, script: ModuleType) -> None:
        with pytest.raises(script.PinDriftError):
            script.assert_pin_matches_lock(
                ["sqlglot==30.12.0"], self.LOCK_DOC, "sqlglot"
            )

    def test_agreement_passes(self, script: ModuleType) -> None:
        script.assert_pin_matches_lock(["sqlglot==30.13.0"], self.LOCK_DOC, "sqlglot")


def test_repo_pins_are_clean(script: ModuleType) -> None:
    """End-to-end: the repo's own pyproject.toml and uv.lock must agree."""
    exit_code = script.main(
        [
            "--pyproject",
            str(REPO_ROOT / "pyproject.toml"),
            "--lock",
            str(REPO_ROOT / "uv.lock"),
        ]
    )
    assert exit_code == 0
