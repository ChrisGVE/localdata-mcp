"""tests/v3/test_config_errors.py — E1.3 typed ConfigurationError taxonomy.

Every NX-2 refusal path has a distinct typed error, all rooted at
ConfigurationError, each carrying the field path and offending source so
refusals are reportable without string-parsing (ARCHITECTURE.md
section 5: typed ConfigurationError, never a silent print).
"""

from __future__ import annotations

import pytest

from localdata_mcp.nexus.config.errors import (
    ConfigurationError,
    InlineCredentialError,
    IntroductionRefusedError,
    InvalidValueError,
    PinShadowingError,
    TypeMismatchError,
    UnknownFieldError,
)

ALL_ERROR_TYPES = [
    UnknownFieldError,
    TypeMismatchError,
    InvalidValueError,
    IntroductionRefusedError,
    PinShadowingError,
    InlineCredentialError,
]


@pytest.mark.parametrize("error_type", ALL_ERROR_TYPES)
def test_every_refusal_type_is_a_configuration_error(error_type: type) -> None:
    assert issubclass(error_type, ConfigurationError)


def test_carries_field_path_source_and_attempted_value() -> None:
    error = PinShadowingError(
        "project layer cannot shadow the pinned value",
        field_path="security.allowed_paths",
        source="project-file",
        attempted_value=("/tmp",),
    )
    assert error.field_path == "security.allowed_paths"
    assert error.source == "project-file"
    assert error.attempted_value == ("/tmp",)
    assert "cannot shadow" in str(error)


def test_context_defaults_are_none() -> None:
    error = ConfigurationError("bad config")
    assert error.field_path is None
    assert error.source is None
    assert error.attempted_value is None


def test_configuration_error_is_an_exception() -> None:
    with pytest.raises(ConfigurationError):
        raise UnknownFieldError("no such field", field_path="query.typo")
