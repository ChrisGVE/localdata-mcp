"""Fixtures shared by the whole suite.

The path boundary is decided by configuration rather than by the environment, so
a test that widens it installs a :class:`~localdata_mcp.config.Config` and this
fixture takes it away again afterwards. Without the teardown a widened boundary
would leak into whichever test ran next, and the leak would look like a passing
test rather than a failure.
"""

from __future__ import annotations

import pytest

from localdata_mcp import config as config_module


@pytest.fixture(autouse=True)
def forget_configuration_afterwards():
    yield
    config_module.reset()


# ---------------------------------------------------------------------------
# Endpoint coverage (issue #46)
# ---------------------------------------------------------------------------
#
# A run against one container used to end on a green line that said nothing
# about the other endpoint targets, which skip silently. Any session that
# selected an endpoint test now ends with a section naming every target in the
# catalogue and what happened to it, read from the probe cache in
# tests/endpoints.py rather than from a second list.

_SELECTED_ENDPOINT_TESTS = pytest.StashKey[bool]()


def pytest_collection_finish(session: pytest.Session) -> None:
    session.config.stash[_SELECTED_ENDPOINT_TESTS] = any(
        item.get_closest_marker("endpoint") is not None for item in session.items
    )


@pytest.hookimpl(trylast=True)
def pytest_terminal_summary(terminalreporter, exitstatus, config) -> None:
    if not config.stash.get(_SELECTED_ENDPOINT_TESTS, False):
        return
    import endpoints

    terminalreporter.section("endpoint coverage")
    for line in endpoints.coverage_report(endpoints._probed):
        terminalreporter.write_line(line)
