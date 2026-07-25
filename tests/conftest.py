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
