"""tests/v3/conftest.py — isolates the v3 suite from legacy test fixtures.

The legacy tests/conftest.py autouse fixture `mock_mcp_framework` patches
`localdata_mcp.localdata_mcp`, whose import chain reaches dependencies the
v3 manifest removed. Legacy code is excluded from new-code gates until E15
deletes it, so the v3 suite shadows that fixture with a no-op: the nearest
same-named fixture wins, and no legacy module is imported for v3 tests.
"""

from typing import Iterator

import pytest


@pytest.fixture()
def mock_mcp_framework() -> Iterator[None]:
    """No-op override of the legacy autouse FastMCP patch (see module docstring)."""
    yield None
