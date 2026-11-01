"""A destination is a file: the write-path boundary refuses a URL by name (#108).

``resolve_write_path`` is the one place every destination passes through —
``query(path=…)`` via ``export_rows`` and ``save`` via the registry — so the
refusal is pinned there, below the tool surface, as well as on it in
:mod:`test_server`. The URL recognition is the same one ``attach`` and
``create`` use, which now lives in :mod:`localdata_mcp.paths`.
"""

from __future__ import annotations

import pytest

from localdata_mcp import config as config_module
from localdata_mcp import slots
from localdata_mcp.config import Config
from localdata_mcp.export import export_rows
from localdata_mcp.paths import PathNotAllowed, resolve_write_path, url_scheme


@pytest.fixture(autouse=True)
def clean_configuration(monkeypatch, tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    config_module.use(Config())
    yield
    config_module.reset()


def test_there_is_one_url_detector():
    # The registry's name for it is the paths module's function, not a copy.
    assert slots.url_scheme is url_scheme


@pytest.mark.parametrize(
    "raw",
    [
        "postgresql://u:hunter2@localhost:5432/db.csv",
        "mysql+pymysql://u:hunter2@localhost/db",
        "postgresql://u:hunter2@localhost:notaport/db",
    ],
)
def test_resolve_write_path_refuses_a_url_by_name(raw):
    with pytest.raises(PathNotAllowed) as refused:
        resolve_write_path(raw)

    message = str(refused.value)
    assert "is a URL" in message
    assert "destination is a file" in message
    assert "create" in message
    assert "irectory" not in message
    assert "hunter2" not in message


def test_export_rows_refuses_a_url_before_choosing_a_writer():
    # No suffix, so the writer lookup would refuse it first — for the wrong
    # reason, as a name with no suffix rather than as a URL.
    with pytest.raises(PathNotAllowed) as refused:
        export_rows(["a"], [(1,)], "postgresql://u:hunter2@localhost/db")

    assert "is a URL" in str(refused.value)


def test_a_windows_drive_is_still_a_path():
    # A one-letter "scheme" is a drive, so this is a path and is judged as one.
    with pytest.raises(PathNotAllowed) as refused:
        resolve_write_path("C:/no/such/place/out.csv")

    assert "is a URL" not in str(refused.value)
