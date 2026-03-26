"""Tests for OS-aware configuration path resolution."""

import warnings
from pathlib import Path
from unittest.mock import patch

import pytest

from localdata_mcp.config_paths import (
    ConfigLocationType,
    ConfigPathInfo,
    emit_deprecation_warning,
    get_config_paths,
    get_recommended_path,
)


class TestGetConfigPaths:
    def test_project_local_always_present(self):
        with patch.dict("os.environ", {}, clear=False):
            paths = get_config_paths()
            types = [p.location_type for p in paths]
            assert ConfigLocationType.PROJECT_LOCAL in types

    def test_legacy_always_present(self):
        paths = get_config_paths()
        legacy = [p for p in paths if p.is_legacy]
        assert len(legacy) == 1
        assert "localdata.yaml" in str(legacy[0].path)

    def test_explicit_env_var_override(self):
        with patch.dict("os.environ", {"LOCALDATA_CONFIG": "/tmp/custom.yaml"}):
            paths = get_config_paths()
            assert paths[0].location_type == ConfigLocationType.EXPLICIT
            assert paths[0].path == Path("/tmp/custom.yaml")

    def test_explicit_is_highest_priority(self):
        with patch.dict("os.environ", {"LOCALDATA_CONFIG": "/tmp/test.yaml"}):
            paths = get_config_paths()
            assert paths[0].priority == 0

    @patch("localdata_mcp.config_paths.platform.system", return_value="Linux")
    def test_linux_xdg_path(self, _mock_sys):
        with patch.dict("os.environ", {}, clear=False):
            # Remove LOCALDATA_CONFIG if set
            env = dict(**{"LOCALDATA_CONFIG": ""})
            with patch.dict("os.environ", env):
                paths = get_config_paths()
                xdg = [
                    p for p in paths if p.location_type == ConfigLocationType.USER_XDG
                ]
                assert len(xdg) == 1
                assert "localdata/config.yaml" in str(xdg[0].path)

    @patch("localdata_mcp.config_paths.platform.system", return_value="Linux")
    def test_linux_xdg_custom(self, _mock_sys):
        with patch.dict("os.environ", {"XDG_CONFIG_HOME": "/custom/config"}):
            paths = get_config_paths()
            xdg = [p for p in paths if p.location_type == ConfigLocationType.USER_XDG]
            assert len(xdg) == 1
            assert str(xdg[0].path).startswith("/custom/config")

    @patch("localdata_mcp.config_paths.platform.system", return_value="Linux")
    def test_linux_has_system_path(self, _mock_sys):
        paths = get_config_paths()
        sys_paths = [p for p in paths if p.location_type == ConfigLocationType.SYSTEM]
        assert len(sys_paths) == 1
        assert sys_paths[0].path == Path("/etc/localdata/config.yaml")

    @patch("localdata_mcp.config_paths.platform.system", return_value="Darwin")
    def test_macos_has_xdg_and_library(self, _mock_sys):
        paths = get_config_paths()
        types = {p.location_type for p in paths}
        assert ConfigLocationType.USER_XDG in types
        assert ConfigLocationType.USER_MACOS in types

    @patch("localdata_mcp.config_paths.platform.system", return_value="Windows")
    def test_windows_appdata(self, _mock_sys):
        with patch.dict("os.environ", {"APPDATA": "C:\\Users\\test\\AppData\\Roaming"}):
            paths = get_config_paths()
            win = [
                p for p in paths if p.location_type == ConfigLocationType.USER_WINDOWS
            ]
            assert len(win) == 1
            assert "localdata" in str(win[0].path)

    @patch("localdata_mcp.config_paths.platform.system", return_value="Windows")
    def test_windows_no_system_path(self, _mock_sys):
        paths = get_config_paths()
        sys_paths = [p for p in paths if p.location_type == ConfigLocationType.SYSTEM]
        assert len(sys_paths) == 0

    def test_priority_order_increasing(self):
        paths = get_config_paths()
        priorities = [p.priority for p in paths]
        assert priorities == sorted(priorities)


class TestGetRecommendedPath:
    @patch("localdata_mcp.config_paths.platform.system", return_value="Linux")
    def test_linux_returns_xdg(self, _mock_sys):
        path = get_recommended_path()
        assert "localdata/config.yaml" in str(path)

    @patch("localdata_mcp.config_paths.platform.system", return_value="Darwin")
    def test_macos_returns_xdg(self, _mock_sys):
        path = get_recommended_path()
        assert "localdata/config.yaml" in str(path)

    @patch("localdata_mcp.config_paths.platform.system", return_value="Windows")
    def test_windows_returns_appdata(self, _mock_sys):
        with patch.dict("os.environ", {"APPDATA": "C:\\Users\\test\\AppData"}):
            path = get_recommended_path()
            assert "localdata" in str(path)


class TestDeprecationWarning:
    def test_emits_warning(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            emit_deprecation_warning(Path("~/.localdata.yaml"))
            deprecations = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(deprecations) == 1
            assert "deprecated" in str(deprecations[0].message).lower()

    def test_warning_mentions_recommended_path(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            emit_deprecation_warning(Path("~/.localdata.yaml"))
            msg = str(w[0].message)
            assert "localdata/config.yaml" in msg
