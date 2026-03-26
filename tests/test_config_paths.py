"""Tests for OS-aware configuration path resolution."""

import warnings
from pathlib import Path
from unittest.mock import patch

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


class TestOSSpecificPaths:
    """Comprehensive OS-specific path resolution tests."""

    @patch("localdata_mcp.config_paths.platform.system", return_value="Linux")
    @patch.dict("os.environ", {"LOCALDATA_CONFIG": ""}, clear=False)
    def test_linux_xdg_default_path(self, _mock_sys):
        import os

        xdg_backup = os.environ.pop("XDG_CONFIG_HOME", None)
        try:
            paths = get_config_paths()
            xdg = [p for p in paths if p.location_type == ConfigLocationType.USER_XDG]
            assert len(xdg) == 1
            assert str(xdg[0].path).endswith(".config/localdata/config.yaml")
        finally:
            if xdg_backup is not None:
                os.environ["XDG_CONFIG_HOME"] = xdg_backup

    @patch("localdata_mcp.config_paths.platform.system", return_value="Linux")
    def test_linux_xdg_custom_path(self, _mock_sys):
        with patch.dict(
            "os.environ",
            {"XDG_CONFIG_HOME": "/custom/config", "LOCALDATA_CONFIG": ""},
        ):
            paths = get_config_paths()
            xdg = [p for p in paths if p.location_type == ConfigLocationType.USER_XDG]
            assert len(xdg) == 1
            assert xdg[0].path == Path("/custom/config/localdata/config.yaml")

    @patch("localdata_mcp.config_paths.platform.system", return_value="Linux")
    def test_linux_has_system_path(self, _mock_sys):
        with patch.dict("os.environ", {"LOCALDATA_CONFIG": ""}, clear=False):
            paths = get_config_paths()
            sys_paths = [
                p for p in paths if p.location_type == ConfigLocationType.SYSTEM
            ]
            assert len(sys_paths) == 1
            assert sys_paths[0].path == Path("/etc/localdata/config.yaml")

    @patch("localdata_mcp.config_paths.platform.system", return_value="Linux")
    def test_linux_no_macos_path(self, _mock_sys):
        with patch.dict("os.environ", {"LOCALDATA_CONFIG": ""}, clear=False):
            paths = get_config_paths()
            macos = [
                p for p in paths if p.location_type == ConfigLocationType.USER_MACOS
            ]
            assert len(macos) == 0

    @patch("localdata_mcp.config_paths.platform.system", return_value="Darwin")
    def test_macos_has_xdg_and_library(self, _mock_sys):
        with patch.dict("os.environ", {"LOCALDATA_CONFIG": ""}, clear=False):
            paths = get_config_paths()
            types = {p.location_type for p in paths}
            assert ConfigLocationType.USER_XDG in types
            assert ConfigLocationType.USER_MACOS in types

    @patch("localdata_mcp.config_paths.platform.system", return_value="Darwin")
    def test_macos_library_path_correct(self, _mock_sys):
        with patch.dict("os.environ", {"LOCALDATA_CONFIG": ""}, clear=False):
            paths = get_config_paths()
            macos = [
                p for p in paths if p.location_type == ConfigLocationType.USER_MACOS
            ]
            assert len(macos) == 1
            assert str(macos[0].path).endswith(
                "Library/Application Support/localdata/config.yaml"
            )

    @patch("localdata_mcp.config_paths.platform.system", return_value="Darwin")
    def test_macos_has_system_path(self, _mock_sys):
        with patch.dict("os.environ", {"LOCALDATA_CONFIG": ""}, clear=False):
            paths = get_config_paths()
            sys_paths = [
                p for p in paths if p.location_type == ConfigLocationType.SYSTEM
            ]
            assert len(sys_paths) == 1

    @patch("localdata_mcp.config_paths.platform.system", return_value="Windows")
    def test_windows_appdata_path(self, _mock_sys):
        with patch.dict(
            "os.environ",
            {
                "APPDATA": "C:\\Users\\test\\AppData\\Roaming",
                "LOCALDATA_CONFIG": "",
            },
        ):
            paths = get_config_paths()
            win = [
                p for p in paths if p.location_type == ConfigLocationType.USER_WINDOWS
            ]
            assert len(win) == 1
            assert str(win[0].path).startswith("C:\\Users\\test\\AppData\\Roaming")

    @patch("localdata_mcp.config_paths.platform.system", return_value="Windows")
    def test_windows_no_system_path(self, _mock_sys):
        with patch.dict("os.environ", {"LOCALDATA_CONFIG": ""}, clear=False):
            paths = get_config_paths()
            sys_paths = [
                p for p in paths if p.location_type == ConfigLocationType.SYSTEM
            ]
            assert len(sys_paths) == 0

    @patch("localdata_mcp.config_paths.platform.system", return_value="Windows")
    def test_windows_no_xdg_path(self, _mock_sys):
        with patch.dict("os.environ", {"LOCALDATA_CONFIG": ""}, clear=False):
            paths = get_config_paths()
            xdg = [p for p in paths if p.location_type == ConfigLocationType.USER_XDG]
            assert len(xdg) == 0

    @patch("localdata_mcp.config_paths.platform.system", return_value="Windows")
    def test_windows_missing_appdata(self, _mock_sys):
        with patch.dict(
            "os.environ",
            {"APPDATA": "", "LOCALDATA_CONFIG": ""},
            clear=False,
        ):
            paths = get_config_paths()
            win = [
                p for p in paths if p.location_type == ConfigLocationType.USER_WINDOWS
            ]
            assert len(win) == 0

    def test_path_priority_ordering(self):
        with patch.dict(
            "os.environ",
            {"LOCALDATA_CONFIG": "/tmp/explicit.yaml"},
            clear=False,
        ):
            paths = get_config_paths()
            type_priorities = {p.location_type: p.priority for p in paths}
            assert type_priorities[ConfigLocationType.EXPLICIT] == 0
            assert type_priorities[ConfigLocationType.PROJECT_LOCAL] == 1
            assert type_priorities[ConfigLocationType.LEGACY] == 5
            # User-level priorities are between project-local and system/legacy
            for p in paths:
                if p.location_type in (
                    ConfigLocationType.USER_XDG,
                    ConfigLocationType.USER_MACOS,
                    ConfigLocationType.USER_WINDOWS,
                ):
                    assert 2 <= p.priority <= 3

    def test_all_paths_have_valid_types(self):
        paths = get_config_paths()
        valid_types = set(ConfigLocationType)
        for p in paths:
            assert isinstance(p, ConfigPathInfo)
            assert p.location_type in valid_types

    @patch("localdata_mcp.config_paths.platform.system", return_value="Linux")
    @patch.dict("os.environ", {"LOCALDATA_CONFIG": ""}, clear=False)
    def test_recommended_path_linux(self, _mock_sys):
        import os

        xdg_backup = os.environ.pop("XDG_CONFIG_HOME", None)
        try:
            path = get_recommended_path()
            assert str(path).endswith(".config/localdata/config.yaml")
        finally:
            if xdg_backup is not None:
                os.environ["XDG_CONFIG_HOME"] = xdg_backup

    @patch("localdata_mcp.config_paths.platform.system", return_value="Windows")
    def test_recommended_path_windows(self, _mock_sys):
        with patch.dict(
            "os.environ",
            {
                "APPDATA": "C:\\Users\\test\\AppData\\Roaming",
                "LOCALDATA_CONFIG": "",
            },
        ):
            path = get_recommended_path()
            assert str(path).startswith("C:\\Users\\test\\AppData\\Roaming")
            assert "localdata" in str(path)

    @patch("localdata_mcp.config_paths.platform.system", return_value="Darwin")
    @patch.dict("os.environ", {"LOCALDATA_CONFIG": ""}, clear=False)
    def test_recommended_path_macos(self, _mock_sys):
        import os

        xdg_backup = os.environ.pop("XDG_CONFIG_HOME", None)
        try:
            path = get_recommended_path()
            assert str(path).endswith(".config/localdata/config.yaml")
        finally:
            if xdg_backup is not None:
                os.environ["XDG_CONFIG_HOME"] = xdg_backup


class TestDeprecationWarningIntegration:
    """Integration tests for deprecation warning behavior."""

    def test_deprecation_warning_message_contains_paths(self):
        legacy = Path("~/.localdata.yaml")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            emit_deprecation_warning(legacy)
            assert len(w) == 1
            msg = str(w[0].message)
            assert str(legacy) in msg
            assert "localdata/config.yaml" in msg

    def test_deprecation_warning_is_deprecation_type(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            emit_deprecation_warning(Path("~/.localdata.yaml"))
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
