"""Tests for CLI argument parsing in main()."""

import sys
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def _isolate_module():
    """Ensure localdata_mcp module internals are mocked so imports don't fail."""
    # The module-level code creates logging_manager, config_manager, etc.
    # We only need the functions under test, so we patch at call sites.


class TestGetVersion:
    """Tests for _get_version()."""

    def test_returns_package_version(self):
        with patch("importlib.metadata.version", return_value="1.5.3"):
            from localdata_mcp.localdata_mcp import _get_version

            assert _get_version() == "1.5.3"

    def test_returns_unknown_on_missing_package(self):
        import importlib.metadata

        with patch(
            "importlib.metadata.version",
            side_effect=importlib.metadata.PackageNotFoundError("localdata-mcp"),
        ):
            from localdata_mcp.localdata_mcp import _get_version

            assert _get_version() == "unknown"


class TestParseCLIArgs:
    """Tests for _parse_cli_args()."""

    def test_config_flag_long(self):
        from localdata_mcp.localdata_mcp import _parse_cli_args

        with patch.object(sys, "argv", ["prog", "--config", "/tmp/test.yaml"]):
            args = _parse_cli_args()
            assert args.config == "/tmp/test.yaml"
            assert args.migrate_config is False

    def test_config_flag_short(self):
        from localdata_mcp.localdata_mcp import _parse_cli_args

        with patch.object(sys, "argv", ["prog", "-c", "/tmp/test.yaml"]):
            args = _parse_cli_args()
            assert args.config == "/tmp/test.yaml"

    def test_migrate_config_flag(self):
        from localdata_mcp.localdata_mcp import _parse_cli_args

        with patch.object(sys, "argv", ["prog", "--migrate-config"]):
            args = _parse_cli_args()
            assert args.migrate_config is True

    def test_no_args(self):
        from localdata_mcp.localdata_mcp import _parse_cli_args

        with patch.object(sys, "argv", ["prog"]):
            args = _parse_cli_args()
            assert args.config is None
            assert args.migrate_config is False

    def test_version_flag_exits(self):
        from localdata_mcp.localdata_mcp import _parse_cli_args

        with patch.object(sys, "argv", ["prog", "--version"]):
            with pytest.raises(SystemExit) as exc_info:
                _parse_cli_args()
            assert exc_info.value.code == 0

    def test_unknown_args_ignored(self):
        """parse_known_args should not fail on unknown arguments."""
        from localdata_mcp.localdata_mcp import _parse_cli_args

        with patch.object(sys, "argv", ["prog", "--unknown-flag", "value"]):
            args = _parse_cli_args()
            assert args.config is None


class TestMain:
    """Tests for main() entry point."""

    @patch("localdata_mcp.localdata_mcp.mcp")
    @patch("localdata_mcp.localdata_mcp.DatabaseManager")
    @patch("localdata_mcp.localdata_mcp.logging_manager")
    @patch("localdata_mcp.localdata_mcp.logging_config")
    @patch("localdata_mcp.localdata_mcp.logger")
    @patch("localdata_mcp.localdata_mcp.initialize_config")
    def test_config_flag_passes_path(
        self,
        mock_init_config,
        mock_logger,
        mock_logging_config,
        mock_logging_manager,
        mock_db_manager,
        mock_mcp,
    ):
        from localdata_mcp.localdata_mcp import main

        mock_logging_config.enable_metrics = False
        mock_logging_config.enable_security_logging = False
        mock_logging_config.level.value = "info"
        mock_logging_manager.context.return_value.__enter__ = MagicMock()
        mock_logging_manager.context.return_value.__exit__ = MagicMock(
            return_value=False
        )

        with patch.object(sys, "argv", ["prog", "--config", "/tmp/test.yaml"]):
            main()

        mock_init_config.assert_called_once_with(config_file="/tmp/test.yaml")
        mock_mcp.run.assert_called_once_with(transport="stdio")

    @patch("localdata_mcp.localdata_mcp.mcp")
    @patch("localdata_mcp.localdata_mcp.DatabaseManager")
    @patch("localdata_mcp.localdata_mcp.logging_manager")
    @patch("localdata_mcp.localdata_mcp.logging_config")
    @patch("localdata_mcp.localdata_mcp.logger")
    @patch("localdata_mcp.localdata_mcp.initialize_config")
    def test_no_args_runs_normally(
        self,
        mock_init_config,
        mock_logger,
        mock_logging_config,
        mock_logging_manager,
        mock_db_manager,
        mock_mcp,
    ):
        from localdata_mcp.localdata_mcp import main

        mock_logging_config.enable_metrics = False
        mock_logging_config.enable_security_logging = False
        mock_logging_config.level.value = "info"
        mock_logging_manager.context.return_value.__enter__ = MagicMock()
        mock_logging_manager.context.return_value.__exit__ = MagicMock(
            return_value=False
        )

        with patch.object(sys, "argv", ["prog"]):
            main()

        mock_init_config.assert_not_called()
        mock_mcp.run.assert_called_once_with(transport="stdio")

    @patch("localdata_mcp.localdata_mcp.initialize_config")
    def test_migrate_config_placeholder(self, mock_init_config, capsys):
        from localdata_mcp.localdata_mcp import main

        with patch.object(sys, "argv", ["prog", "--migrate-config"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0

        output = capsys.readouterr().out
        assert "Migration not yet implemented" in output
        mock_init_config.assert_not_called()

    @patch("localdata_mcp.localdata_mcp.mcp")
    @patch("localdata_mcp.localdata_mcp.DatabaseManager")
    @patch("localdata_mcp.localdata_mcp.logging_manager")
    @patch("localdata_mcp.localdata_mcp.logging_config")
    @patch("localdata_mcp.localdata_mcp.logger")
    @patch("localdata_mcp.localdata_mcp.initialize_config")
    def test_short_config_flag(
        self,
        mock_init_config,
        mock_logger,
        mock_logging_config,
        mock_logging_manager,
        mock_db_manager,
        mock_mcp,
    ):
        from localdata_mcp.localdata_mcp import main

        mock_logging_config.enable_metrics = False
        mock_logging_config.enable_security_logging = False
        mock_logging_config.level.value = "info"
        mock_logging_manager.context.return_value.__enter__ = MagicMock()
        mock_logging_manager.context.return_value.__exit__ = MagicMock(
            return_value=False
        )

        with patch.object(sys, "argv", ["prog", "-c", "/tmp/short.yaml"]):
            main()

        mock_init_config.assert_called_once_with(config_file="/tmp/short.yaml")

    def test_version_flag_prints_version(self, capsys):
        from localdata_mcp.localdata_mcp import main

        with patch.object(sys, "argv", ["prog", "--version"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0

        output = capsys.readouterr().out
        assert "localdata-mcp" in output
