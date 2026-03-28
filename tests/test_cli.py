"""Tests for CLI argument parsing in main()."""

import sys
from pathlib import Path
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

    def test_force_flag(self):
        from localdata_mcp.localdata_mcp import _parse_cli_args

        with patch.object(sys, "argv", ["prog", "--migrate-config", "--force"]):
            args = _parse_cli_args()
            assert args.force is True
            assert args.migrate_config is True

    def test_force_flag_default(self):
        from localdata_mcp.localdata_mcp import _parse_cli_args

        with patch.object(sys, "argv", ["prog"]):
            args = _parse_cli_args()
            assert args.force is False

    def test_validate_config_flag(self):
        from localdata_mcp.localdata_mcp import _parse_cli_args

        with patch.object(sys, "argv", ["prog", "--validate-config"]):
            args = _parse_cli_args()
            assert args.validate_config is True

    def test_show_config_flag(self):
        from localdata_mcp.localdata_mcp import _parse_cli_args

        with patch.object(sys, "argv", ["prog", "--show-config"]):
            args = _parse_cli_args()
            assert args.show_config is True

    def test_init_config_flag(self):
        from localdata_mcp.localdata_mcp import _parse_cli_args

        with patch.object(sys, "argv", ["prog", "--init-config"]):
            args = _parse_cli_args()
            assert args.init_config is True

    def test_new_flags_default_false(self):
        from localdata_mcp.localdata_mcp import _parse_cli_args

        with patch.object(sys, "argv", ["prog"]):
            args = _parse_cli_args()
            assert args.validate_config is False
            assert args.show_config is False
            assert args.init_config is False

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
    def test_migrate_config_success(self, mock_init_config, tmp_path, capsys):
        from localdata_mcp.localdata_mcp import main

        source = tmp_path / ".localdata.yaml"
        source.write_text("databases: {}")
        dest = tmp_path / "localdata" / "config.yaml"

        mock_migrate = MagicMock(return_value=True)
        mock_get_path = MagicMock(return_value=dest)

        with (
            patch.object(sys, "argv", ["prog", "--migrate-config"]),
            patch("localdata_mcp.localdata_mcp.Path", wraps=Path) as mock_path,
            patch(
                "localdata_mcp.config_paths.get_recommended_path",
                mock_get_path,
            ),
            patch(
                "localdata_mcp.config_paths.migrate_config",
                mock_migrate,
            ),
        ):
            # Make Path("~/.localdata.yaml").expanduser() return our tmp source
            original_path = Path.__new__

            def path_new(cls, *args, **kwargs):
                return original_path(cls, *args, **kwargs)

            mock_path.side_effect = lambda p: Path(p)
            # We need to intercept the expanduser call on the result
            with patch.object(
                Path,
                "expanduser",
                return_value=source,
            ):
                with pytest.raises(SystemExit) as exc_info:
                    main()
                assert exc_info.value.code == 0

        output = capsys.readouterr().out
        assert "Config migrated successfully" in output
        mock_migrate.assert_called_once_with(source=source, dest=dest)
        mock_init_config.assert_not_called()

    @patch("localdata_mcp.localdata_mcp.initialize_config")
    def test_migrate_config_no_source(self, mock_init_config, tmp_path, capsys):
        from localdata_mcp.localdata_mcp import main

        source = tmp_path / ".localdata.yaml"  # does not exist
        dest = tmp_path / "localdata" / "config.yaml"

        mock_migrate = MagicMock(side_effect=FileNotFoundError("not found"))
        mock_get_path = MagicMock(return_value=dest)

        with (
            patch.object(sys, "argv", ["prog", "--migrate-config"]),
            patch(
                "localdata_mcp.config_paths.get_recommended_path",
                mock_get_path,
            ),
            patch(
                "localdata_mcp.config_paths.migrate_config",
                mock_migrate,
            ),
            patch.object(Path, "expanduser", return_value=source),
        ):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1

        output = capsys.readouterr().out
        assert "No legacy config found" in output
        mock_init_config.assert_not_called()

    @patch("localdata_mcp.localdata_mcp.initialize_config")
    def test_migrate_config_dest_exists(self, mock_init_config, tmp_path, capsys):
        from localdata_mcp.localdata_mcp import main

        source = tmp_path / ".localdata.yaml"
        source.write_text("databases: {}")
        dest = tmp_path / "localdata" / "config.yaml"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text("existing config")

        mock_migrate = MagicMock(side_effect=FileExistsError(f"exists: {dest}"))
        mock_get_path = MagicMock(return_value=dest)

        with (
            patch.object(sys, "argv", ["prog", "--migrate-config"]),
            patch(
                "localdata_mcp.config_paths.get_recommended_path",
                mock_get_path,
            ),
            patch(
                "localdata_mcp.config_paths.migrate_config",
                mock_migrate,
            ),
            patch.object(Path, "expanduser", return_value=source),
        ):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1

        output = capsys.readouterr().out
        assert "--force" in output
        mock_init_config.assert_not_called()

    @patch("localdata_mcp.localdata_mcp.initialize_config")
    def test_migrate_config_force_overwrites(self, mock_init_config, tmp_path, capsys):
        from localdata_mcp.localdata_mcp import main

        source = tmp_path / ".localdata.yaml"
        source.write_text("databases: {}")
        dest = tmp_path / "localdata" / "config.yaml"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text("existing config")

        mock_migrate = MagicMock(return_value=True)
        mock_get_path = MagicMock(return_value=dest)

        with (
            patch.object(sys, "argv", ["prog", "--migrate-config", "--force"]),
            patch(
                "localdata_mcp.config_paths.get_recommended_path",
                mock_get_path,
            ),
            patch(
                "localdata_mcp.config_paths.migrate_config",
                mock_migrate,
            ),
            patch.object(Path, "expanduser", return_value=source),
        ):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0

        output = capsys.readouterr().out
        assert "Config migrated successfully" in output
        # --force should have removed dest before calling migrate
        assert not dest.exists()
        mock_migrate.assert_called_once_with(source=source, dest=dest)
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

    @patch("localdata_mcp.localdata_mcp.get_config_manager")
    @patch("localdata_mcp.localdata_mcp.initialize_config")
    def test_validate_config_flag(self, mock_init_config, mock_get_cm, capsys):
        from localdata_mcp.localdata_mcp import main

        mock_cm = MagicMock()
        mock_get_cm.return_value = mock_cm

        with patch.object(sys, "argv", ["prog", "--validate-config"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0

        output = capsys.readouterr().out
        assert "Configuration is valid" in output
        mock_init_config.assert_not_called()

    @patch("localdata_mcp.localdata_mcp.get_config_manager")
    @patch("localdata_mcp.localdata_mcp.initialize_config")
    def test_validate_config_flag_invalid(self, mock_init_config, mock_get_cm, capsys):
        from localdata_mcp.localdata_mcp import main

        mock_cm = MagicMock()
        mock_cm._validate_config.side_effect = ValueError("bad config")
        mock_get_cm.return_value = mock_cm

        with patch.object(sys, "argv", ["prog", "--validate-config"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1

        output = capsys.readouterr().out
        assert "validation failed" in output

    @patch("localdata_mcp.localdata_mcp.yaml")
    @patch("localdata_mcp.localdata_mcp.get_config_manager")
    @patch("localdata_mcp.localdata_mcp.initialize_config")
    def test_show_config_flag(self, mock_init_config, mock_get_cm, mock_yaml, capsys):
        from localdata_mcp.localdata_mcp import main

        mock_cm = MagicMock()
        mock_cm._config_data = {
            "databases": {
                "mydb": {
                    "type": "sqlite",
                    "connection_string": "sqlite:///secret.db",
                }
            },
            "logging": {"level": "info"},
        }
        mock_get_cm.return_value = mock_cm
        mock_yaml.dump.return_value = "databases:\n  mydb:\n    connection_string: '***REDACTED***'\n    type: sqlite\n"

        with patch.object(sys, "argv", ["prog", "--show-config"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0

        # Verify yaml.dump was called with redacted data
        call_args = mock_yaml.dump.call_args
        dumped_data = call_args[0][0]
        assert dumped_data["databases"]["mydb"]["connection_string"] == "***REDACTED***"
        mock_init_config.assert_not_called()

    @patch("localdata_mcp.localdata_mcp.initialize_config")
    def test_init_config_flag(self, mock_init_config, tmp_path, capsys):
        from localdata_mcp.localdata_mcp import main

        dest = tmp_path / "localdata" / "config.yaml"
        mock_create = MagicMock(return_value=dest)

        with (
            patch.object(sys, "argv", ["prog", "--init-config"]),
            patch(
                "localdata_mcp.config_paths.create_default_config",
                mock_create,
            ),
        ):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0

        output = capsys.readouterr().out
        assert "Default config created" in output
        mock_create.assert_called_once()
        mock_init_config.assert_not_called()

    @patch("localdata_mcp.localdata_mcp.initialize_config")
    def test_init_config_flag_already_exists(self, mock_init_config, capsys):
        from localdata_mcp.localdata_mcp import main

        mock_create = MagicMock(side_effect=FileExistsError("exists"))

        with (
            patch.object(sys, "argv", ["prog", "--init-config"]),
            patch(
                "localdata_mcp.config_paths.create_default_config",
                mock_create,
            ),
        ):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1

        output = capsys.readouterr().out
        assert "already exists" in output
        mock_init_config.assert_not_called()
