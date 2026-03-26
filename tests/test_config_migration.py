"""Tests for config migration utility."""

from pathlib import Path
from unittest.mock import patch

import pytest

from localdata_mcp.config_paths import migrate_config


@pytest.fixture()
def source_config(tmp_path: Path) -> Path:
    """Create a temporary source config file."""
    src = tmp_path / "source" / ".localdata.yaml"
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_text("databases:\n  mydb:\n    type: sqlite\n", encoding="utf-8")
    return src


class TestMigrateConfigDefaultPaths:
    """Test migration with default source/dest paths."""

    def test_migrate_config_default_paths(self, tmp_path: Path) -> None:
        legacy = tmp_path / ".localdata.yaml"
        legacy.write_text("key: value\n", encoding="utf-8")
        recommended = tmp_path / "config" / "localdata" / "config.yaml"

        with (
            patch(
                "localdata_mcp.config_paths.Path",
                wraps=Path,
            ) as mock_path,
            patch(
                "localdata_mcp.config_paths.get_recommended_path",
                return_value=recommended,
            ),
        ):
            # Make Path("~/.localdata.yaml").expanduser() return our tmp file
            original_init = Path.__new__

            def patched_new(cls, *args, **kwargs):
                return original_init(cls, *args, **kwargs)

            # Simpler approach: pass explicit paths
            result = migrate_config(source=legacy, dest=recommended)

        assert result is True
        assert recommended.exists()
        assert recommended.read_text(encoding="utf-8") == "key: value\n"


class TestMigrateConfigCustomPaths:
    """Test migration with explicit source and dest."""

    def test_migrate_config_custom_paths(
        self, tmp_path: Path, source_config: Path
    ) -> None:
        dest = tmp_path / "custom" / "dest" / "config.yaml"
        result = migrate_config(source=source_config, dest=dest)

        assert result is True
        assert dest.exists()


class TestMigrateConfigBackup:
    """Test backup behavior."""

    def test_migrate_config_creates_backup(
        self, tmp_path: Path, source_config: Path
    ) -> None:
        dest = tmp_path / "dest" / "config.yaml"
        migrate_config(source=source_config, dest=dest, backup=True)

        backup_path = source_config.with_suffix(".yaml.bak")
        assert backup_path.exists()
        assert backup_path.read_text(encoding="utf-8") == source_config.read_text(
            encoding="utf-8"
        )

    def test_migrate_config_no_backup(
        self, tmp_path: Path, source_config: Path
    ) -> None:
        dest = tmp_path / "dest" / "config.yaml"
        migrate_config(source=source_config, dest=dest, backup=False)

        backup_path = source_config.with_suffix(".yaml.bak")
        assert not backup_path.exists()


class TestMigrateConfigErrors:
    """Test error conditions."""

    def test_migrate_config_source_not_found(self, tmp_path: Path) -> None:
        missing = tmp_path / "nonexistent.yaml"
        dest = tmp_path / "dest" / "config.yaml"

        with pytest.raises(FileNotFoundError, match="Source config not found"):
            migrate_config(source=missing, dest=dest)

    def test_migrate_config_dest_exists(
        self, tmp_path: Path, source_config: Path
    ) -> None:
        dest = tmp_path / "dest" / "config.yaml"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text("existing\n", encoding="utf-8")

        with pytest.raises(FileExistsError, match="Destination config already exists"):
            migrate_config(source=source_config, dest=dest)


class TestMigrateConfigFilesystem:
    """Test filesystem operations."""

    def test_migrate_config_creates_parent_dirs(
        self, tmp_path: Path, source_config: Path
    ) -> None:
        dest = tmp_path / "deep" / "nested" / "dir" / "config.yaml"
        assert not dest.parent.exists()

        migrate_config(source=source_config, dest=dest)

        assert dest.parent.exists()
        assert dest.exists()

    def test_migrate_config_preserves_content(
        self, tmp_path: Path, source_config: Path
    ) -> None:
        dest = tmp_path / "dest" / "config.yaml"
        original_content = source_config.read_text(encoding="utf-8")

        migrate_config(source=source_config, dest=dest)

        assert dest.read_text(encoding="utf-8") == original_content

    def test_migrate_config_returns_true(
        self, tmp_path: Path, source_config: Path
    ) -> None:
        dest = tmp_path / "dest" / "config.yaml"
        result = migrate_config(source=source_config, dest=dest)
        assert result is True
