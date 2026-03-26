"""Integration tests for config loading from OS-aware paths.

Creates real temp directories and config files, then verifies
ConfigManager loads everything correctly end-to-end.
"""

import os
from unittest.mock import patch

import pytest

from localdata_mcp.config_manager import ConfigManager


class TestConfigIntegration:
    """End-to-end tests for ConfigManager loading from various paths."""

    def test_load_from_xdg_path(self, tmp_path):
        """ConfigManager discovers config via XDG_CONFIG_HOME."""
        config_dir = tmp_path / "localdata"
        config_dir.mkdir()
        config_file = config_dir / "config.yaml"
        config_file.write_text(
            "databases:\n"
            "  testdb:\n"
            "    type: sqlite\n"
            "    connection_string: /tmp/test.db\n"
        )
        env = {
            "XDG_CONFIG_HOME": str(tmp_path),
            "LOCALDATA_CONFIG": "",
        }
        with patch.dict("os.environ", env, clear=False):
            mgr = ConfigManager()
        db = mgr.get_database_config("testdb")
        assert db is not None
        assert db.connection_string == "/tmp/test.db"

    def test_load_from_project_local(self, tmp_path, monkeypatch):
        """ConfigManager discovers .localdata.yaml in the working directory."""
        config_file = tmp_path / ".localdata.yaml"
        config_file.write_text(
            "databases:\n"
            "  localdb:\n"
            "    type: sqlite\n"
            "    connection_string: /tmp/local.db\n"
        )
        monkeypatch.chdir(tmp_path)
        env = {"LOCALDATA_CONFIG": ""}
        with patch.dict("os.environ", env, clear=False):
            mgr = ConfigManager()
        db = mgr.get_database_config("localdb")
        assert db is not None
        assert db.connection_string == "/tmp/local.db"

    def test_load_staging_section(self, tmp_path):
        """Staging section is loaded and returned as StagingConfig."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("staging:\n  max_concurrent: 20\n")
        mgr = ConfigManager(config_file=str(config_file))
        staging = mgr.get_staging_config()
        assert staging.max_concurrent == 20

    def test_load_memory_section(self, tmp_path):
        """Memory section is loaded and returned as MemoryConfig."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("memory:\n  max_budget_mb: 1024\n")
        mgr = ConfigManager(config_file=str(config_file))
        memory = mgr.get_memory_config()
        assert memory.max_budget_mb == 1024

    def test_load_query_section(self, tmp_path):
        """Query section is loaded and returned as QueryConfig."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("query:\n  blob_handling: placeholder\n")
        mgr = ConfigManager(config_file=str(config_file))
        query = mgr.get_query_config()
        assert query.blob_handling.value == "placeholder"

    def test_load_connections_section(self, tmp_path):
        """Connections section is loaded and returned as ConnectionsConfig."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("connections:\n  timeout_seconds: 60\n")
        mgr = ConfigManager(config_file=str(config_file))
        connections = mgr.get_connections_config()
        assert connections.timeout_seconds == 60

    def test_load_security_section(self, tmp_path):
        """Security section is loaded and returned as SecurityConfig."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("security:\n  max_query_length: 50000\n")
        mgr = ConfigManager(config_file=str(config_file))
        security = mgr.get_security_config()
        assert security.max_query_length == 50000

    def test_env_override_file_values(self, tmp_path):
        """Environment variables override values loaded from the YAML file."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("staging:\n  max_concurrent: 10\n")
        env = {"LOCALDATA_STAGING_MAX_CONCURRENT": "50"}
        with patch.dict("os.environ", env, clear=False):
            mgr = ConfigManager(config_file=str(config_file))
        staging = mgr.get_staging_config()
        assert staging.max_concurrent == 50

    def test_full_config_with_all_sections(self, tmp_path):
        """A comprehensive config with all sections loads correctly."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            "databases:\n"
            "  mydb:\n"
            "    type: sqlite\n"
            "    connection_string: /tmp/full.db\n"
            "staging:\n"
            "  max_concurrent: 15\n"
            "  max_size_mb: 4096\n"
            "memory:\n"
            "  max_budget_mb: 2048\n"
            "  budget_percent: 25\n"
            "query:\n"
            "  default_chunk_size: 200\n"
            "  blob_handling: include\n"
            "connections:\n"
            "  max_concurrent: 20\n"
            "  timeout_seconds: 45\n"
            "security:\n"
            "  max_query_length: 30000\n"
            "  blocked_keywords:\n"
            "    - DROP\n"
            "    - TRUNCATE\n"
        )
        mgr = ConfigManager(config_file=str(config_file))

        db = mgr.get_database_config("mydb")
        assert db is not None
        assert db.connection_string == "/tmp/full.db"

        staging = mgr.get_staging_config()
        assert staging.max_concurrent == 15
        assert staging.max_size_mb == 4096

        memory = mgr.get_memory_config()
        assert memory.max_budget_mb == 2048
        assert memory.budget_percent == 25

        query = mgr.get_query_config()
        assert query.default_chunk_size == 200
        assert query.blob_handling.value == "include"

        connections = mgr.get_connections_config()
        assert connections.max_concurrent == 20
        assert connections.timeout_seconds == 45

        security = mgr.get_security_config()
        assert security.max_query_length == 30000
        assert security.blocked_keywords == ["DROP", "TRUNCATE"]

    def test_config_file_flag_overrides_discovery(self, tmp_path):
        """Passing config_file= bypasses the discovery mechanism."""
        # Create a file at the explicit path
        explicit = tmp_path / "explicit.yaml"
        explicit.write_text(
            "databases:\n"
            "  explicit_db:\n"
            "    type: sqlite\n"
            "    connection_string: /tmp/explicit.db\n"
        )
        # Create a project-local file that should NOT be used
        project_dir = tmp_path / "project"
        project_dir.mkdir()
        project_local = project_dir / ".localdata.yaml"
        project_local.write_text(
            "databases:\n"
            "  local_db:\n"
            "    type: sqlite\n"
            "    connection_string: /tmp/local.db\n"
        )
        mgr = ConfigManager(config_file=str(explicit))
        assert mgr.get_database_config("explicit_db") is not None
        assert mgr.get_database_config("local_db") is None
