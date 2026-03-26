"""Tests for environment variable overrides and validation of new config sections."""

import os
from unittest.mock import patch

import pytest

from localdata_mcp.config_manager import ConfigManager


def _make_manager(**env_vars):
    """Create a ConfigManager with given env vars and no config file."""
    fake_path = "/tmp/nonexistent_localdata_config.yaml"
    with patch.dict(os.environ, {"LOCALDATA_CONFIG": fake_path, **env_vars}):
        mgr = ConfigManager()
    return mgr


class TestStagingEnvVars:
    """Test staging section environment variable overrides."""

    def test_staging_max_concurrent(self):
        mgr = _make_manager(LOCALDATA_STAGING_MAX_CONCURRENT="20")
        assert mgr._config_data.get("staging", {}).get("max_concurrent") == 20

    def test_staging_eviction_policy(self):
        mgr = _make_manager(LOCALDATA_STAGING_EVICTION_POLICY="oldest")
        assert mgr._config_data.get("staging", {}).get("eviction_policy") == "oldest"


class TestMemoryEnvVars:
    """Test memory section environment variable overrides."""

    def test_memory_max_budget_mb(self):
        mgr = _make_manager(LOCALDATA_MEMORY_MAX_BUDGET_MB="1024")
        assert mgr._config_data.get("memory", {}).get("max_budget_mb") == 1024

    def test_memory_budget_percent(self):
        mgr = _make_manager(LOCALDATA_MEMORY_BUDGET_PERCENT="25")
        assert mgr._config_data.get("memory", {}).get("budget_percent") == 25

    def test_memory_low_threshold_gb(self):
        mgr = _make_manager(LOCALDATA_MEMORY_LOW_THRESHOLD_GB="2.5")
        assert mgr._config_data.get("memory", {}).get("low_memory_threshold_gb") == 2.5


class TestQueryEnvVars:
    """Test query section environment variable overrides."""

    def test_query_blob_handling(self):
        mgr = _make_manager(LOCALDATA_QUERY_BLOB_HANDLING="placeholder")
        assert mgr._config_data.get("query", {}).get("blob_handling") == "placeholder"

    def test_query_preflight_default_true(self):
        mgr = _make_manager(LOCALDATA_QUERY_PREFLIGHT_DEFAULT="true")
        assert mgr._config_data.get("query", {}).get("preflight_default") is True

    def test_query_preflight_default_false(self):
        mgr = _make_manager(LOCALDATA_QUERY_PREFLIGHT_DEFAULT="false")
        assert mgr._config_data.get("query", {}).get("preflight_default") is False

    def test_query_preflight_default_yes(self):
        mgr = _make_manager(LOCALDATA_QUERY_PREFLIGHT_DEFAULT="yes")
        assert mgr._config_data.get("query", {}).get("preflight_default") is True

    def test_query_chunk_size(self):
        mgr = _make_manager(LOCALDATA_QUERY_CHUNK_SIZE="500")
        assert mgr._config_data.get("query", {}).get("default_chunk_size") == 500


class TestConnectionsEnvVars:
    """Test connections section environment variable overrides."""

    def test_connections_timeout(self):
        mgr = _make_manager(LOCALDATA_CONNECTIONS_TIMEOUT="60")
        assert mgr._config_data.get("connections", {}).get("timeout_seconds") == 60

    def test_connections_max_concurrent(self):
        mgr = _make_manager(LOCALDATA_CONNECTIONS_MAX_CONCURRENT="25")
        assert mgr._config_data.get("connections", {}).get("max_concurrent") == 25


class TestSecurityEnvVars:
    """Test security section environment variable overrides."""

    def test_security_max_query_length(self):
        mgr = _make_manager(LOCALDATA_SECURITY_MAX_QUERY_LENGTH="50000")
        assert mgr._config_data.get("security", {}).get("max_query_length") == 50000


class TestInvalidEnvVars:
    """Test that invalid environment variable values produce warnings."""

    def test_invalid_int_env_var_warns(self, capsys):
        _make_manager(LOCALDATA_STAGING_MAX_CONCURRENT="notanumber")
        captured = capsys.readouterr()
        assert (
            "Warning: Invalid value for LOCALDATA_STAGING_MAX_CONCURRENT"
            in captured.out
        )

    def test_invalid_float_env_var_warns(self, capsys):
        _make_manager(LOCALDATA_MEMORY_LOW_THRESHOLD_GB="notafloat")
        captured = capsys.readouterr()
        assert (
            "Warning: Invalid value for LOCALDATA_MEMORY_LOW_THRESHOLD_GB"
            in captured.out
        )


class TestConfigValidation:
    """Test _validate_config catches invalid values in new sections."""

    def test_validation_catches_invalid_staging(self, capsys):
        mgr = _make_manager()
        mgr._config_data["staging"] = {"max_concurrent": 0}
        mgr._validate_config()
        captured = capsys.readouterr()
        assert "Configuration validation error in 'staging'" in captured.out

    def test_validation_catches_invalid_memory(self, capsys):
        mgr = _make_manager()
        mgr._config_data["memory"] = {"budget_percent": 200}
        mgr._validate_config()
        captured = capsys.readouterr()
        assert "Configuration validation error in 'memory'" in captured.out

    def test_validation_catches_invalid_query(self, capsys):
        mgr = _make_manager()
        mgr._config_data["query"] = {"default_chunk_size": -1}
        mgr._validate_config()
        captured = capsys.readouterr()
        assert "Configuration validation error in 'query'" in captured.out

    def test_validation_passes_valid_config(self, capsys):
        mgr = _make_manager()
        mgr._config_data["staging"] = {"max_concurrent": 5, "max_size_mb": 1024}
        mgr._config_data["memory"] = {"max_budget_mb": 256, "budget_percent": 15}
        mgr._validate_config()
        captured = capsys.readouterr()
        # No validation errors for new sections
        assert "Configuration validation error" not in captured.out
