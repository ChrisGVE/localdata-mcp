"""Tests for the settings that have two configuration sources.

Three values can be written in two places. `query.default_chunk_size` and
`performance.chunk_size` both describe rows per chunk;
`query.buffer_timeout_seconds` and `performance.query_buffer_timeout`
both describe how long a result buffer lives; `connections.max_concurrent`
and `performance.max_concurrent_connections` both cap simultaneous
connections. The `query`, `connections` and `staging` sections are the
newer, documented surface — they own the environment variables — so they
win, and the older `performance` keys stay honoured as a fallback rather
than being silently ignored.

`ConfigManager` resolves that precedence in one place so no consumer has
to know about it. It returns None when neither section sets the value, so
each consumer keeps its own long-standing default — the two homes carry
different defaults for the same idea, and imposing one here would change
behaviour for every installation that configured nothing.
"""

import pytest

from localdata_mcp.config_manager.manager import ConfigManager


def manager_for(tmp_path, yaml_text: str) -> ConfigManager:
    config_file = tmp_path / "localdata.yaml"
    config_file.write_text(yaml_text)
    return ConfigManager(config_file=str(config_file))


class TestEffectiveChunkSize:
    def test_query_section_wins(self, tmp_path):
        config = manager_for(
            tmp_path,
            "query:\n  default_chunk_size: 250\nperformance:\n  chunk_size: 700\n",
        )
        assert config.get_configured_chunk_size() == 250

    def test_performance_section_is_still_honoured(self, tmp_path):
        config = manager_for(tmp_path, "performance:\n  chunk_size: 700\n")
        assert config.get_configured_chunk_size() == 700

    def test_returns_none_when_neither_is_set(self, tmp_path):
        assert manager_for(tmp_path, "{}\n").get_configured_chunk_size() is None

    def test_environment_variable_takes_effect(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LOCALDATA_QUERY_CHUNK_SIZE", "321")
        assert ConfigManager().get_configured_chunk_size() == 321


class TestEffectiveBufferTimeout:
    def test_query_section_wins(self, tmp_path):
        config = manager_for(
            tmp_path,
            "query:\n  buffer_timeout_seconds: 60\n"
            "performance:\n  query_buffer_timeout: 900\n",
        )
        assert config.get_configured_buffer_timeout() == 60

    def test_performance_section_is_still_honoured(self, tmp_path):
        config = manager_for(tmp_path, "performance:\n  query_buffer_timeout: 900\n")
        assert config.get_configured_buffer_timeout() == 900

    def test_returns_none_when_neither_is_set(self, tmp_path):
        assert manager_for(tmp_path, "{}\n").get_configured_buffer_timeout() is None

    def test_environment_variable_takes_effect(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LOCALDATA_QUERY_BUFFER_TIMEOUT", "45")
        assert ConfigManager().get_configured_buffer_timeout() == 45


class TestEffectiveMaxConcurrentConnections:
    def test_connections_section_wins(self, tmp_path):
        config = manager_for(
            tmp_path,
            "connections:\n  max_concurrent: 3\n"
            "performance:\n  max_concurrent_connections: 25\n",
        )
        assert config.get_configured_max_concurrent_connections() == 3

    def test_performance_section_is_still_honoured(self, tmp_path):
        config = manager_for(
            tmp_path, "performance:\n  max_concurrent_connections: 25\n"
        )
        assert config.get_configured_max_concurrent_connections() == 25

    def test_returns_none_when_neither_is_set(self, tmp_path):
        assert (
            manager_for(tmp_path, "{}\n").get_configured_max_concurrent_connections()
            is None
        )

    def test_environment_variable_takes_effect(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LOCALDATA_CONNECTIONS_MAX_CONCURRENT", "4")
        assert ConfigManager().get_configured_max_concurrent_connections() == 4


class TestSettingsReachTheirConsumers:
    """A resolved value is worth nothing until something reads it."""

    def test_connection_limit_bounds_the_semaphore(self, monkeypatch):
        from localdata_mcp.config_manager import manager as manager_module

        monkeypatch.setenv("LOCALDATA_CONNECTIONS_MAX_CONCURRENT", "2")
        monkeypatch.setattr(manager_module, "_config_manager", None)

        from localdata_mcp import DatabaseManager

        db = DatabaseManager()
        assert db.connection_semaphore._value == 2

    def test_buffer_expiry_uses_the_configured_timeout(self, monkeypatch):
        from localdata_mcp.config_manager import manager as manager_module

        monkeypatch.setenv("LOCALDATA_QUERY_BUFFER_TIMEOUT", "42")
        monkeypatch.setattr(manager_module, "_config_manager", None)

        from localdata_mcp import DatabaseManager

        db = DatabaseManager()
        assert db.buffer_cleanup_interval == 42
        assert db.streaming_executor.buffer_timeout_seconds == 42

    def test_chunk_size_fallback_uses_the_configured_value(self, monkeypatch):
        from localdata_mcp.config_manager import manager as manager_module
        from localdata_mcp.server.query_execution import calculate_dynamic_chunk_size

        monkeypatch.setenv("LOCALDATA_QUERY_CHUNK_SIZE", "77")
        monkeypatch.setattr(manager_module, "_config_manager", None)

        class _Budget:
            budget_bytes = 0

        # No query analysis available, so the configured default applies.
        assert calculate_dynamic_chunk_size(_Budget(), None) == 77

    def test_explicit_fallback_argument_still_wins(self, monkeypatch):
        from localdata_mcp.config_manager import manager as manager_module
        from localdata_mcp.server.query_execution import calculate_dynamic_chunk_size

        monkeypatch.setenv("LOCALDATA_QUERY_CHUNK_SIZE", "77")
        monkeypatch.setattr(manager_module, "_config_manager", None)

        class _Budget:
            budget_bytes = 0

        assert calculate_dynamic_chunk_size(_Budget(), None, fallback=5) == 5


class TestDefaultsAreUnchangedWhenNothingIsConfigured:
    """Wiring the config up must not move any existing default."""

    def unconfigured(self, monkeypatch):
        from localdata_mcp.config_manager import manager as manager_module

        for name in (
            "LOCALDATA_QUERY_CHUNK_SIZE",
            "LOCALDATA_QUERY_BUFFER_TIMEOUT",
            "LOCALDATA_CONNECTIONS_MAX_CONCURRENT",
        ):
            monkeypatch.delenv(name, raising=False)
        monkeypatch.setattr(manager_module, "_config_manager", None)

    def test_connection_cap_stays_at_ten(self, monkeypatch):
        self.unconfigured(monkeypatch)
        from localdata_mcp import DatabaseManager

        assert DatabaseManager().connection_semaphore._value == 10

    def test_buffer_expiries_keep_their_own_defaults(self, monkeypatch):
        self.unconfigured(monkeypatch)
        from localdata_mcp import DatabaseManager

        db = DatabaseManager()
        assert db.buffer_cleanup_interval == 600
        assert db.streaming_executor.buffer_timeout_seconds == 3600

    def test_unknown_row_size_still_falls_back_to_a_generous_chunk(self, monkeypatch):
        self.unconfigured(monkeypatch)
        from localdata_mcp.server.query_execution import calculate_dynamic_chunk_size

        class _Budget:
            budget_bytes = 0

        assert calculate_dynamic_chunk_size(_Budget(), None) == 1000


class TestInvalidValuesAreRejected:
    def test_a_non_positive_chunk_size_is_rejected(self, tmp_path):
        config = manager_for(tmp_path, "query:\n  default_chunk_size: 0\n")
        with pytest.raises(ValueError, match="default_chunk_size"):
            config.get_configured_chunk_size()

    def test_a_non_positive_connection_cap_is_rejected(self, tmp_path):
        config = manager_for(tmp_path, "connections:\n  max_concurrent: 0\n")
        with pytest.raises(ValueError, match="max_concurrent"):
            config.get_configured_max_concurrent_connections()
