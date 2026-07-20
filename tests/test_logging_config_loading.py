"""Tests that every LoggingConfig field is actually loadable.

`ConfigManager.get_logging_config()` used to build `LoggingConfig` from a
hand-written list of six keys, so the other nineteen — `enable_metrics`
among them — were silently discarded and always resolved to their
dataclass defaults. A documented off-switch that cannot be switched off
is worse than an undocumented one, so these tests pin the whole surface:
every field of the dataclass must be settable, and a misspelled key must
be reported rather than swallowed.
"""

import dataclasses
import os
import subprocess
import sys

import pytest

from localdata_mcp.config_manager.manager import ConfigManager
from localdata_mcp.config_manager.models import LoggingConfig
from localdata_mcp.config_manager.types import (
    LogLevel,
    OutputDestination,
    OutputFormat,
)


def config_with_logging(tmp_path, logging_yaml: str) -> ConfigManager:
    """Build a ConfigManager from a YAML file holding a logging section."""
    config_file = tmp_path / "localdata.yaml"
    config_file.write_text(f"logging:\n{logging_yaml}")
    return ConfigManager(config_file=str(config_file))


class TestEveryFieldIsLoadable:
    def test_no_field_is_silently_dropped(self, tmp_path):
        """Guards against the loader drifting behind the model again."""
        # Values chosen to differ from every dataclass default.
        overrides = {
            "level": "error",
            "format": "%(message)s",
            "output_format": "text",
            "destinations": ["file"],
            "file_path": "/tmp/localdata-test.log",
            "json_file_path": "/tmp/localdata-test.json",
            "max_file_size": 123456,
            "backup_count": 9,
            "console_output": False,
            "enable_correlation_ids": False,
            "enable_context_propagation": False,
            "enable_query_audit": False,
            "enable_performance_logging": False,
            "enable_security_logging": False,
            "slow_query_threshold": 2.5,
            "very_slow_query_threshold": 9.5,
            "enable_metrics": False,
            "metrics_port": 9123,
            "metrics_endpoint": "/custom-metrics",
            "log_blocked_queries": False,
            "log_timeout_events": False,
            "log_resource_limits": False,
            "log_failed_connections": False,
            "enable_debug_traces": True,
            "debug_sql_queries": True,
            "debug_connection_pool": True,
        }

        model_fields = {f.name for f in dataclasses.fields(LoggingConfig)}
        assert set(overrides) == model_fields, (
            "this test must cover every LoggingConfig field; update it "
            "alongside the model"
        )

        def as_yaml(value):
            if isinstance(value, list):
                return "[" + ", ".join(str(item) for item in value) + "]"
            if isinstance(value, bool):
                return "true" if value else "false"
            if isinstance(value, str):
                # Quoted, so values like "%(message)s" stay scalars rather
                # than tripping YAML's directive indicator.
                return '"' + value + '"'
            return str(value)

        yaml_body = "".join(
            f"  {key}: {as_yaml(value)}\n" for key, value in overrides.items()
        )
        config = config_with_logging(tmp_path, yaml_body).get_logging_config()

        assert config.level == LogLevel.ERROR
        assert config.format == "%(message)s"
        assert config.output_format == OutputFormat.TEXT
        assert config.destinations == [OutputDestination.FILE]
        assert config.file_path == "/tmp/localdata-test.log"
        assert config.json_file_path == "/tmp/localdata-test.json"
        assert config.max_file_size == 123456
        assert config.backup_count == 9
        assert config.console_output is False
        assert config.enable_correlation_ids is False
        assert config.enable_context_propagation is False
        assert config.enable_query_audit is False
        assert config.enable_performance_logging is False
        assert config.enable_security_logging is False
        assert config.slow_query_threshold == 2.5
        assert config.very_slow_query_threshold == 9.5
        assert config.enable_metrics is False
        assert config.metrics_port == 9123
        assert config.metrics_endpoint == "/custom-metrics"
        assert config.log_blocked_queries is False
        assert config.log_timeout_events is False
        assert config.log_resource_limits is False
        assert config.log_failed_connections is False
        assert config.enable_debug_traces is True
        assert config.debug_sql_queries is True
        assert config.debug_connection_pool is True

    def test_defaults_survive_an_empty_logging_section(self, tmp_path):
        config = config_with_logging(tmp_path, "  {}\n").get_logging_config()
        assert config == LoggingConfig()

    def test_partial_section_leaves_other_fields_at_defaults(self, tmp_path):
        config = config_with_logging(
            tmp_path, "  enable_metrics: false\n"
        ).get_logging_config()

        assert config.enable_metrics is False
        assert config.metrics_port == LoggingConfig().metrics_port
        assert config.level == LoggingConfig().level


class TestEnableMetricsOffSwitch:
    """The switch documented in docs/tools-reference.md must work."""

    def test_metrics_disabled_from_yaml(self, tmp_path):
        config = config_with_logging(
            tmp_path, "  enable_metrics: false\n"
        ).get_logging_config()
        assert config.enable_metrics is False

    def test_metrics_disabled_from_environment(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LOCALDATA_LOGGING_ENABLE_METRICS", "false")
        config = ConfigManager().get_logging_config()
        assert config.enable_metrics is False

    def test_metrics_enabled_by_default(self, tmp_path):
        config = config_with_logging(tmp_path, "  {}\n").get_logging_config()
        assert config.enable_metrics is True


class TestMetricsToolRegistration:
    """The switch must change the registered tool surface, not just a flag.

    `get_metrics` is gated at import time in localdata_mcp.py, so each
    case runs in its own interpreter — reloading the module in-process
    would leave the already-registered tool behind.
    """

    # DatabaseManager registers the other tools from its constructor, so
    # instantiate it to count the surface a running server exposes.
    COUNT_TOOLS = (
        "import asyncio, localdata_mcp.localdata_mcp as m; "
        "m.DatabaseManager(); "
        "print(len(asyncio.run(m.mcp.list_tools())))"
    )

    def run_count(self, env_overrides):
        env = dict(os.environ, **env_overrides)
        completed = subprocess.run(
            [sys.executable, "-c", self.COUNT_TOOLS],
            capture_output=True,
            text=True,
            env=env,
        )
        assert completed.returncode == 0, completed.stderr
        return int(completed.stdout.strip().splitlines()[-1])

    def test_disabling_metrics_removes_the_metrics_tool(self):
        enabled = self.run_count({"LOCALDATA_LOGGING_ENABLE_METRICS": "true"})
        disabled = self.run_count({"LOCALDATA_LOGGING_ENABLE_METRICS": "false"})

        assert (
            disabled == enabled - 1
        ), "disabling metrics must remove exactly one tool (get_metrics)"

    def test_metrics_tool_is_present_by_default(self):
        listing = subprocess.run(
            [
                sys.executable,
                "-c",
                "import asyncio, localdata_mcp.localdata_mcp as m; "
                "print([t.name for t in asyncio.run(m.mcp.list_tools())])",
            ],
            capture_output=True,
            text=True,
            env={
                k: v
                for k, v in os.environ.items()
                if k != "LOCALDATA_LOGGING_ENABLE_METRICS"
            },
        )
        assert listing.returncode == 0, listing.stderr
        assert "get_metrics" in listing.stdout


class TestUnknownKeysAreReported:
    def test_a_misspelled_key_warns_rather_than_being_swallowed(self, tmp_path, capsys):
        config = config_with_logging(
            tmp_path, "  enable_metric: false\n"
        ).get_logging_config()

        assert config.enable_metrics is True, "the typo must not take effect"
        assert "enable_metric" in capsys.readouterr().out

    def test_an_invalid_enum_value_is_reported_with_the_field_name(self, tmp_path):
        manager = config_with_logging(tmp_path, "  level: verbose\n")
        with pytest.raises(ValueError, match="level"):
            manager.get_logging_config()
