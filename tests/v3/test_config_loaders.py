"""tests/v3/test_config_loaders.py — E1.2 layered TOML loading.

One config-file search-path list (the single home of config locations),
TOML per layer, env overrides entering at user rank. File access is
injected so these tests never touch the real filesystem.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import pytest

from localdata_mcp.nexus.config.errors import ConfigurationError
from localdata_mcp.nexus.config.loaders import (
    CONFIG_SEARCH_PATHS,
    load_config,
)
from localdata_mcp.nexus.config.provenance import Layer


def reader(texts: Mapping[str, str]):
    """A read_text stand-in serving from a {path: toml_text} mapping."""

    def read_text(path: Path) -> str | None:
        return texts.get(str(path))

    return read_text


class TestSearchPathList:
    def test_the_one_list_covers_the_three_trust_layers(self) -> None:
        assert [(p.name, p.layer) for p in CONFIG_SEARCH_PATHS] == [
            ("system-file", Layer.SYSTEM),
            ("user-file", Layer.USER),
            ("project-file", Layer.PROJECT),
        ]

    def test_paths_are_the_localdata_toml_conventions(self) -> None:
        assert [p.path for p in CONFIG_SEARCH_PATHS] == [
            "/etc/localdata.toml",
            "~/.localdata.toml",
            "./.localdata.toml",
        ]


class TestLoadConfig:
    def test_no_files_no_env_yields_defaults(self) -> None:
        result = load_config(environ={}, read_text=reader({}))
        assert result.model.query.default_chunk_size == 100
        assert result.refusals == ()

    def test_layers_merge_across_files(self) -> None:
        result = load_config(
            environ={},
            read_text=reader(
                {
                    "/etc/localdata.toml": (
                        "[resources]\nquery_timeout_seconds = 60\n"
                    ),
                    str(Path("~/.localdata.toml").expanduser()): (
                        "[query]\ndefault_chunk_size = 200\n"
                    ),
                    ".localdata.toml": ("[query]\ndefault_chunk_size = 250\n"),
                }
            ),
        )
        # Ordinary field: lowest layer wins; pinned field: system holds.
        assert result.model.query.default_chunk_size == 250
        assert result.model.resources.query_timeout_seconds == 60

    def test_env_enters_at_user_rank(self) -> None:
        result = load_config(
            environ={"LOCALDATA_QUERY_DEFAULT_CHUNK_SIZE": "300"},
            read_text=reader(
                {
                    str(Path("~/.localdata.toml").expanduser()): (
                        "[query]\ndefault_chunk_size = 200\n"
                    )
                }
            ),
        )
        assert result.model.query.default_chunk_size == 300
        assert result.provenance.winner("query.default_chunk_size") == (
            300,
            "env",
        )

    def test_endpoints_load_from_an_operator_file(self) -> None:
        toml = (
            "[endpoints.warehouse]\n"
            'dsn = "postgresql://db.internal/w"\n'
            'posture = "read_write"\n'
            'credentials_ref = "WAREHOUSE_SECRET"\n'
        )
        result = load_config(
            environ={},
            read_text=reader({str(Path("~/.localdata.toml").expanduser()): toml}),
        )
        endpoint = result.model.endpoints["warehouse"]
        assert endpoint.posture == "read_write"
        assert endpoint.credentials_ref == "WAREHOUSE_SECRET"

    def test_malformed_toml_is_a_typed_error(self) -> None:
        with pytest.raises(ConfigurationError) as excinfo:
            load_config(
                environ={},
                read_text=reader({"/etc/localdata.toml": "not = toml ="}),
            )
        assert excinfo.value.source == "system-file"

    def test_project_introduction_refused_end_to_end(self) -> None:
        result = load_config(
            environ={},
            read_text=reader(
                {".localdata.toml": '[security]\nallowed_paths = ["/x"]\n'}
            ),
        )
        assert result.model.security.allowed_paths == ()
        assert len(result.refusals) == 1
