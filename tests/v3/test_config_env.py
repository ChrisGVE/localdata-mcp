"""tests/v3/test_config_env.py — E1.1/E1.2 env-name derivation + parsing.

Every ConfigModel field derives its override name mechanically as
`LOCALDATA_<SECTION>_<FIELD>` (never independently declared), and env
values parse back to the field's declared type. Unknown LOCALDATA_*
variables are refused fail-closed: a typo'd bound silently ignored would
leave the operator believing a limit is set when it is not.
"""

from __future__ import annotations

import pytest

from localdata_mcp.nexus.config.env_derive import (
    env_field_map,
    env_overrides,
    env_var_name,
)
from localdata_mcp.nexus.config.errors import (
    TypeMismatchError,
    UnknownFieldError,
)
from localdata_mcp.nexus.config.models import iter_config_fields


class TestNameDerivation:
    def test_pattern_is_localdata_section_field(self) -> None:
        assert (
            env_var_name("query", "default_chunk_size")
            == "LOCALDATA_QUERY_DEFAULT_CHUNK_SIZE"
        )

    def test_architecture_cited_allowed_paths_name(self) -> None:
        # ARCHITECTURE.md section 7.3 cites this exact derived name.
        assert (
            env_var_name("security", "allowed_paths")
            == "LOCALDATA_SECURITY_ALLOWED_PATHS"
        )

    def test_every_field_has_exactly_one_env_name(self) -> None:
        mapping = env_field_map()
        assert len(mapping) == sum(1 for _ in iter_config_fields())
        assert set(mapping.values()) == {
            (section, fld.name) for section, fld in iter_config_fields()
        }


class TestEnvOverrides:
    def test_int_field_parses(self) -> None:
        overrides = env_overrides({"LOCALDATA_QUERY_DEFAULT_CHUNK_SIZE": "250"})
        assert overrides == {"query": {"default_chunk_size": 250}}

    def test_float_field_parses(self) -> None:
        overrides = env_overrides({"LOCALDATA_TESTBENCH_TOL_ITERATIVE_RTOL": "1e-3"})
        assert overrides == {"testbench": {"tol_iterative_rtol": 1e-3}}

    def test_allowed_paths_split_on_pathsep(self) -> None:
        overrides = env_overrides(
            {"LOCALDATA_SECURITY_ALLOWED_PATHS": "/data/a:/data/b"}
        )
        assert overrides == {"security": {"allowed_paths": ("/data/a", "/data/b")}}

    def test_non_localdata_variables_are_ignored(self) -> None:
        assert env_overrides({"PATH": "/usr/bin", "HOME": "/root"}) == {}

    def test_unknown_localdata_variable_is_refused(self) -> None:
        with pytest.raises(UnknownFieldError) as excinfo:
            env_overrides({"LOCALDATA_QUERY_CHUNKSIZE": "5"})
        assert excinfo.value.source == "env"

    def test_malformed_int_is_refused(self) -> None:
        with pytest.raises(TypeMismatchError) as excinfo:
            env_overrides({"LOCALDATA_QUERY_DEFAULT_CHUNK_SIZE": "many"})
        assert excinfo.value.field_path == "query.default_chunk_size"

    def test_round_trip_int_and_float(self) -> None:
        for name, value in [
            ("LOCALDATA_RESOURCES_MEMORY_CEILING_BYTES", 2**33),
            ("LOCALDATA_PROCESS_SENTINEL_MAX_CONDITION_NUMBER", 1e8),
        ]:
            (section_values,) = env_overrides({name: str(value)}).values()
            ((_, parsed),) = section_values.items()
            assert parsed == value
