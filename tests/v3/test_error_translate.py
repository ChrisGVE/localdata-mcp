"""tests/v3/test_error_translate.py — E4.2 translation + NFR-302 check.

Translation tests drive real exceptions through the kept feeder into
the one wire taxonomy. The NFR-302 half is the always-on import check:
an AST scan over the whole v3 tree asserting translate.py is the ONLY
importer of the legacy error feeder — a second importer would be the
second translation path NX-3 forbids. The feeder module names and the
one sanctioned importer are declared in gated_tree.py (NFR-402), shared
with the import-graph reachability gate (test_nexus_import_graph.py).
"""

from __future__ import annotations

import ast
from pathlib import Path

from localdata_mcp.nexus.error.model import ErrorType
from localdata_mcp.nexus.error.translate import _LEGACY_TO_WIRE, translate
from localdata_mcp.nexus.gated_tree import (
    NX3_FEEDER_ENTRYPOINT,
    NX3_FEEDER_MODULES,
    SRC_ROOT,
    iter_v3_sources,
)

_THE_ONE_IMPORTER = SRC_ROOT / NX3_FEEDER_ENTRYPOINT


class TestTranslation:
    def test_timeout_maps_to_a_faulting_retryable_class(self) -> None:
        result = translate(TimeoutError("connection timed out"), "generic")
        assert result.error_type in (ErrorType.CONNECTION_ERROR, ErrorType.TIMEOUT)
        assert result.retryable
        assert result.error_type.signals_connection_fault

    def test_sqlite_readonly_maps_to_a_permissionish_class(self) -> None:
        result = translate(Exception("attempt to write a readonly database"), "sqlite")
        assert result.error_type is not ErrorType.QUERY_EXECUTION
        assert result.message

    def test_unknown_exception_falls_back_to_query_execution(self) -> None:
        result = translate(Exception("entirely novel condition"), "generic")
        assert result.error_type == ErrorType.QUERY_EXECUTION
        assert not result.retryable

    def test_every_legacy_category_value_is_mapped(self) -> None:
        from localdata_mcp.error_handler.exceptions import ErrorCategory

        unmapped = {c.value for c in ErrorCategory} - set(_LEGACY_TO_WIRE)
        assert unmapped == set(), unmapped


class TestNfr302ImportCheck:
    def test_translate_is_the_only_legacy_feeder_importer(self) -> None:
        offenders: list[str] = []
        for path in iter_v3_sources():
            if Path(path) == _THE_ONE_IMPORTER:
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.ImportFrom)
                    and node.module
                    and any(
                        node.module.startswith(feeder) for feeder in NX3_FEEDER_MODULES
                    )
                ):
                    offenders.append(f"{path}: from {node.module}")
                if isinstance(node, ast.Import) and any(
                    alias.name.startswith(feeder)
                    for alias in node.names
                    for feeder in NX3_FEEDER_MODULES
                ):
                    offenders.append(f"{path}: import")
        assert offenders == [], offenders
