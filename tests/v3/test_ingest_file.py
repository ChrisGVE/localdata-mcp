"""tests/v3/test_ingest_file.py — E8.2: the file family end to end.

I-2's exit-gate battery: one fixture per core format through
read_file, the NFR-107 hardening rows (malicious YAML, XXE
external-file-read and SSRF shapes, crafted-external-link HDF5),
NFR-108 traversal payloads, and query_file's NFR-114 (i-b) branches —
outside allowed_paths refused, inside succeeds read-only, mutation
refused absent the operator grant, the ephemeral_write_paths entry as
the positive control — plus the over-budget refusal's
suggestion-content assertion (NFR-202's load-then-serve cell).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Iterator

import pandas as pd
import pytest

import localdata_mcp.ingest.runtime as runtime
from localdata_mcp.ingest.connectors.file.readers import FileIngestError
from localdata_mcp.ingest.connectors.file.tools import query_file, read_file
from localdata_mcp.nexus.chokepoint.guard import (
    Chokepoint,
    GuardedExecutionError,
    GuardRefusedError,
    ResourceRefusedError,
)
from localdata_mcp.nexus.chokepoint.path_contain import PathRefusedError
from localdata_mcp.nexus.config.models import (
    ConfigModel,
    ResourcesConfig,
    SecurityConfig,
)

_FRAME = pd.DataFrame({"id": [1, 2], "label": ["a", "b"]})


def install(
    tmp_path: Path,
    *,
    write_grants: tuple[str, ...] = (),
    ceiling: int | None = None,
) -> Chokepoint:
    resources = (
        ResourcesConfig(memory_ceiling_bytes=ceiling) if ceiling else ResourcesConfig()
    )
    config = ConfigModel(
        resources=resources,
        security=SecurityConfig(
            allowed_paths=(str(tmp_path),),
            ephemeral_write_paths=write_grants,
        ),
    )
    guard = Chokepoint.boot(config, environ={})
    runtime.configure_ingest(guard)
    return guard


@pytest.fixture(autouse=True)
def reset_runtime() -> Iterator[None]:
    yield
    runtime._CHOKEPOINT = None


def _sqlite_fixture(path: Path) -> None:
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE t (id INTEGER, label TEXT)")
    connection.executemany(
        "INSERT INTO t VALUES (?, ?)", [(1, "a"), (2, "b"), (3, "c")]
    )
    connection.commit()
    connection.close()


class TestReadFilePerFormat:
    def _assert_tabular(self, result: Any) -> None:
        assert result.columns == ("id", "label")
        assert result.rows[0] == (1, "a")

    def test_csv_tsv(self, tmp_path: Path) -> None:
        install(tmp_path)
        _FRAME.to_csv(tmp_path / "d.csv", index=False)
        _FRAME.to_csv(tmp_path / "d.tsv", sep="\t", index=False)
        self._assert_tabular(read_file(str(tmp_path / "d.csv")))
        self._assert_tabular(read_file(str(tmp_path / "d.tsv")))

    def test_json_records_and_document(self, tmp_path: Path) -> None:
        install(tmp_path)
        (tmp_path / "r.json").write_text('[{"id": 1, "label": "a"}]')
        result = read_file(str(tmp_path / "r.json"))
        assert result.rows == ((1, "a"),)
        (tmp_path / "doc.json").write_text('{"meta": {"k": 1}}')
        assert read_file(str(tmp_path / "doc.json")) == {"meta": {"k": 1}}

    def test_yaml_toml_ini(self, tmp_path: Path) -> None:
        install(tmp_path)
        (tmp_path / "d.yaml").write_text("- id: 1\n  label: a\n")
        assert read_file(str(tmp_path / "d.yaml")).rows == ((1, "a"),)
        (tmp_path / "d.toml").write_text('[section]\nkey = "v"\n')
        assert read_file(str(tmp_path / "d.toml")) == {"section": {"key": "v"}}
        (tmp_path / "d.ini").write_text("[s]\nk = v\n")
        assert read_file(str(tmp_path / "d.ini")) == {"s": {"k": "v"}}

    def test_xml_tree_shape(self, tmp_path: Path) -> None:
        install(tmp_path)
        (tmp_path / "d.xml").write_text(
            '<root><item id="1">a</item><item id="2">b</item></root>'
        )
        tree = read_file(str(tmp_path / "d.xml"))
        assert tree["root"]["item"][0] == {"@id": "1", "#text": "a"}

    def test_excel_family(self, tmp_path: Path) -> None:
        install(tmp_path)
        _FRAME.to_excel(tmp_path / "d.xlsx", index=False, engine="openpyxl")
        self._assert_tabular(read_file(str(tmp_path / "d.xlsx")))
        with pd.ExcelWriter(tmp_path / "d.ods", engine="odf") as writer:
            _FRAME.to_excel(writer, index=False)
        self._assert_tabular(read_file(str(tmp_path / "d.ods")))

    def test_numbers(self, tmp_path: Path) -> None:
        from numbers_parser import Document

        install(tmp_path)
        document = Document()
        table = document.sheets[0].tables[0]
        table.write(0, 0, "id")
        table.write(0, 1, "label")
        table.write(1, 0, 1)
        table.write(1, 1, "a")
        document.save(str(tmp_path / "d.numbers"))
        result = read_file(str(tmp_path / "d.numbers"))
        assert result.columns == ("id", "label")
        assert result.rows[0][1] == "a"

    def test_arrow_family(self, tmp_path: Path) -> None:
        import pyarrow
        import pyarrow.feather
        import pyarrow.ipc

        install(tmp_path)
        _FRAME.to_parquet(tmp_path / "d.parquet", index=False)
        self._assert_tabular(read_file(str(tmp_path / "d.parquet")))
        pyarrow.feather.write_feather(
            pyarrow.Table.from_pandas(_FRAME), tmp_path / "d.feather"
        )
        self._assert_tabular(read_file(str(tmp_path / "d.feather")))
        table = pyarrow.Table.from_pandas(_FRAME)
        with pyarrow.ipc.new_file(str(tmp_path / "d.arrow"), table.schema) as w:
            w.write_table(table)
        self._assert_tabular(read_file(str(tmp_path / "d.arrow")))

    def test_hdf5(self, tmp_path: Path) -> None:
        import h5py

        install(tmp_path)
        with h5py.File(tmp_path / "d.h5", "w") as handle:
            handle.create_dataset("values", data=[[1, 2], [3, 4]])
        frame = read_file(str(tmp_path / "d.h5"))
        assert list(frame.rows[0]) == [1, 2]

    def test_unknown_suffix_refused_not_guessed(self, tmp_path: Path) -> None:
        install(tmp_path)
        (tmp_path / "d.mystery").write_text("data")
        with pytest.raises(FileIngestError):
            read_file(str(tmp_path / "d.mystery"))

    def test_explicit_format_overrides_suffix(self, tmp_path: Path) -> None:
        install(tmp_path)
        (tmp_path / "d.dat").write_text("id,label\n1,a\n")
        result = read_file(str(tmp_path / "d.dat"), format="csv")
        assert result.rows == ((1, "a"),)


class TestNfr107Hardening:
    def test_malicious_yaml_python_object_refused(self, tmp_path: Path) -> None:
        install(tmp_path)
        (tmp_path / "evil.yaml").write_text(
            '!!python/object/apply:os.system ["echo owned"]'
        )
        with pytest.raises(Exception) as failure:
            read_file(str(tmp_path / "evil.yaml"))
        assert "python" in str(failure.value).lower() or "refused" in str(failure.value)

    def test_xxe_external_file_read_refused(self, tmp_path: Path) -> None:
        """The classic external-entity file-read shape parses nowhere:
        defusedxml refuses DOCTYPE/entities structurally."""
        install(tmp_path)
        (tmp_path / "xxe.xml").write_text(
            '<?xml version="1.0"?>'
            '<!DOCTYPE r [<!ENTITY x SYSTEM "file:///etc/passwd">]>'
            "<r>&x;</r>"
        )
        with pytest.raises(FileIngestError):
            read_file(str(tmp_path / "xxe.xml"))

    def test_xxe_ssrf_shape_refused(self, tmp_path: Path) -> None:
        install(tmp_path)
        (tmp_path / "ssrf.xml").write_text(
            '<?xml version="1.0"?>'
            '<!DOCTYPE r [<!ENTITY x SYSTEM "http://169.254.169.254/meta">]>'
            "<r>&x;</r>"
        )
        with pytest.raises(FileIngestError):
            read_file(str(tmp_path / "ssrf.xml"))

    def test_crafted_external_link_hdf5_refused(self, tmp_path: Path) -> None:
        """I-2's channel closure: the file's own content points h5py at
        a second path — refused before any second open."""
        import h5py

        install(tmp_path)
        outside = tmp_path / "target.h5"
        with h5py.File(outside, "w") as target:
            target.create_dataset("secret", data=[1])
        hostile = tmp_path / "hostile.h5"
        with h5py.File(hostile, "w") as handle:
            handle["link"] = h5py.ExternalLink(str(outside), "secret")
        with pytest.raises(FileIngestError) as refusal:
            read_file(str(hostile))
        assert "external link" in str(refusal.value)


class TestNfr108Traversal:
    @pytest.mark.parametrize(
        "payload",
        [
            "../../etc/passwd",
            "/etc/passwd",
            "inner/../../../etc/passwd",
        ],
    )
    def test_traversal_payloads_refused(self, tmp_path: Path, payload: str) -> None:
        install(tmp_path)
        with pytest.raises(PathRefusedError):
            read_file(
                str(tmp_path / payload) if not payload.startswith("/") else payload
            )

    def test_symlink_escape_refused(self, tmp_path: Path) -> None:
        outside_dir = tmp_path.parent / f"{tmp_path.name}-outside"
        outside_dir.mkdir(exist_ok=True)
        (outside_dir / "secret.csv").write_text("id\n1\n")
        allowed = tmp_path / "allowed"
        allowed.mkdir()
        install(allowed)
        (allowed / "link.csv").symlink_to(outside_dir / "secret.csv")
        with pytest.raises(PathRefusedError):
            read_file(str(allowed / "link.csv"))


class TestQueryFileNfr114BranchIB:
    def test_outside_allowed_paths_refused(self, tmp_path: Path) -> None:
        allowed = tmp_path / "allowed"
        allowed.mkdir()
        install(allowed)
        outside = tmp_path / "data.db"
        _sqlite_fixture(outside)
        with pytest.raises(PathRefusedError):
            query_file(str(outside), "SELECT * FROM t")

    def test_inside_succeeds_read_only(self, tmp_path: Path) -> None:
        install(tmp_path)
        _sqlite_fixture(tmp_path / "data.db")
        result = query_file(str(tmp_path / "data.db"), "SELECT id FROM t ORDER BY id")
        assert result.rows == ((1,), (2,), (3,))
        assert result.category == "query"

    def test_mutation_refused_on_default_read_only(self, tmp_path: Path) -> None:
        install(tmp_path)
        _sqlite_fixture(tmp_path / "data.db")
        with pytest.raises(GuardRefusedError) as refusal:
            query_file(str(tmp_path / "data.db"), "DELETE FROM t")
        assert "ephemeral_write_paths" in str(refusal.value)

    def test_operator_grant_is_the_positive_control(self, tmp_path: Path) -> None:
        install(tmp_path, write_grants=(str(tmp_path),))
        _sqlite_fixture(tmp_path / "data.db")
        outcome = query_file(str(tmp_path / "data.db"), "DELETE FROM t")
        assert outcome.category == "mutation"
        after = query_file(str(tmp_path / "data.db"), "SELECT count(*) FROM t")
        assert after.rows[0][0] == 0

    def test_non_engine_suffix_routed_to_read_file(self, tmp_path: Path) -> None:
        install(tmp_path)
        with pytest.raises(FileIngestError) as refusal:
            query_file(str(tmp_path / "d.csv"), "SELECT 1")
        assert "read_file" in str(refusal.value)


class TestOverBudgetRefusal:
    def test_suggestion_names_the_sql_narrowing_recovery(self, tmp_path: Path) -> None:
        """NFR-202's load-then-serve cell: over-budget admission is a
        structured refusal naming WHERE/LIMIT and the operator-owned
        ceiling — never a silently broken stream."""
        install(tmp_path, ceiling=16_384)  # derived row cap == 2
        _sqlite_fixture(tmp_path / "data.db")
        with pytest.raises(GuardedExecutionError) as refusal:
            query_file(str(tmp_path / "data.db"), "SELECT * FROM t")
        suggestion = refusal.value.structured.suggestion
        assert "WHERE" in suggestion and "LIMIT" in suggestion
        assert "operator configuration" in suggestion
        assert refusal.value.structured.error_type.value == "resource_error"


class TestDecompressionGateCr005:
    """NFR-105/GP3 fail-safe on the file read path: a high-ratio file
    inside allowed_paths is refused through the memory-admission gate
    BEFORE it materializes — never an OOM (CR-005 wired admit_load)."""

    def test_running_charge_stops_before_reading_the_whole_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The streaming path charges residency per chunk and refuses
        mid-read: with the chunk size shrunk to 100 rows and a fake gate
        that refuses on the 3rd charge, only the first 3 of 10 chunks are
        read — the remaining ~700 rows are never materialized."""
        from localdata_mcp.ingest.connectors.file import readers

        monkeypatch.setattr(readers, "_READ_CHUNK_ROWS", 100)
        csv = tmp_path / "d.csv"
        pd.DataFrame({"a": range(1000)}).to_csv(csv, index=False)
        calls = {"n": 0}

        def admit(estimated: int) -> None:
            calls["n"] += 1
            if calls["n"] >= 3:
                raise ResourceRefusedError("over ceiling", resource_class="memory")

        with pytest.raises(ResourceRefusedError):
            readers.read_path(csv, "csv", admit)
        assert calls["n"] == 3  # refused on chunk 3 of 10, rest unread

    def test_streaming_csv_over_ceiling_is_refused_fail_safe(
        self, tmp_path: Path
    ) -> None:
        install(tmp_path, ceiling=2048)
        bomb = tmp_path / "bomb.csv"
        pd.DataFrame({"a": range(60000), "b": range(60000)}).to_csv(bomb, index=False)
        with pytest.raises(GuardedExecutionError) as refusal:
            read_file(str(bomb))
        assert refusal.value.structured.error_type.value == "resource_error"

    def test_upfront_estimate_refuses_a_non_streaming_format(
        self, tmp_path: Path
    ) -> None:
        """A whole-file format (parquet) is refused by the upfront
        st_size*factor estimate before pandas ever opens it."""
        install(tmp_path, ceiling=1024)
        target = tmp_path / "big.parquet"
        pd.DataFrame({"a": range(5000), "b": range(5000)}).to_parquet(
            target, index=False
        )
        with pytest.raises(GuardedExecutionError) as refusal:
            read_file(str(target))
        assert refusal.value.structured.error_type.value == "resource_error"

    def test_legitimate_large_file_reads_under_a_generous_ceiling(
        self, tmp_path: Path
    ) -> None:
        """Positive control: the same shape reads fine under the default
        ceiling — the gate refuses bombs, not legitimate data."""
        install(tmp_path)
        target = tmp_path / "ok.csv"
        pd.DataFrame({"a": range(60000), "b": range(60000)}).to_csv(target, index=False)
        result = read_file(str(target))
        # 60000 rows > the inline budget → served as a stream
        assert getattr(result, "stream_id", None)
        assert result.columns == ("a", "b")
