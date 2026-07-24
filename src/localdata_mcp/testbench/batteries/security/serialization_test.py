"""testbench/batteries/security/serialization_test.py — NFR-107 at the L3 seam.

The deserialization hardening (E8.2, unit-covered at the reader seam in
tests/v3/test_file_readers.py) proven through the wire the agent uses. A
YAML document carrying a code-execution tag
(`!!python/object/apply:os.system …`) and one carrying a
SafeLoader-accepted-but-non-plain primitive (`!!set`) are each driven
through `read_file`, and each is refused as a structured FR-403 error:
the `SafeLoader` never constructs the object (so the code never runs — a
filesystem sentinel the payload tries to touch never appears) and the
whole-tree primitive scan rejects the non-plain type that a naive safe
loader would otherwise admit. A plain records document is the positive
control proving the reader still serves legitimate YAML.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from . import _seam


@pytest.fixture()
def yaml_bench(tmp_path: Path):
    """A contained root plus a sentinel path just outside every document:
    an out-of-tree file the code-execution payloads try, and fail, to
    create. Its absence after each refusal is the proof no object was
    constructed and no code ran."""
    root = tmp_path / "root"
    root.mkdir()
    sentinel = tmp_path / "pwned.marker"
    with _seam.booted(allowed_paths=(str(root),)):
        yield root, sentinel


# Each payload is a YAML document body; the code-execution variants embed
# the sentinel path so a successful construction would leave a trace on
# disk. `!!python/object/apply` is the classic PyYAML RCE gadget; `!!set`
# is the SafeLoader-legal-but-non-plain type the whole-tree scan closes.
def _payloads(sentinel: Path) -> tuple[tuple[str, str], ...]:
    quoted = str(sentinel)
    return (
        (
            "object_apply_os_system",
            f"!!python/object/apply:os.system ['touch {quoted}']",
        ),
        (
            "object_apply_subprocess",
            f"!!python/object/apply:subprocess.getoutput ['touch {quoted}']",
        ),
        ("object_new_os_system", "!!python/object/new:os.system []"),
        ("non_plain_set", "!!set\n? a\n? b\n"),
    )


@pytest.mark.parametrize(
    "kind",
    [
        "object_apply_os_system",
        "object_apply_subprocess",
        "object_new_os_system",
        "non_plain_set",
    ],
)
def test_hostile_yaml_is_refused(yaml_bench, kind: str) -> None:
    root, sentinel = yaml_bench
    body = dict(_payloads(sentinel))[kind]
    document = root / "payload.yaml"
    document.write_text(body, encoding="utf-8")

    envelope = _seam.call_envelope("read_file", {"path": str(document)})
    _seam.expect_refused(envelope)
    # No object was constructed: the code-execution gadget never ran, so
    # the out-of-tree sentinel it targeted does not exist.
    assert not sentinel.exists()


# -- positive control: the reader still serves legitimate YAML ----------


def test_plain_yaml_records_succeed(yaml_bench) -> None:
    root, _sentinel = yaml_bench
    document = root / "ok.yaml"
    document.write_text(
        "- {id: 1, label: alpha}\n- {id: 2, label: beta}\n", encoding="utf-8"
    )
    data = _seam.expect_ok(_seam.call_envelope("read_file", {"path": str(document)}))
    assert data["rows"][0][0] == 1
