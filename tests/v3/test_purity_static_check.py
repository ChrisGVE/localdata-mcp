"""tests/v3/test_purity_static_check.py — E2.2 no-print/no-second-handler.

The T2 fix made structural: NX-4's one stderr-bound configuration is
the only logging path, so the v3 tree may contain no print() call and
no StreamHandler construction outside the one home
(nexus/observability/config.py) — and a stdout-shaped construction is
banned even there. Synthetic-source cases prove the scanner detects
each violation shape; the tree-wide sweep is the CI gate itself.
Neighbors: test_config_default_site.py runs the sibling NFR-403 check;
test_observability_bootstrap.py covers the runtime half of the T2 fix.
"""

from __future__ import annotations

from localdata_mcp.nexus.observability.purity_check import (
    HANDLER_HOME,
    check_purity,
    scan_source,
)

HOME_FILENAME = "src/localdata_mcp/" + HANDLER_HOME


class TestPrintDetection:
    def test_flags_a_print_call(self) -> None:
        violations = scan_source("print('hello')\n", "mod.py")
        assert len(violations) == 1
        assert "print" in str(violations[0])

    def test_flags_print_inside_a_function(self) -> None:
        source = "def f() -> None:\n    print(1)\n"
        assert scan_source(source, "mod.py")

    def test_ignores_the_word_print_in_strings_and_names(self) -> None:
        source = "note = 'print is banned'\nprints = [note]\n"
        assert scan_source(source, "mod.py") == []


class TestHandlerDetection:
    def test_flags_stdout_stream_handler_anywhere(self) -> None:
        source = "import logging, sys\nh = logging.StreamHandler(sys.stdout)\n"
        violations = scan_source(source, HOME_FILENAME)
        assert len(violations) == 1
        assert "stdout" in str(violations[0])

    def test_flags_bare_name_stream_handler_on_stdout(self) -> None:
        source = (
            "import sys\nfrom logging import StreamHandler\n"
            "h = StreamHandler(sys.stdout)\n"
        )
        assert scan_source(source, HOME_FILENAME)

    def test_flags_any_handler_construction_outside_home(self) -> None:
        source = "import logging, sys\nh = logging.StreamHandler(sys.stderr)\n"
        violations = scan_source(source, "nexus/error/model.py")
        assert len(violations) == 1
        assert "second handler path" in str(violations[0])

    def test_allows_stderr_handler_in_the_home(self) -> None:
        source = "import logging, sys\nh = logging.StreamHandler(sys.stderr)\n"
        assert scan_source(source, HOME_FILENAME) == []

    def test_flags_subclass_construction_outside_home(self) -> None:
        source = (
            "from localdata_mcp.nexus.observability.config import StderrHandler\n"
            "h = StderrHandler()\n"
        )
        assert scan_source(source, "process/domains/x.py")


class TestTreeSweep:
    def test_the_v3_tree_is_clean(self) -> None:
        violations = check_purity()
        assert violations == [], "\n".join(str(v) for v in violations)
