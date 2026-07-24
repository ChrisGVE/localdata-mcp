"""tests/v3/test_config_default_site.py — NFR-403 one-default-site gate.

The importable static check asserting no S8 default value is restated
in the v3 tree outside the ConfigModel declarations. The gate test runs
the check over the real tree (that is its purpose); the unit tests
exercise the scanner on synthetic source text.
"""

from __future__ import annotations

from localdata_mcp.nexus.config.default_site_check import (
    check_one_default_site,
    s8_default_values,
    scan_source,
)


class TestGate:
    def test_the_v3_tree_has_one_default_site(self) -> None:
        violations = check_one_default_site()
        assert violations == [], "\n".join(str(v) for v in violations)


class TestDefaultCollection:
    def test_values_come_from_the_model_not_a_parallel_list(self) -> None:
        values = s8_default_values()
        assert values["query.stream_idle_ttl_seconds"] == 600
        assert values["query.max_analysis_rows"] == 524288  # derived
        assert values["testbench.png_ssim_threshold"] == 0.95


class TestScanner:
    def test_distinctive_int_is_caught_anywhere(self) -> None:
        violations = scan_source(
            "def admit(rows):\n    return rows[:524288]\n", "fake.py"
        )
        assert [v.value for v in violations] == [524288]

    def test_distinctive_float_is_caught_anywhere(self) -> None:
        violations = scan_source("threshold = 1 - 0.95\n", "fake.py")
        assert [v.value for v in violations] == [0.95]

    def test_common_int_is_caught_in_a_module_constant(self) -> None:
        violations = scan_source("CHUNK_SIZE = 100\n", "fake.py")
        assert [v.value for v in violations] == [100]

    def test_common_int_is_caught_in_a_parameter_default(self) -> None:
        violations = scan_source(
            "def fetch(chunk_size=100):\n    return chunk_size\n", "fake.py"
        )
        assert [v.value for v in violations] == [100]

    def test_common_int_inline_use_is_not_flagged(self) -> None:
        # 100 as an ordinary expression operand is unscannable noise,
        # not a restated default.
        assert scan_source("y = x / 100\n", "fake.py") == []

    def test_tiny_ints_are_exempt_everywhere(self) -> None:
        # 0-3 are structurally unscannable (schema versions, indices);
        # their one-home discipline rests on review, not this gate.
        assert scan_source("SCHEMA_VERSION = 1\n", "fake.py") == []

    def test_non_default_values_pass(self) -> None:
        assert scan_source("TIMEOUT = 42\nRATIO = 0.5\n", "fake.py") == []

    def test_string_default_is_caught_in_a_module_constant(self) -> None:
        # "colorblind" is visualize.default_palette's one home; restating
        # it as a module constant is a second default site.
        violations = scan_source('DEFAULT_PALETTE = "colorblind"\n', "fake.py")
        assert [v.value for v in violations] == ["colorblind"]

    def test_string_default_is_caught_in_a_parameter_default(self) -> None:
        violations = scan_source(
            'def render(palette="colorblind"):\n    return palette\n', "fake.py"
        )
        assert [v.value for v in violations] == ["colorblind"]

    def test_string_default_in_a_collection_is_not_flagged(self) -> None:
        # An enumeration of valid presets is not a restated scalar default;
        # only a scalar constant/parameter default re-declares the value.
        assert scan_source('PRESETS = ("deep", "colorblind")\n', "fake.py") == []

    def test_string_default_inline_use_is_not_flagged(self) -> None:
        assert scan_source('use("colorblind")\n', "fake.py") == []

    def test_non_default_string_passes(self) -> None:
        assert scan_source('LABEL = "hello"\n', "fake.py") == []
