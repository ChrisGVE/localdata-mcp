"""tests/v3/test_contract_generators.py — E3.4: the five artifact generators.

Pins nexus/contract/generators/* and generate.py: each renderer
produces deterministic text opening with a MACHINE-WRITTEN banner that
names its generator module, the Python artifacts compile, and the
orchestrator emits all five artifact paths (section 6.1). End-to-end
behavior of the COMMITTED artifacts is covered separately in
test_tools_generated.py; here the renderers are exercised against a
local registry.
"""

from __future__ import annotations

from localdata_mcp.nexus.contract.generate import generate_artifacts
from localdata_mcp.nexus.contract.generators.docs import render_docs
from localdata_mcp.nexus.contract.generators.docstring import render_docstring
from localdata_mcp.nexus.contract.generators.test_stub import (
    render_test_module,
)
from localdata_mcp.nexus.contract.generators.typeshape_registry import (
    render_shapes_module,
)
from localdata_mcp.nexus.contract.generators.wrapper import (
    render_wrapper_module,
)
from localdata_mcp.nexus.contract.registry import ToolRegistry
from localdata_mcp.nexus.contract.spec import Param, ToolSpec, TypeShape


def _registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="echo",
            summary="Echo the given text back unchanged.",
            params=(Param("text", str, "Text to echo."),),
            input_shape=TypeShape.NONE,
            output_shape=TypeShape.SCALAR,
            func=lambda text: text,
        )
    )
    registry.register(
        ToolSpec(
            name="profile_frame",
            summary="Profile a tabular frame.",
            params=(Param("rows", int, "Row count to profile."),),
            input_shape=TypeShape.TABULAR,
            output_shape=TypeShape.TABULAR,
            streaming_capable=True,
            domain="statistical_analysis",
            func=lambda rows: rows,
        )
    )
    return registry


class TestDocstring:
    def test_contains_summary_args_and_shapes(self) -> None:
        text = render_docstring(_registry().lookup("echo"))
        assert "Echo the given text back unchanged." in text
        assert "text: Text to echo." in text
        assert "NONE" in text and "SCALAR" in text

    def test_no_params_renders_without_args_section(self) -> None:
        spec = ToolSpec(
            name="noop",
            summary="Do nothing.",
            params=(),
            input_shape=TypeShape.NONE,
            output_shape=TypeShape.NONE,
            func=lambda: None,
        )
        assert "Args:" not in render_docstring(spec)


class TestWrapperModule:
    def test_banner_names_generator(self) -> None:
        text = render_wrapper_module(_registry())
        assert text.startswith(
            '"""MACHINE-WRITTEN by localdata_mcp.nexus.contract.generators.wrapper'
        )

    def test_compiles_and_defines_each_tool(self) -> None:
        text = render_wrapper_module(_registry())
        compile(text, "tools_generated.py", "exec")
        assert "def echo(text: str)" in text
        assert "def profile_frame(rows: int)" in text
        assert "def register_tools(" in text

    def test_generated_docstring_rides_the_wrapper(self) -> None:
        assert "Echo the given text back unchanged." in render_wrapper_module(
            _registry()
        )


class TestShapesModule:
    def test_banner_and_compile(self) -> None:
        text = render_shapes_module(_registry())
        assert text.startswith(
            '"""MACHINE-WRITTEN by '
            "localdata_mcp.nexus.contract.generators.typeshape_registry"
        )
        compile(text, "generated_shapes.py", "exec")

    def test_entries_present(self) -> None:
        text = render_shapes_module(_registry())
        assert '"echo"' in text
        assert "TypeShape.TABULAR" in text
        assert "'statistical_analysis'" in text


class TestDocs:
    def test_grouped_by_domain_with_banner(self) -> None:
        files = render_docs(_registry())
        assert set(files) == {"core.md", "statistical_analysis.md"}
        for content in files.values():
            assert content.startswith(
                "<!-- MACHINE-WRITTEN by localdata_mcp.nexus.contract.generators.docs"
            )

    def test_rows_carry_tool_facts(self) -> None:
        core = render_docs(_registry())["core.md"]
        assert "`echo`" in core
        assert "SCALAR" in core


class TestTestStub:
    def test_banner_compile_and_entries(self) -> None:
        text = render_test_module(_registry())
        assert text.startswith(
            '"""MACHINE-WRITTEN by localdata_mcp.nexus.contract.generators.test_stub'
        )
        compile(text, "contract_generated_test.py", "exec")
        assert '"echo"' in text and '"profile_frame"' in text

    def test_coverage_check_present(self) -> None:
        text = render_test_module(_registry())
        assert "test_every_registered_spec_has_a_generated_entry" in text

    def test_sample_arguments_follow_annotations(self) -> None:
        text = render_test_module(_registry())
        assert '"text": "probe"' in text
        assert '"rows": 3' in text


class TestOrchestrator:
    def test_emits_all_five_artifact_homes(self) -> None:
        artifacts = generate_artifacts(_registry())
        paths = {str(path) for path in artifacts}
        assert "src/localdata_mcp/server/tools_generated.py" in paths
        assert "src/localdata_mcp/nexus/contract/generated_shapes.py" in paths
        assert (
            "src/localdata_mcp/testbench/batteries/base/"
            "contract_generated_test.py" in paths
        )
        assert "docs/tools/core.md" in paths
        assert "docs/tools/statistical_analysis.md" in paths

    def test_every_artifact_opens_with_machine_written_banner(self) -> None:
        for content in generate_artifacts(_registry()).values():
            first_line = content.splitlines()[0]
            assert "MACHINE-WRITTEN by" in first_line
            assert "localdata_mcp.nexus.contract.generators." in first_line

    def test_deterministic(self) -> None:
        assert generate_artifacts(_registry()) == generate_artifacts(_registry())
