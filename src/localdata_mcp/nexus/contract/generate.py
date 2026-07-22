"""localdata_mcp/nexus/contract/generate.py — the thin codegen orchestrator.

Build/CI-time only, never the request path (section 6.1): loads the
spec-module roster, drives one per-artifact generator each, and writes
the five COMMITTED artifacts at their section-9 homes. check_drift.py
reuses generate_artifacts() against a scratch root to fail CI on hand
edits. Neighbors: generators/ hold the renderers; spec_modules.py
fixes the tool population.
"""

from __future__ import annotations

import sys
from pathlib import Path, PurePosixPath

from localdata_mcp.nexus.contract.generators.docs import render_docs
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

WRAPPER_PATH = PurePosixPath("src/localdata_mcp/server/tools_generated.py")
SHAPES_PATH = PurePosixPath("src/localdata_mcp/nexus/contract/generated_shapes.py")
TEST_STUB_PATH = PurePosixPath(
    "src/localdata_mcp/testbench/batteries/base/contract_generated_test.py"
)
DOCS_DIR = PurePosixPath("docs/tools")

# This file sits at src/localdata_mcp/nexus/contract/generate.py —
# four parents up is the repository root the artifact paths key on.
REPO_ROOT = Path(__file__).resolve().parents[4]


def generate_artifacts(registry: ToolRegistry) -> dict[PurePosixPath, str]:
    """Every artifact as `{repo-relative path: content}` for `registry`."""
    artifacts: dict[PurePosixPath, str] = {
        WRAPPER_PATH: render_wrapper_module(registry),
        SHAPES_PATH: render_shapes_module(registry),
        TEST_STUB_PATH: render_test_module(registry),
    }
    for filename, content in render_docs(registry).items():
        artifacts[DOCS_DIR / filename] = content
    return artifacts


def loaded_default_registry() -> ToolRegistry:
    """The default registry after importing the spec-module roster."""
    from localdata_mcp.nexus.contract.registry import default_registry
    from localdata_mcp.nexus.contract.spec_modules import load_spec_modules

    load_spec_modules()
    return default_registry()


def write_artifacts(root: Path, registry: ToolRegistry) -> list[Path]:
    """Write every artifact under `root`; returns the written paths."""
    written = []
    for relpath, content in generate_artifacts(registry).items():
        target = root / relpath
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        written.append(target)
    return written


def main() -> int:
    """Regenerate the committed artifacts in the working tree."""
    for path in write_artifacts(REPO_ROOT, loaded_default_registry()):
        sys.stderr.write(f"generated {path}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
