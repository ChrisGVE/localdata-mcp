"""Tests for exporting a stored tree as markdown.

``export_structured(name, "markdown")`` is documented as a first-class export
format beside JSON, YAML and TOML, but it returned an empty document: the
reconstructed tree is a plain nested dict, while the markdown renderer expects
nodes carrying "name"/"children"/"properties", so every node rendered as a
bare heading with no text.

The assertions below therefore check the *content* — that the names and values
present in the source tree appear in the document, at the right heading depth —
rather than that a string came back.
"""

import pytest
from sqlalchemy import create_engine

from localdata_mcp.tree_export import tool_export_structured
from localdata_mcp.tree_parsers import parse_dict_to_tree
from localdata_mcp.tree_storage import TreeStorageManager, create_tree_schema

NAME = "markdown_conn"

FLAT_CONFIG = {
    "database": {"host": "localhost", "port": 5432, "enabled": True},
    "logging": {"level": "info", "file": "/var/log/app.log"},
}

NESTED_TREE = {
    "service": {
        "api": {
            "server": {"host": "api.internal", "port": 8080},
            "timeout_seconds": 30,
        },
        "workers": {
            "queue": {"name": "jobs", "concurrency": 4},
        },
    }
}


@pytest.fixture
def manager() -> TreeStorageManager:
    engine = create_engine("sqlite:///:memory:")
    create_tree_schema(engine)
    return TreeStorageManager(engine)


def _markdown(manager: TreeStorageManager, **kwargs) -> str:
    result = tool_export_structured(manager, NAME, "markdown", **kwargs)
    assert result["format"] == "markdown", result
    return result["content"]


def _unescaped(content: str) -> str:
    """Drop the renderer's markdown escapes so names can be matched plainly."""
    return content.replace("\\", "")


class TestMarkdownExportOfFlatConfig:
    def test_every_section_and_key_appears(self, manager: TreeStorageManager) -> None:
        parse_dict_to_tree(FLAT_CONFIG, manager)
        content = _markdown(manager)

        assert "## database" in content
        assert "## logging" in content
        for key in ("host", "port", "enabled", "level", "file"):
            assert key in content, f"key '{key}' missing from:\n{content}"

    def test_values_appear_not_just_names(self, manager: TreeStorageManager) -> None:
        parse_dict_to_tree(FLAT_CONFIG, manager)
        content = _markdown(manager)

        assert "localhost" in content
        assert "5432" in content
        assert "/var/log/app.log" in content

    def test_document_is_not_an_empty_heading(
        self, manager: TreeStorageManager
    ) -> None:
        parse_dict_to_tree(FLAT_CONFIG, manager)
        content = _markdown(manager)

        assert content.strip() != "##"
        assert len(content.splitlines()) > 4, content


class TestMarkdownExportOfNestedTree:
    def test_three_levels_nest_as_deepening_headings(
        self, manager: TreeStorageManager
    ) -> None:
        parse_dict_to_tree(NESTED_TREE, manager)
        content = _markdown(manager)

        assert "## service" in content
        assert "### api" in content
        assert "#### server" in content
        assert "### workers" in content
        assert "#### queue" in content

    def test_leaf_values_survive_the_nesting(self, manager: TreeStorageManager) -> None:
        parse_dict_to_tree(NESTED_TREE, manager)
        content = _markdown(manager)

        assert "api.internal" in content
        assert "8080" in content
        assert "jobs" in content
        assert "timeout_seconds" in _unescaped(content)

    def test_subtree_export_covers_only_that_branch(
        self, manager: TreeStorageManager
    ) -> None:
        parse_dict_to_tree(NESTED_TREE, manager)
        content = _markdown(manager, path="service.workers")

        assert "queue" in content
        assert "jobs" in content
        assert "api.internal" not in content

    def test_md_alias_produces_the_same_document(
        self, manager: TreeStorageManager
    ) -> None:
        parse_dict_to_tree(NESTED_TREE, manager)
        alias = tool_export_structured(manager, NAME, "md")

        assert alias["format"] == "markdown"
        assert alias["content"] == _markdown(manager)
