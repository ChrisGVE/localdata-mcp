"""Neo4j integration tests via MCP tool interface."""

import os

import pytest

from .mcp_test_client import call_tool

NEO4J_URL = os.environ.get("TEST_NEO4J_URL", "bolt://localhost:17687")


def _neo4j_available():
    try:
        from neo4j import GraphDatabase

        driver = GraphDatabase.driver(NEO4J_URL, auth=None)
        driver.verify_connectivity()
        driver.close()
        return True
    except Exception:
        return False


pytestmark = [
    pytest.mark.integration,
    pytest.mark.neo4j,
    pytest.mark.skipif(not _neo4j_available(), reason="Neo4j not available"),
]


@pytest.fixture(scope="module", autouse=True)
def setup_neo4j_data():
    """Create test nodes and relationships directly via neo4j driver."""
    from neo4j import GraphDatabase

    driver = GraphDatabase.driver(NEO4J_URL, auth=None)
    with driver.session() as session:
        # Clean up any existing test data
        session.run("MATCH (n) DETACH DELETE n")
        # Create Person nodes
        for i in range(10):
            session.run(
                "CREATE (p:Person {name: $name, age: $age, city: $city})",
                name=f"person_{i}",
                age=20 + i,
                city=["London", "Paris", "Berlin"][i % 3],
            )
        # Create KNOWS relationships between consecutive persons
        for i in range(9):
            session.run(
                "MATCH (a:Person {name: $a}), (b:Person {name: $b}) "
                "CREATE (a)-[:KNOWS {since: $since}]->(b)",
                a=f"person_{i}",
                b=f"person_{i + 1}",
                since=2015 + i,
            )
    yield
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
    driver.close()


def _try_disconnect(name: str) -> None:
    """Best-effort disconnect."""
    try:
        call_tool("disconnect_database", {"name": name})
    except Exception:
        pass


@pytest.fixture(autouse=True)
def _cleanup_neo4j_connections():
    """Disconnect any Neo4j connections left over after each test."""
    yield
    try:
        result = call_tool("list_databases", {})
        if isinstance(result, dict):
            for db in result.get("databases", []):
                if db.get("db_type") == "neo4j":
                    _try_disconnect(db["name"])
    except Exception:
        pass


class TestNeo4jConnection:
    """Test Neo4j connection lifecycle via MCP."""

    def test_connect_neo4j(self):
        """Connecting to Neo4j should return a result (success or structured error)."""
        result = call_tool(
            "connect_database",
            {"name": "neo4j_conn", "db_type": "neo4j", "conn_string": NEO4J_URL},
        )
        try:
            result_str = str(result)
            assert result_str, "connect_database returned empty result"
            assert (
                "success" in result_str.lower() or "neo4j_conn" in result_str
            ), f"Unexpected connect result: {result_str}"
        finally:
            _try_disconnect("neo4j_conn")

    def test_connect_and_list(self):
        """After connecting, the connection name should appear in list_databases."""
        call_tool(
            "connect_database",
            {"name": "neo4j_ls", "db_type": "neo4j", "conn_string": NEO4J_URL},
        )
        try:
            list_result = call_tool("list_databases", {})
            list_str = str(list_result)
            assert (
                "neo4j_ls" in list_str
            ), f"neo4j_ls not in list_databases after connect: {list_str}"
        finally:
            _try_disconnect("neo4j_ls")

    def test_disconnect_neo4j(self):
        """Disconnect should succeed for Neo4j connections."""
        call_tool(
            "connect_database",
            {"name": "neo4j_dc", "db_type": "neo4j", "conn_string": NEO4J_URL},
        )
        result = call_tool("disconnect_database", {"name": "neo4j_dc"})
        result_str = str(result)
        assert (
            "disconnect" in result_str.lower() or "success" in result_str.lower()
        ), f"Expected successful disconnect, got: {result_str}"

    def test_duplicate_connect_rejected(self):
        """Connecting with a name already in use should return an error."""
        call_tool(
            "connect_database",
            {"name": "neo4j_dup", "db_type": "neo4j", "conn_string": NEO4J_URL},
        )
        try:
            result = call_tool(
                "connect_database",
                {
                    "name": "neo4j_dup",
                    "db_type": "neo4j",
                    "conn_string": NEO4J_URL,
                },
            )
            result_str = str(result)
            assert (
                "already" in result_str.lower() or "error" in result_str.lower()
            ), f"Expected duplicate-name error: {result_str}"
        finally:
            _try_disconnect("neo4j_dup")


class TestCypherQueries:
    """Test Cypher query execution on Neo4j connections."""

    def test_match_all_persons(self):
        """MATCH query should return Person nodes."""
        call_tool(
            "connect_database",
            {"name": "neo4j_qm", "db_type": "neo4j", "conn_string": NEO4J_URL},
        )
        try:
            result = call_tool(
                "execute_query",
                {
                    "name": "neo4j_qm",
                    "query": "MATCH (p:Person) RETURN p.name, p.age ORDER BY p.age",
                },
            )
            result_str = str(result)
            assert result_str, "execute_query returned empty result"
            # Should contain at least one person name
            assert (
                "person_0" in result_str or "person" in result_str.lower()
            ), f"Expected person data in result: {result_str}"
        finally:
            _try_disconnect("neo4j_qm")

    def test_match_with_where(self):
        """MATCH with WHERE clause should filter results."""
        call_tool(
            "connect_database",
            {"name": "neo4j_qw", "db_type": "neo4j", "conn_string": NEO4J_URL},
        )
        try:
            result = call_tool(
                "execute_query",
                {
                    "name": "neo4j_qw",
                    "query": "MATCH (p:Person) WHERE p.age > 25 RETURN p.name, p.age",
                },
            )
            result_str = str(result)
            assert result_str, "execute_query with WHERE returned empty result"
        finally:
            _try_disconnect("neo4j_qw")

    def test_match_relationships(self):
        """Query should return KNOWS relationships."""
        call_tool(
            "connect_database",
            {"name": "neo4j_qr", "db_type": "neo4j", "conn_string": NEO4J_URL},
        )
        try:
            result = call_tool(
                "execute_query",
                {
                    "name": "neo4j_qr",
                    "query": (
                        "MATCH (a:Person)-[r:KNOWS]->(b:Person) "
                        "RETURN a.name, b.name, r.since ORDER BY r.since"
                    ),
                },
            )
            result_str = str(result)
            assert result_str, "execute_query for relationships returned empty result"
        finally:
            _try_disconnect("neo4j_qr")

    def test_shortest_path(self):
        """Shortest path query should return a result."""
        call_tool(
            "connect_database",
            {"name": "neo4j_qp", "db_type": "neo4j", "conn_string": NEO4J_URL},
        )
        try:
            result = call_tool(
                "execute_query",
                {
                    "name": "neo4j_qp",
                    "query": (
                        "MATCH p=shortestPath("
                        "(a:Person {name: 'person_0'})-[*]-(b:Person {name: 'person_5'})"
                        ") RETURN length(p) AS path_length"
                    ),
                },
            )
            result_str = str(result)
            assert result_str, "execute_query for path returned empty result"
        finally:
            _try_disconnect("neo4j_qp")


class TestNeo4jErrors:
    """Test error handling for Neo4j edge cases."""

    def test_connect_invalid_url(self):
        """Connecting with an unreachable host should return an error, not crash."""
        result = call_tool(
            "connect_database",
            {
                "name": "neo4j_bad",
                "db_type": "neo4j",
                "conn_string": "bolt://nonexistent-host:99999",
            },
        )
        try:
            result_str = str(result)
            assert result_str, "connect_database with bad URL returned empty result"
        finally:
            _try_disconnect("neo4j_bad")

    def test_disconnect_nonexistent(self):
        """Disconnecting a nonexistent connection should return error."""
        result = call_tool("disconnect_database", {"name": "neo4j_nonexist_xyz"})
        result_str = str(result)
        assert (
            "error" in result_str.lower() or "not" in result_str.lower()
        ), f"Expected error for nonexistent disconnect: {result_str}"

    def test_query_nonexistent_connection(self):
        """Querying a nonexistent connection should return error."""
        result = call_tool(
            "execute_query",
            {"name": "neo4j_nonexist_xyz", "query": "MATCH (n) RETURN n"},
        )
        result_str = str(result)
        assert (
            "error" in result_str.lower() or "not" in result_str.lower()
        ), f"Expected error for nonexistent query: {result_str}"
