"""Tests for the RDF storage module."""

import os
from pathlib import Path

import pytest
from rdflib import BNode, Graph, Literal, URIRef
from rdflib.namespace import RDF, XSD

from localdata_mcp.rdf_storage import RDFStorageManager

FIXTURES = Path(__file__).parent / "fixtures"
TTL_FILE = str(FIXTURES / "sample.ttl")
NT_FILE = str(FIXTURES / "sample.nt")

SW = "http://example.org/software#"
RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
RDFS_LABEL = "http://www.w3.org/2000/01/rdf-schema#label"


@pytest.fixture()
def mgr() -> RDFStorageManager:
    """Return a fresh, empty RDFStorageManager."""
    return RDFStorageManager()


@pytest.fixture()
def ttl_mgr() -> RDFStorageManager:
    """Return a manager pre-loaded with the Turtle fixture."""
    m = RDFStorageManager()
    m.parse_file(TTL_FILE, format="turtle")
    return m


@pytest.fixture()
def nt_mgr() -> RDFStorageManager:
    """Return a manager pre-loaded with the N-Triples fixture."""
    m = RDFStorageManager()
    m.parse_file(NT_FILE, format="nt")
    return m


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


class TestParsing:
    """Tests for parse_file."""

    def test_parse_turtle_triple_count(self, mgr: RDFStorageManager) -> None:
        stats = mgr.parse_file(TTL_FILE, format="turtle")
        assert stats["triple_count"] == 43

    def test_parse_nt_triple_count(self, mgr: RDFStorageManager) -> None:
        stats = mgr.parse_file(NT_FILE, format="nt")
        assert stats["triple_count"] == 43

    def test_parse_turtle_stats_keys(self, mgr: RDFStorageManager) -> None:
        stats = mgr.parse_file(TTL_FILE, format="turtle")
        assert "subject_count" in stats
        assert "predicate_count" in stats
        assert "object_count" in stats

    def test_parse_invalid_format(self, mgr: RDFStorageManager) -> None:
        with pytest.raises(ValueError, match="Unsupported format"):
            mgr.parse_file(TTL_FILE, format="json-ld")


# ---------------------------------------------------------------------------
# Namespace extraction
# ---------------------------------------------------------------------------


class TestNamespaces:
    """Verify namespace extraction from Turtle."""

    def test_turtle_has_sw_namespace(self, ttl_mgr: RDFStorageManager) -> None:
        stats = ttl_mgr.get_stats()
        prefixes = {ns["prefix"] for ns in stats["namespaces"]}
        assert "sw" in prefixes

    def test_turtle_has_rdf_namespace(self, ttl_mgr: RDFStorageManager) -> None:
        stats = ttl_mgr.get_stats()
        uris = {ns["uri"] for ns in stats["namespaces"]}
        assert "http://www.w3.org/1999/02/22-rdf-syntax-ns#" in uris


# ---------------------------------------------------------------------------
# SPARQL queries
# ---------------------------------------------------------------------------


class TestSPARQLSelect:
    """Tests for SPARQL SELECT queries."""

    def test_select_classes(self, ttl_mgr: RDFStorageManager) -> None:
        query = """
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        SELECT ?cls WHERE { ?cls a rdfs:Class }
        """
        rows = ttl_mgr.execute_sparql(query)
        classes = {r["cls"] for r in rows}
        assert f"{SW}Component" in classes
        assert f"{SW}Service" in classes

    def test_select_instances(self, ttl_mgr: RDFStorageManager) -> None:
        query = f"""
        PREFIX sw: <{SW}>
        SELECT ?inst WHERE {{ ?inst a sw:Service }}
        """
        rows = ttl_mgr.execute_sparql(query)
        instances = {r["inst"] for r in rows}
        assert f"{SW}api" in instances
        assert f"{SW}auth" in instances

    def test_select_with_filter(self, ttl_mgr: RDFStorageManager) -> None:
        query = f"""
        PREFIX sw: <{SW}>
        SELECT ?comp ?ver WHERE {{
            ?comp sw:hasVersion ?ver .
            FILTER(?ver = "5.7.0")
        }}
        """
        rows = ttl_mgr.execute_sparql(query)
        assert len(rows) == 1
        assert rows[0]["comp"] == f"{SW}database"


class TestSPARQLAsk:
    """Tests for SPARQL ASK queries."""

    def test_ask_true(self, ttl_mgr: RDFStorageManager) -> None:
        query = f"""
        PREFIX sw: <{SW}>
        ASK {{ sw:api sw:dependsOn sw:auth }}
        """
        result = ttl_mgr.execute_sparql(query)
        assert result == [{"result": True}]

    def test_ask_false(self, ttl_mgr: RDFStorageManager) -> None:
        query = f"""
        PREFIX sw: <{SW}>
        ASK {{ sw:database sw:dependsOn sw:api }}
        """
        result = ttl_mgr.execute_sparql(query)
        assert result == [{"result": False}]


class TestSPARQLConstruct:
    """Tests for SPARQL CONSTRUCT queries."""

    def test_construct(self, ttl_mgr: RDFStorageManager) -> None:
        query = f"""
        PREFIX sw: <{SW}>
        CONSTRUCT {{ ?s sw:dependsOn ?o }}
        WHERE {{ ?s sw:dependsOn ?o }}
        """
        result = ttl_mgr.execute_sparql(query)
        assert len(result) == 1
        assert "triples" in result[0]
        assert "dependsOn" in result[0]["triples"]


class TestSPARQLErrors:
    """Tests for invalid SPARQL handling."""

    def test_invalid_syntax(self, ttl_mgr: RDFStorageManager) -> None:
        with pytest.raises(ValueError, match="SPARQL error"):
            ttl_mgr.execute_sparql("NOT VALID SPARQL AT ALL {{{}}")


# ---------------------------------------------------------------------------
# Term conversion
# ---------------------------------------------------------------------------


class TestTermConversion:
    """Tests for _term_to_python."""

    def test_uriref(self, mgr: RDFStorageManager) -> None:
        term = URIRef("http://example.org/x")
        assert mgr._term_to_python(term) == "http://example.org/x"

    def test_literal_string(self, mgr: RDFStorageManager) -> None:
        term = Literal("hello")
        assert mgr._term_to_python(term) == "hello"

    def test_literal_integer(self, mgr: RDFStorageManager) -> None:
        term = Literal(42, datatype=XSD.integer)
        assert mgr._term_to_python(term) == 42

    def test_bnode(self, mgr: RDFStorageManager) -> None:
        term = BNode("abc123")
        result = mgr._term_to_python(term)
        assert result.startswith("_:")

    def test_none(self, mgr: RDFStorageManager) -> None:
        assert mgr._term_to_python(None) is None


# ---------------------------------------------------------------------------
# Graph navigation
# ---------------------------------------------------------------------------


class TestNavigation:
    """Tests for get_subjects, get_predicates_for_subject, get_objects."""

    def test_get_subjects_count(self, ttl_mgr: RDFStorageManager) -> None:
        subjects = ttl_mgr.get_subjects()
        assert len(subjects) == 13

    def test_get_subjects_pagination(self, ttl_mgr: RDFStorageManager) -> None:
        page1 = ttl_mgr.get_subjects(offset=0, limit=5)
        page2 = ttl_mgr.get_subjects(offset=5, limit=5)
        assert len(page1) == 5
        assert len(page2) == 5
        assert set(page1).isdisjoint(set(page2))

    def test_get_subjects_offset_beyond(self, ttl_mgr: RDFStorageManager) -> None:
        subjects = ttl_mgr.get_subjects(offset=100)
        assert subjects == []

    def test_get_predicates_for_subject(self, ttl_mgr: RDFStorageManager) -> None:
        preds = ttl_mgr.get_predicates_for_subject(f"{SW}api")
        assert RDF_TYPE in preds
        assert RDFS_LABEL in preds
        assert f"{SW}hasVersion" in preds
        assert f"{SW}hasPort" in preds
        assert f"{SW}dependsOn" in preds

    def test_get_objects_mixed_types(self, ttl_mgr: RDFStorageManager) -> None:
        objects = ttl_mgr.get_objects(f"{SW}api", f"{SW}dependsOn")
        assert len(objects) == 3
        assert f"{SW}auth" in objects
        assert f"{SW}cache" in objects
        assert f"{SW}logger" in objects

    def test_get_objects_literal_value(self, ttl_mgr: RDFStorageManager) -> None:
        objects = ttl_mgr.get_objects(f"{SW}api", f"{SW}hasVersion")
        assert objects == ["2.1.0"]

    def test_navigation_nonexistent_subject(self, ttl_mgr: RDFStorageManager) -> None:
        preds = ttl_mgr.get_predicates_for_subject(f"{SW}nonexistent")
        assert preds == []

    def test_navigation_nonexistent_predicate(self, ttl_mgr: RDFStorageManager) -> None:
        objects = ttl_mgr.get_objects(f"{SW}api", f"{SW}nonexistent")
        assert objects == []


# ---------------------------------------------------------------------------
# Blank node round-trip
# ---------------------------------------------------------------------------


class TestBlankNodeRoundTrip:
    """Tests for blank node handling through get_subjects and navigation."""

    def test_get_subjects_blank_nodes(self, mgr: RDFStorageManager) -> None:
        """Blank nodes in get_subjects must have the '_:' prefix."""
        bnode = BNode("myblank")
        mgr.graph.add((bnode, RDF.type, URIRef("http://example.org/Thing")))
        subjects = mgr.get_subjects()
        blank_subjects = [s for s in subjects if s.startswith("_:")]
        assert len(blank_subjects) == 1
        assert blank_subjects[0] == f"_:{bnode}"

    def test_get_predicates_for_blank_node_subject(
        self, mgr: RDFStorageManager
    ) -> None:
        """get_predicates_for_subject must resolve '_:xxx' back to BNode."""
        bnode = BNode("roundtrip")
        pred = URIRef("http://example.org/name")
        mgr.graph.add((bnode, pred, Literal("test")))

        # Get the blank node string from get_subjects
        subjects = mgr.get_subjects()
        blank_subjects = [s for s in subjects if s.startswith("_:")]
        assert len(blank_subjects) == 1

        # Use the string representation to query predicates
        preds = mgr.get_predicates_for_subject(blank_subjects[0])
        assert str(pred) in preds

        # Also verify get_objects works
        objects = mgr.get_objects(blank_subjects[0], str(pred))
        assert "test" in objects


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


class TestStats:
    """Tests for get_stats accuracy."""

    def test_stats_triple_count(self, ttl_mgr: RDFStorageManager) -> None:
        stats = ttl_mgr.get_stats()
        assert stats["triple_count"] == 43

    def test_stats_subject_count(self, ttl_mgr: RDFStorageManager) -> None:
        stats = ttl_mgr.get_stats()
        assert stats["subject_count"] == 13

    def test_stats_predicate_count(self, ttl_mgr: RDFStorageManager) -> None:
        stats = ttl_mgr.get_stats()
        assert stats["predicate_count"] == 8

    def test_stats_object_count(self, ttl_mgr: RDFStorageManager) -> None:
        stats = ttl_mgr.get_stats()
        assert stats["object_count"] == 27

    def test_stats_empty_graph(self, mgr: RDFStorageManager) -> None:
        stats = mgr.get_stats()
        assert stats["triple_count"] == 0


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


class TestSerialization:
    """Tests for serialize."""

    def test_serialize_turtle(self, ttl_mgr: RDFStorageManager) -> None:
        output = ttl_mgr.serialize(format="turtle")
        assert isinstance(output, str)
        assert "dependsOn" in output

    def test_serialize_nt(self, ttl_mgr: RDFStorageManager) -> None:
        output = ttl_mgr.serialize(format="nt")
        assert isinstance(output, str)
        assert "<http://example.org/software#api>" in output

    def test_serialize_invalid_format(self, ttl_mgr: RDFStorageManager) -> None:
        with pytest.raises(ValueError, match="Unsupported format"):
            ttl_mgr.serialize(format="xml")

    def test_roundtrip_turtle(self, ttl_mgr: RDFStorageManager) -> None:
        """Serialize to Turtle and re-parse; triple count should match."""
        serialized = ttl_mgr.serialize(format="turtle")
        mgr2 = RDFStorageManager()
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".ttl", delete=False) as tmp:
            tmp.write(serialized)
            tmp_path = tmp.name
        try:
            mgr2.parse_file(tmp_path, format="turtle")
            assert len(mgr2.graph) == 43
        finally:
            os.unlink(tmp_path)
