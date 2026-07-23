"""tests/v3/test_sparql_validate.py — E6.2b construct screen.

Legitimate reads pass; SERVICE is refused at any depth (the SSRF
payload row); update forms are refused on the read path and accepted on
the write path; unparseable input is refused either way.
"""

from __future__ import annotations

import pytest

from localdata_mcp.nexus.chokepoint.sparql_validate import (
    DENIED_READ_CONSTRUCTS,
    SparqlRefusedError,
    screen_read,
    screen_update,
)


class TestLegitimateReads:
    @pytest.mark.parametrize(
        "sparql",
        [
            "SELECT ?s WHERE { ?s ?p ?o }",
            "ASK { ?s ?p ?o }",
            "CONSTRUCT { ?s ?p ?o } WHERE { ?s ?p ?o }",
            "DESCRIBE ?s WHERE { ?s ?p ?o }",
            "SELECT ?s WHERE { ?s ?p ?o . OPTIONAL { ?s ?p2 ?o2 } } LIMIT 10",
        ],
    )
    def test_read_query_passes(self, sparql: str) -> None:
        screen_read(sparql)  # no raise


class TestServiceRefusal:
    def test_top_level_service_is_refused(self) -> None:
        with pytest.raises(SparqlRefusedError, match="denied construct"):
            screen_read("SELECT ?s WHERE { SERVICE <http://evil/> { ?s ?p ?o } }")

    def test_nested_service_is_refused(self) -> None:
        with pytest.raises(SparqlRefusedError):
            screen_read(
                "SELECT ?s WHERE { { SELECT ?s WHERE "
                "{ SERVICE <http://evil/> { ?s ?p ?o } } } }"
            )

    def test_service_inside_optional_is_refused(self) -> None:
        with pytest.raises(SparqlRefusedError):
            screen_read(
                "SELECT ?s WHERE { ?s ?p ?o . "
                "OPTIONAL { SERVICE <http://e/> { ?s ?p2 ?o2 } } }"
            )


class TestUpdateFormsOnReadPath:
    @pytest.mark.parametrize(
        "sparql",
        [
            "LOAD <http://evil/data>",
            "INSERT DATA { <a> <b> <c> }",
            "DELETE DATA { <a> <b> <c> }",
            "CLEAR ALL",
            "DROP GRAPH <http://g/>",
        ],
    )
    def test_update_form_is_refused_on_the_read_path(self, sparql: str) -> None:
        with pytest.raises(SparqlRefusedError):
            screen_read(sparql)


class TestWritePath:
    @pytest.mark.parametrize(
        "sparql",
        [
            "INSERT DATA { <a> <b> <c> }",
            "DELETE DATA { <a> <b> <c> }",
            "CLEAR ALL",
        ],
    )
    def test_update_form_passes_the_write_screen(self, sparql: str) -> None:
        screen_update(sparql)  # no raise

    def test_load_is_an_update_form(self) -> None:
        screen_update("LOAD <http://x/data>")


class TestFailSafe:
    def test_unparseable_read_is_refused(self) -> None:
        with pytest.raises(SparqlRefusedError, match="parse"):
            screen_read("this is not sparql {{{")

    def test_unparseable_update_is_refused(self) -> None:
        with pytest.raises(SparqlRefusedError, match="parse"):
            screen_update("this is not sparql {{{")


class TestTheOneConstant:
    def test_denied_read_constructs_names_service(self) -> None:
        assert "ServiceGraphPattern" in DENIED_READ_CONSTRUCTS
