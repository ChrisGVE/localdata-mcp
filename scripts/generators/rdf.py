"""Generate RDF test fixtures (Turtle, N-Triples)."""

from __future__ import annotations

import os

from generators._common import sub_dir, write_text

_TURTLE = """\
@prefix ex: <http://example.org/> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .

ex:Person1 rdf:type ex:Person ;
    rdfs:label "Alice" ;
    ex:age 30 ;
    ex:knows ex:Person2, ex:Person3 .

ex:Person2 rdf:type ex:Person ;
    rdfs:label "Bob" ;
    ex:age 25 ;
    ex:knows ex:Person1 .

ex:Person3 rdf:type ex:Person ;
    rdfs:label "Charlie" ;
    ex:age 35 ;
    ex:knows ex:Person1, ex:Person4 .

ex:Person4 rdf:type ex:Person ;
    rdfs:label "Diana" ;
    ex:age 28 ;
    ex:knows ex:Person3 .

ex:Person5 rdf:type ex:Person ;
    rdfs:label "Eve" ;
    ex:age 22 .
"""

_NTRIPLES = """\
<http://example.org/Person1> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://example.org/Person> .
<http://example.org/Person1> <http://www.w3.org/2000/01/rdf-schema#label> "Alice" .
<http://example.org/Person1> <http://example.org/age> "30"^^<http://www.w3.org/2001/XMLSchema#integer> .
<http://example.org/Person1> <http://example.org/knows> <http://example.org/Person2> .
<http://example.org/Person1> <http://example.org/knows> <http://example.org/Person3> .
<http://example.org/Person2> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://example.org/Person> .
<http://example.org/Person2> <http://www.w3.org/2000/01/rdf-schema#label> "Bob" .
<http://example.org/Person2> <http://example.org/age> "25"^^<http://www.w3.org/2001/XMLSchema#integer> .
<http://example.org/Person2> <http://example.org/knows> <http://example.org/Person1> .
<http://example.org/Person3> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://example.org/Person> .
<http://example.org/Person3> <http://www.w3.org/2000/01/rdf-schema#label> "Charlie" .
<http://example.org/Person3> <http://example.org/age> "35"^^<http://www.w3.org/2001/XMLSchema#integer> .
<http://example.org/Person3> <http://example.org/knows> <http://example.org/Person1> .
<http://example.org/Person3> <http://example.org/knows> <http://example.org/Person4> .
<http://example.org/Person4> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://example.org/Person> .
<http://example.org/Person4> <http://www.w3.org/2000/01/rdf-schema#label> "Diana" .
<http://example.org/Person4> <http://example.org/age> "28"^^<http://www.w3.org/2001/XMLSchema#integer> .
<http://example.org/Person4> <http://example.org/knows> <http://example.org/Person3> .
<http://example.org/Person5> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://example.org/Person> .
<http://example.org/Person5> <http://www.w3.org/2000/01/rdf-schema#label> "Eve" .
<http://example.org/Person5> <http://example.org/age> "22"^^<http://www.w3.org/2001/XMLSchema#integer> .
"""


def generate_rdf(output_dir: str) -> list[str]:
    """Generate Turtle and N-Triples files in rdf/ subdirectory."""
    d = sub_dir(output_dir, "rdf")
    return [
        write_text(os.path.join(d, "sample.ttl"), _TURTLE),
        write_text(os.path.join(d, "sample.nt"), _NTRIPLES),
    ]
