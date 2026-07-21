# Data Sources

LocalData MCP supports five categories of data source, each with its own storage model and tool set: flat files, databases, structured (tree) data, directed graphs, and RDF triple stores. See the [complete reference](complete-reference.md) for connection strings, authentication, and examples for all 30+ supported formats.

RDF sources — `turtle`, `ntriples` and remote `sparql` endpoints — are queried with SPARQL through `execute_query`, which returns bindings under `results` rather than the usual `data`/`metadata` envelope. `rdflib` and `SPARQLWrapper` are required dependencies, so no extra is needed.

```{toctree}
:maxdepth: 2

flat-files
databases
structured-data
directed-graphs
complete-reference
```
