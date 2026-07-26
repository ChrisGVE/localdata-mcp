# non_factual/

**Nothing in this directory is known to be true.** Do not cite it, do not act on it, and
do not hand it to a sub-agent as background. It is quarantined prose, kept only because
deleting it would throw away work that may still be worth salvaging.

## Why this exists

Documents that sit in `docs/` are read as vetted — by people, and much more readily by
agents, which have no way to tell a current specification from an abandoned one and will
take a confident heading at its word. A stale document is therefore worse than a missing
one: a missing document prompts a question, a stale document answers it wrongly and the
error propagates into code.

So `docs/` holds only what somebody has checked against the code **recently and
deliberately**. Everything whose factual state is unknown lives here instead.

## What is in here

Everything under `docs/` that described **LocalData v2** — a data-science platform with
71 tools spanning statistics, regression, time series, geospatial and optimization. That
product was not built. This branch is a database server with seven verbs (`attach`,
`detach`, `query`, `info`, `add_table`, `drop_table`, `save`), and the v2 documents
describe tools that do not exist, at paths that do not exist, with parameters that were
never implemented.

The Sphinx build files (`docs/conf.py`, `docs/requirements.txt`) stayed behind, so the
documentation site currently has almost no content to build. That is a known, deliberate
state, not an accident.

**One file was removed rather than quarantined:** `docs/architecture/FIRST_PRINCIPLES.md`,
the v2 "constitutional foundation". This is a public repository and that filename is on
the leak-guard denylist, so it cannot be committed here under any path. It described the
abandoned data-science platform and was wrong regardless; `git log` still has it if any of
it is ever wanted back.

## Coming back

A file returns to `docs/` when somebody has read it end to end against the code on this
branch and can say which claims they verified. Partial confidence is not enough — half a
vetted document is a document that lies in its second half. Correct it fully, or leave it
here.

Anything moved back should say, near the top, when it was verified and against what.
