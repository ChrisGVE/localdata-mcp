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
product was not built. This branch is a database server with **eight verbs** — `attach`,
`detach`, `query`, `info`, `create`, `update`, `drop`, `save` — and the v2 documents
describe tools that do not exist, at paths that do not exist, with parameters that were
never implemented.

> **Corrected (2026-08-04).** This paragraph named seven verbs including `add_table` and
> `drop_table`, which were renamed to `create` and `drop` before this file was last
> touched, and `update` had been added. A quarantine notice that misdescribes the thing
> it is protecting readers from is the same defect one directory further out.

> **Resolved (2026-07-26).** The Sphinx build files stayed behind when the rest moved, so
> the site had almost nothing left to build and `conf.py` excluded pages that were no
> longer there. `docs/conf.py`, `docs/requirements.txt` and `.readthedocs.yaml` have been
> deleted, and with them the three include-stubs that were their only remaining content.
> Nothing is published: the four documents this project ships — `README.md`, `LEVEL0.md`,
> `CONSTRAINTS.md` and the skill — are read where they live. A generated site for a
> seven-verb server was a fifth surface to keep current and no reader's shortest path to
> anything.
>
> `TROUBLESHOOTING.md` and `DOCKER_USAGE.md` arrived here at the same time and for the
> original reason: both described the v2 product end to end. The Docker image itself still
> ships, so a correct Docker guide is wanted — it is just not this one, and a wrong guide
> is worse than an absent one.

> **Amended (2026-08-04).** The Docker image ships and **does not work on this branch**:
> the `Dockerfile` is 2.x's, and its `HEALTHCHECK` imports `localdata_mcp.localdata_mcp`,
> a module that was deleted, so every container reports itself unhealthy. So the guide is
> not the only thing missing — the image has to be rebuilt before a guide to it can be
> true. `.github/WORKFLOWS.md` records the rest, including that a `v*.*.*` tag would
> publish it as it stands.

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
