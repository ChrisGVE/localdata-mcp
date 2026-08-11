# GitHub workflows

**No workflow currently gates the `new-v3` branch.** Three workflow files exist,
and none of them runs on a push or a pull request to it. Read this before
trusting a green check, and before adding a CI badge to a document.

That state is a defect rather than a policy, and it is recorded here so nobody
re-derives it from a workflow file that looks plausible.

## What is here, and what it actually does

| File | Triggers on | State |
|---|---|---|
| `codeql.yml` | push and PR to `main`; Tuesdays 14:43 UTC | Works. Runs against `main`, which still carries 2.x. |
| `publish-to-pypi.yml` | push to `main`; tags `v*.*.*`; manual | Works, and the registry it publishes to is decided by **the ref, not the trigger** — `main` goes to TestPyPI, a `v*.*.*` tag goes to PyPI, and a manual dispatch goes to whichever of the two the ref it is run from satisfies. See below. |
| `docker-publish.yml` | tags `v*.*.*`; manual | Builds and pushes the image. **The image itself is broken** — see below. |

There is no `ci.yml`, no `release.yml` and no `security.yml` in this repository.
A badge pointing at any of them renders as "no status".

`dependabot.yml` is present and is not a workflow: it opens weekly pull requests
for pip and for GitHub Actions, capped at ten and five respectively, assigned to
`ChristianBerclaz`. Its pull requests land against the default branch, so on this
branch they arrive as noise rather than as updates.

## What was deleted on 2026-08-11 — two workflows and two scripts

Until 2026-08-11 this directory also held `v3-ci.yml` and `v3-nightly.yml`, and
`scripts/` held `build_db_fixtures.py` and `build_oracle_datasets.py`. All four
were **deleted**, and this note exists so nobody restores them from history
believing they were a starting point.

They were written for an earlier, abandoned attempt at v3 — a tree of packages
under `src/localdata_mcp/` named `nexus`, `ingest`, `explore`, `process`,
`visualize`, `output`, `testbench`, and a `server` package that would have
shadowed the live `server.py` this project ships. That tree was never built; the
package has nine modules and no sub-packages at all. So the two workflows
type-checked eleven paths that do not exist, gated a coverage floor over six
packages that do not exist, collected from a `tests/v3` that was never in the
repository, and invoked eight scripts of which six were absent — while the two
present ones died at import on `localdata_mcp.testbench`, the deleted
sub-package. Neither workflow triggered on `new-v3`, so none of it was ever
reported by a run. That closes
[#85](https://github.com/ChrisGVE/localdata-mcp/issues/85) and
[#89](https://github.com/ChrisGVE/localdata-mcp/issues/89).

**Nothing was lost with them.** They described a tree, not this one; whatever CI
this branch eventually gets has to be written against the package that exists.

## The Docker image

`docker-publish.yml` fires on a `v*.*.*` tag, so **tagging 3.0.0 would publish a
broken image.** The `Dockerfile` is 2.x's: it labels itself `version="2.0.0"`,
installs redis, elasticsearch, pymongo, influxdb-client, neo4j and couchdb —
none of which this package uses — and its `HEALTHCHECK` runs

```
python -c "import localdata_mcp.localdata_mcp; print('OK')"
```

against a module that was deleted, so every container reports itself unhealthy.
`docker-compose.yml` (the development stack, not the test one) is the same
vintage: postgres, mysql, mongodb, redis and elasticsearch.

`docker-compose.test.yml` is separate, current, and the only compose file the
test suite uses. See CONTRIBUTING.md for how it is run.

## What running the tests actually looks like

There is no CI to defer to, so run them locally before opening a pull request:

```bash
uv sync --all-extras
.venv/bin/python -m pytest -q -m 'not slow'
```

Every endpoint test skips itself when its container is not answering, so that run
is green with several hundred skips and says nothing about any dialect. The
endpoint batches are in `scripts/endpoint-batch.sh`; CONTRIBUTING.md has the
detail, including why the catalogue cannot be run in one go
([#46](https://github.com/ChrisGVE/localdata-mcp/issues/46)).

## Issue and PR templates

`.github/ISSUE_TEMPLATE/` holds `bug_report.yml`, `feature_request.yml`,
`security_report.md` and a `config.yml`; `.github/pull_request_template.md` is
alongside them. GitHub serves all of these without a workflow. They predate this
rewrite, and one field has gone stale with it: the bug report's **required**
Python dropdown (`bug_report.yml:33-46`) offers 3.8 and 3.9, both below the
`requires-python = ">=3.10"` this project declares, and omits the 3.13 that
`pyproject.toml:21` claims support for. A reporter on 3.13 cannot answer it
truthfully and cannot skip it.

The bug report's `database-info` field and the PR template's "Database Support"
checklist read as 2.x leftovers and are not: each names SQLite, PostgreSQL and
MySQL — three backends this version still supports — plus a free-text catch-all
for the rest.

## PyPI trusted publishing

`publish-to-pypi.yml` has three jobs. `build` produces the sdist and wheel and
runs `twine check`. The two publish jobs are mutually exclusive and go to
different registries:

| Job | Fires on | Registry |
|---|---|---|
| `publish-to-testpypi` | `github.ref == 'refs/heads/main'` — a push to `main`, or a dispatch run from it | **TestPyPI** — `https://test.pypi.org/legacy/`, `skip-existing: true` |
| `publish-to-pypi` | `startsWith(github.ref, 'refs/tags/v')` — a `v*` tag, pushed or dispatched | **PyPI** |

Both use PyPI's trusted publishing, which carries no long-lived token — that is
why it is used. It has two prerequisites, and **both must be in place before a
release tag is pushed**:

1. The publisher registered at
   <https://pypi.org/manage/account/publishing/> against this repository and this
   workflow filename — and separately at
   <https://test.pypi.org/manage/account/publishing/> for the TestPyPI half.
2. **Two GitHub environments, named `testpypi` and `pypi`.** Each publish job
   declares `environment: name: …` alongside `permissions: id-token: write`. A
   job naming an environment that does not exist fails before it uploads
   anything, so this is a hard prerequisite rather than a nicety — and the
   environment is where a protection rule on the release lives.

   **Both exist, and have since 2025-08-29.** They are also the only release gate
   this project actually has, so what they carry matters
   (`gh api repos/ChrisGVE/localdata-mcp/environments`, re-read 2026-08-05):

   | Environment | Deployment branch policy | Other protection |
   |---|---|---|
   | `testpypi` | `main`, and tags matching `v*.*.*` | — |
   | `pypi` | `main`, and tags matching `v*.*.*` | **`wait_timer: 15`** |

   **The PyPI upload sits in a fifteen-minute timed hold before it starts.** Push
   the release tag, watch the build go green, and the upload job will not have
   begun; that is the timer, not a hang, and re-pushing or cancelling the run is
   the wrong reaction. The branch policy is also why the Docker gap's "with no tag
   involved at all" is true of Docker and not of PyPI: `docker-publish.yml`
   names no environment, so nothing restricts what it publishes from. The Docker
   gap is listed under [Before the next release](#before-the-next-release).

**Nothing verifies that the tag and `project.version` agree.** Push `v3.0.0` at
this commit and the job builds `3.0.0.dev0` and uploads it under a tag saying
otherwise. `twine check` passes, because a pre-release is valid metadata — and
because it is a pre-release, `pip install localdata-mcp` and `uvx localdata-mcp`
exclude it from resolution by default, so the release would look green while no
user's install changed at all. A `refs/tags/v*` → `project.version` equality
check in the `build` job is the standard guard.

**Nothing publishes `server.json` either.** No workflow runs `mcp-publisher`,
and this server has never appeared in the MCP registry — a query for `localdata`,
`chrisgve` and `io.github.chrisgve/localdata-mcp` returns nothing. Publishing it
is a manual step somebody has to take, and `server.json` has to be valid first
(see [Before the next release](#before-the-next-release)).

**Nothing validates any manifest.** No workflow runs the `server.json` schema
check or `claude plugin validate`; `twine check` is the only manifest gate there
is, and it does not look at either file.

## What a release does, end to end

Nothing walks a release for you, so this is the whole of it:

1. Bring the version numbers into step — six values across five files, listed in
   `CONTRIBUTING.md` under *Versioning* — then re-run `uv lock`.
2. Run the tests locally, including the endpoint batches, since nothing gates
   this branch.
3. Push a `vX.Y.Z` tag. That is the only *push* that fires `publish-to-pypi.yml`'s
   PyPI job or `docker-publish.yml`, but it is not the only path to either: both
   declare `workflow_dispatch`, `docker-publish.yml`'s job carries no `if:` at
   all, and a dispatch may name a tag — **read the gaps below before doing it.**
   Then wait: the `pypi` environment holds the upload for fifteen minutes, so a
   job that has not started is the timer doing its job.
4. Create the GitHub release from the tag, for the notes; the upload has already
   happened by then.
5. Publish `server.json` to the MCP registry by hand. No workflow does it.

**Branch protection** is not configured and is not described anywhere else. The
previous version of this document listed required checks belonging to a `ci.yml`
that does not exist, which was worse than saying nothing; the honest statement is
that there is no protection posture today, and that setting one up has to wait on
gap 1 below, since there is no passing check to require.

## Before the next release

These are the known gaps, stated so a release does not walk into them:

1. **There is no test workflow at all, and no workflow triggers on `new-v3`.**
   The two that claimed to were deleted on 2026-08-11 (above), which removed a
   misleading file rather than a working gate. A release off this branch is
   therefore gated by whatever was run locally and written into the pull request,
   and by nothing else. Writing one is a matter of `uv sync --all-extras` and the
   endpoint batches — not of restoring what was here.
2. The `Dockerfile` needs rewriting or `docker-publish.yml` needs disabling
   before a `v3.0.0` tag is pushed. `docker-publish.yml` also moves the `latest`
   tag unconditionally, including on a `workflow_dispatch` run from any branch,
   so it can publish the broken image with no tag involved at all.
3. **The tag and `project.version` are not checked against each other**, and
   `3.0.0.dev0` is a pre-release nobody's `pip install` would resolve
   ([#76](https://github.com/ChrisGVE/localdata-mcp/issues/76)).
4. ~~**The `testpypi` and `pypi` GitHub environments have to exist.**~~ Not a
   gap: both have existed since 2025-08-29. Listed as outstanding until
   2026-08-05, when the repository was queried rather than assumed. The
   requirement itself, and the fifteen-minute hold the `pypi` environment
   imposes on the upload, are stated above.
5. **`server.json`'s `packages[0].version` is `2.1.0`, and PyPI 404s on it.**
   The registry resolves that field against the named `registryType` during
   publish, so it has to name a release that exists — which today it does not,
   and never did. It moves with the release version, so it is fixed when that is
   decided ([#78](https://github.com/ChrisGVE/localdata-mcp/issues/78)). The
   over-length `description` in the same file was fixed on 2026-08-04; it had
   survived a deliberate rewrite of that very line, because **nothing in CI
   validates this file** and the error only appears server-side at publish time.
6. `docker-compose.yml`, the issue templates and the PR template describe 2.x.
