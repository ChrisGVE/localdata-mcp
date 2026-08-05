# GitHub workflows

**No workflow currently gates the `new-v3` branch.** Five workflow files exist;
none of them runs on a push or a pull request to it, and the two that were
written for a v3 tree reference source packages and scripts that are not in this
repository. Read this before trusting a green check, and before adding a CI badge
to a document.

That state is a defect rather than a policy, and it is recorded here so nobody
re-derives it from a workflow file that looks plausible.

## What is here, and what it actually does

| File | Triggers on | State |
|---|---|---|
| `codeql.yml` | push and PR to `main`; Tuesdays 14:43 UTC | Works. Runs against `main`, which still carries 2.x. |
| `publish-to-pypi.yml` | push to `main`; tags `v*.*.*`; manual | Works, and the two triggers publish to **different registries** — a push to `main` goes to TestPyPI, a `v*.*.*` tag goes to PyPI. See below. |
| `docker-publish.yml` | tags `v*.*.*`; manual | Builds and pushes the image. **The image itself is broken** — see below. |
| `v3-ci.yml` | push and PR to `v3` or `main` | **Does not run and would not pass.** |
| `v3-nightly.yml` | daily 04:17 UTC; manual | **Does not run to completion.** |

There is no `ci.yml`, no `release.yml` and no `security.yml` in this repository.
A badge pointing at any of them renders as "no status".

`dependabot.yml` is present and is not a workflow: it opens weekly pull requests
for pip and for GitHub Actions, capped at ten and five respectively, assigned to
`ChristianBerclaz`. Its pull requests land against the default branch, so on this
branch they arrive as noise rather than as updates.

## Why the two v3 workflows do not work

Both were written for an earlier, abandoned attempt at v3 — a tree of packages
under `src/localdata_mcp/` named `nexus`, `ingest`, `explore`, `process`,
`visualize` and `testbench`. That tree was deleted. The package now has nine
modules and no sub-packages at all, so:

- `v3-ci.yml`'s mypy job type-checks `src/localdata_mcp/nexus`, which does not
  exist.
- Between them the two files invoke **six** scripts that are not in `scripts/`.
  Three are named by both files — `check_audit_severity.py`,
  `cold_start_smoke.py`, `merge_battery_results.py`; two by `v3-ci.yml` alone —
  `check_battery_run_trailer.py`, `check_pin_drift.py`; and one by
  `v3-nightly.yml` alone — `compare_battery_runs.py`. They also name
  `build_db_fixtures.py` and `build_oracle_datasets.py`, which are the only two
  that are present. Eight paths referenced, two present, six missing.
- **The two that are present do not run either.** Both import
  `localdata_mcp.testbench` — named four lines above as one of the deleted
  sub-packages — and die at import with `ModuleNotFoundError` before reaching
  argument parsing; `v3-nightly.yml` invokes `build_db_fixtures.py` in three
  separate steps, each of them dead. So the count above understates it: **eight
  paths referenced, eight unusable.** Tracked as
  [#85](https://github.com/ChrisGVE/localdata-mcp/issues/85); whether the two
  return with a v3 testbench or are deleted with the workflows is not decided.
- `v3-ci.yml` gates a coverage floor of 85 over `tests/v3`, **which is not in the
  repository at all** — `git ls-files tests/v3` is empty, and what is on disk is
  untracked leftovers. The job would run pytest over a path a clean checkout does
  not have.
- Neither the mypy job nor the coverage job has the tool it invokes. Both run
  `uv sync --frozen --extra dev`, and the `dev` extra is `pytest` and nothing
  else, so `uv run mypy` and `--cov-fail-under` have no mypy and no pytest-cov.
- Neither triggers on `new-v3`, so none of that has ever been reported.

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
rewrite: the bug report's `database-info` field and the PR template's "Database
Support" checklist were written for 2.x's thirteen database types.

## PyPI trusted publishing

`publish-to-pypi.yml` has three jobs. `build` produces the sdist and wheel and
runs `twine check`. The two publish jobs are mutually exclusive and go to
different registries:

| Job | Fires on | Registry |
|---|---|---|
| `publish-to-testpypi` | a push to `main` | **TestPyPI** — `https://test.pypi.org/legacy/`, `skip-existing: true` |
| `publish-to-pypi` | a `refs/tags/v*` tag | **PyPI** |

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
   Then
   wait: the `pypi` environment holds the upload for fifteen minutes, so a job
   that has not started is the timer doing its job.
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

1. `v3-ci.yml` and `v3-nightly.yml` need deleting or rewriting against the tree
   that exists, and whichever survives needs to trigger on the release branch.
   Budget for **eight** scripts, not six: the two that exist are dead on import
   and need writing or removing along with the six that are absent (#85).
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
