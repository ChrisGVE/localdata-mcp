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
| `publish-to-pypi.yml` | push to `main`; tags `v*.*.*`; manual | Works. Trusted publishing to PyPI. |
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
- Between them the two files invoke seven scripts that are not in `scripts/`:
  `check_audit_severity.py`, `check_battery_run_trailer.py`,
  `check_pin_drift.py`, `cold_start_smoke.py`, `merge_battery_results.py`,
  `compare_battery_runs.py`. `build_db_fixtures.py` and
  `build_oracle_datasets.py` are the only two they name that are present.
- `v3-ci.yml` gates a coverage floor of 85 over `tests/v3`, which now holds one
  subdirectory.
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

`publish-to-pypi.yml` publishes on a `v*.*.*` tag through PyPI's trusted
publishing, which needs the publisher registered at
<https://pypi.org/manage/account/publishing/> against this repository and this
workflow filename. Trusted publishing carries no long-lived token, which is why
it is used.

## Before the next release

These are the known gaps, stated so a release does not walk into them:

1. `v3-ci.yml` and `v3-nightly.yml` need deleting or rewriting against the tree
   that exists, and whichever survives needs to trigger on the release branch.
2. The `Dockerfile` needs rewriting or `docker-publish.yml` needs disabling
   before a `v3.0.0` tag is pushed.
3. `docker-compose.yml`, the issue templates and the PR template describe 2.x.
