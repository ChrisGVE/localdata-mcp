"""Which endpoint targets a run reached, measured against the catalogue (issue #46).

Offline. A skipped endpoint test reads like a passing one, and a run against one
container used to end on a green line that said nothing about the other
twenty-three targets. Two surfaces now name the gap, and both read it from the
same place:

* **the catalogue** is :data:`endpoints.TARGETS`, every endpoint once per way of
  reaching it. Nothing here keeps a second list;
* **what a run reached** is the probe cache, :data:`endpoints._probed`, which
  holds one answer per target that any test asked for.

``pytest`` ends every session that selected an endpoint test with a section
naming each target and why it was or was not reached (``conftest.py``), and
``scripts/endpoint-run.sh`` collects the reached line of each service's run and
names every target the whole invocation did not reach.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import endpoints
import pytest
from endpoints import (
    TARGETS,
    DriverMissing,
    Endpoint,
    NotListening,
    Reached,
    Target,
    Unavailable,
)
from test_endpoint_run import SCRIPT, bash_4, machine  # noqa: F401 - a fixture

REPO = Path(__file__).resolve().parent.parent
NAMES = [target.name for target in TARGETS]


def _probed() -> dict[str, Reached | Exception]:
    """One answer of every kind; every other target was never asked for."""
    return {
        "postgres": Reached(url="postgresql+psycopg://u:p@h/db"),
        "mysql": NotListening("Nothing is listening on 127.0.0.1:3306.\nmore"),
        "oracle": DriverMissing("oracle needs the oracledb driver"),
        "mssql": Unavailable("no ODBC driver for SQL Server"),
        "clickhouse": RuntimeError("A container is listening but never completed"),
    }


# ---------------------------------------------------------------------------
# What each target's answer means
# ---------------------------------------------------------------------------


def test_every_catalogue_target_is_accounted_for_once():
    assert [row.target for row in endpoints.coverage(_probed())] == NAMES


@pytest.mark.parametrize(
    ("name", "status", "reason"),
    [
        ("postgres", "reached", ""),
        ("mysql", "no container", "Nothing is listening on 127.0.0.1:3306."),
        ("oracle", "no driver", "oracle needs the oracledb driver"),
        ("mssql", "unavailable", "no ODBC driver for SQL Server"),
        ("clickhouse", "unreachable", "A container is listening but never completed"),
        ("mariadb", "not selected", ""),
        ("postgres[kerberos]", "not selected", ""),
    ],
)
def test_each_answer_is_classified(name, status, reason):
    rows = {row.target: row for row in endpoints.coverage(_probed())}
    assert (rows[name].status, rows[name].reason) == (status, reason)


def test_a_row_carries_the_service_that_serves_its_target():
    rows = {row.target: row for row in endpoints.coverage({})}
    assert rows["postgres[tls-verify-full]"].service == "localdata-test-postgres-tls"
    assert rows["postgres[env-password]"].service == "localdata-test-postgres"


# ---------------------------------------------------------------------------
# The pytest terminal summary
# ---------------------------------------------------------------------------


def test_the_report_counts_against_the_catalogue():
    lines = endpoints.coverage_report(_probed())
    assert lines[0] == (
        f"1 of {len(TARGETS)} endpoint targets reached "
        f"(the catalogue is TARGETS in tests/endpoints.py)"
    )


def test_the_report_names_what_was_reached_on_one_machine_readable_line():
    lines = endpoints.coverage_report(_probed())
    assert f"{endpoints.REACHED_LINE} postgres" in lines


def test_the_report_says_none_when_nothing_was_reached():
    assert f"{endpoints.REACHED_LINE} none" in endpoints.coverage_report({})


def test_the_report_names_every_target_not_reached_and_why():
    lines = endpoints.coverage_report(_probed())
    assert "  mysql: no container — Nothing is listening on 127.0.0.1:3306." in lines
    assert (
        "  clickhouse: unreachable — A container is listening but never completed"
        in lines
    )
    assert "  mariadb: not selected — no test in this run asked for it" in lines
    named = [line.split(":")[0].strip() for line in lines if line.startswith("  ")]
    assert sorted(named) == sorted(set(NAMES) - {"postgres"})


# ---------------------------------------------------------------------------
# The probe raises the kinds the summary tells apart
# ---------------------------------------------------------------------------


def _fake_target(driver: str) -> Target:
    real = TARGETS[0].endpoint
    return Target(
        Endpoint(
            dialect=real.dialect,
            service=real.service,
            container_port=real.container_port,
            driver=driver,
            extra="none",
            url=real.url,
        )
    )


def test_nothing_listening_is_raised_as_no_container(monkeypatch):
    monkeypatch.setattr(endpoints, "_listening", lambda host, port: False)
    with pytest.raises(NotListening, match="Nothing is listening on"):
        endpoints._probe(_fake_target("json"))
    assert issubclass(NotListening, Unavailable)


def test_a_missing_driver_is_raised_as_no_driver():
    with pytest.raises(DriverMissing, match="needs the no_such_driver_46 driver"):
        endpoints._probe(_fake_target("no_such_driver_46"))
    assert issubclass(DriverMissing, Unavailable)


# ---------------------------------------------------------------------------
# What a scripts/endpoint-run.sh invocation covered
# ---------------------------------------------------------------------------


def test_the_run_report_tells_not_run_from_run_and_not_reached():
    lines = endpoints.run_report(
        "postgres,postgres[env-password]", "localdata-test-postgres"
    )
    assert lines[0] == (
        f"2 of {len(TARGETS)} endpoint targets reached by this run "
        f"(the catalogue is TARGETS in tests/endpoints.py)"
    )
    assert "Reached: postgres, postgres[env-password]" in lines
    assert (
        "  postgres[pgpass-file]: its service ran, but it was not reached; "
        "its skip or failure is in localdata-test-postgres's output above"
    ) in lines
    assert (
        "  postgres[trust]: not run — scripts/endpoint-run.sh postgres-trust" in lines
    )
    assert "  mysql: not run — scripts/endpoint-run.sh mysql" in lines
    named = [line.split(":")[0].strip() for line in lines if line.startswith("  ")]
    assert sorted(named) == sorted(set(NAMES) - {"postgres", "postgres[env-password]"})


def test_the_run_report_reads_several_services_reached_lines():
    lines = endpoints.run_report(
        "none,mysql,,mysql[option-file]",
        "localdata-test-postgres localdata-test-mysql",
    )
    assert "Reached: mysql, mysql[option-file]" in lines
    assert lines[0].startswith("2 of ")


def test_the_run_report_says_none_when_nothing_was_reached():
    lines = endpoints.run_report("", "")
    assert "Reached: none" in lines
    assert lines[0].startswith(f"0 of {len(TARGETS)} ")


def test_the_run_report_refuses_a_name_outside_the_catalogue():
    with pytest.raises(ValueError, match="no_such_target"):
        endpoints.run_report("postgres,no_such_target", "localdata-test-postgres")


# ---------------------------------------------------------------------------
# End to end: a pytest session, and the script
# ---------------------------------------------------------------------------


def _pytest(*args: str) -> subprocess.CompletedProcess:
    # In-network addressing from the host: the service names do not resolve
    # here, so every probe is "nothing listening" whatever containers this
    # machine happens to have up. That makes the outcome independent of it.
    env = {**os.environ, endpoints.IN_NETWORK: "1"}
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", *args],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_a_session_with_endpoint_tests_ends_naming_the_targets_it_did_not_reach():
    result = _pytest(
        "-q", "-m", "endpoint", "-k", "postgres", "tests/test_stats_aggregates.py"
    )
    out = result.stdout
    assert result.returncode == 0, out + result.stderr
    assert "endpoint coverage" in out
    assert f"0 of {len(TARGETS)} endpoint targets reached" in out
    assert f"{endpoints.REACHED_LINE} none" in out
    assert (
        "  postgres: no container — Nothing is listening on localdata-test-postgres:5432."
        in out
    )
    assert "  mysql: not selected — no test in this run asked for it" in out


def test_a_session_without_endpoint_tests_prints_no_coverage():
    result = _pytest("-q", "tests/test_names.py")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "endpoint coverage" not in result.stdout


@pytest.mark.skipif(
    not bash_4(), reason="scripts/endpoint-run.sh needs bash 4+ on PATH"
)
def test_the_script_summary_names_what_the_run_did_not_reach(machine):  # noqa: F811
    reached = f"{endpoints.REACHED_LINE} postgres, postgres[env-password]"
    result = subprocess.run(
        [str(SCRIPT), "postgres"],
        env={
            **machine["env"],
            "FAKE_DOCKER_RUN": "coverage",
            "FAKE_DOCKER_TEST_SECONDS": "0",
            "FAKE_DOCKER_PYTEST_OUTPUT": reached,
        },
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    out = result.stdout
    assert result.returncode == 0, out + result.stderr
    assert "Reached: postgres, postgres[env-password]" in out
    assert "  mysql: not run — scripts/endpoint-run.sh mysql" in out
    assert "  postgres[pgpass-file]: its service ran, but it was not reached" in out
    assert "every one not listed above" not in out
