"""``scripts/endpoint-run.sh`` under two concurrent invocations (issue #142).

Offline. ``docker`` on ``PATH`` is ``tests/fake_docker.py``, which models an
image store and a build cache and nothing more, so the script runs end to end
without a daemon and without touching this machine's Docker.

The script's promise to a second invocation is that it waits for the lock. The
defect was that waiting was not enough: both runs tagged their runner image
``localdata-test/runner:local``, built it before taking the lock, and whichever
finished first removed it, so the run still waiting brought its database up and
then failed at ``docker run`` with exit 125. And the end-of-run build cache pass
pruned every record "new since this run started", which included the other
run's.

So the property tested is ownership: a run removes only images and build cache
records it created itself, the runner it needs is there when it needs it, and
once both runs are done the machine holds exactly what it held before.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "endpoint-run.sh"
FAKE = Path(__file__).resolve().parent / "fake_docker.py"

# On this machine before either run starts, and nobody's to remove.
FOREIGN_IMAGE = "someone/else:1"
FOREIGN_RECORD = "rec-foreign"


def bash_4() -> bool:
    bash = shutil.which("bash")
    if bash is None:
        return False
    major = subprocess.run(
        [bash, "-c", "echo ${BASH_VERSINFO[0]}"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    return major.isdigit() and int(major) >= 4


pytestmark = pytest.mark.skipif(
    not bash_4(), reason="scripts/endpoint-run.sh needs bash 4+ on PATH"
)


@pytest.fixture(scope="module")
def machine(tmp_path_factory):
    """A PATH whose docker is the fake, and its state with one foreign image."""
    tmp_path = tmp_path_factory.mktemp("endpoint-run")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    docker = bin_dir / "docker"
    docker.write_text(f'#!/bin/sh\nexec "{sys.executable}" -I "{FAKE}" "$@"\n')
    docker.chmod(0o755)
    context = tmp_path / "context"
    context.mkdir()
    (context / "Dockerfile").write_text("FROM someone/else:1\n")
    state = tmp_path / "state"
    state.mkdir()
    (state / "state.json").write_text(
        json.dumps(
            {
                "images": {FOREIGN_IMAGE: "before"},
                "cache": {FOREIGN_RECORD: "before"},
                "containers": [],
            }
        )
    )
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    env = {
        **os.environ,
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "TMPDIR": str(scratch),
        "LOCALDATA_DOCKER_LOCK": str(tmp_path / "docker.lock"),
        "LOCALDATA_LOCK_POLL": "1",
        "FAKE_DOCKER_STATE": str(state),
        "FAKE_DOCKER_CONTEXT": str(context),
        "FAKE_DOCKER_TEST_SECONDS": "3",
    }
    return {"env": env, "state": state, "lock": tmp_path / "docker.lock"}


def start(machine, run: str, *services: str) -> subprocess.Popen:
    return subprocess.Popen(
        [str(SCRIPT), *services],
        env={**machine["env"], "FAKE_DOCKER_RUN": run},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def wait_for(condition, seconds: float = 30) -> None:
    deadline = time.monotonic() + seconds
    while not condition():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.05)


def events(machine) -> list[dict]:
    lines = (machine["state"] / "log.jsonl").read_text().splitlines()
    return [json.loads(line) for line in lines]


def final_state(machine) -> dict:
    return json.loads((machine["state"] / "state.json").read_text())


@pytest.fixture(scope="module")
def two_runs(machine):
    """Run A over two services; B starts while A holds the lock for the first.

    Once for the module: every test below reads the same two runs' record.
    """
    a = start(machine, "A", "postgres", "mysql")
    wait_for(lambda: machine["lock"].is_dir())
    b = start(machine, "B", "mssql")
    out_a, _ = a.communicate(timeout=120)
    out_b, _ = b.communicate(timeout=120)
    return {"A": (a.returncode, out_a), "B": (b.returncode, out_b)}


def test_both_runs_pass_and_neither_loses_its_runner(machine, two_runs):
    for run, (code, out) in two_runs.items():
        assert code == 0, f"run {run} failed:\n{out}"
    missing = [e for e in events(machine) if e["event"] == "run" and not e["present"]]
    assert missing == [], f"a run used an image that was gone: {missing}"


def test_each_run_builds_a_runner_of_its_own(machine, two_runs):
    runners = {
        e["run"]: e["image"]
        for e in events(machine)
        if e["event"] == "build" and e["image"].startswith("localdata-test/runner:")
    }
    assert set(runners) == {"A", "B"}
    assert runners["A"] != runners["B"]


def test_a_run_removes_only_what_it_created(machine, two_runs):
    removals = [e for e in events(machine) if e["event"] in ("rmi", "cache-prune")]
    assert removals, "no run removed anything"
    foreign = [e for e in removals if e["created_by"] not in (e["run"], None)]
    assert foreign == [], f"a run removed what it did not create: {foreign}"


def test_both_runs_leave_the_machine_as_they_found_it(machine, two_runs):
    state = final_state(machine)
    assert state["images"] == {FOREIGN_IMAGE: "before"}
    assert state["cache"] == {FOREIGN_RECORD: "before"}
    assert state["containers"] == []
    assert not machine["lock"].exists()
