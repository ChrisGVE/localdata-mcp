"""A stand-in for the ``docker`` CLI, for testing ``scripts/endpoint-run.sh`` offline.

``tests/test_endpoint_run.py`` puts a ``docker`` on ``PATH`` that runs this file,
so the script can be driven end to end, two invocations at once, without a
daemon. It models the parts of Docker the script's correctness depends on, and
nothing else:

* an **image store** keyed by reference. ``build -t`` and ``pull`` add to it,
  ``rmi`` removes from it, and ``run`` of a reference that is not in it fails
  the way the real CLI does, with exit 125 and "Unable to find image". That is
  the failure of issue #142.
* a **build cache**: every ``build`` and ``compose build`` adds a record, which
  ``buildx du`` lists and ``buildx prune --filter id=`` removes.
* one container per compose project, healthy as soon as it is up.

``run`` of ``python -c`` executes the code with the interpreter running this
file, in the repository, because that is how the script asks the harness
(``tests/endpoints.py``) which services exist. ``run`` of pytest sleeps for
``FAKE_DOCKER_TEST_SECONDS`` and reports one pass: long enough for a second
invocation to start while the first holds the lock.

State lives in ``FAKE_DOCKER_STATE``, a directory, under a file lock, because
two invocations of the script call this concurrently. Every call is appended to
``log.jsonl`` with ``FAKE_DOCKER_RUN``, the name the test gives each invocation,
so a test can say which run created, removed or pruned what.
"""

from __future__ import annotations

import fcntl
import json
import os
import subprocess
import sys
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

STATE = Path(os.environ["FAKE_DOCKER_STATE"])
RUN = os.environ.get("FAKE_DOCKER_RUN", "?")
REPO = Path(__file__).resolve().parent.parent

# docker run options that take a value, as the script uses them.
RUN_VALUE_OPTIONS = {"--network", "--user", "-e", "-v"}


@contextmanager
def state():
    """The state, read and written back under an exclusive lock."""
    STATE.mkdir(parents=True, exist_ok=True)
    with open(STATE / "lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = STATE / "state.json"
        data = (
            json.loads(path.read_text())
            if path.exists()
            else {"images": {}, "cache": {}, "containers": []}
        )
        yield data
        path.write_text(json.dumps(data))


def log(event: str, **fields: object) -> None:
    with state():  # the lock also serialises the log
        with open(STATE / "log.jsonl", "a") as out:
            out.write(json.dumps({"run": RUN, "event": event, **fields}) + "\n")


def add_cache_record(data: dict, origin: str) -> str:
    record = f"rec-{uuid.uuid4().hex[:12]}"
    data["cache"][record] = RUN
    log_later.append(("cache-add", {"record": record, "origin": origin}))
    return record


log_later: list[tuple[str, dict]] = []


def buildx(args: list[str]) -> int:
    if args[0] == "du":
        fmt = args[args.index("--format") + 1]
        with state() as data:
            records = list(data["cache"])
        for record in records:
            print(f"{record} false" if "{{.Shared}}" in fmt else record)
        return 0
    if args[0] == "prune":
        record = args[args.index("--filter") + 1].removeprefix("id=")
        with state() as data:
            owner = data["cache"].pop(record, None)
        log("cache-prune", record=record, created_by=owner)
        return 0
    raise SystemExit(f"fake docker: unmodelled buildx {args}")


def run(args: list[str]) -> int:
    i = 0
    while args[i].startswith("-"):
        i += 2 if args[i] in RUN_VALUE_OPTIONS else 1
    image, command = args[i], args[i + 1 :]
    with state() as data:
        present = image in data["images"]
    log("run", image=image, present=present)
    if not present:
        print(f"Unable to find image '{image}' locally", file=sys.stderr)
        return 125
    if command[:2] == ["python", "-c"]:
        return subprocess.run(
            [sys.executable, "-c", command[2]], cwd=REPO, check=False
        ).returncode
    if "pytest" in command:
        time.sleep(float(os.environ.get("FAKE_DOCKER_TEST_SECONDS", "0")))
        print("1 passed in 0.01s")
        return 0
    raise SystemExit(f"fake docker: unmodelled run {command}")


def compose(args: list[str]) -> int:
    project = args[args.index("-p") + 1]
    i = args.index("-p") + 2
    while args[i] == "-f":
        i += 2
    sub, rest = args[i], args[i + 1 :]
    if sub == "config":
        context = os.environ["FAKE_DOCKER_CONTEXT"]
        print(json.dumps({"services": {rest[-1]: {"build": {"context": context}}}}))
        return 0
    if sub == "build":
        with state() as data:
            data["images"][f"{project}-{rest[-1]}"] = RUN
            add_cache_record(data, f"compose build {project}")
    elif sub == "up":
        with state() as data:
            data["containers"].append(project)
    elif sub == "ps":
        print(f"cid-{project}")
    elif sub == "port":
        print("127.0.0.1:1")
    elif sub == "down":
        with state() as data:
            for image in [i for i in data["images"] if i.startswith(f"{project}-")]:
                del data["images"][image]
            data["containers"] = [c for c in data["containers"] if c != project]
    else:
        raise SystemExit(f"fake docker: unmodelled compose {sub}")
    return 0


def main(args: list[str]) -> int:
    verb, rest = args[0], args[1:]
    if verb == "buildx":
        return buildx(rest)
    if verb == "compose":
        status = compose(rest)
    elif verb == "image" and rest[0] == "inspect":
        with state() as data:
            return 0 if rest[1] in data["images"] else 1
    elif verb == "pull":
        with state() as data:
            data["images"][rest[-1]] = RUN
        log("pull", image=rest[-1])
        return 0
    elif verb == "build":
        tag = rest[rest.index("-t") + 1]
        with state() as data:
            data["images"][tag] = RUN
            add_cache_record(data, f"build {tag}")
        log("build", image=tag)
        status = 0
    elif verb == "rmi":
        with state() as data:
            owner = data["images"].pop(rest[0], None)
        log("rmi", image=rest[0], created_by=owner)
        return 0 if owner is not None else 1
    elif verb == "run":
        return run(rest)
    elif verb == "inspect":
        print("healthy" if "Health" in rest[1] else "running")
        return 0
    elif verb == "logs":
        return 0
    else:
        raise SystemExit(f"fake docker: unmodelled {args}")
    for event, fields in log_later:
        log(event, **fields)
    return status


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
