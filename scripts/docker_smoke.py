"""Prove a built image answers MCP over stdio before it is published (#76).

``docker build`` succeeds on an image that cannot start, and that is how a
broken v2 image came within one tag of Docker Hub. This runs the image the
way a client does — ``docker run -i`` — sends ``initialize`` and
``tools/list``, and fails unless the server answers both, reports the
expected version, and lists at least one tool.

Usage: python scripts/docker_smoke.py <image> <expected-version>
"""

from __future__ import annotations

import json
import subprocess
import sys

from packaging.version import Version

TIMEOUT_SECONDS = 60

REQUESTS = [
    {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-06-18",
            "capabilities": {},
            "clientInfo": {"name": "docker-smoke", "version": "1"},
        },
    },
    {"jsonrpc": "2.0", "method": "notifications/initialized"},
    {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
]


def main(image: str, expected_version: str) -> int:
    stdin = "".join(json.dumps(r) + "\n" for r in REQUESTS)
    run = subprocess.run(
        ["docker", "run", "--rm", "-i", image],
        input=stdin,
        capture_output=True,
        text=True,
        timeout=TIMEOUT_SECONDS,
    )
    answers = {}
    for line in run.stdout.splitlines():
        message = json.loads(line)
        if "id" in message:
            answers[message["id"]] = message
    if 1 not in answers or 2 not in answers:
        print(f"no answer to initialize and tools/list; stderr:\n{run.stderr}")
        return 1
    reported = answers[1]["result"]["serverInfo"]["version"]
    tools = [t["name"] for t in answers[2]["result"]["tools"]]
    print(f"server version {reported}; tools: {', '.join(tools)}")
    if Version(reported) != Version(expected_version):
        print(f"expected version {expected_version}")
        return 1
    if not tools:
        print("the server lists no tools")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
