"""Load the standard test assets into a running endpoint, for an ad-hoc session.

    python scripts/seed_endpoint.py <service>

``<service>`` is a service in docker-compose.test.yml, with or without its
``localdata-test-`` prefix. The usual way to get one up and seeded is
``scripts/endpoint-run.sh --seed --keep <service>``, which runs this inside the
runner container on the service's own network (task 35, #139).

**The URL is the one tests/endpoints.py builds**, for the service's own
credentialed target: the same address, credentials and authentication mode the
endpoint suite uses, read out of the same compose file. A second way of building
it here would be a second place for it to be wrong.

**The tables are made by the server's own verbs**: ``attach`` with
``writable=true``, then one ``create`` per asset. That is how the suite lands
its data, and the compose file's header says why: a fixture that built the
tables another way would test the fixture.

**The assets are existing fixtures**, from tests/assets, not data invented here.
They are the three clean delimited files, which between them cover text,
integers, decimals, dates and booleans, and a thousand rows for anything that
wants volume. ``messy_mixed_types.csv`` is left out on purpose. It is a loader
stress fixture with an unquoted line break and a broken quote, so it tests the
file reader, not the database, and would not be clean data for an ad-hoc
session.

Seeding is idempotent. A table that is already there is left as it is and
reported, rather than replaced: by the time a seed is run again, the table may
hold the session's own work.

Exit status: 0 when every asset is there afterwards; 1 when the endpoint cannot
be reached or a ``create`` is refused; 2 for a usage error.
"""

from __future__ import annotations

import asyncio
import json
import sys
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ASSET_DIR = REPO / "tests" / "assets"
# tests/endpoints.py is the harness's own module, not part of the package, so its
# directory goes on the path before anything below is imported.
sys.path.insert(0, str(REPO / "tests"))

import endpoints
from fastmcp import Client

from localdata_mcp import config as config_module
from localdata_mcp import server as server_module
from localdata_mcp.config import Config

#: The nickname the endpoint is attached under while seeding.
NICKNAME = "seed"


@dataclass(frozen=True)
class Asset:
    """One file from tests/assets, the separator it is written with, and its table."""

    file: str
    delimiter: str

    @property
    def table(self) -> str:
        return Path(self.file).stem


#: What a seed loads, in order.
ASSETS = (
    Asset("simple.csv", ","),
    Asset("semicolons.csv", ";"),
    Asset("large_dataset.csv", ","),
)


@dataclass(frozen=True)
class Landed:
    """What a seed did with one asset."""

    table: str
    rows: int | None
    #: False when the table was already there and was left alone.
    created: bool


class SeedError(Exception):
    """The seed could not complete, and why, in the server's own words."""


def _call(tool: str, **arguments) -> dict:
    """One tool call over the protocol, as an agent would make it."""

    async def _run():
        async with Client(server_module.mcp) as client:
            result = await client.call_tool(tool, arguments)
            if result.structured_content is not None:
                return result.structured_content
            return json.loads(result.content[0].text)

    return asyncio.run(_run())


def seed(
    database: str,
    *,
    allowed: tuple[Path, ...] = (),
    assets: tuple[Asset, ...] = ASSETS,
) -> list[Landed]:
    """Attach ``database`` writable and land every asset in it that is not there.

    ``database`` is anything ``attach`` takes: an endpoint URL in real use, a
    database file in the offline tests. ``allowed`` adds roots the server may
    open beyond the asset directory, which only a file needs. Network access is
    granted, because an endpoint URL is exactly what that gate is about.
    """
    config_module.use(Config(roots=(ASSET_DIR, *allowed), network_enabled=True))
    server_module._reset()
    try:
        attached = _call("attach", database=database, nickname=NICKNAME, writable=True)
        if not attached.get("ok"):
            raise SeedError(f"attach was refused: {attached.get('error')}")
        nickname = attached["nickname"]
        present = set(_call("directory", nickname=nickname).get("tables", ()))

        landed = []
        for asset in assets:
            if asset.table in present:
                landed.append(Landed(asset.table, rows=None, created=False))
                continue
            made = _call(
                "create",
                nickname=nickname,
                type="table",
                source=str(ASSET_DIR / asset.file),
                table=asset.table,
                delimiter=asset.delimiter,
            )
            if not made.get("ok"):
                raise SeedError(
                    f"create {asset.table} was refused: {made.get('error')}"
                )
            landed.append(Landed(made["table"], rows=made.get("rows"), created=True))
        return landed
    finally:
        server_module._reset()


def target_for(service: str) -> endpoints.Target:
    """The target whose URL a seed of ``service`` uses.

    The endpoint's own credentialed target for a database service. For an
    authentication service it is the first mode that service serves, because
    the endpoint's own container is not the one up.
    """
    if not service.startswith("localdata-test-"):
        service = f"localdata-test-{service}"
    for target in endpoints.TARGETS:
        if target.service == service:
            return target
    raise SeedError(
        f"{service} has no target in tests/endpoints.py: it is a helper with no "
        f"tests of its own, or not a service at all."
    )


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print(__doc__.split("\n\n")[1], file=sys.stderr)
        return 2
    try:
        target = target_for(argv[0])
        reached = endpoints.reach(target)
        with endpoints.applied(reached.environ):
            landed = seed(reached.url)
    except (SeedError, endpoints.Unavailable, RuntimeError) as exc:
        print(f"seed: {exc}", file=sys.stderr)
        return 1
    print(f"seeded {target.name} (as nickname {NICKNAME!r} while seeding):")
    for result in landed:
        if result.created:
            print(f"  {result.table}: {result.rows} rows")
        else:
            print(f"  {result.table}: already there, left as it is")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
