"""Whether the installed SQLAlchemy can carry a URL's database name (#140).

File: ``src/localdata_mcp/urls.py``. Called by :mod:`.slots` at the two places
a URL a caller typed is parsed: ``attach`` of a server or file URL, and
``attach`` of a ``sqlite:`` URL. A path the caller hands over without a URL
never comes here: the backends build a ``URL`` object from it and never render
or re-parse it, so it reaches the driver intact on every SQLAlchemy version.

**What differs between the versions.** SQLAlchemy 2.1 unquotes a URL's database
component when it parses one, so ``postgresql://…/a%3Fb`` names the database
``a?b``. 2.0 does not: it hands ``a%3Fb`` to the driver as it stands, which is a
different database. A raw ``?`` cannot be used either, because it starts the
query string and the name is cut there. So on 2.0 a ``?`` or ``@`` in a database
name has no spelling that arrives intact, and an encoded one arrives as some
other name with nothing said (sqlalchemy/sqlalchemy#11234, fixed in 2.1).

**Why 2.0 is still a version this server runs on.** The project's own lock is
2.1.6, through ``[tool.uv] override-dependencies``. That override never reaches
the published metadata, and ``sqlalchemy-cockroachdb`` 2.0.4 declares
``SQLAlchemy<2.1``, so ``pip install localdata-mcp[cockroachdb]`` resolves to
2.0.55. There the name is refused, with the reason and the remedy, rather than
opened as something else. On 2.1 or later nothing here refuses anything.

Retired together with the override once the CockroachDB dialect supports 2.1.
"""

from __future__ import annotations

import re
from urllib.parse import unquote

import sqlalchemy
from sqlalchemy.engine import URL

from .credentials import redacted_url

__all__ = [
    "CARRYING_RELEASE",
    "UNCARRIED_CHARACTERS",
    "DatabaseNameNotCarried",
    "installed_sqlalchemy",
    "refuse_uncarried_database",
]

#: The first SQLAlchemy release that quotes the database component on the way
#: out and unquotes it on the way back, as (major, minor).
CARRYING_RELEASE = (2, 1)

#: The characters a database name cannot hold on a release before that one.
UNCARRIED_CHARACTERS = ("?", "@")

#: The leading ``major.minor`` of a version string. A pre-release such as
#: ``2.1.0b3`` counts as its release: 2.1.0b3 was measured to round-trip both
#: characters (#70).
_RELEASE = re.compile(r"^(\d+)\.(\d+)")


class DatabaseNameNotCarried(ValueError):
    """A URL whose database name the installed SQLAlchemy would misread."""


def installed_sqlalchemy() -> str:
    """The version of the SQLAlchemy this process imported."""
    return sqlalchemy.__version__


def refuse_uncarried_database(url: URL, *, installed: str | None = None) -> None:
    """Refuse ``url`` if its database name cannot reach the driver intact.

    ``installed`` is the SQLAlchemy version to judge by, read from the imported
    package when not given.

    The name is the one the URL denotes, which is the name 2.1 reads out of it.
    2.1 has already decoded the component by the time it is a ``URL``; 2.0 has
    not, so it is decoded here. Either way the same name is judged, so a URL is
    refused on 2.0 for exactly the name that 2.1 would open.

    **Fail-closed, and deliberately wider than the divergence.** A raw ``@``
    after a password (``u:pw@host/a@b``) is read the same on both versions, and
    is refused here all the same: the rule is one a caller can predict from the
    name alone, and the remedy works for every case it covers.
    """
    version = installed if installed is not None else installed_sqlalchemy()
    if _release(version) >= CARRYING_RELEASE or not url.database:
        return

    name = unquote(url.database)
    found = [c for c in UNCARRIED_CHARACTERS if c in name]
    if not found:
        return

    raise DatabaseNameNotCarried(_explain(url, name, found, version))


def _release(version: str) -> tuple[int, int]:
    """``version`` as (major, minor), compared as numbers and never as text."""
    match = _RELEASE.match(version)
    if match is None:
        raise ValueError(f"SQLAlchemy reports a version this cannot read: {version!r}")
    return int(match.group(1)), int(match.group(2))


def _explain(url: URL, name: str, found: list[str], version: str) -> str:
    """The refusal: what was refused, why, and what to do instead."""
    characters = " and ".join(repr(c) for c in found)
    remedy = (
        "Install SQLAlchemy 2.1 or later (pip install 'sqlalchemy>=2.1'). Note "
        "that the cockroachdb extra holds SQLAlchemy below 2.1, because "
        "sqlalchemy-cockroachdb 2.0.4 declares SQLAlchemy<2.1."
    )
    if url.host is None:
        # A file reached by URL can always be reached by its path instead.
        remedy += (
            " Or attach the file by its path rather than as a URL: a path "
            "carries any file name on this version."
        )
    return (
        f"Refused {redacted_url(url)}: its database name {name!r} contains "
        f"{characters}, and SQLAlchemy {version}, the version installed here, "
        f"cannot carry a '?' or an '@' in a database name. It passes an encoded "
        f"one on to the driver undecoded, cuts a raw '?' off as the start of the "
        f"query, and reads a raw '@' as the end of the user name when a port "
        f"follows, so the database this URL opens is not reliably the one it "
        f"names. {remedy}"
    )
