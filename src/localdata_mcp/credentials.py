"""Taking a password out of text this server hands back, and nothing else.

Every message that leaves the server goes to the caller, and in an MCP
deployment that means into a conversation transcript and whatever logs sit
behind it. A database URL carries its password inline, so any message that
quotes one — the server's own refusal, or a driver's complaint about a failed
connection — would carry it too. This module is the one place that removes it.

**A credential is recognised by where it sits, not by how it is spelled.** The
first version replaced every occurrence of the password's characters, so a
short password rewrote the message around it: password ``e`` turned
"connection refused" into "conn***ction r***fus***d", and the shape of the
damage told the reader the password's length and letters (issue #82). A
password sits in exactly two kinds of place in text, and those are what get
masked:

* **A URL's userinfo** — ``scheme://user:SECRET@host``. Recognised in any text,
  whether or not the URL is known, because the shape alone says it is a
  credential. The scheme may be followed by a single slash: a URL read as a
  filesystem path has its ``//`` collapsed before any message is written, and
  ``postgresql:/u:hunter2@host`` is the same credential (issue #88).
* **A value under a password key** — ``password='e'``, ``"password": "e"``,
  ``PWD=e;``, the forms drivers use when they echo the arguments they were
  called with. Masked only in a complaint about connecting, because elsewhere
  ``password = 'x'`` is more likely a caller's own SQL than a secret, and
  rewriting it would be the corruption this module exists to prevent.

When the URL being connected to is known, its password is also masked in the
spellings a driver may use for it — decoded where the URL percent-encodes it,
or holding a space the generic shapes stop at — but still only in those two
positions, never as a bare substring.
"""

from __future__ import annotations

import re

from sqlalchemy.engine import URL

__all__ = ["MASK", "mask_credentials", "redacted_url"]

#: What a password is replaced with. SQLAlchemy renders a hidden password the
#: same way, so a URL masked here and one rendered by ``redacted_url`` agree.
MASK = "***"

#: A URL's userinfo with a password in it. The scheme is two characters or more,
#: so a Windows drive letter is never one. The password stops at the first ``@``,
#: which is where SQLAlchemy's own parser stops it, and at whitespace, so a
#: password-free URL followed by prose with an ``@`` in it is not swallowed.
_USERINFO = re.compile(
    r"(?P<head>\b[A-Za-z][A-Za-z0-9+.\-]+:/+[^\s:/@]*:)(?P<secret>[^\s@]*)@"
)

#: The keys drivers file a password under when they echo their arguments:
#: libpq's ``password=``, ODBC's ``PWD=``, a kwargs or dict repr's ``password``.
_KEY = r"\b(?:password|passwd|pwd)\b['\"]?\s*[=:]\s*"

#: What a bare value under one of those keys runs to: up to whatever separates
#: one argument from the next in each of the forms drivers echo.
_BARE = r"[^\s;,&)}'\"]"

#: A value under one of those keys, quoted either way or bare.
_KEYED = re.compile(
    rf"(?i)(?P<key>{_KEY})"
    r"(?:'(?P<single>(?:[^'\\]|\\.)*)'|\"(?P<double>(?:[^\"\\]|\\.)*)\""
    rf"|(?P<bare>{_BARE}+))"
)


def redacted_url(url: URL) -> str:
    """The URL as it may be shown: its password replaced, if it had one.

    A URL with no password renders with none rather than with a mask, since a
    mask there would tell the reader a secret was carried when none was.
    """
    return url.render_as_string(hide_password=True)


def mask_credentials(text: str, *, connecting_to: URL | None = None) -> str:
    """``text`` with every password in a credential position masked.

    ``connecting_to`` says the text is a complaint about connecting to that URL —
    a driver's error, typically. It turns on the password-key forms drivers
    echo, and lets the URL's own password be found in spellings the generic
    shapes would miss. Without it only URL userinfo is masked.

    Masking twice changes nothing, so a message that passes through more than
    one layer on its way out is not damaged by the second.
    """
    if connecting_to is not None and connecting_to.password:
        text = _mask_known(text, connecting_to)
    text = _USERINFO.sub(lambda m: f"{m['head']}{MASK}@", text)
    if connecting_to is not None:
        text = _KEYED.sub(_masked_value, text)
    return text


def _mask_known(text: str, url: URL) -> str:
    """Mask this URL's password where it sits as a credential, however spelled.

    The rendered URL percent-encodes the password, and a driver may quote
    either that or the decoded value. The decoded value can hold the very
    characters the generic shapes stop at — ``@``, ``/``, a space — so it is
    found here by its exact text, but still only in a userinfo or key position.
    """
    secret = re.escape(str(url.password))
    text = text.replace(url.render_as_string(hide_password=False), redacted_url(url))
    userinfo = re.compile(rf"(?P<head>:/+{re.escape(url.username or '')}:){secret}@")
    text = userinfo.sub(lambda m: f"{m['head']}{MASK}@", text)
    # Quoted, or bare and ending where a bare value ends — so the password `e`
    # is found in `password=e;` and not at the front of `password=every`.
    keyed = re.compile(
        rf"(?i)(?P<key>{_KEY})"
        rf"(?:(?P<quote>['\"]){secret}(?P=quote)|{secret}(?!{_BARE}))"
    )
    return keyed.sub(_masked_known, text)


def _masked_known(match: re.Match[str]) -> str:
    quote = match["quote"] or ""
    return f"{match['key']}{quote}{MASK}{quote}"


def _masked_value(match: re.Match[str]) -> str:
    """The key as written, its value masked, its quoting kept."""
    if match["single"] is not None:
        return f"{match['key']}'{MASK}'"
    if match["double"] is not None:
        return f'{match["key"]}"{MASK}"'
    return f"{match['key']}{MASK}"
