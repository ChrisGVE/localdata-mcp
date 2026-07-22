"""localdata_mcp/nexus/config/dsn_patterns.py — credential shapes, one home.

The DSN-credential shape knowledge (NFR-110) declared once: URL
userinfo passwords (`scheme://user:secret@host`) and keyword pairs
(`password=` / `pwd=`). Two consumers, two variants per shape:
endpoints.py refuses a DECLARED DSN that matches the anchored/marker
forms; observability/redaction.py (and NX-3's redact.py when E4 lands)
rewrites credential material found ANYWHERE in outbound text via the
search forms. Neighbors: endpoints.py and observability/redaction.py
import from here — a second declaration of these shapes anywhere else
is the drift this module exists to prevent.
"""

from __future__ import annotations

import re

# --- Refusal forms (endpoint declarations, endpoints.py) -------------

# URL userinfo carrying a password component: scheme://user:secret@host
URL_PASSWORD_AT_START = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*://[^/@]*:[^/@]+@")

# Keyword credential markers in non-URL DSNs (ODBC-style strings).
KEYWORD_CREDENTIAL_MARKER = re.compile(r"(?i)\b(password|pwd)\s*=")

# --- Search forms (outbound-text redaction) --------------------------

# The same URL shape found mid-text; group 1 keeps scheme://user so the
# redacted string stays diagnosable, group 2 is the secret.
URL_PASSWORD_SEARCH = re.compile(r"([a-zA-Z][a-zA-Z0-9+.-]*://[^/@\s]*):([^/@\s]+)@")

# The keyword pair with its value; group 2 preserves the separator.
KEYWORD_CREDENTIAL_SEARCH = re.compile(r"(?i)\b(password|pwd)(\s*=\s*)([^;,&\s'\"]+)")

# What a secret becomes on every outbound edge.
REDACTED = "[REDACTED]"


def redact_credentials_text(text: str) -> str:
    """`text` with every credential-shaped span replaced by REDACTED —
    the one transformation both NX-4 and (from E4) NX-3 apply before
    anything leaves their edge."""
    text = URL_PASSWORD_SEARCH.sub(rf"\1:{REDACTED}@", text)
    return KEYWORD_CREDENTIAL_SEARCH.sub(rf"\1\2{REDACTED}", text)
