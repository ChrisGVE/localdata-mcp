"""localdata_mcp/nexus/error/redact.py — NX-3's outbound-edge scrub.

§4b's invariant, NX-3 side: no DSN-shaped string crosses the nexus
boundary un-redacted, on ANY outbound edge. The wire shape's two text
fields (`message`, `suggestion`) pass through here at wire-shaping
time — the credential SHAPES have one home (nexus/config/dsn_patterns,
NFR-110) and this module only applies them to the model. NX-4's
processor covers the log edge; this covers the LLM-bound one.
Neighbors: wire.py calls redact_structured() as the last step before
handing the shape to NX-7.
"""

from __future__ import annotations

from dataclasses import replace

from localdata_mcp.nexus.config.dsn_patterns import redact_credentials_text
from localdata_mcp.nexus.error.model import StructuredError


def redact_structured(error: StructuredError) -> StructuredError:
    """`error` with every credential-shaped span in its text fields
    replaced — value-identical otherwise."""
    return replace(
        error,
        message=redact_credentials_text(error.message),
        suggestion=redact_credentials_text(error.suggestion),
    )
