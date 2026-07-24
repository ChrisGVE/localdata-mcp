"""NX-3 error-taxonomy feeder remnant (post-E15).

The only surviving member of the former v2 error-handling package is the
declared *mapping knowledge* the v3 error nexus keeps unchanged (PRD S4.3,
NFR-302): ``ErrorCategory``, consumed by ``error_mappers.py`` /
``error_classification.py`` and translated onto the closed wire taxonomy
in ``nexus/error/translate.py`` (the tree's ONE importer of this feeder).
E15 deleted the legacy machinery (circuit breaker, retry, recovery,
handler, error logging); the dead v2 exception hierarchy and the
retry/circuit enums went with the E16 audit cleanup (CR-014). What is
left is pure stdlib and imports unconditionally.
"""

from .exceptions import ErrorCategory

__all__ = [
    "ErrorCategory",
]
