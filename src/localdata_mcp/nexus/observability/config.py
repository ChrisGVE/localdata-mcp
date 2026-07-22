"""localdata_mcp/nexus/observability/config.py — the one logging setup.

The kept `logging_manager/config.py` shape with the T2 defect fixed
structurally (ARCHITECTURE.md section 7, logging row): the single
processor chain and the single stdlib handler both bind to stderr —
this module is the only place in the v3 tree where a StreamHandler may
be constructed, and fd 1 belongs exclusively to JSON-RPC frames
(NFR-303/304). Neighbors: manager.py drives the two-phase bring-up by
calling `configure()`; context.py's fields arrive via contextvars.
"""

from __future__ import annotations

import logging
import sys
from typing import TextIO

import structlog


class StderrHandler(logging.StreamHandler):
    """A StreamHandler pinned to the CURRENT `sys.stderr`, late-bound.

    Resolving the stream at emit time (the same design as stdlib's
    `logging._StderrHandler`) keeps every record on stderr even if the
    interpreter's `sys.stderr` object is later replaced — the fd-1
    guard rebinds `sys.stdout`, and test harnesses swap both.
    """

    def __init__(self, level: int = logging.NOTSET) -> None:
        # Skip StreamHandler.__init__: it would freeze one stream object.
        logging.Handler.__init__(self, level)

    @property
    def stream(self) -> TextIO:  # type: ignore[override]
        return sys.stderr


def configure(level: str) -> None:
    """(Re)build the whole logging path — structlog and stdlib — at
    `level`, bound to stderr. Idempotent: repeated calls replace, never
    stack, handlers. Unknown levels raise ValueError."""
    _configure_stdlib(_parse_level(level))
    _configure_structlog(debug=level.upper() == "DEBUG")


def _parse_level(level: str) -> int:
    """The numeric stdlib level for a symbolic name, strictly."""
    numeric = logging.getLevelName(level.upper())
    if not isinstance(numeric, int):
        raise ValueError(f"unknown log level {level!r}")
    return numeric


def _configure_stdlib(numeric_level: int) -> None:
    """One root handler of ours, on stderr — the T2 fix made structural.

    The legacy defect was `StreamHandler(sys.stdout)` here; v3 binds
    stderr and the static purity check (purity_check.py) fails the
    build on any stdout-shaped handler construction anywhere. Repeated
    calls replace this module's own handler only: foreign handlers
    (test harness capture, an embedder's setup) are not ours to remove
    — in-tree second handler paths are banned statically, and a rogue
    fd-1 writer is neutralized by the startup fd guard regardless.
    """
    root_logger = logging.getLogger()
    root_logger.handlers = [
        h for h in root_logger.handlers if not isinstance(h, StderrHandler)
    ]
    root_logger.setLevel(numeric_level)
    handler = StderrHandler(numeric_level)
    handler.setFormatter(logging.Formatter("%(message)s"))
    root_logger.addHandler(handler)


def _configure_structlog(*, debug: bool) -> None:
    """The single processor chain, ending in a renderer chosen by mode:
    human-readable console at DEBUG, JSON lines otherwise (the kept
    legacy convention)."""
    renderer: structlog.typing.Processor
    if debug:
        renderer = structlog.dev.ConsoleRenderer(colors=False)
    else:
        renderer = structlog.processors.JSONRenderer()
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        context_class=dict,
        # Caching would freeze a logger's chain at first use, defeating
        # the two-phase reconfigure (section 4e); left off deliberately.
        cache_logger_on_first_use=False,
    )
