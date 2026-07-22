"""localdata_mcp/nexus/observability — NX-4, the logging nexus.

All log output flows through this package's one structlog
configuration, bound to stderr (fd 1 belongs exclusively to JSON-RPC
frames — NFR-303/304, ARCHITECTURE.md section 8). Importing the
package activates the stderr-only bootstrap phase immediately, before
any config is read; server/mcp_app.py later calls `reconfigure()` with
operator settings (section 4e's two-phase bring-up). Built on the kept
`logging_manager/` shape: config.py, context.py, manager.py
(metrics.py deliberately not carried).
"""

from .context import LogContext
from .manager import (
    bootstrap,
    get_logger,
    log_context,
    logging_phase,
    reconfigure,
)

# Section 4e: stderr-only logging is active from first import — any
# component that can log can only ever log to stderr, config or none.
bootstrap()

__all__ = [
    "LogContext",
    "bootstrap",
    "get_logger",
    "log_context",
    "logging_phase",
    "reconfigure",
]
