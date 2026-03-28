"""Pre-flight query estimation using database EXPLAIN plans."""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ExplainResult:
    """Parsed EXPLAIN plan result."""

    estimated_rows: Optional[int] = None
    estimated_cost: Optional[float] = None
    scan_type: Optional[str] = None
    columns: List[str] = field(default_factory=list)
    confidence: float = 0.3
    raw_plan: Optional[str] = None


@dataclass
class PreflightResult:
    """Pre-flight estimation result for API response."""

    preflight: bool = True
    estimated_rows: Optional[int] = None
    estimated_size_bytes: Optional[int] = None
    estimated_size_mb: Optional[float] = None
    columns: List[str] = field(default_factory=list)
    scan_type: Optional[str] = None
    confidence: float = 0.0
    suggestion: Optional[str] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response."""
        d: Dict[str, Any] = {
            "preflight": self.preflight,
            "estimated_rows": self.estimated_rows,
            "estimated_size_mb": self.estimated_size_mb,
            "columns": self.columns,
            "scan_type": self.scan_type,
            "confidence": self.confidence,
        }
        if self.suggestion:
            d["suggestion"] = self.suggestion
        if self.error:
            d["error"] = self.error
            d["fallback"] = "EXPLAIN not available, consider adding LIMIT"
        return d


_PARSER_REGISTRY: Dict[str, Any] = {}


def register_parser(dialect: str, parser_fn: Any) -> None:
    """Register an EXPLAIN parser function for a dialect."""
    _PARSER_REGISTRY[dialect] = parser_fn


def get_parser(dialect: str) -> Any:
    """Get EXPLAIN parser for a dialect."""
    return _PARSER_REGISTRY.get(dialect)


def run_explain(engine: Any, query: str) -> Optional[ExplainResult]:
    """Run EXPLAIN using the appropriate parser for the engine's dialect."""
    dialect = engine.dialect.name if engine else None
    parser = get_parser(dialect) if dialect else None
    if not parser:
        return None
    try:
        raw = parser(engine, query)
        if raw is None:
            return None
        return ExplainResult(
            estimated_rows=raw.get("estimated_rows"),
            estimated_cost=raw.get("total_cost"),
            scan_type=raw.get("scan_type"),
            confidence=raw.get("confidence", 0.3),
            raw_plan=raw.get("raw_plan"),
        )
    except Exception as e:
        logger.debug("EXPLAIN failed for %s: %s", dialect, e)
        return None


def generate_suggestion(
    estimated_rows: Optional[int],
    estimated_size_mb: Optional[float] = None,
) -> str:
    """Generate LLM-friendly suggestion based on estimates."""
    if estimated_rows is None:
        return "Row count unknown. Consider adding LIMIT for safety."
    if estimated_rows < 1000:
        return "Safe to execute directly."
    if estimated_rows < 100000:
        return "Medium result set. Streaming will be used automatically."
    return (
        f"Large result ({estimated_rows:,} rows). Add LIMIT or WHERE clause to reduce."
    )


def _init_parsers() -> None:
    """Register existing parsers from _explain_parsers module."""
    from ._explain_parsers import (
        parse_explain_mysql,
        parse_explain_postgresql,
        parse_explain_sqlite,
    )

    register_parser("sqlite", parse_explain_sqlite)
    register_parser("postgresql", parse_explain_postgresql)
    register_parser("mysql", parse_explain_mysql)
    try:
        from ._explain_parsers import parse_explain_oracle

        register_parser("oracle", parse_explain_oracle)
    except ImportError:
        pass
    try:
        from ._explain_parsers import parse_explain_mssql

        register_parser("mssql", parse_explain_mssql)
    except ImportError:
        pass


_init_parsers()
