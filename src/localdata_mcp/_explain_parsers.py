"""EXPLAIN query plan parsers for database-specific size estimation.

Provides parsers for SQLite, PostgreSQL, and MySQL EXPLAIN output,
returning normalized result dictionaries for the SizeEstimator.
"""

from typing import Any, Dict, Optional


def parse_explain_sqlite(engine: Any, query: str) -> Optional[Dict[str, Any]]:
    """Parse SQLite EXPLAIN QUERY PLAN output."""
    try:
        from sqlalchemy import text

        with engine.connect() as conn:
            result = conn.execute(text(f"EXPLAIN QUERY PLAN {query}"))
            rows = result.fetchall()
            has_index = any(
                "USING INDEX" in str(r) or "USING COVERING INDEX" in str(r)
                for r in rows
            )
            is_scan = any("SCAN" in str(r) for r in rows)
            return {
                "estimated_rows": None,
                "confidence": 0.3 if is_scan else 0.5,
                "scan_type": "index" if has_index else "scan",
                "raw_plan": "\n".join(str(r) for r in rows),
            }
    except Exception:
        return None


def parse_explain_postgresql(engine: Any, query: str) -> Optional[Dict[str, Any]]:
    """Parse PostgreSQL EXPLAIN (FORMAT JSON) output."""
    try:
        import json as json_mod

        from sqlalchemy import text

        with engine.connect() as conn:
            result = conn.execute(text(f"EXPLAIN (FORMAT JSON) {query}"))
            plan_json = result.scalar()
            if isinstance(plan_json, str):
                plan_json = json_mod.loads(plan_json)
            plan = plan_json[0]["Plan"]
            return {
                "estimated_rows": int(plan.get("Plan Rows", 0)),
                "confidence": 0.7,
                "scan_type": plan.get("Node Type", "unknown"),
                "total_cost": plan.get("Total Cost", 0),
            }
    except Exception:
        return None


def parse_explain_mysql(engine: Any, query: str) -> Optional[Dict[str, Any]]:
    """Parse MySQL EXPLAIN output."""
    try:
        from sqlalchemy import text

        with engine.connect() as conn:
            result = conn.execute(text(f"EXPLAIN {query}"))
            rows = result.fetchall()
            total_rows = 1
            scan_type = "unknown"
            for row in rows:
                row_dict = (
                    row._mapping
                    if hasattr(row, "_mapping")
                    else dict(zip(result.keys(), row))
                )
                row_est = row_dict.get("rows", 1)
                if row_est:
                    total_rows *= int(row_est)
                scan_type = row_dict.get("type", scan_type)
            confidence_map = {
                "const": 0.9,
                "ref": 0.7,
                "range": 0.6,
                "index": 0.5,
                "ALL": 0.3,
            }
            return {
                "estimated_rows": total_rows,
                "confidence": confidence_map.get(scan_type, 0.4),
                "scan_type": scan_type,
            }
    except Exception:
        return None
