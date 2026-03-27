"""EXPLAIN query plan parsers for database-specific size estimation.

Provides parsers for SQLite, PostgreSQL, MySQL, Oracle, and MS SQL EXPLAIN output,
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


def _extract_oracle_rows(rows) -> Optional[int]:
    """Extract row estimate from DBMS_XPLAN output."""
    import re

    for row in rows:
        line = str(row[0]) if row else ""
        m = re.search(r"\|\s*(\d+)\s*\|?\s*$", line)
        if m:
            return int(m.group(1))
    return None


def parse_explain_oracle(engine: Any, query: str) -> Optional[Dict[str, Any]]:
    """Parse Oracle EXPLAIN PLAN output."""
    try:
        from sqlalchemy import text

        with engine.connect() as conn:
            conn.execute(text(f"EXPLAIN PLAN FOR {query}"))
            result = conn.execute(
                text("SELECT * FROM TABLE(DBMS_XPLAN.DISPLAY(format => 'BASIC ROWS'))")
            )
            rows = result.fetchall()
            estimated_rows = _extract_oracle_rows(rows)
            return {
                "estimated_rows": estimated_rows,
                "confidence": 0.7,
                "scan_type": "oracle_plan",
            }
    except Exception:
        return None


def _parse_showplan_xml(xml_str: str) -> int:
    """Extract EstimateRows from SHOWPLAN XML."""
    import xml.etree.ElementTree as ET

    try:
        root = ET.fromstring(xml_str)
        ns = {"sp": "http://schemas.microsoft.com/sqlserver/2004/07/showplan"}
        for relop in root.findall(".//sp:RelOp", ns):
            estimate = relop.get("EstimateRows")
            if estimate:
                return int(float(estimate))
    except ET.ParseError:
        pass
    return 1000


def parse_explain_mssql(engine: Any, query: str) -> Optional[Dict[str, Any]]:
    """Parse MS SQL Server SHOWPLAN_XML output."""
    try:
        from sqlalchemy import text

        with engine.connect() as conn:
            conn.execute(text("SET SHOWPLAN_XML ON"))
            try:
                result = conn.execute(text(query))
                xml_plan = result.fetchone()[0]
                estimated_rows = _parse_showplan_xml(xml_plan)
                return {
                    "estimated_rows": estimated_rows,
                    "confidence": 0.8,
                    "scan_type": "showplan",
                }
            finally:
                conn.execute(text("SET SHOWPLAN_XML OFF"))
    except Exception:
        return None
