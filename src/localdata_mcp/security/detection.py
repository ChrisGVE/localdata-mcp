"""Attack pattern detection and query analysis for LocalData MCP.

Standalone functions for SQL injection detection, query fingerprinting,
and complexity analysis.
"""

import hashlib
import re
from typing import Dict, List

from .models import AttackPattern, QueryComplexity


def compile_attack_patterns() -> Dict[AttackPattern, re.Pattern]:
    """Compile regex patterns for attack detection.

    Returns:
        Dict[AttackPattern, re.Pattern]: Compiled pattern mapping.
    """
    patterns = {
        AttackPattern.UNION_INJECTION: re.compile(
            r"\b(?:union\s+(?:all\s+)?select|union\s+select)\b", re.IGNORECASE
        ),
        AttackPattern.TIME_BASED_BLIND: re.compile(
            r"\b(?:sleep\s*\(|waitfor\s+delay|benchmark\s*\(|pg_sleep\s*\()",
            re.IGNORECASE,
        ),
        AttackPattern.BOOLEAN_BLIND: re.compile(
            r"(?:and|or)\s+(?:\d+=\d+|'[^']*'\s*=\s*'[^']*|true\b|false\b)",
            re.IGNORECASE,
        ),
        AttackPattern.ERROR_BASED: re.compile(
            r"\b(?:cast\s*\(.*?as\s+int\s*\)|extractvalue\s*\(|updatexml\s*\(|floor\s*\(\s*rand\s*\(|concat\s*\(\s*(?:version|user)\s*\()",
            re.IGNORECASE,
        ),
        AttackPattern.INFORMATION_EXTRACTION: re.compile(
            r"\b(?:information_schema|sys\.tables|sys\.columns|pg_tables|sqlite_master)\b",
            re.IGNORECASE,
        ),
        AttackPattern.COMMENT_INJECTION: re.compile(
            r"(?:--[^\r\n]*|/\*(?:.*?\*/|.*$)|#[^\r\n]*)",
            re.DOTALL | re.IGNORECASE,
        ),
        AttackPattern.STACKED_QUERIES: re.compile(
            r";\s*(?:insert|update|delete|drop|create|alter)\b", re.IGNORECASE
        ),
        AttackPattern.FUNCTION_ABUSE: re.compile(
            r"\b(?:load_file\s*\(|into\s+outfile|into\s+dumpfile|exec\s+\w+|exec\s*\(|eval\s*\()",
            re.IGNORECASE,
        ),
    }
    return patterns


def detect_attack_patterns(
    query: str, patterns: Dict[AttackPattern, re.Pattern]
) -> List[AttackPattern]:
    """Detect known attack patterns in query.

    Args:
        query: SQL query string.
        patterns: Compiled attack pattern mapping.

    Returns:
        List[AttackPattern]: List of detected attack patterns.
    """
    detected_patterns = []

    for pattern_type, regex in patterns.items():
        if regex.search(query):
            detected_patterns.append(pattern_type)

    return detected_patterns


def create_query_fingerprint(query: str) -> str:
    """Create a cryptographic fingerprint of the query.

    Args:
        query: SQL query string.

    Returns:
        str: SHA-256 hexdigest of the normalized query.
    """
    # Normalize query for consistent fingerprinting
    normalized = query.strip().lower()
    normalized = re.sub(r"--[^\r\n]*", "", normalized)
    normalized = re.sub(r"/\*.*?\*/", "", normalized, flags=re.DOTALL)
    normalized = re.sub(r"\s+", " ", normalized).strip()

    # Create fingerprint
    fingerprint = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]
    return fingerprint


def analyze_query_complexity(query: str) -> QueryComplexity:
    """Analyze query complexity for resource limit enforcement.

    Args:
        query: SQL query string.

    Returns:
        QueryComplexity: Detailed complexity analysis.
    """
    complexity = QueryComplexity()
    query_lower = query.lower()

    # Basic metrics
    complexity.length = len(query)
    complexity.joins = len(
        re.findall(
            r"\b(?:inner\s+join|left\s+join|right\s+join|full\s+join|join)\b",
            query_lower,
        )
    )
    complexity.subqueries = len(re.findall(r"\(\s*select\b", query_lower))
    complexity.functions = len(re.findall(r"\b\w+\s*\(", query_lower))
    complexity.unions = len(re.findall(r"\bunion\b", query_lower))
    complexity.conditions = len(re.findall(r"\b(?:where|and|or|having)\b", query_lower))

    # Count unique table references
    table_pattern = re.compile(r"\bfrom\s+(\w+)|join\s+(\w+)", re.IGNORECASE)
    tables = set()
    for match in table_pattern.finditer(query):
        table = match.group(1) or match.group(2)
        if table:
            tables.add(table.lower())
    complexity.tables = len(tables)

    # Calculate overall score
    complexity.calculate_score()

    return complexity
