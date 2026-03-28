"""Safe regex execution with ReDoS prevention for LocalData MCP."""

import logging
import re
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

try:
    import re._parser as sre_parse  # Python 3.11+
except ImportError:
    import sre_parse  # type: ignore[import-not-found]  # Python 3.10

logger = logging.getLogger(__name__)

MAX_PATTERN_LENGTH = 200
MAX_CAPTURE_GROUPS = 10
EXECUTION_TIMEOUT_SECONDS = 5


@dataclass
class PatternValidationResult:
    """Result of regex pattern safety validation."""

    is_valid: bool
    error_message: Optional[str] = None
    group_count: int = 0
    complexity_score: float = 0.0
    detected_issues: List[str] = field(default_factory=list)


@dataclass
class SearchMatch:
    """A single regex match found in data."""

    row_index: int
    column: str
    value: str
    match: str
    start: int
    end: int


class RegexSafetyValidator:
    """Validates regex patterns for safety before execution."""

    def validate(self, pattern: str) -> PatternValidationResult:
        """Validate a regex pattern for length, syntax, groups, and complexity."""
        if len(pattern) > MAX_PATTERN_LENGTH:
            return PatternValidationResult(
                is_valid=False,
                error_message=(
                    f"Pattern too long ({len(pattern)} > {MAX_PATTERN_LENGTH})"
                ),
            )

        try:
            compiled = re.compile(pattern)
        except re.error as e:
            return PatternValidationResult(
                is_valid=False, error_message=f"Invalid regex: {e}"
            )

        if compiled.groups > MAX_CAPTURE_GROUPS:
            return PatternValidationResult(
                is_valid=False,
                error_message=(
                    f"Too many groups ({compiled.groups} > {MAX_CAPTURE_GROUPS})"
                ),
            )

        issues = self._detect_dangerous_patterns(pattern)
        score = len(issues) * 25.0 + compiled.groups * 2.0

        if score > 80:
            return PatternValidationResult(
                is_valid=False,
                error_message=f"Pattern too complex (score {score:.0f})",
                group_count=compiled.groups,
                complexity_score=score,
                detected_issues=issues,
            )

        return PatternValidationResult(
            is_valid=True,
            group_count=compiled.groups,
            complexity_score=score,
            detected_issues=issues,
        )

    def _detect_dangerous_patterns(self, pattern: str) -> List[str]:
        """Detect patterns that could cause catastrophic backtracking."""
        issues: List[str] = []
        try:
            parsed = sre_parse.parse(pattern)
            self._check_nested_quantifiers(parsed, issues, depth=0)
        except Exception:
            pass
        return issues

    def _check_nested_quantifiers(
        self,
        parsed: sre_parse.SubPattern,
        issues: List[str],
        depth: int,
    ) -> None:
        """Recursively check for nested quantifiers in parsed pattern."""
        for op, av in parsed:
            if op in (sre_parse.MAX_REPEAT, sre_parse.MIN_REPEAT):
                if depth > 0:
                    issues.append("Nested quantifier detected (potential ReDoS)")
                if isinstance(av[2], sre_parse.SubPattern):
                    self._check_nested_quantifiers(av[2], issues, depth + 1)
            elif op == sre_parse.SUBPATTERN and av[3]:
                self._check_nested_quantifiers(av[3], issues, depth)
            elif op == sre_parse.BRANCH:
                for branch in av[1]:
                    self._check_nested_quantifiers(branch, issues, depth)


def _execute_with_timeout(func: Callable, timeout: float, *args: Any) -> Any:
    """Execute a function with timeout protection."""
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args)
        try:
            return future.result(timeout=timeout)
        except FuturesTimeout:
            raise TimeoutError(f"Regex execution timed out after {timeout}s") from None


class RegexEngine:
    """Safe regex engine with validation and timeout."""

    def __init__(self, timeout: float = EXECUTION_TIMEOUT_SECONDS) -> None:
        self._validator = RegexSafetyValidator()
        self._timeout = timeout
        self._cache: Dict[Tuple[str, int], re.Pattern] = {}

    def validate_pattern(self, pattern: str) -> Tuple[bool, Optional[str]]:
        """Validate a pattern and return (is_valid, error_message)."""
        result = self._validator.validate(pattern)
        return result.is_valid, result.error_message

    def _compile(self, pattern: str, flags: int = 0) -> re.Pattern:
        """Compile and cache a regex pattern."""
        key = (pattern, flags)
        if key not in self._cache:
            self._cache[key] = re.compile(pattern, flags)
        return self._cache[key]

    def _validate_and_compile(self, pattern: str, flags: int = 0) -> re.Pattern:
        """Validate a pattern then compile it, raising ValueError if unsafe."""
        valid, error = self.validate_pattern(pattern)
        if not valid:
            raise ValueError(f"Unsafe regex pattern: {error}")
        return self._compile(pattern, flags)

    def search(self, text: str, pattern: str, flags: int = 0) -> Optional[re.Match]:
        """Search text for pattern with timeout protection."""
        compiled = self._validate_and_compile(pattern, flags)
        return _execute_with_timeout(compiled.search, self._timeout, text)

    def findall(self, text: str, pattern: str, flags: int = 0) -> List[str]:
        """Find all matches with timeout protection."""
        compiled = self._validate_and_compile(pattern, flags)
        return _execute_with_timeout(compiled.findall, self._timeout, text)

    def sub(
        self,
        pattern: str,
        replacement: str,
        text: str,
        count: int = 0,
        flags: int = 0,
    ) -> str:
        """Substitute matches with timeout protection."""
        compiled = self._validate_and_compile(pattern, flags)
        return _execute_with_timeout(
            compiled.sub, self._timeout, replacement, text, count
        )


_engine: Optional[RegexEngine] = None


def get_regex_engine(timeout: float = EXECUTION_TIMEOUT_SECONDS) -> RegexEngine:
    """Get the singleton RegexEngine instance."""
    global _engine
    if _engine is None:
        _engine = RegexEngine(timeout=timeout)
    return _engine


def search_data(
    data: List[Dict[str, Any]],
    pattern: str,
    columns: Optional[List[str]] = None,
    case_sensitive: bool = True,
    max_matches: int = 100,
) -> Dict[str, Any]:
    """Search query results for regex pattern matches."""
    engine = get_regex_engine()
    valid, error = engine.validate_pattern(pattern)
    if not valid:
        return {"error": error, "matches": [], "total_matches": 0}

    flags = 0 if case_sensitive else re.IGNORECASE
    compiled = engine._compile(pattern, flags)
    matches: List[SearchMatch] = []

    for i, row in enumerate(data):
        search_cols = columns or list(row.keys())
        for col in search_cols:
            if col not in row:
                continue
            text = str(row[col]) if row[col] is not None else ""
            for m in compiled.finditer(text):
                matches.append(SearchMatch(i, col, text, m.group(), m.start(), m.end()))
                if len(matches) >= max_matches:
                    break
            if len(matches) >= max_matches:
                break
        if len(matches) >= max_matches:
            break

    return {
        "matches": [
            {
                "row_index": m.row_index,
                "column": m.column,
                "value": m.value,
                "match": m.match,
                "start": m.start,
                "end": m.end,
            }
            for m in matches
        ],
        "total_matches": len(matches),
        "rows_searched": len(data),
        "pattern": pattern,
        "truncated": len(matches) >= max_matches,
    }


def transform_data(
    data: List[Dict[str, Any]],
    column: str,
    find: str,
    replace: str,
    max_rows: int = 1000,
) -> Dict[str, Any]:
    """Apply regex find/replace to a column in query result data."""
    engine = get_regex_engine()
    valid, error = engine.validate_pattern(find)
    if not valid:
        return {"error": error, "transformed_count": 0, "data": data}

    compiled = engine._compile(find)
    transformed = 0
    sample: List[Dict[str, str]] = []
    result: List[Dict[str, Any]] = []

    for row in data[:max_rows]:
        new_row = dict(row)
        if column in new_row and new_row[column] is not None:
            original = str(new_row[column])
            new_val = compiled.sub(replace, original)
            if new_val != original:
                transformed += 1
                if len(sample) < 5:
                    sample.append({"original": original, "transformed": new_val})
                new_row[column] = new_val
        result.append(new_row)

    return {
        "transformed_count": transformed,
        "total_rows": len(result),
        "sample": sample,
        "data": result,
    }
