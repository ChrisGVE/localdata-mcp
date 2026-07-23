"""localdata_mcp/explore/safe_regex.py — hardened pattern screen (E9.3).

The harvested successor of `regex_tools.py`'s `RegexSafetyValidator`:
FR-203's search accepts caller-supplied regular expressions, so every
pattern is screened BEFORE compilation is used — length cap, capture
-group cap, and a parse-tree walk refusing nested quantifiers (the
catastrophic-backtracking shape). The whole row scan additionally
runs under a wall-clock backstop (`scan_with_timeout`) — one worker
for the WHOLE scan, not `main`'s per-execution thread pool (the
static screen refuses the pathological class; the timeout catches
what static analysis cannot prove, at one thread's cost). Caps are
named non-config constants: legibility bounds on a screen, not
operator resource knobs. Neighbors: search.py is the only consumer.
"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout
from typing import Any, Callable

import re._parser as sre_parse

# Screen caps (non-config legibility bounds; values chosen off the S8
# default set so the one-default-site gate stays meaningful).
MAX_PATTERN_LENGTH = 256
MAX_CAPTURE_GROUPS = 12
SCAN_TIMEOUT_SECONDS = 5.0


class UnsafePatternError(ValueError):
    """The pattern failed the safety screen — never compiled for use."""


def validated_pattern(pattern: str, case_sensitive: bool) -> "re.Pattern[str]":
    """The compiled pattern, or an `UnsafePatternError` refusal."""
    if len(pattern) > MAX_PATTERN_LENGTH:
        raise UnsafePatternError(
            f"pattern too long ({len(pattern)} > {MAX_PATTERN_LENGTH})"
        )
    try:
        compiled = re.compile(pattern, 0 if case_sensitive else re.IGNORECASE)
    except re.error as invalid:
        raise UnsafePatternError(f"invalid regex: {invalid}") from invalid
    if compiled.groups > MAX_CAPTURE_GROUPS:
        raise UnsafePatternError(
            f"too many capture groups ({compiled.groups} > {MAX_CAPTURE_GROUPS})"
        )
    issues = _dangerous_shapes(pattern)
    if issues:
        # Deny-by-default: any detected backtracking shape refuses
        # outright — STRICTER than `main`'s weighted score, which let
        # the canonical `(a+)+` through and leaned on its per-call
        # timeout; here the static screen carries the refusal and the
        # scan timeout backstops only what parsing cannot prove.
        raise UnsafePatternError(f"dangerous pattern: {'; '.join(issues)}")
    return compiled


def _dangerous_shapes(pattern: str) -> list[str]:
    """Nested-quantifier detection over the parse tree (harvested)."""
    issues: list[str] = []
    try:
        parsed = sre_parse.parse(pattern)
        _walk_quantifiers(parsed, issues, depth=0)
    except Exception:  # nosec B110 — an unparseable tree already failed compile
        pass
    return issues


def _walk_quantifiers(parsed: Any, issues: list[str], depth: int) -> None:
    for op, av in parsed:
        if op in (sre_parse.MAX_REPEAT, sre_parse.MIN_REPEAT):
            if depth > 0:
                issues.append("nested quantifier (potential ReDoS)")
            if isinstance(av[2], sre_parse.SubPattern):
                _walk_quantifiers(av[2], issues, depth + 1)
        elif op == sre_parse.SUBPATTERN and av[3]:
            _walk_quantifiers(av[3], issues, depth)
        elif op == sre_parse.BRANCH:
            for branch in av[1]:
                _walk_quantifiers(branch, issues, depth)


def scan_with_timeout(scan: Callable[[], Any]) -> Any:
    """Run the whole row scan under the wall-clock backstop."""
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(scan)
        try:
            return future.result(timeout=SCAN_TIMEOUT_SECONDS)
        except FuturesTimeout:
            raise TimeoutError(
                f"search timed out after {SCAN_TIMEOUT_SECONDS}s — narrow "
                "the source (WHERE/LIMIT) or simplify the pattern"
            ) from None
