"""localdata_mcp/nexus/chokepoint/sql_validate/cache.py — the bounded LRU (E6.3).

A bounded LRU keyed on `(dialect, normalized statement text)` storing
ONLY the parse/walk classification (§7): the composition engine and the
batteries drive many textually-identical statements through the gate,
which is exactly the case the cache serves. It is fail-closed by
construction:

- Normalization is strictly semantics- and literal-preserving —
  whitespace runs are collapsed ONLY OUTSIDE quoted literals (SQL is
  whitespace-insensitive there) and leading/trailing trimmed, NEVER
  literal masking or fingerprinting. Whitespace INSIDE a quoted literal
  is preserved byte-for-byte, so two statements differing only in a
  literal's interior whitespace (`'a b'` vs `'a  b'`, distinct executed
  bytes and distinct paths) never share an entry — the cached verdict
  can never reason over a reconstruction of different executed text.
- Only the classification is cached; posture and NFR-108 containment
  are evaluated per call by guard.py, OUTSIDE this cache, so a cached
  verdict never carries a containment decision and never goes stale
  against config.
- A REFUSAL is cached too (as the raised message): a statement that
  cannot pass is deterministically refused per (dialect, text), so
  re-refusing from cache is correct and cheap.

The bound is the NX-2 field `security.validation_cache_entries`, passed
in — never a module literal. Neighbors: walker.py produces the
classification; guard.py owns one cache instance sized from NX-2.
"""

from __future__ import annotations

import threading
from collections import OrderedDict

from .walker import SqlClassification, SqlRefusedError, classify


def _collapse_outside_literals(sql: str) -> str:
    """Collapse every whitespace run to one space, but ONLY outside
    quoted literals — the interior of a `'…'`, `"…"`, or `` `…` `` span
    is copied byte-for-byte.

    The scanner errs toward staying INSIDE a literal (a doubled
    delimiter is an escaped quote, a backslash escapes the next
    character): mis-reading a literal as closed would re-open the exact
    whitespace-collision hole this normalization exists to close, so the
    only tolerated error is the safe one — preserving whitespace that
    did not strictly need preserving (a cache miss, never a collision).
    """
    out: list[str] = []
    i = 0
    n = len(sql)
    quote = ""
    while i < n:
        ch = sql[i]
        if quote:
            out.append(ch)
            if ch == "\\" and i + 1 < n:  # backslash escape (e.g. MySQL)
                out.append(sql[i + 1])
                i += 2
                continue
            if ch == quote:
                if i + 1 < n and sql[i + 1] == quote:  # doubled = escaped quote
                    out.append(sql[i + 1])
                    i += 2
                    continue
                quote = ""  # the closing delimiter
            i += 1
            continue
        if ch in "'\"`":
            quote = ch
            out.append(ch)
            i += 1
            continue
        if ch.isspace():
            out.append(" ")
            i += 1
            while i < n and sql[i].isspace():
                i += 1
            continue
        out.append(ch)
        i += 1
    return "".join(out).strip()


def _cache_key(sql: str, backend_kind: str) -> str:
    """Semantics- and literal-preserving normalization: collapse
    whitespace OUTSIDE quoted literals only. Case is NOT folded — a
    quoted identifier's case is semantic — and no literal interior is
    ever touched, so distinct executed bytes cannot collide (§7)."""
    return f"{backend_kind}\x00{_collapse_outside_literals(sql)}"


class ValidationCache:
    """A thread-safe bounded LRU over classifications AND refusals."""

    def __init__(self, max_entries: int) -> None:
        if max_entries < 1:
            raise ValueError("validation cache must hold at least one entry")
        self._max = max_entries
        self._entries: "OrderedDict[str, SqlClassification | SqlRefusedError]" = (
            OrderedDict()
        )
        self._lock = threading.Lock()

    def classify(self, sql: str, backend_kind: str) -> SqlClassification:
        """The cached classification, computing and storing on a miss.

        A cached refusal is re-raised — the verdict is deterministic per
        (dialect, normalized text), so serving it from cache is correct.
        """
        key = _cache_key(sql, backend_kind)
        cached = self._get(key)
        if cached is not None:
            if isinstance(cached, SqlRefusedError):
                raise SqlRefusedError(str(cached))
            return cached
        try:
            result = classify(sql, backend_kind)
        except SqlRefusedError as refusal:
            self._put(key, refusal)
            raise
        self._put(key, result)
        return result

    def _get(self, key: str) -> "SqlClassification | SqlRefusedError | None":
        with self._lock:
            if key not in self._entries:
                return None
            self._entries.move_to_end(key)
            return self._entries[key]

    def _put(self, key: str, value: "SqlClassification | SqlRefusedError") -> None:
        with self._lock:
            self._entries[key] = value
            self._entries.move_to_end(key)
            while len(self._entries) > self._max:
                self._entries.popitem(last=False)

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)
