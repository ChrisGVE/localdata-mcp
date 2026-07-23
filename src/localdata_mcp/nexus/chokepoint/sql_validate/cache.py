"""localdata_mcp/nexus/chokepoint/sql_validate/cache.py — the bounded LRU (E6.3).

A bounded LRU keyed on `(dialect, normalized statement text)` storing
ONLY the parse/walk classification (§7): the composition engine and the
batteries drive many textually-identical statements through the gate,
which is exactly the case the cache serves. It is fail-closed by
construction:

- Normalization is strictly semantics- and literal-preserving —
  whitespace runs collapsed and leading/trailing trimmed, NEVER literal
  masking or fingerprinting — so two statements differing in a path or
  value literal never share an entry.
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

import re
import threading
from collections import OrderedDict

from .walker import SqlClassification, SqlRefusedError, classify

_WHITESPACE = re.compile(r"\s+")


def _cache_key(sql: str, backend_kind: str) -> str:
    """Semantics- and literal-preserving normalization: collapse
    whitespace only. Case is NOT folded — a quoted identifier's case is
    semantic — and no literal is ever masked, so distinct literals
    cannot collide (§7)."""
    return f"{backend_kind}\x00{_WHITESPACE.sub(' ', sql).strip()}"


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
