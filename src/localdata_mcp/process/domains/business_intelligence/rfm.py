"""localdata_mcp/process/domains/business_intelligence/rfm.py — FR-301/307.

`analyze_rfm`'s computation with the FR-307 defect closure. `main`'s
cascade had two DEAD branches (#37): "Loyal Customers" (f>=3, m>=3)
shadowed "Need Attention" (r<=2, f>=3, m>=3), and "At Risk" (f>=1,
m>=1 — always true for 1..5 scores) shadowed "Lost". The reordered
cascade below places every specific rule before the general rule
that would swallow it, and narrows "At Risk" so "Lost" keeps a
reachable remainder — E10.x2's enumeration test constructs an input
for every segment and asserts each is assigned. Scores are quintile
ranks (1..5; recency inverted — smaller days-since = higher score).
Neighbors: tools.py declares the ToolSpec; clv.py is the sibling.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from ..support import invalid_source_refusal, require_columns

# The reordered cascade (FR-307): first matching rule wins; specific
# rules precede the general rules that would otherwise shadow them.
SEGMENT_RULES: tuple[tuple[str, Any], ...] = (
    ("Champions", lambda r, f, m: r >= 4 and f >= 4 and m >= 3),
    ("Need Attention", lambda r, f, m: r <= 2 and f >= 3 and m >= 3),
    ("Loyal Customers", lambda r, f, m: f >= 3 and m >= 3),
    ("Potential Loyalists", lambda r, f, m: r >= 3 and f >= 2),
    ("Recent Customers", lambda r, f, m: r >= 3),
    ("Promising", lambda r, f, m: f >= 2 and m >= 2),
    ("About to Sleep", lambda r, f, m: r <= 2 and f >= 2),
    ("At Risk", lambda r, f, m: f >= 2 or m >= 2),
    ("Lost", lambda r, f, m: True),
)


def analyze_rfm_segments(
    frame: pd.DataFrame,
    customer_column: str,
    date_column: str,
    value_column: str,
) -> dict[str, Any]:
    """Quintile RFM scores and the reordered segment assignment."""
    require_columns(frame, customer_column, date_column, value_column)
    dates = pd.to_datetime(frame[date_column], errors="coerce")
    amounts = pd.to_numeric(frame[value_column], errors="coerce")
    rows = pd.DataFrame(
        {"customer": frame[customer_column], "date": dates, "amount": amounts}
    ).dropna()
    if rows.empty:
        raise invalid_source_refusal(
            "No usable (customer, date, amount) rows in the addressed data."
        )
    observation_point = rows["date"].max()
    metrics = rows.groupby("customer").agg(
        recency=("date", lambda d: int((observation_point - d.max()).days)),
        frequency=("date", "count"),
        monetary=("amount", "sum"),
    )
    scores = pd.DataFrame(
        {
            "R": _quintile_scores(metrics["recency"], descending=True),
            "F": _quintile_scores(metrics["frequency"], descending=False),
            "M": _quintile_scores(metrics["monetary"], descending=False),
        },
        index=metrics.index,
    )
    segments = scores.apply(
        lambda row: _assign_segment(int(row["R"]), int(row["F"]), int(row["M"])),
        axis=1,
    )
    combined = metrics.join(scores)
    combined["segment"] = segments
    summary = {
        str(name): {
            "customer_count": int(len(group)),
            "monetary_sum": float(group["monetary"].sum()),
            "recency_mean": float(group["recency"].mean()),
            "frequency_mean": float(group["frequency"].mean()),
        }
        for name, group in combined.groupby("segment", sort=True)
    }
    return {
        "n_customers": int(len(combined)),
        "observation_point": str(observation_point),
        "segments": summary,
        "customers": [
            {
                "customer": str(customer),
                "recency_days": int(row["recency"]),
                "frequency": int(row["frequency"]),
                "monetary": float(row["monetary"]),
                "R": int(row["R"]),
                "F": int(row["F"]),
                "M": int(row["M"]),
                "segment": str(row["segment"]),
            }
            for customer, row in combined.iterrows()
        ],
    }


def _quintile_scores(values: "pd.Series[Any]", descending: bool) -> "pd.Series[int]":
    """Quintile ranks 1..5 (rank-based, stable under heavy ties);
    descending inverts so that smaller raw values score higher."""
    ranks = values.rank(method="first", ascending=not descending)
    bins = pd.qcut(ranks, q=5, labels=False, duplicates="drop")
    return (bins + 1).astype(int)


def _assign_segment(r: int, f: int, m: int) -> str:
    for name, rule in SEGMENT_RULES:
        if rule(r, f, m):
            return name
    raise AssertionError("the cascade ends in a catch-all")  # pragma: no cover
