"""localdata_mcp/process/domains/business_intelligence/clv.py — FR-301/309.

`calculate_clv`'s computation — v3's FIRST registration of the tool
(an unregistered orphan on `main`, S9.1) with the FR-309 closure:
`main`'s implementation accepted a `customer_column` parameter and
then hardcoded `customer_id`/`date`/`amount` in the aggregation
(#24); here the caller's column names are the ONLY names used. The
formula is `main`'s historical model kept intact: per-customer
average order value × purchase frequency (orders per active day) ×
gross margin × annualization. Neighbors: tools.py declares the
ToolSpec; rfm.py is the sibling.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from ..support import invalid_source_refusal, require_columns

# main's assumptions, now caller-overridable: 20% gross margin,
# annualized over 365 days.
_DEFAULT_MARGIN = 0.2
_DAYS_PER_YEAR = 365


def calculate_lifetime_value(
    frame: pd.DataFrame,
    customer_column: str,
    date_column: str,
    value_column: str,
    gross_margin: float = _DEFAULT_MARGIN,
) -> dict[str, Any]:
    """Historical CLV per customer plus the distribution summary."""
    require_columns(frame, customer_column, date_column, value_column)
    if not (0.0 < gross_margin <= 1.0):
        raise invalid_source_refusal("gross_margin must be inside (0, 1].")
    dates = pd.to_datetime(frame[date_column], errors="coerce")
    amounts = pd.to_numeric(frame[value_column], errors="coerce")
    rows = pd.DataFrame(
        {"customer": frame[customer_column], "date": dates, "amount": amounts}
    ).dropna()
    if rows.empty:
        raise invalid_source_refusal(
            "No usable (customer, date, amount) rows in the addressed data."
        )
    metrics = rows.groupby("customer").agg(
        total_spent=("amount", "sum"),
        avg_order_value=("amount", "mean"),
        order_count=("amount", "count"),
        first_purchase=("date", "min"),
        last_purchase=("date", "max"),
    )
    lifespan_days = (metrics["last_purchase"] - metrics["first_purchase"]).dt.days
    purchase_frequency = metrics["order_count"] / (lifespan_days + 1)
    clv = (
        metrics["avg_order_value"] * purchase_frequency * gross_margin * _DAYS_PER_YEAR
    )
    return {
        "customer_column": customer_column,
        "gross_margin": gross_margin,
        "n_customers": int(len(metrics)),
        "clv_distribution": {
            "mean": float(clv.mean()),
            "median": float(clv.median()),
            "min": float(clv.min()),
            "max": float(clv.max()),
        },
        "customers": [
            {
                "customer": str(customer),
                "total_spent": float(metrics.loc[customer, "total_spent"]),
                "avg_order_value": float(metrics.loc[customer, "avg_order_value"]),
                "order_count": int(metrics.loc[customer, "order_count"]),
                "lifespan_days": int(lifespan_days.loc[customer]),
                "clv_estimate": float(clv.loc[customer]),
            }
            for customer in metrics.index
        ],
    }
