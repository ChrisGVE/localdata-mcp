"""
Business Intelligence Domain - High-level convenience functions.

This module provides simple top-level functions that wrap the transformer
classes for common BI analysis tasks including RFM, cohort, CLV, A/B testing,
attribution, funnel analysis, and enhanced A/B testing with cross-domain
statistical integration.
"""

from typing import Any, Dict, List, Optional

import pandas as pd

from ...logging_manager import get_logger
from .models import (
    ABTestResult,
    AttributionResult,
    CLVResult,
    CohortAnalysisResult,
    FunnelAnalysisResult,
    RFMResult,
)
from .customer_analytics import RFMAnalysisTransformer
from .cohort_clv import CohortAnalysisTransformer, CLVCalculator
from .ab_testing import ABTestAnalyzer
from .attribution import AttributionAnalyzer
from .funnel import FunnelAnalyzer

logger = get_logger(__name__)


# High-level convenience functions
def analyze_rfm(
    data: pd.DataFrame,
    customer_column: str = "customer_id",
    date_column: str = "date",
    amount_column: str = "amount",
    analysis_date: Optional[str] = None,
) -> RFMResult:
    """
    Perform RFM analysis on customer transaction data.

    Parameters:
    -----------
    data : pd.DataFrame
        Transaction data containing customer, date, and amount information
    customer_column : str, default='customer_id'
        Name of the customer identifier column
    date_column : str, default='date'
        Name of the date column
    amount_column : str, default='amount'
        Name of the transaction amount column
    analysis_date : str, optional
        Reference date for recency calculation

    Returns:
    --------
    result : RFMResult
        Complete RFM analysis results
    """
    transformer = RFMAnalysisTransformer(
        customer_column=customer_column,
        date_column=date_column,
        amount_column=amount_column,
        analysis_date=analysis_date,
    )

    transformer.fit(data)
    return transformer.transform(data)


def perform_cohort_analysis(
    data: pd.DataFrame,
    customer_column: str = "customer_id",
    date_column: str = "date",
    period_type: str = "monthly",
) -> CohortAnalysisResult:
    """
    Perform cohort analysis for customer retention tracking.

    Parameters:
    -----------
    data : pd.DataFrame
        Transaction data containing customer and date information
    customer_column : str, default='customer_id'
        Name of the customer identifier column
    date_column : str, default='date'
        Name of the date column
    period_type : str, default='monthly'
        Period for cohort analysis: 'daily', 'weekly', 'monthly', 'quarterly'

    Returns:
    --------
    result : CohortAnalysisResult
        Complete cohort analysis results
    """
    transformer = CohortAnalysisTransformer(
        customer_column=customer_column,
        date_column=date_column,
        period_type=period_type,
    )

    transformer.fit(data)
    return transformer.transform(data)


def calculate_clv(
    data: pd.DataFrame,
    customer_column: str = "customer_id",
    date_column: str = "date",
    amount_column: str = "amount",
    method: str = "historical",
) -> CLVResult:
    """
    Calculate Customer Lifetime Value for customers.

    Parameters:
    -----------
    data : pd.DataFrame
        Transaction data containing customer, date, and amount information
    customer_column : str, default='customer_id'
        Name of the customer identifier column
    date_column : str, default='date'
        Name of the date column
    amount_column : str, default='amount'
        Name of the transaction amount column
    method : str, default='historical'
        CLV calculation method

    Returns:
    --------
    result : CLVResult
        Complete CLV analysis results
    """
    # Ensure required columns are present and properly formatted
    required_columns = [customer_column, date_column, amount_column]
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    transformer = CLVCalculator(method=method)
    transformer.fit(data)
    return transformer.transform(data)


def perform_ab_test(
    data: pd.DataFrame,
    group_column: str = "group",
    outcome_column: str = "converted",
    test_type: str = "proportion",
    alpha: float = 0.05,
) -> ABTestResult:
    """
    Perform A/B test statistical analysis.

    Parameters:
    -----------
    data : pd.DataFrame
        A/B test data with group assignments and outcomes
    group_column : str, default='group'
        Name of the group assignment column
    outcome_column : str, default='converted'
        Name of the outcome/conversion column
    test_type : str, default='proportion'
        Type of statistical test to perform
    alpha : float, default=0.05
        Significance level for hypothesis testing

    Returns:
    --------
    result : ABTestResult
        Complete A/B test analysis results
    """
    # Rename columns to match expected format
    data_renamed = data.rename(
        columns={group_column: "group", outcome_column: "converted"}
    )

    transformer = ABTestAnalyzer(alpha=alpha, test_type=test_type)
    transformer.fit(data_renamed)
    return transformer.transform(data_renamed)


# Additional high-level convenience functions
def analyze_attribution(
    data: pd.DataFrame,
    customer_column: str = "customer_id",
    channel_column: str = "channel",
    timestamp_column: str = "timestamp",
    model: str = "last_touch",
) -> AttributionResult:
    """
    Perform marketing attribution analysis.

    Parameters:
    -----------
    data : pd.DataFrame
        Touchpoint data with customer, channel, and timestamp information
    customer_column : str, default='customer_id'
        Name of the customer identifier column
    channel_column : str, default='channel'
        Name of the marketing channel column
    timestamp_column : str, default='timestamp'
        Name of the timestamp column
    model : str, default='last_touch'
        Attribution model to use

    Returns:
    --------
    result : AttributionResult
        Complete attribution analysis results
    """
    # Rename columns to match expected format
    data_renamed = data.rename(
        columns={
            customer_column: "customer_id",
            channel_column: "channel",
            timestamp_column: "timestamp",
        }
    )

    transformer = AttributionAnalyzer(attribution_model=model)
    transformer.fit(data_renamed)
    return transformer.transform(data_renamed)


def analyze_funnel(
    data: pd.DataFrame, steps: Optional[List[str]] = None
) -> FunnelAnalysisResult:
    """
    Perform marketing funnel analysis.

    Parameters:
    -----------
    data : pd.DataFrame
        Funnel data with columns for each step (boolean values)
    steps : list of str, optional
        Ordered list of funnel step names

    Returns:
    --------
    result : FunnelAnalysisResult
        Complete funnel analysis results
    """
    transformer = FunnelAnalyzer(steps=steps)
    transformer.fit(data)
    return transformer.transform(data)


# Integration with existing statistical domain for enhanced A/B testing
def enhanced_ab_test(
    data: pd.DataFrame,
    group_column: str = "group",
    outcome_column: str = "converted",
    use_statistical_domain: bool = True,
    **kwargs,
) -> Dict[str, Any]:
    """
    Enhanced A/B test analysis leveraging both BI and statistical analysis domains.

    Parameters:
    -----------
    data : pd.DataFrame
        A/B test data
    group_column : str, default='group'
        Group assignment column
    outcome_column : str, default='converted'
        Outcome variable column
    use_statistical_domain : bool, default=True
        Whether to include statistical domain analysis
    **kwargs : additional arguments
        Additional arguments for statistical tests

    Returns:
    --------
    results : Dict[str, Any]
        Combined BI and statistical analysis results
    """
    results = {}

    # Business Intelligence A/B test analysis
    bi_result = perform_ab_test(data, group_column, outcome_column, **kwargs)
    results["business_intelligence"] = bi_result.to_dict()

    if use_statistical_domain:
        try:
            # Import and use statistical analysis domain for additional rigor
            from ..statistical_analysis import run_hypothesis_test

            # Prepare data for statistical analysis
            groups = data[group_column].unique()
            if len(groups) == 2:
                group_a_data = data[data[group_column] == groups[0]][outcome_column]
                group_b_data = data[data[group_column] == groups[1]][outcome_column]

                # Run statistical hypothesis test
                stat_result = run_hypothesis_test(
                    group_a_data.values,
                    group_b_data.values,
                    test_type="ttest_ind",
                    alpha=kwargs.get("alpha", 0.05),
                )
                results["statistical_analysis"] = stat_result

        except ImportError:
            logger.warning(
                "Statistical analysis domain not available for enhanced A/B testing"
            )

    return results
