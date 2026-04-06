"""
Business Intelligence Domain - Cohort Analysis and CLV Calculation.

This module implements cohort analysis for customer retention tracking
and Customer Lifetime Value (CLV) calculation using various methods.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from ...logging_manager import get_logger
from .models import CLVResult, CohortAnalysisResult

logger = get_logger(__name__)


class CohortAnalysisTransformer(BaseEstimator, TransformerMixin):
    """
    sklearn-compatible transformer for cohort analysis and retention tracking.

    Analyzes customer retention patterns over time by grouping customers into
    cohorts based on their first purchase date and tracking their subsequent
    purchase behavior.

    Parameters:
    -----------
    date_column : str, default='date'
        Name of the column containing transaction dates
    customer_column : str, default='customer_id'
        Name of the column containing customer identifiers
    period_type : str, default='monthly'
        Cohort period type: 'daily', 'weekly', 'monthly', 'quarterly'
    """

    def __init__(
        self, date_column="date", customer_column="customer_id", period_type="monthly"
    ):
        self.date_column = date_column
        self.customer_column = customer_column
        self.period_type = period_type

    def fit(self, X, y=None):
        """
        Fit the cohort analyzer to the data.

        Parameters:
        -----------
        X : pd.DataFrame
            Transaction data with customer and date columns
        y : ignored
            Not used, present for sklearn compatibility

        Returns:
        --------
        self : object
            Fitted transformer
        """
        # Validate input data
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")

        required_columns = [self.customer_column, self.date_column]
        missing_columns = [col for col in required_columns if col not in X.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        self.is_fitted_ = True
        return self

    def transform(self, X):
        """
        Transform the data to perform cohort analysis.

        Parameters:
        -----------
        X : pd.DataFrame
            Transaction data to analyze

        Returns:
        --------
        result : CohortAnalysisResult
            Complete cohort analysis results
        """
        check_is_fitted(self, "is_fitted_")

        logger.info(f"Performing {self.period_type} cohort analysis")

        # Prepare data
        X_work = X.copy()
        X_work[self.date_column] = pd.to_datetime(X_work[self.date_column])

        # Create cohort and period columns
        cohort_data = self._prepare_cohort_data(X_work)

        # Create cohort table
        cohort_table = self._create_cohort_table(cohort_data)

        # Calculate cohort sizes
        cohort_sizes = self._calculate_cohort_sizes(cohort_data)

        # Calculate retention rates
        retention_rates = self._calculate_retention_rates(cohort_table, cohort_sizes)

        # Generate period summary
        period_summary = self._generate_period_summary(retention_rates)

        return CohortAnalysisResult(
            cohort_table=cohort_table,
            cohort_sizes=cohort_sizes,
            retention_rates=retention_rates,
            period_summary=period_summary,
        )

    def _prepare_cohort_data(self, X):
        """Prepare data with cohort and period columns."""
        # Determine first purchase date for each customer
        customer_cohorts = (
            X.groupby(self.customer_column)[self.date_column].min().reset_index()
        )
        customer_cohorts.columns = [self.customer_column, "cohort_group"]

        # Create period columns based on period_type
        if self.period_type == "daily":
            X["order_period"] = X[self.date_column].dt.date
            customer_cohorts["cohort_group"] = customer_cohorts["cohort_group"].dt.date
        elif self.period_type == "weekly":
            X["order_period"] = X[self.date_column].dt.to_period("W")
            customer_cohorts["cohort_group"] = customer_cohorts[
                "cohort_group"
            ].dt.to_period("W")
        elif self.period_type == "monthly":
            X["order_period"] = X[self.date_column].dt.to_period("M")
            customer_cohorts["cohort_group"] = customer_cohorts[
                "cohort_group"
            ].dt.to_period("M")
        elif self.period_type == "quarterly":
            X["order_period"] = X[self.date_column].dt.to_period("Q")
            customer_cohorts["cohort_group"] = customer_cohorts[
                "cohort_group"
            ].dt.to_period("Q")
        else:
            raise ValueError(f"Unsupported period_type: {self.period_type}")

        # Merge cohort info back to transaction data
        cohort_data = X.merge(customer_cohorts, on=self.customer_column)

        return cohort_data

    def _create_cohort_table(self, cohort_data):
        """Create cohort table showing customer counts by cohort and period."""
        cohort_table = (
            cohort_data.groupby(["cohort_group", "order_period"])[self.customer_column]
            .nunique()
            .reset_index()
        )
        cohort_table = cohort_table.pivot(
            index="cohort_group", columns="order_period", values=self.customer_column
        )
        cohort_table.fillna(0, inplace=True)

        return cohort_table

    def _calculate_cohort_sizes(self, cohort_data):
        """Calculate the size of each cohort."""
        cohort_sizes = cohort_data.groupby("cohort_group")[
            self.customer_column
        ].nunique()
        return cohort_sizes.to_frame("cohort_size")

    def _calculate_retention_rates(self, cohort_table, cohort_sizes):
        """Calculate retention rates as percentage of original cohort."""
        retention_rates = cohort_table.divide(cohort_sizes["cohort_size"], axis=0)
        return retention_rates

    def _generate_period_summary(self, retention_rates):
        """Generate summary statistics for retention analysis."""
        # Calculate average retention rates by period
        period_averages = retention_rates.mean(axis=0).to_dict()

        # Calculate retention rate ranges
        period_summary = {
            "average_retention_by_period": period_averages,
            "overall_average_retention": retention_rates.values[
                retention_rates.values > 0
            ].mean(),
            "cohort_count": len(retention_rates),
            "analysis_period_type": self.period_type,
        }

        return period_summary


class CLVCalculator(BaseEstimator, TransformerMixin):
    """
    sklearn-compatible transformer for Customer Lifetime Value (CLV) calculation.

    Calculates CLV using various methods including historical average,
    predictive modeling, and cohort-based approaches.

    Parameters:
    -----------
    method : str, default='historical'
        CLV calculation method: 'historical', 'predictive', 'cohort'
    prediction_periods : int, default=12
        Number of periods to predict for CLV calculation
    model_type : str, default='random_forest'
        Machine learning model for predictive CLV: 'random_forest', 'gradient_boosting'
    """

    def __init__(
        self, method="historical", prediction_periods=12, model_type="random_forest"
    ):
        self.method = method
        self.prediction_periods = prediction_periods
        self.model_type = model_type

    def fit(self, X, y=None):
        """Fit the CLV calculator to the data."""
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")

        self.is_fitted_ = True
        return self

    def transform(self, X):
        """Calculate CLV for customers."""
        check_is_fitted(self, "is_fitted_")

        if self.method == "historical":
            return self._calculate_historical_clv(X)
        elif self.method == "predictive":
            return self._calculate_predictive_clv(X)
        elif self.method == "cohort":
            return self._calculate_cohort_clv(X)
        else:
            raise ValueError(f"Unsupported CLV method: {self.method}")

    def _calculate_historical_clv(self, X):
        """Calculate CLV based on historical data."""
        # Implementation for historical CLV
        logger.info("Calculating historical CLV")

        # Simple CLV = Average Order Value * Purchase Frequency * Gross Margin * Lifespan
        customer_metrics = X.groupby("customer_id").agg(
            total_spent=("amount", "sum"),
            avg_order_value=("amount", "mean"),
            order_count=("amount", "count"),
            first_purchase=("date", "min"),
            last_purchase=("date", "max"),
        )
        # Round only numeric columns to avoid warnings on datetime columns
        numeric_cols = customer_metrics.select_dtypes(include="number").columns
        customer_metrics[numeric_cols] = customer_metrics[numeric_cols].round(2)

        # Calculate customer lifespan in days
        customer_metrics["lifespan_days"] = (
            customer_metrics["last_purchase"] - customer_metrics["first_purchase"]
        ).dt.days
        customer_metrics["lifespan_days"] = customer_metrics["lifespan_days"].fillna(0)

        # Calculate purchase frequency (orders per day)
        customer_metrics["purchase_frequency"] = customer_metrics["order_count"] / (
            customer_metrics["lifespan_days"] + 1
        )

        # Simple CLV calculation (assuming 20% gross margin)
        gross_margin = 0.2
        customer_metrics["clv_estimate"] = (
            customer_metrics["avg_order_value"]
            * customer_metrics["purchase_frequency"]
            * gross_margin
            * 365  # annualize
        )

        # Distribution statistics
        clv_distribution = {
            "mean": customer_metrics["clv_estimate"].mean(),
            "median": customer_metrics["clv_estimate"].median(),
            "std": customer_metrics["clv_estimate"].std(),
            "min": customer_metrics["clv_estimate"].min(),
            "max": customer_metrics["clv_estimate"].max(),
        }

        return CLVResult(
            clv_scores=customer_metrics.reset_index(),
            model_metrics={
                "method": "historical",
                "gross_margin_assumed": gross_margin,
            },
            clv_distribution=clv_distribution,
            segment_clv={},
        )
