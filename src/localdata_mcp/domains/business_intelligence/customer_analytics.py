"""
Business Intelligence Domain - RFM Analysis Transformer.

This module implements the RFM (Recency, Frequency, Monetary) analysis
transformer for customer segmentation based on transaction behavior patterns.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from ...logging_manager import get_logger
from .models import RFMResult

logger = get_logger(__name__)


class RFMAnalysisTransformer(BaseEstimator, TransformerMixin):
    """
    sklearn-compatible transformer for RFM (Recency, Frequency, Monetary) analysis.

    Performs customer segmentation based on transaction behavior patterns,
    calculating RFM scores and creating meaningful customer segments for
    targeted marketing and retention strategies.

    Parameters:
    -----------
    date_column : str, default='date'
        Name of the column containing transaction dates
    customer_column : str, default='customer_id'
        Name of the column containing customer identifiers
    amount_column : str, default='amount'
        Name of the column containing transaction amounts
    analysis_date : str or datetime, default=None
        Reference date for recency calculation (defaults to max date in data)
    quartiles : bool, default=True
        Whether to use quartiles for scoring (True) or custom bins (False)
    """

    def __init__(
        self,
        date_column="date",
        customer_column="customer_id",
        amount_column="amount",
        analysis_date=None,
        quartiles=True,
    ):
        self.date_column = date_column
        self.customer_column = customer_column
        self.amount_column = amount_column
        self.analysis_date = analysis_date
        self.quartiles = quartiles

    def fit(self, X, y=None):
        """
        Fit the RFM analyzer to the data.

        Parameters:
        -----------
        X : pd.DataFrame
            Transaction data with customer, date, and amount columns
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

        required_columns = [self.customer_column, self.date_column, self.amount_column]
        missing_columns = [col for col in required_columns if col not in X.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Convert date column to datetime if not already
        X_work = X.copy()
        X_work[self.date_column] = pd.to_datetime(X_work[self.date_column])

        # Set analysis date
        if self.analysis_date is None:
            self.analysis_date_ = X_work[self.date_column].max()
        else:
            self.analysis_date_ = pd.to_datetime(self.analysis_date)

        # Calculate RFM metrics
        rfm_data = self._calculate_rfm_metrics(X_work)

        # Calculate scoring thresholds
        if rfm_data.empty:
            # Use reasonable business-based thresholds for empty data
            self.recency_thresholds_ = np.array([30, 90, 180])
            self.frequency_thresholds_ = np.array([2, 5, 10])
            self.monetary_thresholds_ = np.array([100, 500, 1000])
        elif self.quartiles:
            self.recency_thresholds_ = (
                rfm_data["recency"].quantile([0.25, 0.5, 0.75]).values
            )
            self.frequency_thresholds_ = (
                rfm_data["frequency"].quantile([0.25, 0.5, 0.75]).values
            )
            self.monetary_thresholds_ = (
                rfm_data["monetary"].quantile([0.25, 0.5, 0.75]).values
            )
        else:
            # Use reasonable business-based thresholds
            self.recency_thresholds_ = [30, 90, 180]  # days
            self.frequency_thresholds_ = [2, 5, 10]  # number of orders
            self.monetary_thresholds_ = [100, 500, 1000]  # currency units

        self.is_fitted_ = True
        return self

    def transform(self, X):
        """
        Transform the data to include RFM scores and segments.

        Parameters:
        -----------
        X : pd.DataFrame
            Transaction data to transform

        Returns:
        --------
        result : RFMResult
            Complete RFM analysis results including scores, segments, and summary
        """
        check_is_fitted(self, "is_fitted_")

        # Calculate RFM metrics
        X_work = X.copy()
        X_work[self.date_column] = pd.to_datetime(X_work[self.date_column])
        rfm_data = self._calculate_rfm_metrics(X_work)

        # Handle empty data
        if rfm_data.empty:
            empty_scores = rfm_data.copy()
            empty_segments = rfm_data.copy()
            empty_summary = pd.DataFrame()
            return RFMResult(
                rfm_scores=empty_scores,
                segments=empty_segments,
                segment_summary=empty_summary,
                quartile_boundaries={
                    "recency": self.recency_thresholds_.tolist(),
                    "frequency": self.frequency_thresholds_.tolist(),
                    "monetary": self.monetary_thresholds_.tolist(),
                },
            )

        # Calculate RFM scores
        rfm_scores = self._calculate_rfm_scores(rfm_data)

        # Create customer segments
        segments = self._create_customer_segments(rfm_scores)

        # Generate segment summary
        segment_summary = self._generate_segment_summary(segments)

        # Prepare quartile boundaries info
        quartile_boundaries = {
            "recency": self.recency_thresholds_.tolist(),
            "frequency": self.frequency_thresholds_.tolist(),
            "monetary": self.monetary_thresholds_.tolist(),
        }

        return RFMResult(
            rfm_scores=rfm_scores,
            segments=segments,
            segment_summary=segment_summary,
            quartile_boundaries=quartile_boundaries,
        )

    def _calculate_rfm_metrics(self, X):
        """Calculate base RFM metrics from transaction data."""
        logger.info("Calculating RFM metrics for customer segmentation")

        # Use named aggregation to avoid column duplication when the groupby
        # key would otherwise collide with an aggregation column on reset_index.
        rfm_data = (
            X.groupby(self.customer_column)
            .agg(
                recency=(
                    self.date_column,
                    lambda x: (self.analysis_date_ - x.max()).days,
                ),
                frequency=(self.amount_column, "count"),
                monetary=(self.amount_column, "sum"),
            )
            .reset_index()
        )

        # Handle empty data: pandas infers datetime64 dtype for recency from
        # NaT propagation on empty groups, which makes numeric clip fail.
        if rfm_data.empty:
            rfm_data["recency"] = rfm_data["recency"].astype("int64")
            rfm_data["monetary"] = rfm_data["monetary"].astype("float64")
            return rfm_data

        # Handle edge cases
        rfm_data["recency"] = rfm_data["recency"].clip(lower=0)
        rfm_data["frequency"] = rfm_data["frequency"].clip(lower=1)
        rfm_data["monetary"] = rfm_data["monetary"].clip(lower=0)

        return rfm_data

    def _calculate_rfm_scores(self, rfm_data):
        """Calculate RFM scores based on thresholds."""
        rfm_scores = rfm_data.copy()

        # Recency score (lower recency = higher score)
        rfm_scores["R"] = pd.cut(
            rfm_scores["recency"],
            bins=[-np.inf] + self.recency_thresholds_.tolist() + [np.inf],
            labels=[4, 3, 2, 1],
            include_lowest=True,
        ).astype(int)

        # Frequency score (higher frequency = higher score)
        rfm_scores["F"] = pd.cut(
            rfm_scores["frequency"],
            bins=[-np.inf] + self.frequency_thresholds_.tolist() + [np.inf],
            labels=[1, 2, 3, 4],
            include_lowest=True,
        ).astype(int)

        # Monetary score (higher monetary = higher score)
        rfm_scores["M"] = pd.cut(
            rfm_scores["monetary"],
            bins=[-np.inf] + self.monetary_thresholds_.tolist() + [np.inf],
            labels=[1, 2, 3, 4],
            include_lowest=True,
        ).astype(int)

        # Combined RFM score
        rfm_scores["RFM_Score"] = (
            rfm_scores["R"].astype(str)
            + rfm_scores["F"].astype(str)
            + rfm_scores["M"].astype(str)
        )

        return rfm_scores

    def _create_customer_segments(self, rfm_scores):
        """Create meaningful customer segments from RFM scores."""
        segments = rfm_scores.copy()

        # Define segment rules based on RFM scores
        def assign_segment(row):
            r, f, m = row["R"], row["F"], row["M"]

            # Champions: High value, frequent, recent customers
            if r >= 4 and f >= 4 and m >= 3:
                return "Champions"
            # Loyal Customers: High frequency, good monetary
            elif f >= 3 and m >= 3:
                return "Loyal Customers"
            # Potential Loyalists: Recent customers with potential
            elif r >= 3 and f >= 2:
                return "Potential Loyalists"
            # Recent Customers: Recent but low frequency/monetary
            elif r >= 3:
                return "Recent Customers"
            # Promising: Decent frequency and monetary but not recent
            elif f >= 2 and m >= 2:
                return "Promising"
            # Need Attention: Good customers who haven't purchased recently
            elif r <= 2 and f >= 3 and m >= 3:
                return "Need Attention"
            # About to Sleep: Low recency, was frequent
            elif r <= 2 and f >= 2:
                return "About to Sleep"
            # At Risk: Low across metrics but some history
            elif f >= 1 and m >= 1:
                return "At Risk"
            # Lost: Very low across all metrics
            else:
                return "Lost"

        segments["Segment"] = segments.apply(assign_segment, axis=1)

        return segments

    def _generate_segment_summary(self, segments):
        """Generate summary statistics for each segment."""
        summary = (
            segments.groupby("Segment")
            .agg(
                customer_count=(self.customer_column, "count"),
                recency_mean=("recency", "mean"),
                recency_median=("recency", "median"),
                frequency_mean=("frequency", "mean"),
                frequency_median=("frequency", "median"),
                monetary_mean=("monetary", "mean"),
                monetary_median=("monetary", "median"),
                monetary_sum=("monetary", "sum"),
            )
            .round(2)
        )

        # Add percentage of total customers
        total_customers = len(segments)
        summary["percentage"] = (
            summary["customer_count"] / total_customers * 100
        ).round(1)

        return summary.reset_index()
