"""
DistributionAnalyzerTransformer - sklearn-compatible transformer for distribution analysis.
"""

import json
import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted

from ...logging_manager import get_logger
from ..base import PipelineState

logger = get_logger(__name__)


class DistributionAnalyzerTransformer(BaseEstimator, TransformerMixin):
    """
    sklearn-compatible transformer for distribution analysis.

    Wraps the existing analyze_distributions functionality in a sklearn pipeline-compatible interface
    while preserving histogram generation, percentile calculations, and statistical pattern detection.

    Parameters:
    -----------
    sample_size : int, default=10000
        Number of rows to sample for distribution analysis
    bins : int, default=20
        Number of bins for histogram generation
    percentiles : list, default=None
        List of percentiles to calculate (if None, uses default set)

    Attributes:
    -----------
    distributions_ : dict
        Analyzed distributions after fitting
    feature_names_in_ : ndarray of shape (n_features,)
        Names of features seen during fit
    n_features_in_ : int
        Number of features seen during fit
    state_ : PipelineState
        Current transformer state
    """

    def __init__(
        self,
        sample_size: int = 10000,
        bins: int = 20,
        percentiles: Optional[List[float]] = None,
    ):
        self.sample_size = sample_size
        self.bins = bins
        self.percentiles = percentiles or [1, 5, 10, 25, 50, 75, 90, 95, 99]

        # Internal state
        self.state_ = PipelineState.INITIALIZED
        self.distributions_ = None
        self.feature_names_in_ = None
        self.n_features_in_ = None

    def fit(self, X, y=None):
        """
        Fit the distribution analyzer by analyzing the input data.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features) or pandas.DataFrame
            Training data to analyze distributions
        y : array-like of shape (n_samples,), default=None
            Target values (ignored)

        Returns:
        --------
        self : DistributionAnalyzerTransformer
            Fitted transformer
        """
        # Input validation and conversion
        if isinstance(X, pd.DataFrame):
            df = X.copy()
            self.feature_names_in_ = np.array(X.columns)
        else:
            X = check_array(X, accept_sparse=False, force_all_finite=False)
            df = pd.DataFrame(X)
            self.feature_names_in_ = np.array(
                [f"feature_{i}" for i in range(X.shape[1])]
            )

        self.n_features_in_ = df.shape[1]
        self.state_ = PipelineState.EXECUTING

        try:
            # Apply sampling if specified
            if self.sample_size > 0 and len(df) > self.sample_size:
                df = df.sample(n=self.sample_size, random_state=42)

            # Analyze distributions using existing logic
            self.distributions_ = self._analyze_distributions(df)

            # Add metadata
            self.distributions_["metadata"] = {
                "source_type": "transformer_input",
                "sample_size": self.sample_size,
                "actual_rows_analyzed": len(df),
                "analysis_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "bins": self.bins,
                "percentiles": self.percentiles,
                "pipeline_state": self.state_.value,
            }

            self.state_ = PipelineState.FITTED
            logger.info(
                f"DistributionAnalyzerTransformer fitted successfully with {len(df)} rows, {df.shape[1]} columns"
            )

        except Exception as e:
            self.state_ = PipelineState.ERROR
            logger.error(f"Error fitting DistributionAnalyzerTransformer: {e}")
            raise

        return self

    def transform(self, X):
        """
        Transform is identity for distribution analysis - returns input unchanged.
        Distribution data is available via get_distributions() method.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Input data

        Returns:
        --------
        X : array-like of shape (n_samples, n_features)
            Unchanged input data
        """
        check_is_fitted(self)

        if isinstance(X, pd.DataFrame):
            return X
        else:
            return check_array(X, accept_sparse=False, force_all_finite=False)

    def get_distributions(self) -> Dict[str, Any]:
        """
        Get the analyzed distributions.

        Returns:
        --------
        distributions : dict
            Distribution analysis with histograms, percentiles, and patterns
        """
        check_is_fitted(self)
        return self.distributions_

    def get_distributions_json(self) -> str:
        """
        Get the distribution analysis as JSON string (backward compatibility).

        Returns:
        --------
        distributions_json : str
            Distribution analysis in JSON format
        """
        check_is_fitted(self)
        return json.dumps(self.distributions_, indent=2, default=str)

    def get_feature_names_out(self, input_features=None):
        """
        Get output feature names for transformation (sklearn compatibility).

        Parameters:
        -----------
        input_features : array-like of str or None, default=None
            Input features. If None, uses feature_names_in_.

        Returns:
        --------
        feature_names_out : ndarray of str
            Transformed feature names (same as input for distribution analysis)
        """
        check_is_fitted(self)

        if input_features is None:
            return self.feature_names_in_.copy()
        else:
            return np.array(input_features)

    def get_composition_metadata(self) -> Dict[str, Any]:
        """
        Get metadata for pipeline composition and tool chaining.

        Returns:
        --------
        composition_metadata : dict
            Metadata for downstream pipeline composition including:
            - Distribution characteristics and patterns
            - Statistical summaries for each column
            - Normality and outlier information
            - Processing hints for downstream tools
        """
        check_is_fitted(self)

        if not self.distributions_:
            return {}

        # Extract composition-relevant metadata
        metadata = {
            "tool_type": "distribution_analyzer",
            "processing_stage": "statistical_analysis",
            "summary": self.distributions_.get("summary", {}),
            "distribution_patterns": {},
            "normality_tests": {},
            "outlier_information": {},
            "processing_hints": {},
            "recommended_next_steps": [],
        }

        # Extract distribution information for each column
        for col_name, dist_info in self.distributions_.get("distributions", {}).items():
            if dist_info.get("type") == "numeric":
                self._extract_numeric_metadata(col_name, dist_info, metadata)

            elif dist_info.get("type") == "categorical":
                self._extract_categorical_metadata(col_name, dist_info, metadata)

        # Generate recommended next steps based on patterns
        self._generate_recommended_steps(metadata)

        return metadata

    def _extract_numeric_metadata(
        self, col_name: str, dist_info: Dict[str, Any], metadata: Dict[str, Any]
    ) -> None:
        """Extract numeric distribution metadata for composition."""
        shape_metrics = dist_info.get("shape_metrics", {})
        metadata["distribution_patterns"][col_name] = {
            "type": "numeric",
            "is_normal": shape_metrics.get("is_normal_distributed", False),
            "skewness": shape_metrics.get("skewness", 0),
            "kurtosis": shape_metrics.get("kurtosis", 0),
            "outliers_count": shape_metrics.get("outliers_count", 0),
        }

        # Normality test results
        metadata["normality_tests"][col_name] = {
            "is_normal": shape_metrics.get("is_normal_distributed", False),
            "recommendation": "parametric_tests"
            if shape_metrics.get("is_normal_distributed")
            else "non_parametric_tests",
        }

        # Outlier information
        outliers_count = shape_metrics.get("outliers_count", 0)
        if outliers_count > 0:
            metadata["outlier_information"][col_name] = {
                "count": outliers_count,
                "percentage": (
                    outliers_count / dist_info.get("summary_stats", {}).get("count", 1)
                )
                * 100,
                "treatment_needed": outliers_count
                > dist_info.get("summary_stats", {}).get("count", 0)
                * 0.05,  # More than 5%
            }

        # Processing hints for numeric columns
        hints = []
        if not shape_metrics.get("is_normal_distributed", True):
            hints.extend(["log_transformation", "box_cox_transformation"])
        if outliers_count > 0:
            hints.extend(["outlier_treatment", "robust_scaling"])
        if abs(shape_metrics.get("skewness", 0)) > 1:
            hints.append("skewness_correction")
        hints.append("standardization")

        metadata["processing_hints"][col_name] = hints

    def _extract_categorical_metadata(
        self, col_name: str, dist_info: Dict[str, Any], metadata: Dict[str, Any]
    ) -> None:
        """Extract categorical distribution metadata for composition."""
        dist_metrics = dist_info.get("distribution_metrics", {})
        metadata["distribution_patterns"][col_name] = {
            "type": "categorical",
            "entropy": dist_metrics.get("entropy", 0),
            "uniformity": dist_metrics.get("uniformity_score", 0),
            "concentration_ratio": dist_metrics.get("concentration_ratio", 0),
        }

        # Processing hints for categorical columns
        hints = []
        if dist_metrics.get("entropy", 0) > 3:  # High entropy
            hints.append("dimensionality_reduction")
        if dist_metrics.get("concentration_ratio", 0) > 0.8:  # Highly concentrated
            hints.append("rare_category_handling")
        hints.extend(["one_hot_encoding", "label_encoding"])

        metadata["processing_hints"][col_name] = hints

    def _generate_recommended_steps(self, metadata: Dict[str, Any]) -> None:
        """Generate recommended next steps based on distribution patterns."""
        numeric_cols = [
            col
            for col, info in metadata["distribution_patterns"].items()
            if info.get("type") == "numeric"
        ]
        categorical_cols = [
            col
            for col, info in metadata["distribution_patterns"].items()
            if info.get("type") == "categorical"
        ]

        if numeric_cols:
            metadata["recommended_next_steps"].extend(
                ["feature_scaling", "correlation_analysis"]
            )
            if any(
                info.get("outliers_count", 0) > 0
                for info in metadata["outlier_information"].values()
            ):
                metadata["recommended_next_steps"].append("outlier_treatment")
            if any(
                not info.get("is_normal")
                for info in metadata["normality_tests"].values()
            ):
                metadata["recommended_next_steps"].append("normality_transformation")

        if categorical_cols:
            metadata["recommended_next_steps"].extend(
                ["categorical_encoding", "feature_engineering"]
            )

        if numeric_cols and categorical_cols:
            metadata["recommended_next_steps"].append("mixed_type_preprocessing")

    def _analyze_distributions(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze distributions for all columns in DataFrame.

        This method replicates the existing analyze_distributions logic
        to maintain 100% compatibility with the original implementation.
        """
        results = {
            "summary": {
                "total_columns": len(df.columns),
                "numeric_columns": 0,
                "categorical_columns": 0,
                "analyzed_columns": 0,
            },
            "distributions": {},
        }

        for column in df.columns:
            col_data = df[column].dropna()

            if len(col_data) == 0:
                results["distributions"][column] = {
                    "type": "empty",
                    "analysis": "No data available for analysis",
                }
                continue

            results["summary"]["analyzed_columns"] += 1

            # Analyze based on data type
            if pd.api.types.is_numeric_dtype(col_data):
                results["summary"]["numeric_columns"] += 1
                results["distributions"][column] = self._analyze_numeric_distribution(
                    col_data
                )
            else:
                results["summary"]["categorical_columns"] += 1
                results["distributions"][column] = (
                    self._analyze_categorical_distribution(col_data)
                )

        return results

    def _analyze_numeric_distribution(self, series: pd.Series) -> Dict[str, Any]:
        """Analyze numeric column distribution."""
        analysis = {
            "type": "numeric",
            "summary_stats": {
                "count": len(series),
                "mean": float(series.mean()),
                "std": float(series.std()) if len(series) > 1 else 0,
                "min": float(series.min()),
                "max": float(series.max()),
            },
        }

        # Percentiles
        try:
            percentile_values = series.quantile([p / 100 for p in self.percentiles])
            analysis["percentiles"] = {
                f"p{p}": float(percentile_values[p / 100]) for p in self.percentiles
            }
        except Exception as e:
            logger.warning(f"Could not calculate percentiles: {e}")
            analysis["percentiles"] = {}

        # Histogram
        try:
            hist, bin_edges = np.histogram(series.values, bins=self.bins)
            analysis["histogram"] = {
                "bins": self.bins,
                "counts": [int(x) for x in hist],
                "bin_edges": [float(x) for x in bin_edges],
                "bin_centers": [
                    float((bin_edges[i] + bin_edges[i + 1]) / 2)
                    for i in range(len(bin_edges) - 1)
                ],
            }
        except Exception as e:
            logger.warning(f"Could not generate histogram: {e}")
            analysis["histogram"] = None

        # Distribution shape analysis
        try:
            analysis["shape_metrics"] = {
                "skewness": float(series.skew()) if len(series) > 2 else 0,
                "kurtosis": float(series.kurtosis()) if len(series) > 3 else 0,
                "is_normal_distributed": self._test_normality(series),
                "outliers_count": self._count_outliers(series),
            }
        except Exception as e:
            logger.warning(f"Could not analyze distribution shape: {e}")
            analysis["shape_metrics"] = {}

        return analysis

    def _analyze_categorical_distribution(self, series: pd.Series) -> Dict[str, Any]:
        """Analyze categorical column distribution."""
        value_counts = series.value_counts()

        analysis = {
            "type": "categorical",
            "summary_stats": {
                "count": len(series),
                "unique_values": len(value_counts),
                "most_frequent": str(value_counts.index[0])
                if len(value_counts) > 0
                else None,
                "most_frequent_count": int(value_counts.iloc[0])
                if len(value_counts) > 0
                else 0,
            },
        }

        # Value distribution
        analysis["value_distribution"] = {
            str(val): int(count) for val, count in value_counts.head(20).items()
        }

        # Distribution metrics
        analysis["distribution_metrics"] = {
            "entropy": self._calculate_entropy(value_counts),
            "concentration_ratio": float(value_counts.iloc[0] / len(series))
            if len(value_counts) > 0
            else 0,
            "uniformity_score": self._calculate_uniformity(value_counts),
        }

        return analysis

    def _test_normality(self, series: pd.Series) -> bool:
        """Simple normality test based on skewness and kurtosis."""
        try:
            skew = abs(series.skew())
            kurt = abs(series.kurtosis())
            # Simple heuristic: normal if skewness < 1 and kurtosis < 3
            return skew < 1.0 and kurt < 3.0
        except:
            return False

    def _count_outliers(self, series: pd.Series) -> int:
        """Count outliers using IQR method."""
        try:
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers = series[(series < lower_bound) | (series > upper_bound)]
            return len(outliers)
        except:
            return 0

    def _calculate_entropy(self, value_counts: pd.Series) -> float:
        """Calculate entropy of categorical distribution."""
        try:
            probabilities = value_counts / value_counts.sum()
            entropy = -sum(p * np.log2(p) for p in probabilities if p > 0)
            return float(entropy)
        except:
            return 0.0

    def _calculate_uniformity(self, value_counts: pd.Series) -> float:
        """Calculate uniformity score (0 = highly skewed, 1 = perfectly uniform)."""
        try:
            if len(value_counts) <= 1:
                return 1.0
            expected_count = value_counts.sum() / len(value_counts)
            deviations = [(count - expected_count) ** 2 for count in value_counts]
            mse = sum(deviations) / len(value_counts)
            # Normalize to 0-1 scale
            max_possible_mse = (value_counts.sum() ** 2) / len(value_counts)
            uniformity = 1 - (mse / max_possible_mse) if max_possible_mse > 0 else 1
            return float(max(0, min(1, uniformity)))
        except:
            return 0.0
