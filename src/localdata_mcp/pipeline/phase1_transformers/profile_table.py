"""
ProfileTableTransformer - sklearn-compatible transformer for comprehensive data profiling.
"""

import json
import time
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted

from ...logging_manager import get_logger
from ..base import PipelineState

logger = get_logger(__name__)


class ProfileTableTransformer(BaseEstimator, TransformerMixin):
    """
    sklearn-compatible transformer for comprehensive data profiling.

    Wraps the existing profile_table functionality in a sklearn pipeline-compatible interface
    while preserving all original capabilities including streaming, sampling, and distribution analysis.

    Parameters:
    -----------
    sample_size : int, default=10000
        Number of rows to sample for analysis (0 = all rows)
    include_distributions : bool, default=True
        Whether to include distribution analysis for numeric columns
    connection_name : str, default=None
        Database connection name (if using database source)
    table_name : str, default=None
        Table name to profile (mutually exclusive with query)
    query : str, default=None
        Custom SQL query to profile (mutually exclusive with table_name)

    Attributes:
    -----------
    profile_ : dict
        Generated data profile after fitting
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
        include_distributions: bool = True,
        connection_name: Optional[str] = None,
        table_name: Optional[str] = None,
        query: Optional[str] = None,
    ):
        self.sample_size = sample_size
        self.include_distributions = include_distributions
        self.connection_name = connection_name
        self.table_name = table_name
        self.query = query

        # Internal state
        self.state_ = PipelineState.INITIALIZED
        self.profile_ = None
        self.feature_names_in_ = None
        self.n_features_in_ = None

    def fit(self, X, y=None):
        """
        Fit the profiler by analyzing the input data.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features) or pandas.DataFrame
            Training data to profile
        y : array-like of shape (n_samples,), default=None
            Target values (ignored)

        Returns:
        --------
        self : ProfileTableTransformer
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

            # Generate comprehensive profile using the existing logic
            self.profile_ = self._generate_data_profile(df, self.include_distributions)

            # Add metadata
            self.profile_["metadata"] = {
                "source_type": "transformer_input",
                "sample_size": self.sample_size,
                "actual_rows_analyzed": len(df),
                "profiling_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "include_distributions": self.include_distributions,
                "pipeline_state": self.state_.value,
            }

            self.state_ = PipelineState.FITTED
            logger.info(
                f"ProfileTableTransformer fitted successfully with {len(df)} rows, {df.shape[1]} columns"
            )

        except Exception as e:
            self.state_ = PipelineState.ERROR
            logger.error(f"Error fitting ProfileTableTransformer: {e}")
            raise

        return self

    def transform(self, X):
        """
        Transform is identity for profiling - returns input unchanged.
        Profile data is available via get_profile() method.

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

    def get_profile(self) -> Dict[str, Any]:
        """
        Get the generated data profile.

        Returns:
        --------
        profile : dict
            Comprehensive data profile with statistics and quality metrics
        """
        check_is_fitted(self)
        return self.profile_

    def get_profile_json(self) -> str:
        """
        Get the generated data profile as JSON string (backward compatibility).

        Returns:
        --------
        profile_json : str
            Comprehensive data profile in JSON format
        """
        check_is_fitted(self)
        return json.dumps(self.profile_, indent=2, default=str)

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
            Transformed feature names (same as input for profiling)
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
            - Statistical summaries for each column
            - Data quality scores
            - Processing hints for downstream tools
            - Recommended next steps
        """
        check_is_fitted(self)

        if not self.profile_:
            return {}

        # Extract composition-relevant metadata
        metadata = {
            "tool_type": "profiler",
            "processing_stage": "data_understanding",
            "data_shape": {
                "rows": self.profile_["summary"]["total_rows"],
                "columns": self.profile_["summary"]["total_columns"],
            },
            "data_quality": self.profile_.get("data_quality", {}),
            "column_types": {},
            "processing_hints": {},
            "recommended_next_steps": [],
        }

        # Extract column-level information
        for col_name, col_info in self.profile_.get("columns", {}).items():
            metadata["column_types"][col_name] = {
                "data_type": col_info.get("data_type", "unknown"),
                "null_percentage": col_info.get("null_percentage", 0),
                "unique_percentage": col_info.get("unique_percentage", 0),
                "has_outliers": col_info.get("outliers", {}).get("count", 0) > 0
                if "outliers" in col_info
                else False,
            }

            # Generate processing hints
            hints = []
            if col_info.get("null_percentage", 0) > 5:
                hints.append("missing_value_imputation")
            if "outliers" in col_info and col_info["outliers"].get("count", 0) > 0:
                hints.append("outlier_handling")
            if pd.api.types.is_numeric_dtype(col_info.get("data_type", "")):
                hints.append("scaling_normalization")
            if col_info.get("unique_percentage", 0) > 95:
                hints.append("potential_identifier")

            metadata["processing_hints"][col_name] = hints

        # Generate recommended next steps
        overall_quality = metadata["data_quality"].get("overall_score", 100)
        if overall_quality < 80:
            metadata["recommended_next_steps"].append("data_cleaning")
        if any(
            "missing_value_imputation" in hints
            for hints in metadata["processing_hints"].values()
        ):
            metadata["recommended_next_steps"].append("missing_value_treatment")
        if any(
            "outlier_handling" in hints
            for hints in metadata["processing_hints"].values()
        ):
            metadata["recommended_next_steps"].append("outlier_analysis")
        if any(
            "scaling_normalization" in hints
            for hints in metadata["processing_hints"].values()
        ):
            metadata["recommended_next_steps"].append("feature_scaling")

        return metadata

    def _generate_data_profile(
        self, df: pd.DataFrame, include_distributions: bool = True
    ) -> Dict[str, Any]:
        """
        Generate comprehensive data profile for a DataFrame.

        This method replicates the existing _generate_data_profile logic
        to maintain 100% compatibility with the original implementation.

        Args:
            df: Pandas DataFrame to profile
            include_distributions: Whether to include distribution analysis

        Returns:
            Dictionary containing comprehensive profile data
        """
        profile = {
            "summary": {
                "total_rows": len(df),
                "total_columns": len(df.columns),
                "memory_usage_mb": df.memory_usage(deep=True).sum() / (1024 * 1024),
                "completeness_score": (
                    (df.notna().sum().sum()) / (len(df) * len(df.columns))
                )
                * 100
                if len(df) > 0
                else 0,
            },
            "columns": {},
        }

        # Profile each column
        for column in df.columns:
            col_data = df[column]
            col_profile = {
                "data_type": str(col_data.dtype),
                "non_null_count": int(col_data.notna().sum()),
                "null_count": int(col_data.isnull().sum()),
                "null_percentage": float(
                    (col_data.isnull().sum() / len(col_data)) * 100
                )
                if len(col_data) > 0
                else 0,
                "unique_count": int(col_data.nunique()),
                "unique_percentage": float((col_data.nunique() / len(col_data)) * 100)
                if len(col_data) > 0
                else 0,
                "memory_usage_bytes": int(col_data.memory_usage(deep=True)),
            }

            # Add basic statistics for non-null values
            non_null_data = col_data.dropna()

            if len(non_null_data) > 0:
                # Most common values
                value_counts = non_null_data.value_counts().head(5)
                col_profile["top_values"] = {
                    str(val): int(count) for val, count in value_counts.items()
                }

                # Type-specific analysis
                if pd.api.types.is_numeric_dtype(col_data):
                    col_profile.update(
                        self._profile_numeric_column(
                            non_null_data, include_distributions
                        )
                    )
                elif pd.api.types.is_datetime64_any_dtype(col_data):
                    col_profile.update(self._profile_datetime_column(non_null_data))
                else:
                    col_profile.update(self._profile_text_column(non_null_data))

            profile["columns"][column] = col_profile

        # Calculate data quality metrics
        profile["data_quality"] = self._calculate_data_quality_metrics(df)

        return profile

    def _profile_numeric_column(
        self, series: pd.Series, include_distributions: bool
    ) -> Dict[str, Any]:
        """Profile numeric column with statistical analysis."""
        profile = {
            "min_value": float(series.min()),
            "max_value": float(series.max()),
            "mean": float(series.mean()),
            "median": float(series.median()),
            "std_deviation": float(series.std()) if len(series) > 1 else 0,
            "variance": float(series.var()) if len(series) > 1 else 0,
            "skewness": float(series.skew()) if len(series) > 2 else 0,
            "kurtosis": float(series.kurtosis()) if len(series) > 3 else 0,
        }

        # Quartiles
        try:
            quartiles = series.quantile([0.25, 0.5, 0.75])
            q25_val = float(quartiles[0.25])
            q75_val = float(quartiles[0.75])

            # Add quartiles both as top-level values for easy access and nested for detail
            profile["q25"] = q25_val  # 25th percentile (Q1)
            profile["q75"] = q75_val  # 75th percentile (Q3)

            profile["quartiles"] = {
                "q1": q25_val,
                "q2": float(quartiles[0.5]),
                "q3": q75_val,
                "iqr": float(q75_val - q25_val),
            }
        except Exception:
            profile["quartiles"] = None
            profile["q25"] = None
            profile["q75"] = None

        # Outlier detection using IQR method
        if profile["quartiles"]:
            q1, q3 = profile["quartiles"]["q1"], profile["quartiles"]["q3"]
            iqr = profile["quartiles"]["iqr"]
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers = series[(series < lower_bound) | (series > upper_bound)]
            profile["outliers"] = {
                "count": int(len(outliers)),
                "percentage": float((len(outliers) / len(series)) * 100),
                "lower_bound": float(lower_bound),
                "upper_bound": float(upper_bound),
            }

        # Distribution analysis if requested
        if include_distributions:
            try:
                # Create histogram data
                hist, bin_edges = np.histogram(series.values, bins=20)
                profile["histogram"] = {
                    "counts": [int(x) for x in hist],
                    "bin_edges": [float(x) for x in bin_edges],
                }

                # Percentile distribution
                percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
                profile["percentiles"] = {
                    f"p{p}": float(series.quantile(p / 100)) for p in percentiles
                }
            except Exception as e:
                logger.warning(f"Could not generate distribution data: {e}")
                profile["histogram"] = None
                profile["percentiles"] = None

        return profile

    def _profile_datetime_column(self, series: pd.Series) -> Dict[str, Any]:
        """Profile datetime column with temporal analysis."""
        profile = {
            "min_date": str(series.min()),
            "max_date": str(series.max()),
            "date_range_days": (series.max() - series.min()).days,
        }

        # Extract time components for analysis
        try:
            profile["year_range"] = {
                "min_year": int(series.dt.year.min()),
                "max_year": int(series.dt.year.max()),
            }
            profile["month_distribution"] = series.dt.month.value_counts().to_dict()
            profile["weekday_distribution"] = (
                series.dt.day_name().value_counts().to_dict()
            )
        except Exception as e:
            logger.warning(f"Could not analyze datetime components: {e}")

        return profile

    def _profile_text_column(self, series: pd.Series) -> Dict[str, Any]:
        """Profile text column with string analysis."""
        str_series = series.astype(str)

        profile = {
            "min_length": int(str_series.str.len().min()),
            "max_length": int(str_series.str.len().max()),
            "avg_length": float(str_series.str.len().mean()),
            "std_length": float(str_series.str.len().std())
            if len(str_series) > 1
            else 0,
        }

        # Pattern analysis
        try:
            profile["patterns"] = {
                "contains_digits": int(str_series.str.contains(r"\d", na=False).sum()),
                "contains_letters": int(
                    str_series.str.contains(r"[a-zA-Z]", na=False).sum()
                ),
                "contains_special_chars": int(
                    str_series.str.contains(r"[^a-zA-Z0-9\s]", na=False).sum()
                ),
                "all_uppercase": int(str_series.str.isupper().sum()),
                "all_lowercase": int(str_series.str.islower().sum()),
            }
        except Exception as e:
            logger.warning(f"Could not analyze text patterns: {e}")
            profile["patterns"] = {}

        return profile

    def _calculate_data_quality_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate comprehensive data quality metrics."""
        total_cells = len(df) * len(df.columns)

        if total_cells == 0:
            return {
                "completeness": 0,
                "consistency": 0,
                "validity": 0,
                "accuracy": 0,
                "overall_score": 0,
            }

        # Completeness: percentage of non-null values
        non_null_cells = df.notna().sum().sum()
        completeness = (non_null_cells / total_cells) * 100

        # Consistency: low variance in data types and formats per column
        consistency_scores = []
        for column in df.columns:
            col_data = df[column].dropna()
            if len(col_data) > 0:
                # Simple consistency metric based on data type uniformity
                if pd.api.types.is_numeric_dtype(col_data):
                    consistency_scores.append(95)  # Numeric data generally consistent
                elif pd.api.types.is_datetime64_any_dtype(col_data):
                    consistency_scores.append(90)  # Datetime data generally consistent
                else:
                    # Text data consistency based on length variance
                    lengths = col_data.astype(str).str.len()
                    if len(lengths) > 1:
                        cv = lengths.std() / lengths.mean() if lengths.mean() > 0 else 1
                        consistency_scores.append(max(0, 100 - (cv * 50)))
                    else:
                        consistency_scores.append(100)
            else:
                consistency_scores.append(0)

        consistency = np.mean(consistency_scores) if consistency_scores else 0

        # Validity: percentage of values that conform to expected patterns
        validity_scores = []
        for column in df.columns:
            col_data = df[column].dropna()
            if len(col_data) > 0:
                # Simple validity check - non-empty strings for text, finite numbers for numeric
                if pd.api.types.is_numeric_dtype(col_data):
                    valid_count = np.isfinite(col_data).sum()
                    validity_scores.append((valid_count / len(col_data)) * 100)
                else:
                    # For text, check for non-empty strings
                    str_data = col_data.astype(str)
                    valid_count = (str_data.str.len() > 0).sum()
                    validity_scores.append((valid_count / len(str_data)) * 100)
            else:
                validity_scores.append(0)

        validity = np.mean(validity_scores) if validity_scores else 0

        # Accuracy: placeholder (would need reference data for true accuracy)
        accuracy = min(completeness, consistency, validity)  # Conservative estimate

        # Overall score: weighted average
        overall_score = (
            completeness * 0.3 + consistency * 0.25 + validity * 0.25 + accuracy * 0.2
        )

        return {
            "completeness": round(completeness, 2),
            "consistency": round(consistency, 2),
            "validity": round(validity, 2),
            "accuracy": round(accuracy, 2),
            "overall_score": round(overall_score, 2),
        }
