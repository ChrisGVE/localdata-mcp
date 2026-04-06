"""
DataTypeDetectorTransformer - sklearn-compatible transformer for advanced data type detection.
"""

import json
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted

from ...logging_manager import get_logger
from ..base import PipelineState

logger = get_logger(__name__)


class DataTypeDetectorTransformer(BaseEstimator, TransformerMixin):
    """
    sklearn-compatible transformer for advanced data type detection.

    Wraps the existing detect_data_types functionality in a sklearn pipeline-compatible interface
    while preserving semantic type detection, confidence scoring, and pattern recognition.

    Parameters:
    -----------
    sample_size : int, default=1000
        Number of rows to sample for type detection
    confidence_threshold : float, default=0.8
        Minimum confidence threshold for type detection
    include_semantic_types : bool, default=True
        Whether to include semantic type detection (email, phone, etc.)

    Attributes:
    -----------
    detected_types_ : dict
        Detected data types after fitting
    feature_names_in_ : ndarray of shape (n_features,)
        Names of features seen during fit
    n_features_in_ : int
        Number of features seen during fit
    state_ : PipelineState
        Current transformer state
    """

    def __init__(
        self,
        sample_size: int = 1000,
        confidence_threshold: float = 0.8,
        include_semantic_types: bool = True,
    ):
        self.sample_size = sample_size
        self.confidence_threshold = confidence_threshold
        self.include_semantic_types = include_semantic_types

        # Internal state
        self.state_ = PipelineState.INITIALIZED
        self.detected_types_ = None
        self.feature_names_in_ = None
        self.n_features_in_ = None

    def fit(self, X, y=None):
        """
        Fit the type detector by analyzing the input data.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features) or pandas.DataFrame
            Training data to analyze types
        y : array-like of shape (n_samples,), default=None
            Target values (ignored)

        Returns:
        --------
        self : DataTypeDetectorTransformer
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

            # Detect data types using existing logic
            self.detected_types_ = self._detect_column_types(df)

            # Add metadata
            self.detected_types_["metadata"] = {
                "source_type": "transformer_input",
                "sample_size": self.sample_size,
                "actual_rows_analyzed": len(df),
                "detection_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "confidence_threshold": self.confidence_threshold,
                "include_semantic_types": self.include_semantic_types,
                "pipeline_state": self.state_.value,
            }

            self.state_ = PipelineState.FITTED
            logger.info(
                f"DataTypeDetectorTransformer fitted successfully with {len(df)} rows, {df.shape[1]} columns"
            )

        except Exception as e:
            self.state_ = PipelineState.ERROR
            logger.error(f"Error fitting DataTypeDetectorTransformer: {e}")
            raise

        return self

    def transform(self, X):
        """
        Transform is identity for type detection - returns input unchanged.
        Detected types are available via get_detected_types() method.

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

    def get_detected_types(self) -> Dict[str, Any]:
        """
        Get the detected data types.

        Returns:
        --------
        types : dict
            Detected data types with confidence scores and semantic information
        """
        check_is_fitted(self)
        return self.detected_types_

    def get_detected_types_json(self) -> str:
        """
        Get the detected data types as JSON string (backward compatibility).

        Returns:
        --------
        types_json : str
            Detected data types in JSON format
        """
        check_is_fitted(self)
        return json.dumps(self.detected_types_, indent=2, default=str)

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
            Transformed feature names (same as input for type detection)
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
            - Detected types and confidence scores
            - Semantic type information
            - Type conversion recommendations
            - Processing hints for downstream tools
        """
        check_is_fitted(self)

        if not self.detected_types_:
            return {}

        # Extract composition-relevant metadata
        metadata = {
            "tool_type": "type_detector",
            "processing_stage": "data_understanding",
            "overall_confidence": self.detected_types_.get("summary", {}).get(
                "detection_confidence", 0
            ),
            "type_conversions": {},
            "semantic_types": {},
            "processing_hints": {},
            "recommended_next_steps": [],
        }

        # Extract column-level type information
        for col_name, col_info in self.detected_types_.get("columns", {}).items():
            detected_type = col_info.get("detected_type", "unknown")
            confidence = col_info.get("confidence", 0)
            semantic_type = col_info.get("semantic_type")

            # Type conversion recommendations
            if (
                detected_type == "numeric_string"
                and confidence > self.confidence_threshold
            ):
                metadata["type_conversions"][col_name] = {
                    "from": "string",
                    "to": "numeric",
                    "confidence": confidence,
                    "conversion_function": "pd.to_numeric",
                }
            elif (
                detected_type == "date_string"
                and confidence > self.confidence_threshold
            ):
                metadata["type_conversions"][col_name] = {
                    "from": "string",
                    "to": "datetime",
                    "confidence": confidence,
                    "conversion_function": "pd.to_datetime",
                }
            elif (
                detected_type == "boolean_string"
                and confidence > self.confidence_threshold
            ):
                metadata["type_conversions"][col_name] = {
                    "from": "string",
                    "to": "boolean",
                    "confidence": confidence,
                    "conversion_function": "astype(bool)",
                }

            # Semantic type information
            if semantic_type:
                metadata["semantic_types"][col_name] = {
                    "type": semantic_type,
                    "validation_required": True,
                    "special_handling": self._get_semantic_handling_hints(
                        semantic_type
                    ),
                }

            # Processing hints
            hints = []
            if confidence < self.confidence_threshold:
                hints.append("manual_type_verification")
            if semantic_type == "email":
                hints.append("email_validation")
            elif semantic_type == "phone":
                hints.append("phone_formatting")
            elif semantic_type == "url":
                hints.append("url_validation")
            elif detected_type in ["integer", "float"]:
                hints.append("numeric_analysis")
            elif detected_type == "datetime":
                hints.append("temporal_analysis")

            metadata["processing_hints"][col_name] = hints

        # Generate recommended next steps
        if len(metadata["type_conversions"]) > 0:
            metadata["recommended_next_steps"].append("type_conversion")
        if any(
            info["type"] in ["email", "phone", "url"]
            for info in metadata["semantic_types"].values()
        ):
            metadata["recommended_next_steps"].append("data_validation")
        if metadata["overall_confidence"] < 0.9:
            metadata["recommended_next_steps"].append("manual_type_review")

        return metadata

    def _get_semantic_handling_hints(self, semantic_type: str) -> List[str]:
        """Get special handling hints for semantic types."""
        hints_map = {
            "email": ["validation", "privacy_masking", "domain_analysis"],
            "phone": ["formatting", "country_code_detection", "privacy_masking"],
            "url": ["validation", "domain_extraction", "security_check"],
            "zip_code": ["geographic_analysis", "validation"],
        }
        return hints_map.get(semantic_type, [])

    def _detect_column_types(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect data types for all columns in DataFrame.

        This method replicates the existing detect_data_types logic
        to maintain 100% compatibility with the original implementation.
        """
        results = {
            "summary": {
                "total_columns": len(df.columns),
                "columns_analyzed": len(df.columns),
                "detection_confidence": 0.0,
            },
            "columns": {},
        }

        confidence_scores = []

        for column in df.columns:
            col_data = df[column].dropna()

            if len(col_data) == 0:
                results["columns"][column] = {
                    "detected_type": "unknown",
                    "confidence": 0.0,
                    "pandas_dtype": str(df[column].dtype),
                    "semantic_type": None,
                    "patterns": [],
                    "sample_values": [],
                }
                confidence_scores.append(0.0)
                continue

            # Get basic type information
            pandas_dtype = str(col_data.dtype)
            sample_values = [str(val) for val in col_data.head(5).tolist()]

            # Detect primary type
            detected_type, confidence = self._detect_primary_type(col_data)

            # Detect semantic type if enabled
            semantic_type = None
            if self.include_semantic_types:
                semantic_type = self._detect_semantic_type(col_data)

            # Detect patterns
            patterns = self._detect_patterns(col_data)

            results["columns"][column] = {
                "detected_type": detected_type,
                "confidence": confidence,
                "pandas_dtype": pandas_dtype,
                "semantic_type": semantic_type,
                "patterns": patterns,
                "sample_values": sample_values,
            }

            confidence_scores.append(confidence)

        # Calculate overall confidence
        results["summary"]["detection_confidence"] = (
            np.mean(confidence_scores) if confidence_scores else 0.0
        )

        return results

    def _detect_primary_type(self, series: pd.Series) -> Tuple[str, float]:
        """Detect primary data type with confidence score."""
        # Start with pandas dtype
        if pd.api.types.is_numeric_dtype(series):
            if pd.api.types.is_integer_dtype(series):
                return "integer", 0.95
            else:
                return "float", 0.95
        elif pd.api.types.is_datetime64_any_dtype(series):
            return "datetime", 0.95
        elif pd.api.types.is_bool_dtype(series):
            return "boolean", 0.95

        # For object dtype, do deeper analysis
        str_series = series.astype(str)

        # Check for numeric strings
        try:
            pd.to_numeric(str_series)
            return "numeric_string", 0.9
        except:
            pass

        # Check for date strings
        try:
            pd.to_datetime(str_series)
            return "date_string", 0.85
        except:
            pass

        # Check for boolean strings
        bool_values = {"true", "false", "yes", "no", "1", "0", "y", "n"}
        if set(str_series.str.lower().unique()).issubset(bool_values):
            return "boolean_string", 0.8

        # Default to string
        return "string", 0.7

    def _detect_semantic_type(self, series: pd.Series) -> Optional[str]:
        """Detect semantic type (email, phone, URL, etc.)."""
        str_series = series.astype(str)

        # Email pattern
        if str_series.str.match(
            r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
        ).any():
            return "email"

        # Phone pattern (simple)
        if str_series.str.match(
            r"^\+?1?[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}$"
        ).any():
            return "phone"

        # URL pattern
        if str_series.str.match(r"^https?://").any():
            return "url"

        # ZIP code pattern
        if str_series.str.match(r"^\d{5}(-\d{4})?$").any():
            return "zip_code"

        return None

    def _detect_patterns(self, series: pd.Series) -> List[str]:
        """Detect common patterns in the data."""
        patterns = []
        str_series = series.astype(str)

        # Check for common patterns
        if str_series.str.contains(r"^\d+$", na=False).any():
            patterns.append("digits_only")

        if str_series.str.contains(r"^[A-Z]+$", na=False).any():
            patterns.append("uppercase_only")

        if str_series.str.contains(r"^[a-z]+$", na=False).any():
            patterns.append("lowercase_only")

        if str_series.str.contains(r"^\d{4}-\d{2}-\d{2}$", na=False).any():
            patterns.append("date_iso")

        if str_series.str.contains(r"^[A-Z]{2,}$", na=False).any():
            patterns.append("abbreviation")

        return patterns
