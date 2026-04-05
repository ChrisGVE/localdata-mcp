"""
Categorical Encoding Pipeline - Sklearn-based categorical feature encoding.

Implements the CategoricalEncodingPipeline class with automatic strategy
selection based on cardinality analysis, unknown category handling, and
streaming compatibility.
"""

import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder

from ..base import (
    AnalysisPipelineBase,
    StreamingConfig,
)
from ...logging_manager import get_logger

logger = get_logger(__name__)


class CategoricalEncodingPipeline(AnalysisPipelineBase):
    """
    Categorical encoding pipeline using sklearn preprocessing encoders.

    Provides comprehensive categorical encoding with automatic strategy selection
    based on cardinality analysis, unknown category handling, and streaming compatibility.
    """

    def __init__(
        self,
        analytical_intention: str = "encode categorical features for analysis",
        encoding_strategy: str = "auto",  # "auto", "onehot", "label", "ordinal", "target"
        cardinality_threshold: int = 10,
        handle_unknown: str = "ignore",  # "error", "ignore", "infrequent_if_exist"
        streaming_config: Optional[StreamingConfig] = None,
        custom_parameters: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize categorical encoding pipeline.

        Args:
            analytical_intention: Natural language description of encoding goal
            encoding_strategy: Encoding method or "auto" for automatic selection
            cardinality_threshold: Threshold for high/low cardinality encoding decisions
            handle_unknown: How to handle unknown categories in production
            streaming_config: Configuration for streaming execution
            custom_parameters: Additional custom parameters
        """
        super().__init__(
            analytical_intention=analytical_intention,
            streaming_config=streaming_config or StreamingConfig(),
            progressive_complexity="auto",
            composition_aware=True,
            custom_parameters=custom_parameters or {},
        )

        self.encoding_strategy = encoding_strategy
        self.cardinality_threshold = cardinality_threshold
        self.handle_unknown = handle_unknown

        # Encoding state for streaming compatibility
        self._encoders: Dict[str, Any] = {}
        self._encoding_metadata: Dict[str, Any] = {}
        self._category_mappings: Dict[str, Dict[str, Any]] = {}

        logger.info(
            "CategoricalEncodingPipeline initialized",
            intention=analytical_intention,
            strategy=encoding_strategy,
        )

    def get_analysis_type(self) -> str:
        """Get the analysis type - categorical encoding."""
        return "categorical_encoding"

    def _configure_analysis_pipeline(self) -> List[Callable]:
        """Configure encoding pipeline steps."""
        return [
            self._analyze_categorical_data,
            self._select_encoding_strategies,
            self._apply_categorical_encoding,
            self._validate_encoding_results,
        ]

    def _execute_analysis_step(
        self, step: Callable, data: pd.DataFrame, context: Dict[str, Any]
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Execute individual encoding step with error handling."""
        step_name = step.__name__
        start_time = time.time()

        try:
            # Execute the encoding step
            encoded_data, step_metadata = step(data)

            execution_time = time.time() - start_time

            metadata = {
                "step": step_name,
                "execution_time": execution_time,
                "success": True,
                "step_metadata": step_metadata,
            }

            logger.info(
                f"Encoding step {step_name} completed successfully",
                execution_time=execution_time,
            )

            return encoded_data, metadata

        except Exception as e:
            execution_time = time.time() - start_time

            logger.error(f"Encoding step {step_name} failed: {e}")

            metadata = {
                "step": step_name,
                "execution_time": execution_time,
                "success": False,
                "error": str(e),
            }

            return data, metadata  # Return original data on failure

    def _execute_streaming_analysis(
        self, data: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Execute encoding with streaming support."""
        processed_data = data.copy()

        # Apply each encoding step in the pipeline
        for encode_func in self._analysis_pipeline:
            processed_data, step_metadata = self._execute_analysis_step(
                encode_func, processed_data, self.get_execution_context()
            )

        metadata = self._build_encoding_metadata(processed_data, streaming_enabled=True)
        return processed_data, metadata

    def _execute_standard_analysis(
        self, data: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Execute encoding on full dataset in memory."""
        processed_data = data.copy()

        # Apply each encoding step in the pipeline
        for encode_func in self._analysis_pipeline:
            processed_data, step_metadata = self._execute_analysis_step(
                encode_func, processed_data, self.get_execution_context()
            )

        metadata = self._build_encoding_metadata(
            processed_data, streaming_enabled=False
        )
        return processed_data, metadata

    # Encoding method implementations
    def _analyze_categorical_data(
        self, data: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Analyze categorical data characteristics."""
        categorical_cols = data.select_dtypes(include=["object", "category"]).columns
        cardinality_analysis = {}

        for col in categorical_cols:
            col_data = data[col].dropna()
            unique_values = col_data.unique()

            cardinality_analysis[col] = {
                "cardinality": len(unique_values),
                "unique_values": unique_values.tolist()[
                    :20
                ],  # Limit to first 20 for metadata
                "missing_percentage": (data[col].isnull().sum() / len(data)) * 100,
                "is_high_cardinality": len(unique_values) > self.cardinality_threshold,
                "is_binary": len(unique_values) == 2,
                "is_ordinal": self._detect_ordinal_nature(unique_values),
            }

        metadata = {
            "cardinality_analysis": cardinality_analysis,
            "categorical_columns_analyzed": len(categorical_cols),
        }

        return data, metadata

    def _select_encoding_strategies(
        self, data: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Select optimal encoding strategies for each categorical column."""
        categorical_cols = data.select_dtypes(include=["object", "category"]).columns
        selected_strategies = {}

        for col in categorical_cols:
            if self.encoding_strategy != "auto":
                # Use global strategy
                selected_strategies[col] = self.encoding_strategy
            else:
                # Auto-select based on data characteristics
                col_data = data[col].dropna()
                cardinality = col_data.nunique()

                if cardinality == 2:
                    selected_strategies[col] = "label"  # Binary encoding
                elif cardinality <= self.cardinality_threshold:
                    selected_strategies[col] = "onehot"  # One-hot for low cardinality
                elif self._detect_ordinal_nature(col_data.unique()):
                    selected_strategies[col] = "ordinal"  # Preserve order
                else:
                    selected_strategies[col] = (
                        "label"  # Label encoding for high cardinality
                    )

        self._encoding_metadata["selected_strategies"] = selected_strategies

        metadata = {
            "selected_strategies": selected_strategies,
            "total_columns": len(selected_strategies),
        }

        return data, metadata

    def _apply_categorical_encoding(
        self, data: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Apply the selected encoding transformations."""
        result_data = data.copy()
        encoding_log = {}

        strategies = self._encoding_metadata.get("selected_strategies", {})

        for col, strategy in strategies.items():
            try:
                if strategy == "onehot":
                    # One-hot encoding
                    encoder = OneHotEncoder(
                        handle_unknown=self.handle_unknown, sparse_output=False
                    )
                    encoded_data = encoder.fit_transform(data[[col]])

                    # Create column names
                    feature_names = [
                        f"{col}_{category}" for category in encoder.categories_[0]
                    ]
                    encoded_df = pd.DataFrame(
                        encoded_data, columns=feature_names, index=data.index
                    )

                    # Replace original column with encoded columns
                    result_data = result_data.drop(col, axis=1)
                    result_data = pd.concat([result_data, encoded_df], axis=1)

                    self._encoders[col] = encoder
                    encoding_log[col] = {
                        "strategy": strategy,
                        "new_columns": feature_names,
                        "original_cardinality": len(encoder.categories_[0]),
                    }

                elif strategy == "label":
                    # Label encoding
                    encoder = LabelEncoder()
                    encoded_data = encoder.fit_transform(
                        data[col].fillna("__missing__")
                    )
                    result_data[f"{col}_encoded"] = encoded_data

                    self._encoders[col] = encoder
                    self._category_mappings[col] = dict(
                        zip(encoder.classes_, encoder.transform(encoder.classes_))
                    )

                    encoding_log[col] = {
                        "strategy": strategy,
                        "new_column": f"{col}_encoded",
                        "category_mapping": self._category_mappings[col],
                    }

                elif strategy == "ordinal":
                    # Ordinal encoding
                    categories = self._get_ordinal_categories(
                        data[col].dropna().unique()
                    )
                    encoder = OrdinalEncoder(
                        categories=[categories], handle_unknown=self.handle_unknown
                    )
                    encoded_data = encoder.fit_transform(data[[col]])
                    result_data[f"{col}_ordinal"] = encoded_data.flatten()

                    self._encoders[col] = encoder
                    encoding_log[col] = {
                        "strategy": strategy,
                        "new_column": f"{col}_ordinal",
                        "ordinal_mapping": dict(
                            zip(categories, range(len(categories)))
                        ),
                    }

            except Exception as e:
                logger.warning(f"Failed to encode column {col} with {strategy}: {e}")
                encoding_log[col] = {
                    "strategy": strategy,
                    "error": str(e),
                    "fallback": "no_encoding",
                }

        metadata = {
            "encoding_log": encoding_log,
            "successful_encodings": sum(
                1 for log in encoding_log.values() if "error" not in log
            ),
            "failed_encodings": sum(
                1 for log in encoding_log.values() if "error" in log
            ),
        }

        return result_data, metadata

    def _validate_encoding_results(
        self, data: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Validate encoding results and generate quality metrics."""
        validation_results = {}

        # Check for encoded columns
        encoded_columns = []
        for col_name in data.columns:
            if any(
                col_name.startswith(f"{original_col}_")
                for original_col in self._encoders.keys()
            ):
                encoded_columns.append(col_name)

        validation_results = {
            "encoded_columns_created": len(encoded_columns),
            "encoders_fitted": len(self._encoders),
            "memory_increase_mb": data.memory_usage(deep=True).sum() / (1024 * 1024),
            "encoding_quality": "good"
            if len(encoded_columns) > 0
            else "no_encoding_applied",
        }

        metadata = {
            "validation_results": validation_results,
            "overall_quality": validation_results["encoding_quality"],
        }

        return data, metadata

    def _detect_ordinal_nature(self, unique_values) -> bool:
        """Detect if categorical values have ordinal nature."""
        # Simple heuristic - look for size/rating patterns
        size_patterns = ["xs", "s", "m", "l", "xl", "xxl"]
        rating_patterns = ["poor", "fair", "good", "excellent"]
        numeric_patterns = [str(i) for i in range(1, 11)]

        values_str = [str(v).lower() for v in unique_values if pd.notna(v)]

        # Check if values follow known ordinal patterns
        for pattern in [size_patterns, rating_patterns, numeric_patterns]:
            if all(val in pattern for val in values_str):
                return True

        return False

    def _get_ordinal_categories(self, unique_values):
        """Get ordered categories for ordinal encoding."""
        # Simple ordering - could be enhanced with more sophisticated logic
        values_str = [str(v) for v in unique_values if pd.notna(v)]
        return sorted(values_str)

    def _build_encoding_metadata(
        self, encoded_data: pd.DataFrame, streaming_enabled: bool
    ) -> Dict[str, Any]:
        """Build comprehensive metadata for encoding results."""
        return {
            "encoding_pipeline": {
                "analytical_intention": self.analytical_intention,
                "encoding_strategy": self.encoding_strategy,
                "cardinality_threshold": self.cardinality_threshold,
                "streaming_enabled": streaming_enabled,
                "encoders_fitted": len(self._encoders),
            },
            "encoding_results": self._encoding_metadata,
            "category_mappings": self._category_mappings,
            "data_characteristics": {
                "shape": encoded_data.shape,
                "categorical_columns_remaining": encoded_data.select_dtypes(
                    include=["object", "category"]
                ).columns.tolist(),
                "numeric_columns": encoded_data.select_dtypes(
                    include=["number"]
                ).columns.tolist(),
                "memory_usage_mb": encoded_data.memory_usage(deep=True).sum()
                / (1024 * 1024),
            },
            "composition_context": {
                "ready_for_ml": len(
                    encoded_data.select_dtypes(include=["object", "category"]).columns
                )
                == 0,
                "encoding_artifacts": {
                    "fitted_encoders": list(self._encoders.keys()),
                    "category_mappings": self._category_mappings,
                },
                "suggested_next_steps": [
                    {
                        "analysis_type": "feature_scaling",
                        "reason": "Categorical features encoded, ready for scaling",
                        "confidence": 0.9,
                    },
                    {
                        "analysis_type": "machine_learning",
                        "reason": "All features properly encoded for ML",
                        "confidence": 0.8,
                    },
                ],
            },
        }

    # Public utility methods
    def get_encoder(self, column: str) -> Optional[Any]:
        """Get fitted encoder for specific column."""
        return self._encoders.get(column)

    def get_category_mapping(self, column: str) -> Optional[Dict[str, Any]]:
        """Get category mapping for specific column."""
        return self._category_mappings.get(column)
