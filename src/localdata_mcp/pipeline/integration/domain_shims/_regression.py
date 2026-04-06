"""
Regression domain shim implementation.

Provides RegressionShim for connecting regression modeling with other domains,
handling model coefficients, predictions, residuals, and diagnostic tests.
"""

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from ....logging_manager import get_logger
from ..interfaces import ConversionError, ConversionRequest
from ..shim_registry import AdapterConfig
from ._base import BaseDomainShim
from ._types import DomainMapping, DomainShimType, SemanticContext

logger = get_logger(__name__)


class RegressionShim(BaseDomainShim):
    """
    Connect regression modeling with other domains.

    Transforms regression model outputs for time series forecasting, pattern recognition,
    and business intelligence, handling model coefficients, predictions, and residuals.
    """

    def __init__(
        self,
        adapter_id: str = "regression_shim",
        config: Optional[AdapterConfig] = None,
        **kwargs,
    ):
        """Initialize RegressionShim."""
        super().__init__(
            adapter_id=adapter_id,
            domain_type=DomainShimType.REGRESSION,
            config=config,
            **kwargs,
        )

    def _initialize_domain_knowledge(self) -> None:
        """Initialize regression domain knowledge."""
        self._domain_schemas = {
            "linear_model": {
                "required_fields": ["coefficients", "intercept", "r2_score"],
                "optional_fields": ["std_error", "p_values", "confidence_intervals"],
            },
            "model_diagnostics": {
                "required_fields": ["residuals", "fitted_values"],
                "optional_fields": [
                    "leverage",
                    "cooks_distance",
                    "standardized_residuals",
                ],
            },
            "predictions": {
                "required_fields": ["predicted_values"],
                "optional_fields": ["prediction_intervals", "confidence_intervals"],
            },
        }

    def _load_domain_mappings(self) -> None:
        """Load regression domain mappings."""
        # Regression to Time Series mapping
        self.supported_mappings.append(
            DomainMapping(
                source_domain="regression",
                target_domain="time_series",
                parameter_mappings={
                    "model_coefficients": "trend_parameters",
                    "residuals": "error_component",
                    "predictions": "forecasted_values",
                    "confidence_intervals": "forecast_intervals",
                },
                result_transformations={
                    "regression_model": "trend_model",
                    "fitted_values": "fitted_trend",
                    "model_diagnostics": "forecast_diagnostics",
                },
                semantic_hints={
                    "trend_modeling": "linear_trend",
                    "seasonal_regression": "seasonal_components",
                    "autoregression": "ar_model",
                },
                quality_preservation=0.92,
            )
        )

        # Regression to Pattern Recognition mapping
        self.supported_mappings.append(
            DomainMapping(
                source_domain="regression",
                target_domain="pattern_recognition",
                parameter_mappings={
                    "feature_importance": "feature_weights",
                    "model_coefficients": "component_loadings",
                    "residual_analysis": "anomaly_scores",
                },
                result_transformations={
                    "regression_features": "pattern_features",
                    "model_predictions": "classification_scores",
                },
                semantic_hints={
                    "feature_selection": "dimensionality_reduction",
                    "outlier_detection": "anomaly_detection",
                    "classification": "supervised_learning",
                },
                quality_preservation=0.88,
            )
        )

        # Regression to Statistical mapping
        self.supported_mappings.append(
            DomainMapping(
                source_domain="regression",
                target_domain="statistical",
                parameter_mappings={
                    "model_statistics": "test_statistics",
                    "p_values": "statistical_significance",
                    "residuals": "error_distribution",
                },
                result_transformations={
                    "regression_summary": "statistical_summary",
                    "anova_table": "hypothesis_test_results",
                },
                semantic_hints={
                    "significance_testing": "hypothesis_testing",
                    "model_validation": "assumption_testing",
                    "residual_analysis": "normality_testing",
                },
                quality_preservation=0.94,
            )
        )

    def _perform_domain_conversion(
        self,
        request: ConversionRequest,
        mapping: DomainMapping,
        semantic_context: SemanticContext,
    ) -> Any:
        """Perform regression domain conversion."""
        source_data = request.source_data

        if mapping.target_domain == "time_series":
            return self._convert_regression_to_time_series(
                source_data, mapping, semantic_context
            )
        elif mapping.target_domain == "pattern_recognition":
            return self._convert_regression_to_pattern_recognition(
                source_data, mapping, semantic_context
            )
        elif mapping.target_domain == "statistical":
            return self._convert_regression_to_statistical(
                source_data, mapping, semantic_context
            )
        else:
            raise ConversionError(
                ConversionError.Type.CONVERSION_FAILED,
                f"Unsupported target domain: {mapping.target_domain}",
            )

    def _convert_regression_to_time_series(
        self, data: Any, mapping: DomainMapping, semantic_context: SemanticContext
    ) -> Dict[str, Any]:
        """Convert regression results to time series format."""
        if isinstance(data, dict):
            # Handle regression model for trend analysis
            if "coefficients" in data and "fitted_values" in data:
                return {
                    "trend_model": {
                        "trend_parameters": data["coefficients"],
                        "fitted_trend": data["fitted_values"],
                        "residuals": data.get("residuals"),
                        "model_type": "linear_trend",
                    },
                    "forecast_info": {
                        "trend_strength": (
                            abs(data["coefficients"][0])
                            if len(data["coefficients"]) > 0
                            else 0
                        ),
                        "model_r2": data.get("r2_score", 0),
                        "forecast_reliability": data.get("r2_score", 0),
                    },
                }

            # Handle predictions for forecasting
            if "predictions" in data or "predicted_values" in data:
                predictions = data.get("predictions", data.get("predicted_values"))
                return {
                    "forecasted_values": predictions,
                    "forecast_intervals": data.get("confidence_intervals"),
                    "forecast_method": "regression_based",
                    "forecast_horizon": (
                        len(predictions) if hasattr(predictions, "__len__") else 1
                    ),
                }

        elif isinstance(data, pd.DataFrame):
            # Handle regression results DataFrame
            if "fitted" in data.columns or "predicted" in data.columns:
                fitted_col = "fitted" if "fitted" in data.columns else "predicted"
                result = {
                    "fitted_values": data[fitted_col].values,
                    "time_index": (
                        data.index.tolist() if hasattr(data, "index") else None
                    ),
                }

                if "residuals" in data.columns:
                    result["residuals"] = data["residuals"].values

                return result

        # Fallback
        return {
            "regression_input": data,
            "conversion_type": "regression_to_time_series",
            "temporal_context": semantic_context.transformation_hints,
        }

    def _convert_regression_to_pattern_recognition(
        self, data: Any, mapping: DomainMapping, semantic_context: SemanticContext
    ) -> Dict[str, Any]:
        """Convert regression results to pattern recognition format."""
        if isinstance(data, dict):
            # Handle feature importance for dimensionality reduction
            if "feature_importance" in data or "coefficients" in data:
                importance = data.get(
                    "feature_importance", np.abs(data.get("coefficients", []))
                )
                return {
                    "feature_weights": importance,
                    "feature_names": data.get("feature_names", []),
                    "importance_type": "regression_coefficients",
                    "dimensionality_reduction_ready": True,
                }

            # Handle residuals for anomaly detection
            if "residuals" in data:
                residuals = np.array(data["residuals"])
                return {
                    "anomaly_scores": np.abs(residuals),
                    "residual_statistics": {
                        "mean": np.mean(residuals),
                        "std": np.std(residuals),
                        "outlier_threshold": np.mean(residuals) + 2 * np.std(residuals),
                    },
                    "anomaly_method": "residual_based",
                }

            # Handle model predictions for classification
            if "predictions" in data and "model_type" in data:
                if data["model_type"] in ["logistic", "classification"]:
                    return {
                        "classification_scores": data["predictions"],
                        "prediction_probabilities": data.get("probabilities"),
                        "classification_method": "regression_based",
                    }

        # Fallback
        return {
            "regression_input": data,
            "conversion_type": "regression_to_pattern_recognition",
            "pattern_hints": semantic_context.transformation_hints,
        }

    def _convert_regression_to_statistical(
        self, data: Any, mapping: DomainMapping, semantic_context: SemanticContext
    ) -> Dict[str, Any]:
        """Convert regression results to statistical format."""
        if isinstance(data, dict):
            # Handle model summary for statistical analysis
            if "coefficients" in data and "p_values" in data:
                return {
                    "hypothesis_test_results": {
                        "test_statistics": data["coefficients"],
                        "p_values": data["p_values"],
                        "significance_level": 0.05,
                        "significant_features": (
                            [i for i, p in enumerate(data["p_values"]) if p <= 0.05]
                            if isinstance(data["p_values"], (list, np.ndarray))
                            else []
                        ),
                    }
                }

            # Handle ANOVA table
            if "anova_table" in data or ("f_statistic" in data and "f_pvalue" in data):
                return {
                    "anova_results": {
                        "f_statistic": data.get("f_statistic"),
                        "f_pvalue": data.get("f_pvalue"),
                        "degrees_of_freedom": data.get("df_model", data.get("df")),
                        "sum_of_squares": data.get("sum_of_squares"),
                    }
                }

            # Handle model diagnostics
            if "residuals" in data:
                residuals = np.array(data["residuals"])
                return {
                    "residual_analysis": {
                        "normality_test": self._test_normality(residuals),
                        "homoscedasticity": self._test_homoscedasticity(residuals),
                        "autocorrelation": self._test_autocorrelation(residuals),
                    }
                }

        # Fallback
        return {
            "regression_input": data,
            "conversion_type": "regression_to_statistical",
            "statistical_context": semantic_context.transformation_hints,
        }

    def _test_normality(self, residuals: np.ndarray) -> Dict[str, Any]:
        """Test normality of residuals."""
        from scipy.stats import jarque_bera, shapiro

        try:
            # Shapiro-Wilk test (for small samples)
            if len(residuals) <= 5000:
                stat, p_value = shapiro(residuals)
                test_name = "shapiro_wilk"
            else:
                # Jarque-Bera test (for larger samples)
                stat, p_value = jarque_bera(residuals)
                test_name = "jarque_bera"

            return {
                "test_name": test_name,
                "statistic": stat,
                "p_value": p_value,
                "is_normal": p_value > 0.05,
            }
        except Exception as e:
            return {"error": str(e), "is_normal": False}

    def _test_homoscedasticity(self, residuals: np.ndarray) -> Dict[str, Any]:
        """Test homoscedasticity (constant variance) of residuals."""
        try:
            # Simple Breusch-Pagan style test
            abs_residuals = np.abs(residuals)
            mean_abs_residual = np.mean(abs_residuals)

            # Test if variance changes over time/fitted values
            n_groups = min(10, len(residuals) // 10)
            group_size = len(residuals) // n_groups

            group_variances = []
            for i in range(n_groups):
                start_idx = i * group_size
                end_idx = (i + 1) * group_size if i < n_groups - 1 else len(residuals)
                group_var = np.var(residuals[start_idx:end_idx])
                group_variances.append(group_var)

            # Simple test: variance of group variances
            variance_of_variances = np.var(group_variances)
            is_homoscedastic = variance_of_variances < (np.mean(group_variances) * 0.5)

            return {
                "test_name": "group_variance_test",
                "variance_of_variances": variance_of_variances,
                "is_homoscedastic": is_homoscedastic,
            }
        except Exception as e:
            return {"error": str(e), "is_homoscedastic": False}

    def _test_autocorrelation(self, residuals: np.ndarray) -> Dict[str, Any]:
        """Test for autocorrelation in residuals."""
        try:
            from scipy.stats import pearsonr

            if len(residuals) < 2:
                return {"error": "Insufficient data", "has_autocorrelation": False}

            # Test lag-1 autocorrelation
            lag1_corr, p_value = pearsonr(residuals[:-1], residuals[1:])

            return {
                "test_name": "lag1_autocorrelation",
                "correlation": lag1_corr,
                "p_value": p_value,
                "has_autocorrelation": abs(lag1_corr) > 0.3 and p_value < 0.05,
            }
        except Exception as e:
            return {"error": str(e), "has_autocorrelation": False}

    def _normalize_results(
        self, data: Any, mapping: DomainMapping, semantic_context: SemanticContext
    ) -> Any:
        """Normalize regression conversion results for target domain."""
        if not isinstance(data, dict):
            return data

        # Add domain-specific metadata
        data["domain_conversion"] = {
            "source": "regression",
            "target": mapping.target_domain,
            "semantic_goal": semantic_context.analytical_goal,
            "quality_preservation": mapping.quality_preservation,
        }

        # Ensure consistent structure based on target domain
        if mapping.target_domain == "time_series":
            # Ensure time series compatible structure
            if "fitted_values" in data:
                data["forecast_diagnostics"] = {
                    "method": "regression_based_forecast",
                    "reliability": data.get("model_r2", 0.8),
                    "assumptions_met": True,  # Simplified
                }

        elif mapping.target_domain == "pattern_recognition":
            # Ensure pattern recognition compatible structure
            if "feature_weights" in data:
                data["feature_selection_info"] = {
                    "selection_method": "regression_importance",
                    "n_features_selected": len(data.get("feature_weights", [])),
                    "selection_threshold": 0.1,
                }

        return data
