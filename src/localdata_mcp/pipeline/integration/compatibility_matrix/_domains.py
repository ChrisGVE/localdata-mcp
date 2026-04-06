"""
Standard domain profile initialization for the Pipeline Compatibility Matrix.

Defines the built-in domain profiles for Statistical Analysis, Regression & Modeling,
Time Series Analysis, and Advanced Pattern Recognition domains.
"""

from ..interfaces import DataFormat, DataFormatSpec, DomainRequirements
from ._types import DomainProfile


def build_standard_domain_profiles() -> dict[str, DomainProfile]:
    """Build and return the standard LocalData MCP domain profiles.

    Returns:
        Dictionary mapping domain names to their DomainProfile instances.
    """
    profiles: dict[str, DomainProfile] = {}

    # Statistical Analysis Domain
    stats_requirements = DomainRequirements(
        domain_name="statistical_analysis",
        input_formats=[
            DataFormat.PANDAS_DATAFRAME,
            DataFormat.NUMPY_ARRAY,
            DataFormat.TIME_SERIES,
            DataFormat.PYTHON_LIST,
        ],
        output_formats=[DataFormat.STATISTICAL_RESULT, DataFormat.PANDAS_DATAFRAME],
        preferred_format=DataFormat.PANDAS_DATAFRAME,
        metadata_requirements=["column_names", "data_types", "sample_size"],
        quality_requirements={
            "min_sample_size": 30,
            "max_missing_ratio": 0.2,
            "numeric_data_required": True,
        },
    )

    profiles["statistical_analysis"] = DomainProfile(
        domain_name="statistical_analysis",
        base_requirements=stats_requirements,
        tool_specifications={
            "hypothesis_testing": DataFormatSpec(
                format_type=DataFormat.PANDAS_DATAFRAME,
                schema_requirements={"min_columns": 2, "numeric_columns": True},
            ),
            "correlation_analysis": DataFormatSpec(
                format_type=DataFormat.PANDAS_DATAFRAME,
                schema_requirements={"all_numeric": True, "min_columns": 2},
            ),
            "descriptive_stats": DataFormatSpec(
                format_type=DataFormat.PANDAS_DATAFRAME,
                schema_requirements={"numeric_columns": True},
            ),
        },
        compatibility_preferences={
            DataFormat.PANDAS_DATAFRAME: 1.0,
            DataFormat.NUMPY_ARRAY: 0.8,
            DataFormat.TIME_SERIES: 0.9,
            DataFormat.PYTHON_LIST: 0.6,
        },
    )

    # Regression & Modeling Domain
    regression_requirements = DomainRequirements(
        domain_name="regression_modeling",
        input_formats=[
            DataFormat.PANDAS_DATAFRAME,
            DataFormat.NUMPY_ARRAY,
            DataFormat.SCIPY_SPARSE,
        ],
        output_formats=[DataFormat.REGRESSION_MODEL, DataFormat.PANDAS_DATAFRAME],
        preferred_format=DataFormat.PANDAS_DATAFRAME,
        metadata_requirements=["feature_names", "target_column", "data_types"],
        quality_requirements={
            "min_samples_per_feature": 10,
            "feature_target_separation": True,
        },
    )

    profiles["regression_modeling"] = DomainProfile(
        domain_name="regression_modeling",
        base_requirements=regression_requirements,
        tool_specifications={
            "linear_regression": DataFormatSpec(
                format_type=DataFormat.PANDAS_DATAFRAME,
                schema_requirements={"features_numeric": True, "target_numeric": True},
            ),
            "logistic_regression": DataFormatSpec(
                format_type=DataFormat.PANDAS_DATAFRAME,
                schema_requirements={
                    "features_numeric": True,
                    "target_categorical": True,
                },
            ),
        },
        compatibility_preferences={
            DataFormat.PANDAS_DATAFRAME: 1.0,
            DataFormat.NUMPY_ARRAY: 0.85,
            DataFormat.SCIPY_SPARSE: 0.9,
        },
    )

    # Time Series Analysis Domain
    timeseries_requirements = DomainRequirements(
        domain_name="time_series",
        input_formats=[DataFormat.TIME_SERIES, DataFormat.PANDAS_DATAFRAME],
        output_formats=[DataFormat.FORECAST_RESULT, DataFormat.TIME_SERIES],
        preferred_format=DataFormat.TIME_SERIES,
        metadata_requirements=["datetime_index", "frequency", "seasonality"],
        quality_requirements={
            "min_periods": 24,
            "regular_frequency": True,
            "datetime_sorted": True,
        },
    )

    profiles["time_series"] = DomainProfile(
        domain_name="time_series",
        base_requirements=timeseries_requirements,
        tool_specifications={
            "arima_forecasting": DataFormatSpec(
                format_type=DataFormat.TIME_SERIES,
                schema_requirements={"univariate": True, "regular_intervals": True},
            ),
            "seasonal_decomposition": DataFormatSpec(
                format_type=DataFormat.TIME_SERIES,
                schema_requirements={"min_periods_per_season": 2},
            ),
        },
        compatibility_preferences={
            DataFormat.TIME_SERIES: 1.0,
            DataFormat.PANDAS_DATAFRAME: 0.9,
        },
    )

    # Advanced Pattern Recognition Domain
    pattern_requirements = DomainRequirements(
        domain_name="pattern_recognition",
        input_formats=[
            DataFormat.NUMPY_ARRAY,
            DataFormat.PANDAS_DATAFRAME,
            DataFormat.SCIPY_SPARSE,
        ],
        output_formats=[
            DataFormat.CLUSTERING_RESULT,
            DataFormat.PATTERN_RECOGNITION_RESULT,
        ],
        preferred_format=DataFormat.NUMPY_ARRAY,
        metadata_requirements=["feature_names", "dimensionality"],
        quality_requirements={"min_samples": 50, "feature_scaling_recommended": True},
    )

    profiles["pattern_recognition"] = DomainProfile(
        domain_name="pattern_recognition",
        base_requirements=pattern_requirements,
        tool_specifications={
            "clustering": DataFormatSpec(
                format_type=DataFormat.NUMPY_ARRAY,
                schema_requirements={"numeric_features": True, "feature_scaling": True},
            ),
            "dimensionality_reduction": DataFormatSpec(
                format_type=DataFormat.NUMPY_ARRAY,
                schema_requirements={"high_dimensional": True},
            ),
        },
        compatibility_preferences={
            DataFormat.NUMPY_ARRAY: 1.0,
            DataFormat.PANDAS_DATAFRAME: 0.8,
            DataFormat.SCIPY_SPARSE: 0.9,
        },
    )

    return profiles
