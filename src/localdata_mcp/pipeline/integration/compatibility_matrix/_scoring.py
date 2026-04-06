"""
Format compatibility scoring and conversion path logic.

Contains the format compatibility matrix construction, score calculation,
family-based fallback scoring, conversion path finding, and issue/recommendation
generation used by PipelineCompatibilityMatrix.
"""

from typing import Dict, List, Optional, Tuple

from ..interfaces import (
    ConversionCost,
    ConversionPath,
    ConversionStep,
    DataFormat,
)


def build_format_compatibility_matrix() -> Dict[Tuple[DataFormat, DataFormat], float]:
    """Build the base format compatibility matrix."""
    matrix = {}

    # Perfect compatibility (same format)
    for fmt in DataFormat:
        matrix[(fmt, fmt)] = 1.0

    # High compatibility conversions (0.8-0.94)
    high_compat = [
        # Pandas DataFrame conversions
        (DataFormat.PANDAS_DATAFRAME, DataFormat.NUMPY_ARRAY, 0.9),
        (DataFormat.PANDAS_DATAFRAME, DataFormat.TIME_SERIES, 0.92),
        (DataFormat.PANDAS_DATAFRAME, DataFormat.PYTHON_DICT, 0.85),
        (DataFormat.TIME_SERIES, DataFormat.PANDAS_DATAFRAME, 0.95),
        # NumPy array conversions
        (DataFormat.NUMPY_ARRAY, DataFormat.PANDAS_DATAFRAME, 0.85),
        (DataFormat.NUMPY_ARRAY, DataFormat.SCIPY_SPARSE, 0.88),
        (DataFormat.NUMPY_ARRAY, DataFormat.PYTHON_LIST, 0.92),
        # Result format conversions
        (DataFormat.STATISTICAL_RESULT, DataFormat.PYTHON_DICT, 0.9),
        (DataFormat.REGRESSION_MODEL, DataFormat.PYTHON_DICT, 0.85),
        (DataFormat.CLUSTERING_RESULT, DataFormat.PANDAS_DATAFRAME, 0.8),
        (DataFormat.FORECAST_RESULT, DataFormat.TIME_SERIES, 0.9),
    ]

    # Moderate compatibility conversions (0.6-0.79)
    moderate_compat = [
        (DataFormat.PANDAS_DATAFRAME, DataFormat.SCIPY_SPARSE, 0.75),
        (DataFormat.SCIPY_SPARSE, DataFormat.PANDAS_DATAFRAME, 0.7),
        (DataFormat.TIME_SERIES, DataFormat.NUMPY_ARRAY, 0.75),
        (DataFormat.CATEGORICAL, DataFormat.PANDAS_DATAFRAME, 0.8),
        (DataFormat.CATEGORICAL, DataFormat.NUMPY_ARRAY, 0.65),
        (DataFormat.PYTHON_DICT, DataFormat.PANDAS_DATAFRAME, 0.7),
    ]

    # Low compatibility conversions (0.3-0.59)
    low_compat = [
        (DataFormat.STATISTICAL_RESULT, DataFormat.NUMPY_ARRAY, 0.4),
        (DataFormat.REGRESSION_MODEL, DataFormat.PANDAS_DATAFRAME, 0.45),
        (DataFormat.PYTHON_LIST, DataFormat.SCIPY_SPARSE, 0.5),
        (DataFormat.JSON, DataFormat.PANDAS_DATAFRAME, 0.55),
        (DataFormat.CSV, DataFormat.NUMPY_ARRAY, 0.5),
    ]

    # Add all compatibility scores
    for source, target, score in high_compat + moderate_compat + low_compat:
        matrix[(source, target)] = score
        # Add reverse with slight penalty
        matrix[(target, source)] = score * 0.95

    return matrix


def calculate_family_compatibility(
    source_format: DataFormat, target_format: DataFormat
) -> float:
    """Calculate compatibility based on format families."""
    tabular_formats = {
        DataFormat.PANDAS_DATAFRAME,
        DataFormat.TIME_SERIES,
        DataFormat.CATEGORICAL,
        DataFormat.MULTI_INDEX,
    }

    array_formats = {DataFormat.NUMPY_ARRAY, DataFormat.SCIPY_SPARSE}

    result_formats = {
        DataFormat.STATISTICAL_RESULT,
        DataFormat.REGRESSION_MODEL,
        DataFormat.CLUSTERING_RESULT,
        DataFormat.FORECAST_RESULT,
        DataFormat.PATTERN_RECOGNITION_RESULT,
    }

    collection_formats = {DataFormat.PYTHON_LIST, DataFormat.PYTHON_DICT}

    # Same family compatibility
    families = [tabular_formats, array_formats, result_formats, collection_formats]
    for family in families:
        if source_format in family and target_format in family:
            return 0.7

    # Cross-family compatibility
    if (source_format in tabular_formats and target_format in array_formats) or (
        source_format in array_formats and target_format in tabular_formats
    ):
        return 0.6

    if (source_format in collection_formats and target_format in tabular_formats) or (
        source_format in tabular_formats and target_format in collection_formats
    ):
        return 0.5

    # Default low compatibility for unknown combinations
    return 0.2


def calculate_compatibility_score(
    format_matrix: Dict[Tuple[DataFormat, DataFormat], float],
    source_format: DataFormat,
    target_format: DataFormat,
) -> float:
    """Calculate base compatibility score between formats."""
    score = format_matrix.get((source_format, target_format))
    if score is not None:
        return score
    return calculate_family_compatibility(source_format, target_format)


def find_conversion_path(
    format_matrix: Dict[Tuple[DataFormat, DataFormat], float],
    source_format: DataFormat,
    target_format: DataFormat,
) -> Optional[ConversionPath]:
    """Find conversion path between formats."""
    if source_format == target_format:
        return None

    compatibility_score = calculate_compatibility_score(
        format_matrix, source_format, target_format
    )

    base_cost = 1.0 - compatibility_score

    cost = ConversionCost(
        computational_cost=base_cost * 0.5,
        memory_cost_mb=base_cost * 100,
        time_estimate_seconds=base_cost * 2.0,
        quality_impact=max(0.0, 0.1 - compatibility_score * 0.1),
    )

    step = ConversionStep(
        adapter_id=f"converter_{source_format.value}_to_{target_format.value}",
        source_format=source_format,
        target_format=target_format,
        estimated_cost=cost,
        confidence=min(compatibility_score + 0.1, 1.0),
    )

    return ConversionPath(
        source_format=source_format,
        target_format=target_format,
        steps=[step],
        total_cost=cost,
        success_probability=min(compatibility_score + 0.2, 1.0),
    )


def identify_compatibility_issues(
    source_format: DataFormat, target_format: DataFormat, score: float
) -> List[str]:
    """Identify potential compatibility issues."""
    issues = []

    if score < 0.3:
        issues.append(
            f"Very low compatibility between {source_format.value} and {target_format.value}"
        )

    if (
        source_format == DataFormat.SCIPY_SPARSE
        and target_format == DataFormat.PANDAS_DATAFRAME
    ):
        issues.append(
            "Sparse to DataFrame conversion may significantly increase memory usage"
        )

    if (
        source_format == DataFormat.TIME_SERIES
        and target_format == DataFormat.NUMPY_ARRAY
    ):
        issues.append("Time series to array conversion will lose temporal indexing")

    if target_format in [
        DataFormat.STATISTICAL_RESULT,
        DataFormat.REGRESSION_MODEL,
    ] and source_format in [DataFormat.PYTHON_LIST, DataFormat.PYTHON_DICT]:
        issues.append(
            "Converting simple data structures to complex result formats may not be meaningful"
        )

    return issues


def generate_recommendations(
    source_format: DataFormat, target_format: DataFormat, score: float
) -> List[str]:
    """Generate recommendations for improving compatibility."""
    recommendations = []

    if score < 0.6:
        recommendations.append(
            f"Consider using {DataFormat.PANDAS_DATAFRAME.value} as an intermediate format"
        )

    if (
        source_format == DataFormat.PYTHON_LIST
        and target_format != DataFormat.NUMPY_ARRAY
    ):
        recommendations.append(
            "Convert Python lists to NumPy arrays for better data science compatibility"
        )

    if target_format == DataFormat.TIME_SERIES:
        recommendations.append(
            "Ensure source data has proper datetime indexing for time series conversion"
        )

    if (
        source_format in [DataFormat.NUMPY_ARRAY, DataFormat.SCIPY_SPARSE]
        and target_format == DataFormat.PANDAS_DATAFRAME
    ):
        recommendations.append(
            "Consider providing column names for better DataFrame structure"
        )

    return recommendations
