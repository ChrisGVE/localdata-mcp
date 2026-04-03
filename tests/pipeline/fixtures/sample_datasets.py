"""
Sample Dataset Creation for Integration Shims Testing

This module provides comprehensive dataset generation functions for testing
the Integration Shims Framework. All datasets are designed to exercise
different aspects of the framework while remaining realistic and representative
of actual data science workflows.

Dataset Types:
- Statistical analysis datasets
- Time series datasets
- Regression modeling datasets
- Pattern recognition datasets
- Mixed domain datasets
- Streaming data sources
- High-dimensional datasets
- Sparse data structures
"""

import random
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.datasets import make_classification, make_regression, make_blobs


def create_statistical_dataset(
    size: str = "medium",
    complexity: str = "medium",
    features: Optional[int] = None,
    include_correlations: bool = True,
    include_outliers: bool = True,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Create a dataset optimized for statistical analysis testing.

    Args:
        size: Dataset size ("small", "medium", "large", "xlarge")
        complexity: Statistical complexity ("low", "medium", "high")
        features: Number of features (auto-determined if None)
        include_correlations: Whether to include correlated features
        include_outliers: Whether to include statistical outliers
        random_seed: Random seed for reproducibility

    Returns:
        DataFrame with statistical characteristics suitable for testing
    """
    np.random.seed(random_seed)

    # Define size parameters
    size_configs = {
        "small": {"rows": 100, "base_features": 5},
        "medium": {"rows": 1000, "base_features": 15},
        "large": {"rows": 10000, "base_features": 25},
        "xlarge": {"rows": 50000, "base_features": 50},
    }

    config = size_configs[size]
    n_samples = config["rows"]
    n_features = features or config["base_features"]

    # Adjust features based on complexity
    complexity_multipliers = {"low": 0.7, "medium": 1.0, "high": 1.5}
    n_features = int(n_features * complexity_multipliers[complexity])

    # Generate base data
    data = {}

    # Normal distributed features
    for i in range(n_features // 3):
        data[f"normal_{i}"] = np.random.normal(
            loc=random.uniform(-5, 5), scale=random.uniform(0.5, 3.0), size=n_samples
        )

    # Skewed distributions
    for i in range(n_features // 3):
        data[f"skewed_{i}"] = np.random.exponential(
            scale=random.uniform(0.5, 2.0), size=n_samples
        ) * random.choice([-1, 1])  # Random direction

    # Uniform distributions
    remaining_features = n_features - len(data)
    for i in range(remaining_features):
        data[f"uniform_{i}"] = np.random.uniform(
            low=random.uniform(-10, 0), high=random.uniform(0, 10), size=n_samples
        )

    df = pd.DataFrame(data)

    # Add correlations if requested
    if include_correlations and complexity in ["medium", "high"]:
        # Create some correlated features
        n_correlated = min(5, n_features // 4)
        base_feature = df.columns[0]

        for i in range(n_correlated):
            correlation_strength = random.uniform(0.3, 0.9)
            noise_level = 1.0 - correlation_strength

            correlated_data = correlation_strength * df[
                base_feature
            ] + noise_level * np.random.normal(0, df[base_feature].std(), n_samples)
            df[f"corr_with_{base_feature}_{i}"] = correlated_data

    # Add outliers if requested
    if include_outliers:
        outlier_ratio = 0.05  # 5% outliers
        n_outliers = int(n_samples * outlier_ratio)

        for col in df.select_dtypes(include=[np.number]).columns:
            outlier_indices = np.random.choice(n_samples, n_outliers, replace=False)
            col_std = df[col].std()
            col_mean = df[col].mean()

            # Create outliers at 3-5 standard deviations
            outlier_multiplier = random.uniform(3, 5) * random.choice([-1, 1])
            df.loc[outlier_indices, col] = col_mean + outlier_multiplier * col_std

    # Add categorical features for complexity
    if complexity == "high":
        df["category_low_card"] = np.random.choice(["A", "B", "C"], n_samples)
        df["category_high_card"] = np.random.choice(
            [f"cat_{i}" for i in range(20)], n_samples
        )
        df["binary_flag"] = np.random.choice([0, 1], n_samples)

    # Add metadata
    df.attrs = {
        "dataset_type": "statistical_analysis",
        "size": size,
        "complexity": complexity,
        "n_features": len(df.columns),
        "has_correlations": include_correlations,
        "has_outliers": include_outliers,
        "creation_timestamp": datetime.now().isoformat(),
    }

    return df


def create_time_series_dataset(
    length: int = 1000,
    features: int = 10,
    freq: str = "D",
    seasonality: bool = True,
    trend: bool = True,
    noise_level: float = 0.1,
    missing_data: bool = False,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Create a time series dataset for temporal analysis testing.

    Args:
        length: Number of time points
        features: Number of time series features
        freq: Pandas frequency string ('D', 'H', 'T', etc.)
        seasonality: Whether to include seasonal patterns
        trend: Whether to include trend components
        noise_level: Amount of noise (0.0 to 1.0)
        missing_data: Whether to include missing values
        random_seed: Random seed for reproducibility

    Returns:
        DataFrame with temporal structure and patterns
    """
    np.random.seed(random_seed)

    # Create time index
    start_date = datetime(2020, 1, 1)
    date_range = pd.date_range(start=start_date, periods=length, freq=freq)

    data = {}

    for i in range(features):
        # Base trend component
        if trend:
            trend_component = np.linspace(0, random.uniform(-5, 5), length)
            trend_component += np.random.normal(0, 0.5, length)  # Trend noise
        else:
            trend_component = np.zeros(length)

        # Seasonal component
        if seasonality:
            # Primary seasonality
            seasonal_period = random.choice(
                [12, 24, 7, 365]
            )  # Different seasonal patterns
            seasonal_amplitude = random.uniform(0.5, 3.0)
            seasonal_component = seasonal_amplitude * np.sin(
                2 * np.pi * np.arange(length) / seasonal_period
                + random.uniform(0, 2 * np.pi)
            )

            # Secondary seasonality for complexity
            if random.random() > 0.5:
                secondary_period = seasonal_period // random.choice([2, 3, 4])
                secondary_amplitude = seasonal_amplitude * random.uniform(0.2, 0.5)
                seasonal_component += secondary_amplitude * np.sin(
                    2 * np.pi * np.arange(length) / secondary_period
                )
        else:
            seasonal_component = np.zeros(length)

        # Noise component
        noise_component = np.random.normal(
            0,
            noise_level
            * (abs(trend_component).max() + abs(seasonal_component).max() + 1),
            length,
        )

        # Combine components
        series = trend_component + seasonal_component + noise_component

        # Add some non-linear patterns
        if random.random() > 0.7:  # 30% chance
            series += 0.1 * np.sin(0.1 * np.arange(length)) * series

        data[f"ts_feature_{i}"] = series

    df = pd.DataFrame(data, index=date_range)

    # Add missing data if requested
    if missing_data:
        missing_ratio = 0.05  # 5% missing
        total_values = length * features
        n_missing = int(total_values * missing_ratio)

        # Random missing values
        for _ in range(n_missing):
            row_idx = np.random.randint(0, length)
            col_idx = np.random.randint(0, features)
            df.iloc[row_idx, col_idx] = np.nan

    # Add metadata
    df.attrs = {
        "dataset_type": "time_series",
        "length": length,
        "features": features,
        "frequency": freq,
        "has_seasonality": seasonality,
        "has_trend": trend,
        "noise_level": noise_level,
        "has_missing_data": missing_data,
        "creation_timestamp": datetime.now().isoformat(),
    }

    return df


def create_regression_dataset(
    samples: int = 1000,
    features: int = 20,
    targets: int = 1,
    target_correlation: float = 0.8,
    noise: float = 0.1,
    include_categorical: bool = True,
    feature_interactions: bool = True,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Create a dataset optimized for regression modeling testing.

    Args:
        samples: Number of samples
        features: Number of input features
        targets: Number of target variables
        target_correlation: Correlation between features and targets
        noise: Noise level in the relationship
        include_categorical: Include categorical features
        feature_interactions: Include feature interaction terms
        random_seed: Random seed for reproducibility

    Returns:
        DataFrame suitable for regression analysis
    """
    np.random.seed(random_seed)

    # Generate base regression dataset
    X, y = make_regression(
        n_samples=samples,
        n_features=features,
        n_targets=targets,
        noise=noise * 10,  # Scale noise appropriately
        random_state=random_seed,
    )

    # Create feature names
    feature_names = [f"feature_{i}" for i in range(features)]

    # Build DataFrame
    data = pd.DataFrame(X, columns=feature_names)

    # Add target columns
    if targets == 1:
        data["target"] = y
    else:
        for i in range(targets):
            data[f"target_{i}"] = y[:, i]

    # Add categorical features if requested
    if include_categorical:
        # Low cardinality categorical
        data["category_A"] = np.random.choice(["cat1", "cat2", "cat3"], samples)

        # Medium cardinality categorical
        data["category_B"] = np.random.choice([f"level_{i}" for i in range(8)], samples)

        # Binary categorical
        data["binary_feature"] = np.random.choice([0, 1], samples)

        # Ordinal categorical (affects target)
        ordinal_values = ["low", "medium", "high"]
        data["ordinal_feature"] = np.random.choice(ordinal_values, samples)

        # Create some relationship with target
        ordinal_effect = (
            pd.Categorical(
                data["ordinal_feature"], categories=ordinal_values, ordered=True
            ).codes
            * 2.0
        )

        if targets == 1:
            data["target"] += ordinal_effect
        else:
            data["target_0"] += ordinal_effect

    # Add feature interactions if requested
    if feature_interactions:
        # Add some polynomial features
        data["poly_interaction_1"] = data["feature_0"] * data["feature_1"]
        data["poly_interaction_2"] = data["feature_2"] ** 2

        # Add some conditional interactions
        mask = data["feature_3"] > data["feature_3"].median()
        data["conditional_interaction"] = np.where(
            mask,
            data["feature_4"] * data["feature_5"],
            data["feature_4"] + data["feature_5"],
        )

    # Add some realistic business-like features
    if samples >= 500:  # Only for larger datasets
        # Simulate business metrics that might affect target
        data["customer_age"] = np.random.gamma(2, 20) + 18  # Age distribution
        data["purchase_history"] = np.random.poisson(5, samples)  # Purchase count
        data["account_balance"] = np.random.lognormal(8, 1.5)  # Financial data

        # Create realistic relationships
        if targets == 1:
            data["target"] += (
                0.1 * (data["customer_age"] - 40)
                + 0.05 * data["purchase_history"]
                + 0.0001 * data["account_balance"]
            )

    # Add metadata
    data.attrs = {
        "dataset_type": "regression",
        "n_samples": samples,
        "n_features": features,
        "n_targets": targets,
        "target_correlation": target_correlation,
        "noise_level": noise,
        "has_categorical": include_categorical,
        "has_interactions": feature_interactions,
        "creation_timestamp": datetime.now().isoformat(),
    }

    return data


def create_pattern_recognition_dataset(
    samples: int = 1000,
    dimensions: int = 50,
    clusters: int = 5,
    cluster_std: float = 1.0,
    anomaly_fraction: float = 0.05,
    sparsity: float = 0.0,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Create a dataset for pattern recognition and clustering testing.

    Args:
        samples: Number of samples
        dimensions: Number of dimensions/features
        clusters: Number of natural clusters
        cluster_std: Standard deviation of clusters
        anomaly_fraction: Fraction of anomalous samples
        sparsity: Fraction of zero values (0.0 = dense, 1.0 = all zeros)
        random_seed: Random seed for reproducibility

    Returns:
        DataFrame suitable for pattern recognition tasks
    """
    np.random.seed(random_seed)

    # Generate base clustering dataset
    n_normal_samples = int(samples * (1 - anomaly_fraction))

    X, y = make_blobs(
        n_samples=n_normal_samples,
        centers=clusters,
        n_features=dimensions,
        cluster_std=cluster_std,
        random_state=random_seed,
    )

    # Add anomalies
    n_anomalies = samples - n_normal_samples
    if n_anomalies > 0:
        # Create anomalies as outliers
        anomaly_X = np.random.uniform(
            low=X.min(axis=0) - 3 * cluster_std,
            high=X.max(axis=0) + 3 * cluster_std,
            size=(n_anomalies, dimensions),
        )
        anomaly_y = np.full(n_anomalies, -1)  # Mark as anomalies

        X = np.vstack([X, anomaly_X])
        y = np.hstack([y, anomaly_y])

    # Apply sparsity if requested
    if sparsity > 0:
        mask = np.random.random(X.shape) < sparsity
        X[mask] = 0

    # Create feature names
    feature_names = [f"dim_{i}" for i in range(dimensions)]

    # Build DataFrame
    data = pd.DataFrame(X, columns=feature_names)
    data["true_cluster"] = y

    # Add some derived features for pattern recognition
    if dimensions >= 10:
        # Distance from origin
        data["distance_from_origin"] = np.sqrt(np.sum(X**2, axis=1))

        # Principal angles
        data["angle_pc1"] = np.arctan2(X[:, 1], X[:, 0])

        # Local density approximation
        from sklearn.neighbors import NearestNeighbors

        nn = NearestNeighbors(n_neighbors=5)
        nn.fit(X)
        distances, _ = nn.kneighbors(X)
        data["local_density"] = 1.0 / (np.mean(distances[:, 1:], axis=1) + 1e-10)

    # Add some categorical representations of continuous features
    if dimensions >= 20:
        # Discretize some continuous features
        data["high_dim_category"] = pd.qcut(
            data["dim_0"],
            q=5,
            labels=["very_low", "low", "medium", "high", "very_high"],
        )

        data["pattern_type"] = np.where(
            data["distance_from_origin"] > data["distance_from_origin"].median(),
            "peripheral",
            "central",
        )

    # Shuffle the data to remove ordering effects
    data = data.sample(frac=1, random_state=random_seed).reset_index(drop=True)

    # Add metadata
    data.attrs = {
        "dataset_type": "pattern_recognition",
        "n_samples": samples,
        "n_dimensions": dimensions,
        "n_clusters": clusters,
        "cluster_std": cluster_std,
        "anomaly_fraction": anomaly_fraction,
        "sparsity": sparsity,
        "creation_timestamp": datetime.now().isoformat(),
    }

    return data


def create_mixed_domain_dataset(
    statistical_features: int = 10,
    temporal_length: int = 200,
    regression_targets: int = 2,
    pattern_dimensions: int = 30,
    complexity: str = "medium",
    include_semantic_labels: bool = False,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Create a comprehensive dataset suitable for multi-domain analysis.

    Args:
        statistical_features: Number of features for statistical analysis
        temporal_length: Length of time series components
        regression_targets: Number of regression target variables
        pattern_dimensions: Dimensions for pattern recognition
        complexity: Overall complexity level
        include_semantic_labels: Include semantic metadata for interpretation
        random_seed: Random seed for reproducibility

    Returns:
        DataFrame combining characteristics from all domains
    """
    np.random.seed(random_seed)

    # Create base time series structure
    ts_data = create_time_series_dataset(
        length=temporal_length,
        features=statistical_features // 2,
        seasonality=True,
        trend=True,
        random_seed=random_seed,
    )

    # Reset index to have integer index instead of datetime
    ts_data = ts_data.reset_index()
    ts_data.rename(columns={"index": "timestamp"}, inplace=True)

    # Add cross-sectional statistical features
    for i in range(statistical_features):
        feature_name = f"stat_feature_{i}"

        if complexity == "high":
            # Complex distributions
            distribution_type = random.choice(["normal", "gamma", "beta", "lognormal"])
            if distribution_type == "normal":
                ts_data[feature_name] = np.random.normal(
                    loc=random.uniform(-2, 2),
                    scale=random.uniform(0.5, 2.0),
                    size=temporal_length,
                )
            elif distribution_type == "gamma":
                ts_data[feature_name] = np.random.gamma(
                    shape=random.uniform(1, 3),
                    scale=random.uniform(0.5, 2.0),
                    size=temporal_length,
                )
            elif distribution_type == "beta":
                ts_data[feature_name] = np.random.beta(
                    a=random.uniform(0.5, 3),
                    b=random.uniform(0.5, 3),
                    size=temporal_length,
                )
            else:  # lognormal
                ts_data[feature_name] = np.random.lognormal(
                    mean=random.uniform(0, 1),
                    sigma=random.uniform(0.5, 1.0),
                    size=temporal_length,
                )
        else:
            # Simpler distributions
            ts_data[feature_name] = np.random.normal(
                loc=0, scale=1, size=temporal_length
            )

    # Add regression targets with realistic relationships
    for i in range(regression_targets):
        target_name = f"regression_target_{i}"

        # Create target as combination of features
        feature_cols = [col for col in ts_data.columns if "feature" in col]
        if len(feature_cols) > 0:
            # Linear combination with some selected features
            selected_features = random.sample(feature_cols, min(5, len(feature_cols)))

            coefficients = np.random.normal(0, 1, len(selected_features))
            target_value = np.zeros(temporal_length)

            for j, feature in enumerate(selected_features):
                target_value += coefficients[j] * ts_data[feature]

            # Add noise and non-linear terms
            noise = np.random.normal(0, 0.1 * np.std(target_value), temporal_length)
            nonlinear_term = 0.1 * np.sin(0.1 * np.arange(temporal_length))

            ts_data[target_name] = target_value + noise + nonlinear_term

    # Add high-dimensional features for pattern recognition
    pattern_features = np.random.multivariate_normal(
        mean=np.zeros(pattern_dimensions),
        cov=np.eye(pattern_dimensions),
        size=temporal_length,
    )

    # Create some cluster structure in pattern features
    n_clusters = 3
    cluster_centers = np.random.normal(0, 3, (n_clusters, pattern_dimensions))
    cluster_assignments = np.random.choice(n_clusters, temporal_length)

    for i in range(temporal_length):
        cluster_id = cluster_assignments[i]
        pattern_features[i] += cluster_centers[cluster_id]

    # Add pattern features to main dataset
    for i in range(pattern_dimensions):
        ts_data[f"pattern_dim_{i}"] = pattern_features[:, i]

    ts_data["cluster_assignment"] = cluster_assignments

    # Add categorical variables
    ts_data["category_low"] = np.random.choice(["A", "B", "C"], temporal_length)
    ts_data["category_high"] = np.random.choice(
        [f"cat_{i}" for i in range(15)], temporal_length
    )
    ts_data["binary_indicator"] = np.random.choice([0, 1], temporal_length)

    # Add some business-like semantic features if requested
    if include_semantic_labels:
        ts_data["customer_segment"] = np.random.choice(
            ["premium", "standard", "basic"], temporal_length
        )
        ts_data["product_category"] = np.random.choice(
            ["electronics", "clothing", "home", "books"], temporal_length
        )
        ts_data["transaction_amount"] = np.random.lognormal(3, 1, temporal_length)

        # Store semantic mappings
        feature_meanings = {
            "customer_segment": "Customer tier classification",
            "product_category": "Product type classification",
            "transaction_amount": "Transaction value in currency units",
            "timestamp": "Time of observation",
        }

        for col in ts_data.columns:
            if "stat_feature" in col:
                feature_meanings[col] = "Cross-sectional statistical measure"
            elif "ts_feature" in col:
                feature_meanings[col] = "Time series component"
            elif "regression_target" in col:
                feature_meanings[col] = "Target variable for predictive modeling"
            elif "pattern_dim" in col:
                feature_meanings[col] = "High-dimensional pattern feature"

        ts_data.feature_meanings = feature_meanings

    # Add comprehensive metadata
    ts_data.attrs = {
        "dataset_type": "mixed_domain",
        "statistical_features": statistical_features,
        "temporal_length": temporal_length,
        "regression_targets": regression_targets,
        "pattern_dimensions": pattern_dimensions,
        "complexity": complexity,
        "has_semantic_labels": include_semantic_labels,
        "domain_coverage": [
            "statistical",
            "temporal",
            "regression",
            "pattern_recognition",
        ],
        "creation_timestamp": datetime.now().isoformat(),
    }

    return ts_data


# Additional utility functions for creating specialized datasets


def create_pandas_dataframe(
    rows: int = 1000, columns: int = 10, mixed_types: bool = True
) -> pd.DataFrame:
    """Create a pandas DataFrame with optional mixed data types."""
    np.random.seed(42)

    data = {}

    if mixed_types:
        # Numeric columns
        for i in range(columns // 3):
            data[f"numeric_{i}"] = np.random.normal(0, 1, rows)

        # String columns
        for i in range(columns // 3):
            data[f"string_{i}"] = [f"str_{j}_{i}" for j in range(rows)]

        # Categorical columns
        remaining = columns - len(data)
        for i in range(remaining):
            data[f"category_{i}"] = np.random.choice(["A", "B", "C"], rows)
    else:
        # All numeric
        for i in range(columns):
            data[f"col_{i}"] = np.random.normal(0, 1, rows)

    return pd.DataFrame(data)


def create_numpy_array(
    shape: Tuple[int, ...], dtype: np.dtype = np.float64
) -> np.ndarray:
    """Create a numpy array with specified shape and dtype."""
    np.random.seed(42)
    return np.random.random(shape).astype(dtype)


def create_sparse_matrix(
    shape: Tuple[int, int], density: float = 0.1, format: str = "csr"
) -> sparse.spmatrix:
    """Create a scipy sparse matrix with specified characteristics."""
    np.random.seed(42)
    return sparse.random(shape[0], shape[1], density=density, format=format)


def create_mixed_type_dataframe(
    rows: int = 1000,
    numeric_cols: int = 10,
    categorical_cols: int = 3,
    temporal_cols: int = 2,
    text_cols: int = 2,
) -> pd.DataFrame:
    """Create a DataFrame with mixed data types for conversion testing."""
    np.random.seed(42)

    data = {}

    # Numeric columns
    for i in range(numeric_cols):
        data[f"numeric_{i}"] = np.random.normal(i, 1, rows)

    # Categorical columns
    for i in range(categorical_cols):
        categories = [f"cat_{j}" for j in range(random.randint(3, 8))]
        data[f"categorical_{i}"] = np.random.choice(categories, rows)

    # Temporal columns
    for i in range(temporal_cols):
        start_date = datetime.now() - timedelta(days=rows)
        dates = [start_date + timedelta(days=j) for j in range(rows)]
        data[f"temporal_{i}"] = dates

    # Text columns
    for i in range(text_cols):
        texts = [f"text_sample_{j}_{i}" * random.randint(1, 5) for j in range(rows)]
        data[f"text_{i}"] = texts

    return pd.DataFrame(data)


def create_temporal_dataframe(rows: int = 1000, freq: str = "D") -> pd.DataFrame:
    """Create a DataFrame with temporal index and time-based features."""
    np.random.seed(42)

    date_range = pd.date_range(start="2020-01-01", periods=rows, freq=freq)

    data = {
        "value": np.random.normal(0, 1, rows),
        "trend": np.linspace(0, 10, rows),
        "seasonal": np.sin(2 * np.pi * np.arange(rows) / 365),
        "day_of_week": date_range.dayofweek,
        "month": date_range.month,
    }

    return pd.DataFrame(data, index=date_range)


def create_high_dimensional_array(
    shape: Tuple[int, int], dtype: np.dtype = np.float64, sparsity: float = 0.0
) -> np.ndarray:
    """Create a high-dimensional array suitable for performance testing."""
    np.random.seed(42)

    array = np.random.normal(0, 1, shape).astype(dtype)

    if sparsity > 0:
        mask = np.random.random(shape) < sparsity
        array[mask] = 0

    return array


class StreamingDataSource:
    """Mock streaming data source for testing streaming conversions."""

    def __init__(
        self, total_rows: int, chunk_size: int, columns: int, data_types: List[str]
    ):
        self.total_rows = total_rows
        self.chunk_size = chunk_size
        self.columns = columns
        self.data_types = data_types
        self.current_row = 0
        np.random.seed(42)

    async def get_next_chunk(self) -> Optional[pd.DataFrame]:
        """Get the next chunk of data."""
        if self.current_row >= self.total_rows:
            return None

        chunk_end = min(self.current_row + self.chunk_size, self.total_rows)
        chunk_rows = chunk_end - self.current_row

        data = {}

        for i in range(self.columns):
            if "numeric" in self.data_types:
                data[f"numeric_{i}"] = np.random.normal(0, 1, chunk_rows)
            if "categorical" in self.data_types and i % 3 == 0:
                data[f"categorical_{i}"] = np.random.choice(["A", "B", "C"], chunk_rows)
            if "temporal" in self.data_types and i % 5 == 0:
                dates = pd.date_range(
                    start=datetime.now() - timedelta(days=chunk_rows),
                    periods=chunk_rows,
                    freq="D",
                )
                data[f"temporal_{i}"] = dates

        chunk = pd.DataFrame(data)
        self.current_row = chunk_end

        return chunk

    @property
    def estimated_total_size(self) -> int:
        """Estimate total size in bytes."""
        return self.total_rows * self.columns * 8  # Rough estimate


def create_streaming_data_source(
    total_size: str = "1GB",
    chunk_size: str = "100MB",
    columns: int = 20,
    data_types: List[str] = None,
) -> StreamingDataSource:
    """Create a mock streaming data source."""

    if data_types is None:
        data_types = ["numeric", "categorical"]

    # Convert size strings to bytes
    size_multipliers = {"KB": 1024, "MB": 1024**2, "GB": 1024**3}

    def parse_size(size_str: str) -> int:
        for unit, multiplier in size_multipliers.items():
            if unit in size_str:
                return int(size_str.replace(unit, "")) * multiplier
        return int(size_str)

    total_bytes = parse_size(total_size)
    chunk_bytes = parse_size(chunk_size)

    # Estimate rows (assuming 8 bytes per numeric value)
    bytes_per_row = columns * 8
    total_rows = total_bytes // bytes_per_row
    chunk_rows = chunk_bytes // bytes_per_row

    return StreamingDataSource(total_rows, chunk_rows, columns, data_types)


# Aliases for backward compatibility with extension_v2 test names
def create_high_dimensional_dataset(
    n_samples: int = 500, n_features: int = 50, **kwargs
):
    """Alias for create_high_dimensional_array returning a DataFrame."""
    arr = create_high_dimensional_array(shape=(n_samples, n_features), **kwargs)
    return pd.DataFrame(arr, columns=[f"feature_{i}" for i in range(arr.shape[1])])


def create_streaming_dataset(**kwargs):
    """Alias for create_streaming_data_source."""
    return create_streaming_data_source(**kwargs)


# Export all dataset creation functions
__all__ = [
    "create_statistical_dataset",
    "create_time_series_dataset",
    "create_regression_dataset",
    "create_pattern_recognition_dataset",
    "create_mixed_domain_dataset",
    "create_pandas_dataframe",
    "create_numpy_array",
    "create_sparse_matrix",
    "create_mixed_type_dataframe",
    "create_temporal_dataframe",
    "create_high_dimensional_array",
    "create_streaming_data_source",
    "StreamingDataSource",
]
