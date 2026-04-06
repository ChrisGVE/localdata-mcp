"""
Business Intelligence Domain - Attribution Analysis.

This module implements marketing attribution analysis using various models
including first-touch, last-touch, linear, time-decay, and position-based
attribution for customer conversion path analysis.
"""

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from ...logging_manager import get_logger
from .models import AttributionModel, AttributionResult

logger = get_logger(__name__)


class AttributionAnalyzer(BaseEstimator, TransformerMixin):
    """
    sklearn-compatible transformer for marketing attribution analysis.

    Analyzes customer conversion paths and assigns attribution weights to
    different marketing touchpoints using various attribution models.

    Parameters:
    -----------
    attribution_model : AttributionModel or str, default='last_touch'
        Attribution model to use for analysis
    lookback_window : int, default=30
        Number of days to look back for touchpoints
    """

    def __init__(self, attribution_model="last_touch", lookback_window=30):
        if isinstance(attribution_model, str):
            self.attribution_model = AttributionModel(attribution_model)
        else:
            self.attribution_model = attribution_model
        self.lookback_window = lookback_window

    def fit(self, X, y=None):
        """Fit the attribution analyzer."""
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")

        # Expect columns: customer_id, channel, timestamp, converted
        required_columns = ["customer_id", "channel", "timestamp"]
        missing_columns = [col for col in required_columns if col not in X.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        self.is_fitted_ = True
        return self

    def transform(self, X):
        """Perform attribution analysis."""
        check_is_fitted(self, "is_fitted_")

        logger.info(f"Performing {self.attribution_model.value} attribution analysis")

        # Prepare conversion paths
        conversion_paths = self._prepare_conversion_paths(X)

        # Apply attribution model
        if self.attribution_model == AttributionModel.FIRST_TOUCH:
            attribution_weights = self._first_touch_attribution(conversion_paths)
        elif self.attribution_model == AttributionModel.LAST_TOUCH:
            attribution_weights = self._last_touch_attribution(conversion_paths)
        elif self.attribution_model == AttributionModel.LINEAR:
            attribution_weights = self._linear_attribution(conversion_paths)
        elif self.attribution_model == AttributionModel.TIME_DECAY:
            attribution_weights = self._time_decay_attribution(conversion_paths)
        elif self.attribution_model == AttributionModel.POSITION_BASED:
            attribution_weights = self._position_based_attribution(conversion_paths)
        else:
            raise ValueError(f"Unsupported attribution model: {self.attribution_model}")

        # Calculate channel-level attribution
        channel_attribution = self._calculate_channel_attribution(attribution_weights)

        # Create model comparison
        model_comparison = self._compare_attribution_models(conversion_paths)

        return AttributionResult(
            attribution_weights=attribution_weights,
            channel_attribution=channel_attribution,
            model_comparison=model_comparison,
            conversion_paths=conversion_paths,
        )

    def _prepare_conversion_paths(self, X):
        """Prepare customer conversion paths from touchpoint data."""
        # Sort by customer and timestamp
        X_sorted = X.sort_values(["customer_id", "timestamp"])

        # Group by customer to create paths
        paths = []
        for customer_id, group in X_sorted.groupby("customer_id"):
            path_data = {
                "customer_id": customer_id,
                "touchpoints": group["channel"].tolist(),
                "timestamps": group["timestamp"].tolist(),
                "converted": group.get("converted", [False] * len(group)).iloc[-1],
            }
            paths.append(path_data)

        return pd.DataFrame(paths)

    def _first_touch_attribution(self, conversion_paths):
        """Apply first-touch attribution model."""
        attribution_data = []

        for _, path in conversion_paths.iterrows():
            if path["converted"] and len(path["touchpoints"]) > 0:
                # First touchpoint gets 100% credit
                first_channel = path["touchpoints"][0]
                attribution_data.append(
                    {
                        "customer_id": path["customer_id"],
                        "channel": first_channel,
                        "attribution_weight": 1.0,
                        "model": "first_touch",
                    }
                )

        return pd.DataFrame(attribution_data)

    def _last_touch_attribution(self, conversion_paths):
        """Apply last-touch attribution model."""
        attribution_data = []

        for _, path in conversion_paths.iterrows():
            if path["converted"] and len(path["touchpoints"]) > 0:
                # Last touchpoint gets 100% credit
                last_channel = path["touchpoints"][-1]
                attribution_data.append(
                    {
                        "customer_id": path["customer_id"],
                        "channel": last_channel,
                        "attribution_weight": 1.0,
                        "model": "last_touch",
                    }
                )

        return pd.DataFrame(attribution_data)

    def _linear_attribution(self, conversion_paths):
        """Apply linear attribution model."""
        attribution_data = []

        for _, path in conversion_paths.iterrows():
            if path["converted"] and len(path["touchpoints"]) > 0:
                # Each touchpoint gets equal credit
                weight_per_touchpoint = 1.0 / len(path["touchpoints"])

                for channel in path["touchpoints"]:
                    attribution_data.append(
                        {
                            "customer_id": path["customer_id"],
                            "channel": channel,
                            "attribution_weight": weight_per_touchpoint,
                            "model": "linear",
                        }
                    )

        return pd.DataFrame(attribution_data)

    def _time_decay_attribution(self, conversion_paths):
        """Apply time-decay attribution model (placeholder)."""
        # For now, fallback to linear attribution
        return self._linear_attribution(conversion_paths)

    def _position_based_attribution(self, conversion_paths):
        """Apply position-based attribution model (placeholder)."""
        # For now, fallback to linear attribution
        return self._linear_attribution(conversion_paths)

    def _calculate_channel_attribution(self, attribution_weights):
        """Calculate total attribution by channel."""
        if attribution_weights.empty:
            return pd.DataFrame(columns=["channel", "total_attribution", "conversions"])

        channel_totals = (
            attribution_weights.groupby("channel")
            .agg({"attribution_weight": "sum", "customer_id": "count"})
            .reset_index()
        )

        channel_totals.columns = ["channel", "total_attribution", "conversions"]
        channel_totals = channel_totals.sort_values(
            "total_attribution", ascending=False
        )

        return channel_totals

    def _compare_attribution_models(self, conversion_paths):
        """Compare results across different attribution models."""
        models = [
            AttributionModel.FIRST_TOUCH,
            AttributionModel.LAST_TOUCH,
            AttributionModel.LINEAR,
        ]
        comparison = {}

        for model in models:
            temp_analyzer = AttributionAnalyzer(attribution_model=model)
            temp_analyzer.is_fitted_ = True

            if model == AttributionModel.FIRST_TOUCH:
                weights = temp_analyzer._first_touch_attribution(conversion_paths)
            elif model == AttributionModel.LAST_TOUCH:
                weights = temp_analyzer._last_touch_attribution(conversion_paths)
            elif model == AttributionModel.LINEAR:
                weights = temp_analyzer._linear_attribution(conversion_paths)

            if not weights.empty:
                channel_totals = (
                    weights.groupby("channel")["attribution_weight"].sum().to_dict()
                )
            else:
                channel_totals = {}

            comparison[model.value] = channel_totals

        return comparison
