"""
Business Intelligence Domain - Funnel Analysis.

This module implements marketing funnel analysis for identifying bottlenecks
and optimization opportunities in the customer journey.
"""

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from ...logging_manager import get_logger
from .models import FunnelAnalysisResult

logger = get_logger(__name__)


class FunnelAnalyzer(BaseEstimator, TransformerMixin):
    """
    sklearn-compatible transformer for marketing funnel analysis.

    Analyzes conversion funnels to identify bottlenecks and optimization
    opportunities in the customer journey.

    Parameters:
    -----------
    steps : list of str
        Ordered list of funnel step names
    """

    def __init__(self, steps=None):
        self.steps = steps or ["awareness", "interest", "consideration", "purchase"]

    def fit(self, X, y=None):
        """Fit the funnel analyzer."""
        self.is_fitted_ = True
        return self

    def transform(self, X):
        """Perform funnel analysis."""
        check_is_fitted(self, "is_fitted_")

        logger.info("Performing marketing funnel analysis")

        # Calculate funnel metrics
        funnel_steps = self._calculate_funnel_steps(X)
        conversion_rates = self._calculate_conversion_rates(funnel_steps)
        drop_off_rates = self._calculate_drop_off_rates(funnel_steps)
        bottlenecks = self._identify_bottlenecks(drop_off_rates)
        recommendations = self._generate_recommendations(bottlenecks, drop_off_rates)

        return FunnelAnalysisResult(
            funnel_steps=funnel_steps,
            conversion_rates=conversion_rates,
            drop_off_rates=drop_off_rates,
            bottlenecks=bottlenecks,
            optimization_recommendations=recommendations,
        )

    def _calculate_funnel_steps(self, X):
        """Calculate user counts at each funnel step."""
        # Expect data with columns for each step (boolean values)
        step_counts = []

        for i, step in enumerate(self.steps):
            if step in X.columns:
                count = X[step].sum()
            else:
                # If step column doesn't exist, assume 0
                count = 0

            step_counts.append({"step": step, "step_number": i + 1, "users": count})

        return pd.DataFrame(step_counts)

    def _calculate_conversion_rates(self, funnel_steps):
        """Calculate conversion rates between steps."""
        conversion_rates = {}

        for i in range(len(funnel_steps) - 1):
            current_step = funnel_steps.iloc[i]
            next_step = funnel_steps.iloc[i + 1]

            if current_step["users"] > 0:
                rate = next_step["users"] / current_step["users"]
            else:
                rate = 0

            conversion_rates[f"{current_step['step']}_to_{next_step['step']}"] = rate

        return conversion_rates

    def _calculate_drop_off_rates(self, funnel_steps):
        """Calculate drop-off rates between steps."""
        drop_off_rates = {}

        for i in range(len(funnel_steps) - 1):
            current_step = funnel_steps.iloc[i]
            next_step = funnel_steps.iloc[i + 1]

            if current_step["users"] > 0:
                rate = (current_step["users"] - next_step["users"]) / current_step[
                    "users"
                ]
            else:
                rate = 1

            drop_off_rates[f"{current_step['step']}_to_{next_step['step']}"] = rate

        return drop_off_rates

    def _identify_bottlenecks(self, drop_off_rates):
        """Identify funnel bottlenecks with high drop-off rates."""
        # Sort drop-off rates and identify top bottlenecks
        sorted_drops = sorted(drop_off_rates.items(), key=lambda x: x[1], reverse=True)

        # Consider bottlenecks as steps with >50% drop-off
        bottlenecks = [step for step, rate in sorted_drops if rate > 0.5]

        return bottlenecks

    def _generate_recommendations(self, bottlenecks, drop_off_rates):
        """Generate optimization recommendations."""
        recommendations = []

        for bottleneck in bottlenecks:
            rate = drop_off_rates[bottleneck]
            recommendations.append(
                f"High drop-off at {bottleneck} ({rate:.1%}). Consider improving user experience or reducing friction."
            )

        if not recommendations:
            recommendations.append(
                "Funnel performance looks good. Focus on incremental improvements."
            )

        return recommendations
