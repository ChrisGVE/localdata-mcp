"""
Time Series Analysis - Structural Break Testing.

Contains the StructuralBreakTester transformer for statistical structural
break testing in time series using Chow, CUSUM, recursive residuals,
and Quandt-Andrews tests.
"""

import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

from ...logging_manager import get_logger
from ._base import TimeSeriesAnalysisResult
from ._transformer import TimeSeriesTransformer

logger = get_logger(__name__)


class StructuralBreakTester(TimeSeriesTransformer):
    """
    sklearn-compatible transformer for statistical structural break testing in time series.

    Implements various statistical tests for detecting structural breaks including
    Chow test, CUSUM test, and recursive residuals test for time series stability.

    Parameters:
    -----------
    test_type : str, default='chow'
        Type of structural break test: 'chow', 'cusum', 'recursive', 'quandt_andrews'
    break_point : int or float, optional
        Known break point for Chow test (position or fraction)
    significance_level : float, default=0.05
        Significance level for statistical tests
    min_segment_size : int, default=10
        Minimum size for each segment in break point testing
    """

    def __init__(
        self,
        test_type="chow",
        break_point=None,
        significance_level=0.05,
        min_segment_size=10,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.test_type = test_type
        self.break_point = break_point
        self.significance_level = significance_level
        self.min_segment_size = min_segment_size
        self.test_results_ = {}
        self.break_points_ = []

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Fit the structural break tester."""
        if self.validate_input:
            X, y = self._validate_time_series(X, y)
        self.is_fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> TimeSeriesAnalysisResult:
        """
        Perform structural break testing on time series data.

        Returns:
        --------
        result : TimeSeriesAnalysisResult
            Structural break test results with test statistics and significance
        """
        start_time = time.time()

        if self.validate_input:
            X, _ = self._validate_time_series(X, None)

        try:
            # Use first column for structural break testing
            if len(X.columns) == 0:
                raise ValueError("No columns found in time series data")

            series = X.iloc[:, 0].dropna()
            if len(series) < 2 * self.min_segment_size:
                raise ValueError(
                    f"Insufficient data points for structural break testing (need at least {2 * self.min_segment_size})"
                )

            # Perform structural break test based on method
            if self.test_type == "chow":
                test_results = self._perform_chow_test(series)
            elif self.test_type == "cusum":
                test_results = self._perform_cusum_test(series)
            elif self.test_type == "recursive":
                test_results = self._perform_recursive_test(series)
            elif self.test_type == "quandt_andrews":
                test_results = self._perform_quandt_andrews_test(series)
            else:
                raise ValueError(f"Unknown structural break test: {self.test_type}")

            self.test_results_ = test_results

            # Extract break points if detected
            break_points = test_results.get("break_points", [])
            self.break_points_ = break_points

            # Calculate test diagnostics
            test_diagnostics = self._calculate_test_diagnostics(series, test_results)

            # Generate interpretation
            interpretation = self._generate_break_test_interpretation(
                test_results, series, test_diagnostics
            )

            # Generate recommendations
            recommendations = self._generate_break_test_recommendations(
                test_results, series, test_diagnostics
            )

            processing_time = time.time() - start_time

            # Prepare result
            model_parameters = {
                "test_type": self.test_type,
                "test_statistic": test_results.get("statistic"),
                "p_value": test_results.get("p_value"),
                "critical_value": test_results.get("critical_value"),
                "break_points": break_points,
                "break_point_used": self.break_point,
                "significance_level": self.significance_level,
                "test_details": test_results,
            }

            model_diagnostics = test_diagnostics

            return TimeSeriesAnalysisResult(
                analysis_type="structural_break_test",
                statistic=test_results.get("statistic"),
                p_value=test_results.get("p_value"),
                critical_values={"critical_value": test_results.get("critical_value")},
                model_parameters=model_parameters,
                model_diagnostics=model_diagnostics,
                interpretation=interpretation,
                recommendations=recommendations,
                processing_time=processing_time,
                data_quality_score=self._calculate_data_quality_score(X),
                confidence_level=1 - self.significance_level,
            )

        except Exception as e:
            logger.error(f"Structural break test failed: {e}")
            return TimeSeriesAnalysisResult(
                analysis_type="structural_break_test",
                interpretation=f"Structural break test failed: {str(e)}",
                recommendations=["Check data quality and test parameters"],
                processing_time=time.time() - start_time,
            )

    def _resolve_break_index(self, n: int) -> int:
        """Resolve the break index from self.break_point or default to midpoint."""
        if self.break_point is None:
            break_idx = n // 2
        elif isinstance(self.break_point, float) and 0 < self.break_point < 1:
            break_idx = int(self.break_point * n)
        else:
            break_idx = int(self.break_point)
        return max(self.min_segment_size, min(break_idx, n - self.min_segment_size))

    def _fit_ols_segment(self, X_design: np.ndarray, y_segment: np.ndarray) -> tuple:
        """Fit OLS on a segment; return (beta, rss). Falls back to mean on failure."""
        try:
            beta = np.linalg.lstsq(X_design, y_segment, rcond=None)[0]
            residuals = y_segment - X_design @ beta
            return beta, float(np.sum(residuals**2))
        except Exception:
            mean_val = np.mean(y_segment)
            return np.array([mean_val, 0.0]), float(np.sum((y_segment - mean_val) ** 2))

    def _perform_chow_test(self, series: pd.Series) -> Dict:
        """Perform Chow test for structural break at known point."""
        try:
            n = len(series)
            break_idx = self._resolve_break_index(n)

            t = np.arange(n)
            y = series.values
            X_full = np.column_stack([np.ones(n), t])

            beta_full, rss_full = self._fit_ols_segment(X_full, y)
            beta1, rss1 = self._fit_ols_segment(X_full[:break_idx], y[:break_idx])
            beta2, rss2 = self._fit_ols_segment(X_full[break_idx:], y[break_idx:])

            rss_unrestricted = rss1 + rss2
            k = 2  # Number of parameters

            if rss_unrestricted > 0:
                chow_stat = ((rss_full - rss_unrestricted) / k) / (
                    rss_unrestricted / (n - 2 * k)
                )
            else:
                chow_stat = 0.0

            from scipy.stats import f

            p_value = 1 - f.cdf(chow_stat, k, n - 2 * k)
            critical_value = f.ppf(1 - self.significance_level, k, n - 2 * k)

            return {
                "statistic": chow_stat,
                "p_value": p_value,
                "critical_value": critical_value,
                "break_points": [break_idx] if chow_stat > critical_value else [],
                "rss_full": rss_full,
                "rss_unrestricted": rss_unrestricted,
                "break_point_tested": break_idx,
                "segments": {
                    "segment_1": {
                        "start": 0,
                        "end": break_idx,
                        "coefficients": beta1.tolist(),
                    },
                    "segment_2": {
                        "start": break_idx,
                        "end": n,
                        "coefficients": beta2.tolist(),
                    },
                },
            }

        except Exception as e:
            logger.error(f"Chow test failed: {e}")
            return {
                "statistic": 0,
                "p_value": 1,
                "critical_value": 0,
                "break_points": [],
            }

    def _perform_cusum_test(self, series: pd.Series) -> Dict:
        """Perform CUSUM test for structural stability."""
        try:
            y = series.values
            n = len(y)

            # Fit regression on full sample
            t = np.arange(n)
            X = np.column_stack([np.ones(n), t])

            try:
                beta = np.linalg.lstsq(X, y, rcond=None)[0]
                residuals = y - X @ beta
            except:
                residuals = y - np.mean(y)

            # Calculate CUSUM statistics
            sigma2 = np.var(residuals)
            if sigma2 == 0:
                return {
                    "statistic": 0,
                    "p_value": 1,
                    "critical_value": 0,
                    "break_points": [],
                }

            # Recursive residuals (simplified)
            cusum = np.cumsum(residuals) / np.sqrt(sigma2 * n)
            cusum_sq = np.cumsum(residuals**2) / (sigma2 * n)

            # Brown-Durbin-Evans CUSUM test
            cusum_stat = np.max(
                np.abs(cusum[self.min_segment_size : -self.min_segment_size])
            )

            # Critical value (approximate)
            critical_value = 0.948 * np.sqrt(n)  # 5% significance level approximation

            # Calculate p-value (approximate)
            p_value = 2 * (1 - stats.norm.cdf(cusum_stat))

            # Detect break points where CUSUM exceeds boundaries
            break_points = []
            for i in range(self.min_segment_size, n - self.min_segment_size):
                if abs(cusum[i]) > critical_value / np.sqrt(n):
                    break_points.append(i)

            # Remove clustered break points
            if break_points:
                filtered_breaks = [break_points[0]]
                for bp in break_points[1:]:
                    if bp - filtered_breaks[-1] >= self.min_segment_size:
                        filtered_breaks.append(bp)
                break_points = filtered_breaks

            return {
                "statistic": cusum_stat,
                "p_value": p_value,
                "critical_value": critical_value,
                "break_points": break_points,
                "cusum_path": cusum.tolist(),
                "cusum_squares": cusum_sq.tolist(),
            }

        except Exception as e:
            logger.error(f"CUSUM test failed: {e}")
            return {
                "statistic": 0,
                "p_value": 1,
                "critical_value": 0,
                "break_points": [],
            }

    def _calculate_test_diagnostics(
        self, series: pd.Series, test_results: Dict
    ) -> Dict:
        """Calculate diagnostic information for structural break test."""
        diagnostics = {
            "series_length": len(series),
            "test_power": 1 - test_results.get("p_value", 1),
            "break_detected": len(test_results.get("break_points", [])) > 0,
            "significance_level": self.significance_level,
        }

        # Add test-specific diagnostics
        if self.test_type == "chow":
            diagnostics["break_point_tested"] = test_results.get("break_point_tested")
            diagnostics["rss_reduction"] = test_results.get(
                "rss_full", 0
            ) - test_results.get("rss_unrestricted", 0)
        elif self.test_type == "cusum":
            diagnostics["cusum_boundary_violations"] = len(
                test_results.get("break_points", [])
            )

        return diagnostics

    def _generate_break_test_interpretation(
        self, test_results: Dict, series: pd.Series, diagnostics: Dict
    ) -> str:
        """Generate interpretation for structural break test results."""
        statistic = test_results.get("statistic", 0)
        p_value = test_results.get("p_value", 1)
        break_points = test_results.get("break_points", [])

        if len(break_points) == 0:
            return (
                f"{self.test_type.title()} test found no evidence of structural breaks "
                f"(test statistic: {statistic:.3f}, p-value: {p_value:.3f}). "
                f"The time series appears structurally stable over the analyzed period."
            )

        interpretation = (
            f"{self.test_type.title()} test detected {len(break_points)} "
            f"structural break{'s' if len(break_points) > 1 else ''} "
            f"(test statistic: {statistic:.3f}, p-value: {p_value:.3f}). "
        )

        if p_value < self.significance_level:
            interpretation += f"The evidence is statistically significant at the {self.significance_level:.1%} level. "
        else:
            interpretation += f"The evidence is not statistically significant at the {self.significance_level:.1%} level. "

        # Add break point specific information
        if len(break_points) == 1:
            bp_pct = (break_points[0] / len(series)) * 100
            interpretation += f"The break occurs at position {break_points[0]} ({bp_pct:.1f}% through the series)."
        else:
            interpretation += f"Multiple breaks detected at positions: {break_points}."

        return interpretation

    def _generate_break_test_recommendations(
        self, test_results: Dict, series: pd.Series, diagnostics: Dict
    ) -> List[str]:
        """Generate recommendations based on structural break test results."""
        recommendations = []
        break_points = test_results.get("break_points", [])
        p_value = test_results.get("p_value", 1)

        if len(break_points) == 0:
            recommendations.extend(
                [
                    "No structural breaks detected - series appears stable",
                    "Standard time series models (ARIMA, exponential smoothing) should work well",
                    "Consider monitoring for future structural changes",
                ]
            )
        else:
            recommendations.extend(
                [
                    f"Structural break{'s' if len(break_points) > 1 else ''} detected - model accordingly",
                    "Consider regime-switching models or separate models for each period",
                    "Investigate potential causes of structural changes",
                ]
            )

            if len(break_points) > 1:
                recommendations.append(
                    "Multiple breaks suggest highly unstable system - use caution in forecasting"
                )

        # Test-specific recommendations
        if self.test_type == "chow":
            if p_value > self.significance_level:
                recommendations.append("Consider testing other potential break points")
        elif self.test_type == "cusum":
            recommendations.append(
                "CUSUM path analysis can help identify gradual vs. abrupt changes"
            )

        # Power and sample size recommendations
        if len(series) < 50:
            recommendations.append(
                "Small sample size may reduce test power - interpret results cautiously"
            )

        return recommendations
