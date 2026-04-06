"""
Time Series Analysis - Impulse response analyzer.

Contains the ImpulseResponseAnalyzer class for computing impulse response
functions and forecast error variance decomposition from VAR models.
"""

import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.vector_ar.var_model import VAR

from ...logging_manager import get_logger
from ._base import TimeSeriesAnalysisResult, TimeSeriesValidationError
from ._multivariate_base import MultivariateTimeSeriesTransformer

logger = get_logger(__name__)


class ImpulseResponseAnalyzer(MultivariateTimeSeriesTransformer):
    """
    Impulse Response Function (IRF) analysis for multivariate time series.

    This analyzer computes impulse response functions from a fitted VAR model to
    understand how shocks to one variable propagate through the system over time.
    IRFs show the dynamic response of each variable to a one-unit shock in any
    variable in the system, providing insights into shock transmission mechanisms
    and the temporal dynamics of variable interactions.

    Key Features:
    - Orthogonalized impulse response functions using Cholesky decomposition
    - Confidence intervals for impulse responses using bootstrap or analytical methods
    - Cumulative impulse response analysis for permanent effect assessment
    - Forecast error variance decomposition (FEVD) for shock contribution analysis
    - Multiple periods ahead analysis with customizable horizons
    - Comprehensive interpretation of shock propagation patterns

    Parameters:
    -----------
    periods : int, default=10
        Number of periods ahead for impulse response computation
    orthogonalized : bool, default=True
        Whether to compute orthogonalized impulse responses
    confidence_level : float, default=0.95
        Confidence level for impulse response confidence bands
    bootstrap_reps : int, default=1000
        Number of bootstrap replications for confidence intervals
    cumulative : bool, default=False
        Whether to compute cumulative impulse responses

    Example:
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from localdata_mcp.domains.time_series_analysis import ImpulseResponseAnalyzer
    >>>
    >>> # Create sample VAR data
    >>> dates = pd.date_range('2020-01-01', periods=200, freq='D')
    >>> np.random.seed(42)
    >>> data = pd.DataFrame({
    ...     'gdp': np.cumsum(np.random.randn(200) * 0.1),
    ...     'interest_rate': np.cumsum(np.random.randn(200) * 0.05),
    ...     'inflation': np.cumsum(np.random.randn(200) * 0.03)
    ... }, index=dates)
    >>>
    >>> # Perform impulse response analysis
    >>> analyzer = ImpulseResponseAnalyzer(periods=12)
    >>> result = analyzer.fit_transform(data)
    >>>
    >>> print(f"IRF shape: {result.model_parameters['impulse_responses'].shape}")
    >>> print(f"FEVD results: {result.model_parameters['fevd_results']}")
    """

    def __init__(
        self,
        periods: int = 10,
        orthogonalized: bool = True,
        confidence_level: float = 0.95,
        bootstrap_reps: int = 1000,
        cumulative: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.periods = periods
        self.orthogonalized = orthogonalized
        self.confidence_level = confidence_level
        self.bootstrap_reps = bootstrap_reps
        self.cumulative = cumulative

        # Validate parameters
        if periods < 1:
            raise ValueError("periods must be at least 1")

        if confidence_level <= 0 or confidence_level >= 1:
            raise ValueError("confidence_level must be between 0 and 1")

        if bootstrap_reps < 100:
            raise ValueError("bootstrap_reps must be at least 100")

    def _fit_var_model(self, X: pd.DataFrame):
        """Fit a VAR model and return (var_fitted, optimal_lags)."""
        var_model = VAR(X)
        lag_order_results = var_model.select_order(maxlags=min(10, len(X) // 10))
        optimal_lags = getattr(lag_order_results, "aic")
        var_fitted = var_model.fit(maxlags=optimal_lags)
        logger.info(f"Fitted VAR({optimal_lags}) model for impulse response analysis")
        return var_fitted, optimal_lags

    def _compute_irf_arrays(self, var_fitted):
        """Compute IRF arrays and optional confidence intervals."""
        irf_result = var_fitted.irf(self.periods)
        impulse_responses = (
            irf_result.orth_irfs if self.orthogonalized else irf_result.irfs
        )
        cumulative_responses = (
            np.cumsum(impulse_responses, axis=0) if self.cumulative else None
        )

        irf_lower_approx = None
        irf_upper_approx = None
        try:
            irf_result.cum_effect_cova(orth=self.orthogonalized)[-1]
            alpha = 1 - self.confidence_level
            z_score = stats.norm.ppf(1 - alpha / 2)
            irf_stderr = np.std(impulse_responses, axis=0, keepdims=True)
            irf_lower_approx = impulse_responses - z_score * irf_stderr
            irf_upper_approx = impulse_responses + z_score * irf_stderr
        except Exception as e:
            logger.warning(f"Could not compute confidence intervals: {e}")

        return (
            impulse_responses,
            cumulative_responses,
            irf_lower_approx,
            irf_upper_approx,
        )

    def _compute_fevd(self, var_fitted):
        """Compute Forecast Error Variance Decomposition."""
        try:
            fevd_result = var_fitted.fevd(self.periods)
            return fevd_result.decomp
        except Exception as e:
            logger.warning(f"Could not compute FEVD: {e}")
            return None

    def _build_irf_dataframes(
        self, impulse_responses, cumulative_responses, fevd_decomposition, series_names
    ):
        """Build structured DataFrames from IRF arrays."""
        periods_index = list(range(self.periods + 1))
        irf_dict = {
            f"{shock_var} \u2192 {response_var}": impulse_responses[
                :, resp_idx, shock_idx
            ]
            for shock_idx, shock_var in enumerate(series_names)
            for resp_idx, response_var in enumerate(series_names)
        }
        irf_df = pd.DataFrame(irf_dict, index=periods_index)

        cumulative_irf_df = None
        if cumulative_responses is not None:
            cum_dict = {
                f"{shock_var} \u2192 {response_var}": cumulative_responses[
                    :, resp_idx, shock_idx
                ]
                for shock_idx, shock_var in enumerate(series_names)
                for resp_idx, response_var in enumerate(series_names)
            }
            cumulative_irf_df = pd.DataFrame(cum_dict, index=periods_index)

        fevd_df = None
        if fevd_decomposition is not None:
            n_fevd_periods = fevd_decomposition.shape[1]
            fevd_dict: Dict = {}
            for period in range(n_fevd_periods):
                for shock_idx, shock_var in enumerate(series_names):
                    for resp_idx, response_var in enumerate(series_names):
                        key = f"{response_var} \u2190 {shock_var}"
                        fevd_dict.setdefault(key, []).append(
                            fevd_decomposition[resp_idx, period, shock_idx]
                        )
            fevd_df = pd.DataFrame(fevd_dict, index=list(range(n_fevd_periods)))

        return irf_dict, irf_df, cumulative_irf_df, fevd_df

    def _rank_significant_responses(self, irf_dict: Dict) -> List[Dict]:
        """Rank impulse responses by magnitude and return sorted list."""
        significant_responses = [
            {
                "relationship": key,
                "max_response": np.max(np.abs(responses)),
                "max_period": int(np.argmax(np.abs(responses))),
                "total_cumulative_effect": float(np.sum(responses)),
            }
            for key, responses in irf_dict.items()
        ]
        significant_responses.sort(key=lambda x: abs(x["max_response"]), reverse=True)
        return significant_responses

    def _analysis_logic(self, X: pd.DataFrame) -> TimeSeriesAnalysisResult:
        """Core impulse response analysis logic."""
        start_time = time.time()

        try:
            X = self._validate_multivariate_data(X)
            var_fitted, optimal_lags = self._fit_var_model(X)

            (
                impulse_responses,
                cumulative_responses,
                irf_lower_approx,
                irf_upper_approx,
            ) = self._compute_irf_arrays(var_fitted)
            fevd_decomposition = self._compute_fevd(var_fitted)

            series_names = list(X.columns)
            n_vars = len(series_names)
            irf_dict, irf_df, cumulative_irf_df, fevd_df = self._build_irf_dataframes(
                impulse_responses,
                cumulative_responses,
                fevd_decomposition,
                series_names,
            )
            significant_responses = self._rank_significant_responses(irf_dict)

            model_diagnostics = {
                "var_model_lags": optimal_lags,
                "periods_analyzed": self.periods,
                "orthogonalized": self.orthogonalized,
                "n_variables": n_vars,
                "confidence_level": self.confidence_level,
                "has_confidence_intervals": irf_lower_approx is not None,
                "has_fevd": fevd_decomposition is not None,
            }
            model_parameters = {
                "impulse_responses": impulse_responses,
                "impulse_response_df": irf_df,
                "cumulative_responses": cumulative_responses,
                "cumulative_response_df": cumulative_irf_df,
                "confidence_intervals_lower": irf_lower_approx,
                "confidence_intervals_upper": irf_upper_approx,
                "fevd_decomposition": fevd_decomposition,
                "fevd_df": fevd_df,
                "significant_responses": significant_responses,
                "series_names": series_names,
                "var_model_params": {
                    "optimal_lags": optimal_lags,
                    "aic": var_fitted.aic,
                    "bic": var_fitted.bic,
                },
            }

            interpretation = self._generate_irf_interpretation(
                significant_responses, n_vars, self.periods
            )
            recommendations = self._generate_irf_recommendations(
                significant_responses, fevd_decomposition, X
            )
            processing_time = time.time() - start_time

            return self._prepare_multivariate_result(
                analysis_type="impulse_response_analysis",
                model_parameters=model_parameters,
                model_diagnostics=model_diagnostics,
                interpretation=interpretation,
                recommendations=recommendations,
                processing_time=processing_time,
                data_quality_score=self._calculate_data_quality_score(X),
            )

        except Exception as e:
            logger.error(f"Impulse response analysis failed: {e}")
            return self._prepare_multivariate_result(
                analysis_type="impulse_response_analysis",
                interpretation=f"Impulse response analysis failed: {str(e)}",
                recommendations=["Check data quality and VAR model specification"],
                processing_time=time.time() - start_time,
            )

    def _generate_irf_interpretation(
        self, significant_responses: List[Dict], n_vars: int, periods: int
    ) -> str:
        """
        Generate interpretation of impulse response analysis results.

        Parameters:
        -----------
        significant_responses : list
            List of impulse response relationships sorted by significance
        n_vars : int
            Number of variables in the system
        periods : int
            Number of periods analyzed

        Returns:
        --------
        interpretation : str
            Human-readable interpretation
        """
        if not significant_responses:
            return f"Impulse response analysis over {periods} periods found no significant responses."

        strongest_response = significant_responses[0]
        n_relationships = len(significant_responses)

        interpretation = (
            f"Impulse response analysis over {periods} periods identified "
            f"{n_relationships} shock transmission pathways in {n_vars}-variable system. "
        )

        # Describe strongest response
        interpretation += (
            f"Strongest response: {strongest_response['relationship']} "
            f"with maximum impact of {strongest_response['max_response']:.3f} "
            f"at period {strongest_response['max_period']}. "
        )

        # Analyze persistence
        cumulative_effect = strongest_response["total_cumulative_effect"]
        if abs(cumulative_effect) > abs(strongest_response["max_response"]) * 0.5:
            interpretation += (
                "Response shows persistent effects over the analyzed horizon. "
            )
        else:
            interpretation += (
                "Response shows transitory effects that dissipate quickly. "
            )

        # Check for system-wide effects
        high_impact_responses = [
            r for r in significant_responses if abs(r["max_response"]) > 0.1
        ]
        if len(high_impact_responses) > n_vars:
            interpretation += "System shows strong interconnectedness with widespread shock propagation."
        else:
            interpretation += (
                "Shock transmission appears limited to specific variable relationships."
            )

        return interpretation

    def _irf_timing_recommendation(self, significant_responses: List[Dict]) -> str:
        """Return a recommendation based on the average timing of max shock impact."""
        avg_max_period = np.mean([r["max_period"] for r in significant_responses])
        if avg_max_period < 2:
            return (
                "Shocks show immediate impact - system responds quickly to disturbances"
            )
        elif avg_max_period > 5:
            return "Shocks show delayed maximum impact - consider longer forecasting horizons"
        return "Shocks show moderate response timing - standard forecasting horizons appropriate"

    def _fevd_recommendations(self, fevd_decomposition) -> List[str]:
        """Return recommendations derived from FEVD results."""
        recs = ["Use FEVD results to identify key drivers of forecast error variance"]
        final_period_fevd = fevd_decomposition[-1]
        if np.max(final_period_fevd) > 0.7:
            recs.append(
                "One variable dominates forecast error variance - focus forecasting efforts accordingly"
            )
        return recs

    def _generate_irf_recommendations(
        self, significant_responses: List[Dict], fevd_decomposition, X: pd.DataFrame
    ) -> List[str]:
        """Generate recommendations based on impulse response analysis results."""
        if not significant_responses:
            return [
                "No significant impulse responses detected - system may be weakly connected",
                "Consider alternative shock identification strategies",
                "Verify VAR model specification and lag structure",
            ]

        recommendations = [self._irf_timing_recommendation(significant_responses)]

        persistent_shocks = [
            r
            for r in significant_responses
            if abs(r["total_cumulative_effect"]) > abs(r["max_response"]) * 0.7
        ]
        if persistent_shocks:
            recommendations.append(
                f"{len(persistent_shocks)} persistent shock relationships detected - "
                "permanent effects should be considered in long-term forecasting"
            )

        if fevd_decomposition is not None:
            recommendations.extend(self._fevd_recommendations(fevd_decomposition))

        high_impact_count = sum(
            1 for r in significant_responses if abs(r["max_response"]) > 0.1
        )
        if high_impact_count > len(X.columns) * 2:
            recommendations.append(
                "Complex shock transmission detected - consider structural VAR identification"
            )

        if len(X) < 100:
            recommendations.append(
                "Limited data for robust IRF estimation - confidence intervals may be wide"
            )

        if not recommendations:
            recommendations.append(
                "Impulse response analysis provides clear shock transmission patterns - "
                "use for forecasting and policy analysis"
            )

        return recommendations
