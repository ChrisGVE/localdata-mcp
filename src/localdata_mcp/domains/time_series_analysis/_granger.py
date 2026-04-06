"""
Time Series Analysis - Granger causality analyzer.

Contains the GrangerCausalityAnalyzer class for pairwise Granger causality
testing between multivariate time series.
"""

import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests

from ...logging_manager import get_logger
from ._base import TimeSeriesAnalysisResult, TimeSeriesValidationError
from ._multivariate_base import MultivariateTimeSeriesTransformer

logger = get_logger(__name__)


class GrangerCausalityAnalyzer(MultivariateTimeSeriesTransformer):
    """
    Granger causality analysis for multivariate time series.

    This analyzer tests whether one time series can help predict another time series,
    which is the statistical definition of Granger causality. The test examines if
    lagged values of variable X provide statistically significant information about
    variable Y, beyond what is already contained in lagged values of Y itself.

    Key Features:
    - Pairwise Granger causality testing for all variable combinations
    - Multiple lag lengths testing with automatic optimal lag selection
    - F-statistic computation with p-values for significance testing
    - Directional causality analysis (X->Y vs Y->X)
    - Comprehensive interpretation of causal relationships
    - Support for different significance levels

    Parameters:
    -----------
    max_lags : int, default=4
        Maximum number of lags to test for Granger causality
    significance_level : float, default=0.05
        Significance level for Granger causality tests
    test_all_pairs : bool, default=True
        Whether to test all pairwise combinations of variables

    Example:
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from localdata_mcp.domains.time_series_analysis import GrangerCausalityAnalyzer
    >>>
    >>> # Create sample data with causal relationship
    >>> dates = pd.date_range('2020-01-01', periods=200, freq='D')
    >>> np.random.seed(42)
    >>> x = np.random.randn(200)
    >>> y = np.zeros(200)
    >>> for i in range(1, 200):
    ...     y[i] = 0.5 * x[i-1] + 0.3 * y[i-1] + np.random.randn() * 0.1
    >>>
    >>> data = pd.DataFrame({
    ...     'cause': x,
    ...     'effect': y
    ... }, index=dates)
    >>>
    >>> # Perform Granger causality analysis
    >>> analyzer = GrangerCausalityAnalyzer()
    >>> result = analyzer.fit_transform(data)
    >>>
    >>> print(f"Causality results: {result.model_parameters['causality_results']}")
    """

    def __init__(
        self,
        max_lags: int = 4,
        significance_level: float = 0.05,
        test_all_pairs: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.max_lags = max_lags
        self.significance_level = significance_level
        self.test_all_pairs = test_all_pairs

        # Validate parameters
        if max_lags < 1:
            raise ValueError("max_lags must be at least 1")

        if significance_level <= 0 or significance_level >= 1:
            raise ValueError("significance_level must be between 0 and 1")

    def _test_pair(
        self, X: pd.DataFrame, cause_var: str, effect_var: str
    ) -> Optional[Dict]:
        """Run Granger causality test for one (cause, effect) pair; return result dict or None."""
        pair_name = f"{cause_var} \u2192 {effect_var}"
        test_data = X[[effect_var, cause_var]].dropna()
        if len(test_data) < self.max_lags + 20:
            logger.warning(f"Insufficient data for {pair_name} causality test")
            return None

        granger_result = grangercausalitytests(
            test_data, maxlag=self.max_lags, verbose=False
        )

        lag_results: Dict = {}
        min_p_value = 1.0
        best_lag = 1
        for lag in range(1, self.max_lags + 1):
            if lag in granger_result:
                f_stat = granger_result[lag][0]["ssr_ftest"][0]
                p_val = granger_result[lag][0]["ssr_ftest"][1]
                lag_results[lag] = {
                    "f_statistic": f_stat,
                    "p_value": p_val,
                    "significant": p_val < self.significance_level,
                }
                if p_val < min_p_value:
                    min_p_value = p_val
                    best_lag = lag

        return {
            "pair_name": pair_name,
            "cause": cause_var,
            "effect": effect_var,
            "lag_results": lag_results,
            "best_lag": best_lag,
            "min_p_value": min_p_value,
            "significant": min_p_value < self.significance_level,
            "f_statistic_best": (
                lag_results[best_lag]["f_statistic"]
                if best_lag in lag_results
                else None
            ),
        }

    def _run_pairwise_tests(self, X: pd.DataFrame, series_names: List[str]):
        """Run all pairwise Granger causality tests and collect results."""
        causality_results: Dict = {}
        f_statistics: Dict = {}
        p_values: Dict = {}
        significant_relationships: List[Dict] = []

        for i, cause_var in enumerate(series_names):
            for j, effect_var in enumerate(series_names):
                if i == j:
                    continue
                pair_name = f"{cause_var} \u2192 {effect_var}"
                try:
                    result = self._test_pair(X, cause_var, effect_var)
                    if result is None:
                        continue
                    causality_results[pair_name] = {
                        k: v for k, v in result.items() if k != "pair_name"
                    }
                    f_statistics[pair_name] = result["f_statistic_best"]
                    p_values[pair_name] = result["min_p_value"]
                    if result["significant"]:
                        significant_relationships.append(
                            {
                                "relationship": pair_name,
                                "p_value": result["min_p_value"],
                                "lag": result["best_lag"],
                                "f_statistic": result["f_statistic_best"],
                            }
                        )
                except Exception as e:
                    logger.warning(
                        f"Granger causality test failed for {pair_name}: {e}"
                    )

        significant_relationships.sort(key=lambda x: x["p_value"])
        return causality_results, f_statistics, p_values, significant_relationships

    def _analysis_logic(self, X: pd.DataFrame) -> TimeSeriesAnalysisResult:
        """Core Granger causality analysis logic."""
        start_time = time.time()

        try:
            X = self._validate_multivariate_data(X)
            series_names = list(X.columns)

            causality_results, f_statistics, p_values, significant_relationships = (
                self._run_pairwise_tests(X, series_names)
            )

            causality_matrix = self._create_causality_matrix(
                causality_results, series_names
            )

            model_diagnostics = {
                "n_relationships_tested": len(causality_results),
                "n_significant_relationships": len(significant_relationships),
                "significance_level": self.significance_level,
                "max_lags_tested": self.max_lags,
                "causality_matrix": causality_matrix,
            }
            model_parameters = {
                "causality_results": causality_results,
                "significant_relationships": significant_relationships,
                "f_statistics": f_statistics,
                "p_values": p_values,
                "causality_matrix": causality_matrix,
                "series_names": series_names,
            }

            interpretation = self._generate_granger_interpretation(
                significant_relationships, len(causality_results), series_names
            )
            recommendations = self._generate_granger_recommendations(
                significant_relationships, causality_results, X
            )
            processing_time = time.time() - start_time

            return self._prepare_multivariate_result(
                analysis_type="granger_causality",
                statistic=(
                    significant_relationships[0]["f_statistic"]
                    if significant_relationships
                    else None
                ),
                p_value=(
                    significant_relationships[0]["p_value"]
                    if significant_relationships
                    else None
                ),
                model_parameters=model_parameters,
                model_diagnostics=model_diagnostics,
                interpretation=interpretation,
                recommendations=recommendations,
                processing_time=processing_time,
                data_quality_score=self._calculate_data_quality_score(X),
            )

        except Exception as e:
            logger.error(f"Granger causality analysis failed: {e}")
            return self._prepare_multivariate_result(
                analysis_type="granger_causality",
                interpretation=f"Granger causality analysis failed: {str(e)}",
                recommendations=["Check data quality and stationarity"],
                processing_time=time.time() - start_time,
            )

    def _create_causality_matrix(
        self, causality_results: Dict, series_names: List[str]
    ) -> pd.DataFrame:
        """
        Create a matrix showing causality relationships between all variables.

        Parameters:
        -----------
        causality_results : dict
            Results from Granger causality tests
        series_names : list
            Names of the time series

        Returns:
        --------
        matrix : pd.DataFrame
            Causality matrix where entry (i,j) indicates whether i causes j
        """
        n_series = len(series_names)
        matrix = np.zeros((n_series, n_series))

        for relationship, results in causality_results.items():
            cause_var = results["cause"]
            effect_var = results["effect"]

            try:
                cause_idx = series_names.index(cause_var)
                effect_idx = series_names.index(effect_var)

                # Store p-value (0 = strong causality, 1 = no causality)
                matrix[cause_idx, effect_idx] = results["min_p_value"]
            except ValueError:
                continue

        causality_df = pd.DataFrame(matrix, index=series_names, columns=series_names)

        return causality_df

    def _generate_granger_interpretation(
        self,
        significant_relationships: List[Dict],
        total_tests: int,
        series_names: List[str],
    ) -> str:
        """
        Generate interpretation of Granger causality results.

        Parameters:
        -----------
        significant_relationships : list
            List of significant causality relationships
        total_tests : int
            Total number of causality tests performed
        series_names : list
            Names of time series

        Returns:
        --------
        interpretation : str
            Human-readable interpretation
        """
        n_significant = len(significant_relationships)
        n_series = len(series_names)

        interpretation = (
            f"Granger causality analysis tested {total_tests} directional relationships "
            f"between {n_series} time series. "
            f"Found {n_significant} significant causal relationships "
            f"at {self.significance_level * 100:.0f}% significance level. "
        )

        if n_significant == 0:
            interpretation += (
                "No significant causal relationships detected. "
                "Variables appear to evolve independently or relationships "
                "may be non-linear or at different time scales."
            )
        elif n_significant == 1:
            rel = significant_relationships[0]
            interpretation += (
                f"One significant relationship found: {rel['relationship']} "
                f"(p-value: {rel['p_value']:.4f}, optimal lag: {rel['lag']}). "
                "This suggests predictive information flows in one direction."
            )
        else:
            interpretation += (
                f"Multiple causal relationships detected. "
                f"Strongest relationship: {significant_relationships[0]['relationship']} "
                f"(p-value: {significant_relationships[0]['p_value']:.4f}). "
            )

            # Check for bidirectional causality
            relationships_set = {
                rel["relationship"] for rel in significant_relationships
            }
            bidirectional = []

            for rel in significant_relationships:
                cause, effect = rel["relationship"].split(" \u2192 ")
                reverse_rel = f"{effect} \u2192 {cause}"
                if reverse_rel in relationships_set:
                    bidirectional.append((cause, effect))

            if bidirectional:
                interpretation += f" Detected {len(bidirectional)} bidirectional causal relationships."

        return interpretation

    def _generate_granger_recommendations(
        self,
        significant_relationships: List[Dict],
        causality_results: Dict,
        X: pd.DataFrame,
    ) -> List[str]:
        """
        Generate recommendations based on Granger causality results.

        Parameters:
        -----------
        significant_relationships : list
            List of significant causality relationships
        causality_results : dict
            All causality test results
        X : pd.DataFrame
            Original data

        Returns:
        --------
        recommendations : List[str]
            List of recommendations
        """
        recommendations = []
        n_significant = len(significant_relationships)

        if n_significant == 0:
            recommendations.extend(
                [
                    "No Granger causality detected - consider alternative modeling approaches",
                    "Check for non-linear relationships using non-parametric methods",
                    "Consider different lag structures or transformation of variables",
                ]
            )
        else:
            recommendations.extend(
                [
                    f"Incorporate {n_significant} causal relationships in forecasting models",
                    "Use causal variables as predictors in targeted forecasting",
                    "Consider VAR model structure based on causality patterns",
                ]
            )

            # Check for complex causal patterns
            if n_significant > len(X.columns):
                recommendations.append(
                    "Complex causal network detected - consider structural VAR modeling"
                )

            # Lag-specific recommendations
            optimal_lags = [rel["lag"] for rel in significant_relationships]
            if max(optimal_lags) > 1:
                recommendations.append(
                    f"Multi-period causality detected (up to {max(optimal_lags)} lags) - "
                    "consider longer-term predictive relationships"
                )

        # Data quality recommendations
        if len(X) < self.max_lags * 20:
            recommendations.append(
                "Limited sample size relative to lag order - results may be unreliable"
            )

        # Statistical power recommendations
        weak_relationships = [
            rel
            for rel in significant_relationships
            if rel["p_value"] > self.significance_level * 2  # Marginally significant
        ]
        if weak_relationships:
            recommendations.append(
                "Some marginally significant relationships detected - verify with larger sample"
            )

        if not recommendations:
            recommendations.append(
                "Granger causality analysis appears robust - use results for model specification"
            )

        return recommendations
