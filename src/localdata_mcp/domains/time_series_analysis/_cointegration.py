"""
Time Series Analysis - Cointegration analyzer.

Contains the CointegrationAnalyzer class for Johansen cointegration testing
and long-run equilibrium relationship analysis.
"""

import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from statsmodels.tsa.vector_ar.vecm import coint_johansen

from ...logging_manager import get_logger
from ._base import TimeSeriesAnalysisResult, TimeSeriesValidationError
from ._multivariate_base import MultivariateTimeSeriesTransformer

logger = get_logger(__name__)


class CointegrationAnalyzer(MultivariateTimeSeriesTransformer):
    """
    Cointegration analysis for multivariate time series using Johansen test.

    This analyzer identifies long-run equilibrium relationships between multiple
    non-stationary time series. The Johansen cointegration test determines the
    number of cointegrating relationships and provides error correction model
    components for understanding short-term dynamics and long-term equilibrium.

    Key Features:
    - Johansen cointegration test with trace and maximum eigenvalue statistics
    - Automatic determination of optimal lag order for VECM
    - Cointegrating vectors and adjustment coefficients estimation
    - Error correction model analysis
    - Comprehensive interpretation of long-term relationships
    - Support for different deterministic trend specifications

    Parameters:
    -----------
    det_order : int, default=-1
        Deterministic order for cointegration test
        (-1: no deterministic trend, 0: constant, 1: linear trend)
    k_ar_diff : int, default=1
        Number of lags for VECM in differences
    significance_level : float, default=0.05
        Significance level for cointegration tests

    Example:
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from localdata_mcp.domains.time_series_analysis import CointegrationAnalyzer
    >>>
    >>> # Create sample cointegrated time series
    >>> dates = pd.date_range('2020-01-01', periods=200, freq='D')
    >>> np.random.seed(42)
    >>> # Create common stochastic trend
    >>> common_trend = np.cumsum(np.random.randn(200))
    >>> data = pd.DataFrame({
    ...     'series1': common_trend + np.random.randn(200) * 0.1,
    ...     'series2': 2 * common_trend + np.random.randn(200) * 0.1,
    ...     'series3': common_trend - np.random.randn(200) * 0.1
    ... }, index=dates)
    >>>
    >>> # Perform cointegration analysis
    >>> analyzer = CointegrationAnalyzer()
    >>> result = analyzer.fit_transform(data)
    >>>
    >>> print(f"Number of cointegrating relationships: {result.model_parameters['n_coint']}")
    >>> print(f"Trace statistics: {result.model_parameters['trace_stat']}")
    """

    def __init__(
        self,
        det_order: int = -1,
        k_ar_diff: int = 1,
        significance_level: float = 0.05,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.det_order = det_order
        self.k_ar_diff = k_ar_diff
        self.significance_level = significance_level

        # Validate parameters
        if det_order not in [-1, 0, 1]:
            raise ValueError(
                "det_order must be -1 (no trend), 0 (constant), or 1 (linear trend)"
            )

        if significance_level <= 0 or significance_level >= 1:
            raise ValueError("significance_level must be between 0 and 1")

    def _count_cointegrating_relationships(self, coint_result, n_series: int):
        """Count cointegrating relationships using trace and max-eigenvalue tests."""
        n_coint_trace = sum(
            1 for i in range(n_series) if coint_result.lr1[i] > coint_result.cvt[i, 1]
        )
        n_coint_max_eigen = sum(
            1 for i in range(n_series) if coint_result.lr2[i] > coint_result.cvm[i, 1]
        )
        return n_coint_trace, n_coint_max_eigen, min(n_coint_trace, n_coint_max_eigen)

    def _extract_coint_vectors(self, coint_result, n_coint: int, X: pd.DataFrame):
        """Extract cointegrating vectors, adjustment coefficients and their DataFrames."""
        coint_vectors = coint_result.evec[:, :n_coint] if n_coint > 0 else None
        adjustment_coeffs = None
        if n_coint > 0:
            try:
                beta = coint_result.evec[:, :n_coint]
                rkt_beta = coint_result.rkt @ beta
                adjustment_coeffs = (
                    coint_result.r0t.T @ rkt_beta
                ) / coint_result.r0t.shape[0]
                adjustment_coeffs = adjustment_coeffs.T
            except Exception:
                pass

        coint_relationships = None
        if n_coint > 0 and coint_vectors is not None:
            coint_relationships = pd.DataFrame(
                coint_vectors,
                index=X.columns,
                columns=[f"Coint_Relationship_{i + 1}" for i in range(n_coint)],
            )

        adjustment_df = None
        if n_coint > 0 and adjustment_coeffs is not None:
            adjustment_df = pd.DataFrame(
                adjustment_coeffs,
                index=[f"Coint_Relationship_{i + 1}" for i in range(n_coint)],
                columns=X.columns,
            )

        return coint_vectors, adjustment_coeffs, coint_relationships, adjustment_df

    def _analysis_logic(self, X: pd.DataFrame) -> TimeSeriesAnalysisResult:
        """Core cointegration analysis logic."""
        start_time = time.time()

        try:
            X = self._validate_multivariate_data(X)
            coint_result = coint_johansen(
                X.values, det_order=self.det_order, k_ar_diff=self.k_ar_diff
            )
            n_series = X.shape[1]

            n_coint_trace, n_coint_max_eigen, n_coint = (
                self._count_cointegrating_relationships(coint_result, n_series)
            )
            logger.info(f"Detected {n_coint} cointegrating relationships")

            coint_vectors, adjustment_coeffs, coint_relationships, adjustment_df = (
                self._extract_coint_vectors(coint_result, n_coint, X)
            )

            model_diagnostics = {
                "n_cointegrating_relationships_trace": n_coint_trace,
                "n_cointegrating_relationships_max_eigen": n_coint_max_eigen,
                "n_cointegrating_relationships_final": n_coint,
                "deterministic_order": self.det_order,
                "vecm_lags": self.k_ar_diff,
                "trace_statistics": coint_result.lr1.tolist(),
                "max_eigen_statistics": coint_result.lr2.tolist(),
                "trace_critical_values": coint_result.cvt.tolist(),
                "max_eigen_critical_values": coint_result.cvm.tolist(),
                "eigenvalues": coint_result.eig.tolist(),
            }
            model_parameters = {
                "n_coint": n_coint,
                "det_order": self.det_order,
                "k_ar_diff": self.k_ar_diff,
                "trace_stat": coint_result.lr1,
                "max_eigen_stat": coint_result.lr2,
                "eigenvalues": coint_result.eig,
                "cointegrating_vectors": coint_vectors,
                "adjustment_coefficients": adjustment_coeffs,
                "cointegrating_relationships": coint_relationships,
                "adjustment_coefficients_df": adjustment_df,
            }

            interpretation = self._generate_cointegration_interpretation(
                n_coint, n_series, coint_result, X.columns
            )
            recommendations = self._generate_cointegration_recommendations(
                n_coint, n_series, coint_result, X
            )
            processing_time = time.time() - start_time

            return self._prepare_multivariate_result(
                analysis_type="cointegration_analysis",
                statistic=(
                    float(coint_result.lr1[0]) if len(coint_result.lr1) > 0 else None
                ),
                p_value=None,
                model_parameters=model_parameters,
                model_diagnostics=model_diagnostics,
                interpretation=interpretation,
                recommendations=recommendations,
                processing_time=processing_time,
                data_quality_score=self._calculate_data_quality_score(X),
            )

        except Exception as e:
            logger.error(f"Cointegration analysis failed: {e}")
            return self._prepare_multivariate_result(
                analysis_type="cointegration_analysis",
                interpretation=f"Cointegration analysis failed: {str(e)}",
                recommendations=["Check data quality and ensure series are I(1)"],
                processing_time=time.time() - start_time,
            )

    def _generate_cointegration_interpretation(
        self, n_coint: int, n_series: int, coint_result, series_names
    ) -> str:
        """
        Generate interpretation of cointegration analysis results.

        Parameters:
        -----------
        n_coint : int
            Number of cointegrating relationships detected
        n_series : int
            Total number of time series
        coint_result
            Johansen cointegration test result
        series_names
            Names of the time series

        Returns:
        --------
        interpretation : str
            Human-readable interpretation
        """
        interpretation = (
            f"Johansen cointegration test on {n_series} time series "
            f"detected {n_coint} cointegrating relationships. "
        )

        if n_coint == 0:
            interpretation += (
                "No long-term equilibrium relationships found. "
                "Time series appear to drift apart over time without "
                "a stable long-term relationship."
            )
        elif n_coint == 1:
            interpretation += (
                "One cointegrating relationship found, indicating a "
                "single long-term equilibrium binding the time series together. "
                "The series share a common stochastic trend."
            )
        elif n_coint == n_series - 1:
            interpretation += (
                f"Maximum number of cointegrating relationships ({n_coint}) found. "
                "This suggests the system is stationary in levels, with strong "
                "long-term equilibrium relationships."
            )
        else:
            interpretation += (
                f"Multiple cointegrating relationships ({n_coint}) found, "
                "indicating several independent long-term equilibria. "
                "The system has both common trends and stable relationships."
            )

        # Add information about the strongest relationship
        if n_coint > 0 and len(coint_result.lr1) > 0:
            max_trace_stat = max(coint_result.lr1)
            interpretation += (
                f" Strongest relationship has trace statistic: {max_trace_stat:.2f}."
            )

        return interpretation

    def _generate_cointegration_recommendations(
        self, n_coint: int, n_series: int, coint_result, X: pd.DataFrame
    ) -> List[str]:
        """
        Generate recommendations based on cointegration analysis results.

        Parameters:
        -----------
        n_coint : int
            Number of cointegrating relationships detected
        n_series : int
            Total number of time series
        coint_result
            Johansen cointegration test result
        X : pd.DataFrame
            Original data

        Returns:
        --------
        recommendations : List[str]
            List of recommendations
        """
        recommendations = []

        if n_coint == 0:
            recommendations.extend(
                [
                    "No cointegration detected - consider VAR model in first differences",
                    "Check for structural breaks that might mask cointegrating relationships",
                    "Verify that all series are integrated of the same order (I(1))",
                ]
            )
        elif n_coint > 0:
            recommendations.extend(
                [
                    f"Use Vector Error Correction Model (VECM) with {n_coint} cointegrating relationships",
                    "Analyze error correction coefficients for adjustment speed to equilibrium",
                    "Consider economic interpretation of cointegrating vectors",
                ]
            )

            if n_coint == n_series - 1:
                recommendations.append(
                    "System appears stationary - consider VAR model in levels"
                )

        # Check eigenvalues for stability
        if hasattr(coint_result, "eig") and len(coint_result.eig) > 0:
            largest_eigenvalue = max(coint_result.eig)
            if largest_eigenvalue > 0.9:
                recommendations.append(
                    "Large eigenvalue detected - verify system stability"
                )

        # Data quality recommendations
        if len(X) < 100:
            recommendations.append(
                "Limited sample size - cointegration tests have low power with small samples"
            )

        # Check for sufficient variation
        try:
            coeff_vars = X.std() / X.mean()
            if any(cv < 0.1 for cv in coeff_vars.abs()):
                recommendations.append(
                    "Low variation in some series - may affect cointegration test power"
                )
        except Exception:
            pass

        if not recommendations:
            recommendations.append(
                "Cointegration analysis appears robust - proceed with VECM modeling"
            )

        return recommendations
