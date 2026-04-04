"""Time Series Analysis - Stationarity transformers."""

import time
from typing import List, Optional

import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss

from ...logging_manager import get_logger
from ._base import TimeSeriesAnalysisResult
from ._transformer import TimeSeriesTransformer

logger = get_logger(__name__)


def _pp_test(series, lags="auto", regression="c"):
    """Phillips-Perron test shim — raises because ``arch`` is not installed."""
    raise NotImplementedError(
        "Phillips-Perron test requires the 'arch' package which is not installed"
    )


class StationarityTestTransformer(TimeSeriesTransformer):
    """Stationarity testing via ADF, KPSS, and Phillips-Perron tests."""

    def __init__(
        self,
        tests=["adf", "kpss", "pp"],
        alpha=0.05,
        auto_differencing=False,
        max_diff_order=2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.tests = tests
        self.alpha = alpha
        self.auto_differencing = auto_differencing
        self.max_diff_order = max_diff_order

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Fit the stationarity test transformer."""
        if self.validate_input:
            X, y = self._validate_time_series(X, y)
        self.is_fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> TimeSeriesAnalysisResult:
        """Perform stationarity tests on time series data."""
        start_time = time.time()
        if self.validate_input:
            X, _ = self._validate_time_series(X, None)
        try:
            if len(X.columns) == 0:
                raise ValueError("No columns found in time series data")
            series = X.iloc[:, 0].dropna()
            if len(series) < 10:
                raise ValueError("Insufficient data points for stationarity testing")

            test_results = {}
            overall_stationary = True
            recommendations = []
            warnings_list = []

            overall_stationary = self._run_stationarity_tests(
                series, test_results, warnings_list
            )

            if not overall_stationary:
                recommendations.append("Time series appears to be non-stationary")
                recommendations.append(
                    "Consider applying differencing or transformation"
                )
                if self.auto_differencing:
                    recommendations.extend(self._suggest_differencing(series))
            else:
                recommendations.append("Time series appears to be stationary")
                recommendations.append(
                    "Suitable for ARMA modeling without differencing"
                )

            stationary_tests = sum(
                result.get("is_stationary", False) for result in test_results.values()
            )
            total_tests = len(test_results)
            if total_tests == 0:
                interpretation = "No stationarity tests could be performed"
            elif stationary_tests == total_tests:
                interpretation = (
                    f"All {total_tests} stationarity tests indicate STATIONARY series"
                )
            elif stationary_tests == 0:
                interpretation = f"All {total_tests} stationarity tests indicate NON-STATIONARY series"
            else:
                interpretation = f"Mixed results: {stationary_tests}/{total_tests} tests indicate stationarity"

            return TimeSeriesAnalysisResult(
                analysis_type="stationarity_testing",
                interpretation=interpretation,
                model_diagnostics=test_results,
                recommendations=recommendations,
                warnings=warnings_list,
                processing_time=time.time() - start_time,
            )
        except Exception as e:
            logger.error(f"Error in stationarity testing: {e}")
            return TimeSeriesAnalysisResult(
                analysis_type="stationarity_testing_error",
                interpretation=f"Error during stationarity testing: {str(e)}",
                warnings=[str(e)],
                processing_time=time.time() - start_time,
            )

    def _suggest_differencing(self, series: pd.Series) -> List[str]:
        """Suggest appropriate differencing order for non-stationary series."""
        suggestions = []
        try:
            for diff_order in range(1, self.max_diff_order + 1):
                diff_series = series.diff(diff_order).dropna()
                if len(diff_series) < 10:
                    break
                try:
                    adf_result = adfuller(diff_series, autolag="AIC")
                    if adf_result[1] <= self.alpha:
                        suggestions.append(
                            f"First-order differencing (d={diff_order}) achieves stationarity"
                        )
                        break
                except Exception:
                    continue
            if not suggestions:
                suggestions.append(
                    "Consider seasonal differencing or transformation (log, sqrt)"
                )
        except Exception as e:
            suggestions.append(f"Could not test differencing: {e}")
        return suggestions

    def _run_stationarity_tests(self, series, test_results, warnings_list):
        """Execute requested stationarity tests, return overall_stationary flag."""
        overall = True
        _label = {"adf": "ADF", "kpss": "KPSS", "pp": "Phillips-Perron"}
        _cv = {("adf", 4), ("kpss", 3), ("pp", 4)}
        for name in ("adf", "kpss", "pp"):
            if name not in self.tests:
                continue
            try:
                if name == "adf":
                    r = adfuller(series, autolag="AIC")
                elif name == "kpss":
                    r = kpss(series, regression="c", nlags="auto")
                else:
                    r = _pp_test(series, lags="auto")
                is_stat = r[1] > self.alpha if name == "kpss" else r[1] <= self.alpha
                d = {"statistic": r[0], "p_value": r[1], "used_lag": r[2]}
                d["critical_values"] = r[4] if name != "kpss" else r[3]
                if name != "kpss":
                    d["n_observations"] = r[3]
                d["is_stationary"] = is_stat
                d["interpretation"] = "Stationary" if is_stat else "Non-stationary"
                test_results[name] = d
                if not is_stat:
                    overall = False
            except Exception as e:
                warnings_list.append(f"{_label[name]} test failed: {e}")
        return overall


class UnitRootTestTransformer(TimeSeriesTransformer):
    """Comprehensive unit root testing for integration order detection."""

    def __init__(
        self,
        test_type="adf",
        regression_type="c",
        max_lags=None,
        autolag="AIC",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.test_type = test_type
        self.regression_type = regression_type
        self.max_lags = max_lags
        self.autolag = autolag

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Fit the unit root test transformer."""
        if self.validate_input:
            X, y = self._validate_time_series(X, y)
        self.is_fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> TimeSeriesAnalysisResult:
        """Perform unit root tests on time series data."""
        start_time = time.time()
        if self.validate_input:
            X, _ = self._validate_time_series(X, None)
        try:
            if len(X.columns) == 0:
                raise ValueError("No columns found in time series data")
            series = X.iloc[:, 0].dropna()
            if len(series) < 10:
                raise ValueError("Insufficient data points for unit root testing")

            test_results = {}
            recommendations = []
            warnings_list = []

            test_kwargs = {"regression": self.regression_type}
            if self.max_lags is not None:
                test_kwargs["maxlag"] = self.max_lags
            if self.autolag is not None:
                test_kwargs["autolag"] = self.autolag

            has_unit_root, interpretation = self._run_test(
                series, test_kwargs, test_results, warnings_list
            )

            self._add_recommendations(has_unit_root, series, recommendations)

            return TimeSeriesAnalysisResult(
                analysis_type="unit_root_testing",
                interpretation=interpretation,
                model_diagnostics=test_results,
                recommendations=recommendations,
                warnings=warnings_list,
                processing_time=time.time() - start_time,
            )
        except Exception as e:
            logger.error(f"Error in unit root testing: {e}")
            return TimeSeriesAnalysisResult(
                analysis_type="unit_root_testing_error",
                interpretation=f"Error during unit root testing: {str(e)}",
                warnings=[str(e)],
                processing_time=time.time() - start_time,
            )

    def _run_test(self, series, test_kwargs, test_results, warnings_list):
        """Run the configured unit root test, return (has_unit_root, interpretation)."""
        _interp_root = "Series has unit root (non-stationary)"
        _interp_ok = "Series does not have unit root (stationary)"
        if self.test_type == "adf":
            try:
                r = adfuller(series, **test_kwargs)
                test_results["adf"] = {
                    "test_statistic": r[0],
                    "p_value": r[1],
                    "lags_used": r[2],
                    "n_observations": r[3],
                    "critical_values": r[4],
                    "ic_best": r[5] if len(r) > 5 else None,
                }
                has = r[1] > 0.05
                return has, _interp_root if has else _interp_ok
            except Exception as e:
                warnings_list.append(f"ADF test failed: {e}")
                return False, "Unit root test failed"
        elif self.test_type == "kpss":
            try:
                r = kpss(series, regression=self.regression_type, nlags=self.autolag)
                test_results["kpss"] = {
                    "test_statistic": r[0],
                    "p_value": r[1],
                    "lags_used": r[2],
                    "critical_values": r[3],
                }
                has = r[1] <= 0.05  # KPSS null is stationarity
                return has, _interp_root if has else _interp_ok
            except Exception as e:
                warnings_list.append(f"KPSS test failed: {e}")
                return False, "Unit root test failed"
        elif self.test_type == "pp":
            try:
                r = _pp_test(series, lags=self.autolag, regression=self.regression_type)
                test_results["pp"] = {
                    "test_statistic": r[0],
                    "p_value": r[1],
                    "lags_used": r[2],
                    "n_observations": r[3],
                    "critical_values": r[4],
                }
                has = r[1] > 0.05
                return has, _interp_root if has else _interp_ok
            except Exception as e:
                warnings_list.append(f"Phillips-Perron test failed: {e}")
                return False, "Unit root test failed"
        else:
            raise ValueError(f"Unsupported test type: {self.test_type}")

    @staticmethod
    def _add_recommendations(has_unit_root, series, recommendations):
        """Populate *recommendations* based on unit root result."""
        if has_unit_root:
            recommendations.extend(
                [
                    "Series contains unit root - requires differencing",
                    "Consider first differencing: y(t) - y(t-1)",
                    "Test differenced series for stationarity",
                ]
            )
            try:
                diff_series = series.diff().dropna()
                if len(diff_series) >= 10:
                    diff_result = adfuller(diff_series)
                    if diff_result[1] <= 0.05:
                        recommendations.append(
                            "First differencing should achieve stationarity (I(1) process)"
                        )
                    else:
                        recommendations.append(
                            "May require second differencing or seasonal differencing"
                            " (I(2) or seasonal process)"
                        )
            except Exception:
                pass
        else:
            recommendations.extend(
                [
                    "Series appears stationary - suitable for ARMA modeling",
                    "No differencing required",
                ]
            )
