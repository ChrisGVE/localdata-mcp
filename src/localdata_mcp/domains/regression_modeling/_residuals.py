"""
Regression & Modeling Domain - Residual Analysis Transformer.

Performs detailed residual diagnostics including normality tests,
homoscedasticity tests, outlier detection, and influence measures.
"""

import time

import numpy as np
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, check_X_y
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan, het_white
from statsmodels.stats.outliers_influence import OLSInfluence

from ._base import ResidualAnalysisResult, logger


class ResidualAnalysisTransformer(BaseEstimator, TransformerMixin):
    """
    sklearn-compatible transformer for comprehensive residual analysis.

    Performs detailed residual diagnostics including normality tests,
    homoscedasticity tests, outlier detection, and influence measures.

    Parameters:
    -----------
    alpha : float, default=0.05
        Significance level for statistical tests
    outlier_threshold : float, default=2.5
        Threshold for outlier detection (in standard deviations)
    include_influence : bool, default=True
        Whether to compute influence measures (leverage, Cook's distance)
    """

    def __init__(self, alpha=0.05, outlier_threshold=2.5, include_influence=True):
        self.alpha = alpha
        self.outlier_threshold = outlier_threshold
        self.include_influence = include_influence
        self.result_ = None

    def fit(self, X, y, residuals=None, fitted_values=None):
        """
        Perform comprehensive residual analysis.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Original input data
        y : array-like, shape (n_samples,)
            Original target values
        residuals : array-like, shape (n_samples,), optional
            Model residuals. If None, computed from y - fitted_values
        fitted_values : array-like, shape (n_samples,), optional
            Model predictions. If None, y is used as fitted values

        Returns:
        --------
        self : ResidualAnalysisTransformer
            Returns self for method chaining
        """
        start_time = time.time()
        logger.info("Starting residual analysis")

        # Validate inputs
        X, y = check_X_y(X, y, accept_sparse=False, y_numeric=True)

        # Handle residuals and fitted values
        if fitted_values is None:
            fitted_values = y  # Fallback if not provided
        if residuals is None:
            residuals = y - fitted_values

        try:
            # Standardized residuals
            residual_std = np.std(residuals)
            standardized_residuals = (
                residuals / residual_std if residual_std > 0 else residuals
            )

            # Basic residual statistics
            residual_stats = {
                "mean": np.mean(residuals),
                "std": residual_std,
                "min": np.min(residuals),
                "max": np.max(residuals),
                "skewness": stats.skew(residuals),
                "kurtosis": stats.kurtosis(residuals),
            }

            # 1. Normality tests
            normality_test = {}

            # Shapiro-Wilk test (for n < 5000)
            if len(residuals) < 5000:
                sw_stat, sw_p = stats.shapiro(residuals)
                normality_test["shapiro_wilk"] = {
                    "statistic": sw_stat,
                    "p_value": sw_p,
                    "is_normal": sw_p > self.alpha,
                }

            # Anderson-Darling test
            try:
                ad_stat, ad_critical, ad_p = stats.anderson(residuals, dist="norm")
                normality_test["anderson_darling"] = {
                    "statistic": ad_stat,
                    "critical_values": ad_critical.tolist(),
                    "significance_levels": [15, 10, 5, 2.5, 1],
                    "is_normal": ad_stat < ad_critical[2],  # 5% level
                }
            except:
                pass

            # Jarque-Bera test
            try:
                jb_stat, jb_p = stats.jarque_bera(residuals)
                normality_test["jarque_bera"] = {
                    "statistic": jb_stat,
                    "p_value": jb_p,
                    "is_normal": jb_p > self.alpha,
                }
            except:
                pass

            # 2. Homoscedasticity tests
            homoscedasticity_test = {}

            # Breusch-Pagan test (requires statsmodels)
            try:
                X_sm = sm.add_constant(X)
                bp_stat, bp_p, bp_f, bp_fp = het_breuschpagan(residuals, X_sm)
                homoscedasticity_test["breusch_pagan"] = {
                    "lm_statistic": bp_stat,
                    "p_value": bp_p,
                    "f_statistic": bp_f,
                    "f_p_value": bp_fp,
                    "is_homoscedastic": bp_p > self.alpha,
                }
            except Exception as e:
                logger.debug(f"Breusch-Pagan test failed: {e}")

            # White test
            try:
                X_sm = sm.add_constant(X)
                w_stat, w_p, w_f, w_fp = het_white(residuals, X_sm)
                homoscedasticity_test["white"] = {
                    "lm_statistic": w_stat,
                    "p_value": w_p,
                    "f_statistic": w_f,
                    "f_p_value": w_fp,
                    "is_homoscedastic": w_p > self.alpha,
                }
            except Exception as e:
                logger.debug(f"White test failed: {e}")

            # 3. Autocorrelation test (Durbin-Watson)
            autocorrelation_test = {}
            try:
                from statsmodels.stats.diagnostic import durbin_watson

                dw_stat = durbin_watson(residuals)
                autocorrelation_test["durbin_watson"] = {
                    "statistic": dw_stat,
                    "interpretation": self._interpret_durbin_watson(dw_stat),
                    "no_autocorrelation": 1.5 <= dw_stat <= 2.5,
                }
            except Exception as e:
                logger.debug(f"Durbin-Watson test failed: {e}")

            # 4. Outlier detection
            outliers = []
            outlier_threshold_abs = self.outlier_threshold

            # Using standardized residuals
            outlier_mask = np.abs(standardized_residuals) > outlier_threshold_abs
            outliers = np.where(outlier_mask)[0].tolist()

            # 5. Influence measures (if requested and data allows)
            leverage = None
            cooks_distance = None
            studentized_residuals = None

            if self.include_influence:
                try:
                    X_sm = sm.add_constant(X)
                    ols_model = sm.OLS(y, X_sm).fit()
                    influence = OLSInfluence(ols_model)

                    leverage = influence.hat_matrix_diag
                    cooks_distance = influence.cooks_distance[0]
                    studentized_residuals = influence.resid_studentized_external

                except Exception as e:
                    logger.debug(f"Influence measures computation failed: {e}")

            # Create result object
            self.result_ = ResidualAnalysisResult(
                residuals=residuals,
                standardized_residuals=standardized_residuals,
                studentized_residuals=studentized_residuals,
                fitted_values=fitted_values,
                normality_test=normality_test,
                homoscedasticity_test=homoscedasticity_test,
                autocorrelation_test=autocorrelation_test,
                outliers=outliers,
                leverage=leverage,
                cooks_distance=cooks_distance,
                residual_stats=residual_stats,
            )

            fit_time = time.time() - start_time
            logger.info(
                f"Residual analysis completed in {fit_time:.3f}s - {len(outliers)} outliers detected"
            )

        except Exception as e:
            logger.error(f"Error in residual analysis: {e}")
            raise

        return self

    def transform(self, X):
        """Transform method returns standardized residuals for pipeline compatibility."""
        check_is_fitted(self)
        return self.result_.standardized_residuals.reshape(-1, 1)

    def _interpret_durbin_watson(self, dw_stat):
        """Interpret Durbin-Watson statistic."""
        if dw_stat < 1.5:
            return "Positive autocorrelation likely"
        elif dw_stat > 2.5:
            return "Negative autocorrelation likely"
        else:
            return "No strong evidence of autocorrelation"

    def get_result(self):
        """Get the comprehensive residual analysis result."""
        check_is_fitted(self)
        return self.result_
