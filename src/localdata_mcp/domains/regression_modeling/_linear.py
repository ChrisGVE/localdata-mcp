"""
Regression & Modeling Domain - Linear Regression Transformer.

Implements ordinary least squares regression with comprehensive
statistical analysis, diagnostics, and model evaluation.
"""

import time

import numpy as np
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan, het_white, normal_ad

from ._base import RegressionModelResult, logger


class LinearRegressionTransformer(BaseEstimator, TransformerMixin, RegressorMixin):
    """
    sklearn-compatible transformer for linear regression analysis.

    Implements ordinary least squares regression with comprehensive
    statistical analysis, diagnostics, and model evaluation.

    Parameters:
    -----------
    fit_intercept : bool, default=True
        Whether to calculate the intercept for this model
    normalize : bool, default=False
        Whether to normalize the features before regression
    include_diagnostics : bool, default=True
        Whether to include comprehensive model diagnostics
    alpha : float, default=0.05
        Significance level for statistical tests
    """

    def __init__(
        self, fit_intercept=True, normalize=False, include_diagnostics=True, alpha=0.05
    ):
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.include_diagnostics = include_diagnostics
        self.alpha = alpha

        # Initialize model
        self.model = LinearRegression(fit_intercept=fit_intercept)
        self.result_ = None

    def _compute_adjusted_r2(self, r2: float, n_samples: int, n_features: int):
        """Compute adjusted R-squared; returns None if denominator is zero."""
        denominator = n_samples - n_features - 1
        if denominator <= 0:
            return None
        return 1 - (1 - r2) * (n_samples - 1) / denominator

    def _run_statsmodels_diagnostics(self, X, y, residuals):
        """Fit OLS via statsmodels and return (aic, bic, log_likelihood, assumptions_met)."""
        if not self.include_diagnostics:
            return None, None, None, {}
        try:
            X_sm = sm.add_constant(X) if self.fit_intercept else X
            sm_model = sm.OLS(y, X_sm).fit()
            assumptions_met = self._check_assumptions(X, y, residuals, sm_model)
            return sm_model.aic, sm_model.bic, sm_model.llf, assumptions_met
        except Exception as e:
            logger.warning(f"Failed to compute statistical diagnostics: {e}")
            return None, None, None, {}

    def fit(self, X, y, sample_weight=None, feature_names=None):
        """
        Fit linear regression model.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,)
            Target values
        sample_weight : array-like, shape (n_samples,), optional
            Individual weights for each sample
        feature_names : list of str, optional
            Names of features

        Returns:
        --------
        self : LinearRegressionTransformer
            Returns self for method chaining
        """
        start_time = time.time()
        logger.info("Starting linear regression fit")

        X, y = check_X_y(X, y, accept_sparse=False, y_numeric=True)

        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        self.feature_names_ = feature_names

        try:
            self.model.fit(X, y, sample_weight=sample_weight)

            y_pred = self.model.predict(X)
            residuals = y - y_pred

            r2 = r2_score(y, y_pred)
            mse = mean_squared_error(y, y_pred)
            mae = mean_absolute_error(y, y_pred)
            rmse = np.sqrt(mse)
            n_samples, n_features = X.shape
            adjusted_r2 = self._compute_adjusted_r2(r2, n_samples, n_features)
            aic, bic, log_likelihood, assumptions_met = (
                self._run_statsmodels_diagnostics(X, y, residuals)
            )

            self.result_ = RegressionModelResult(
                model_type="Linear Regression",
                model_params={
                    "fit_intercept": self.fit_intercept,
                    "normalize": self.normalize,
                },
                fitted_model=self.model,
                r2_score=r2,
                adjusted_r2=adjusted_r2,
                mse=mse,
                mae=mae,
                rmse=rmse,
                aic=aic,
                bic=bic,
                log_likelihood=log_likelihood,
                residuals=residuals,
                fitted_values=y_pred,
                standardized_residuals=residuals / np.std(residuals),
                feature_names=feature_names,
                coefficients=self.model.coef_,
                n_features=n_features,
                n_samples=n_samples,
                assumptions_met=assumptions_met,
            )

            if self.fit_intercept:
                self.result_.coefficients = np.concatenate(
                    [[self.model.intercept_], self.model.coef_]
                )
                self.result_.feature_names = ["intercept"] + feature_names

            fit_time = time.time() - start_time
            logger.info(
                f"Linear regression fit completed in {fit_time:.3f}s - R² = {r2:.4f}"
            )

        except Exception as e:
            logger.error(f"Error fitting linear regression: {e}")
            raise

        return self

    def transform(self, X):
        """
        Transform method returns predictions for pipeline compatibility.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Input data

        Returns:
        --------
        y_pred : ndarray, shape (n_samples,)
            Predicted values
        """
        check_is_fitted(self)
        return self.predict(X)

    def predict(self, X):
        """
        Make predictions using the fitted model.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Input data

        Returns:
        --------
        y_pred : ndarray, shape (n_samples,)
            Predicted values
        """
        check_is_fitted(self)
        X = check_array(X, accept_sparse=False)
        return self.model.predict(X)

    def score(self, X, y, sample_weight=None):
        """
        Return the coefficient of determination R².

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Test samples
        y : array-like, shape (n_samples,)
            True values for X
        sample_weight : array-like, shape (n_samples,), optional
            Sample weights

        Returns:
        --------
        score : float
            R² of self.predict(X) wrt. y
        """
        return self.model.score(X, y, sample_weight=sample_weight)

    def _check_assumptions(self, X, y, residuals, sm_model):
        """Check linear regression assumptions."""
        assumptions = {}

        try:
            # 1. Normality of residuals (Anderson-Darling test)
            ad_stat, ad_p = normal_ad(residuals)
            assumptions["normality"] = ad_p > self.alpha

            # 2. Homoscedasticity (Breusch-Pagan test)
            try:
                bp_stat, bp_p, _, _ = het_breuschpagan(residuals, sm_model.model.exog)
                assumptions["homoscedasticity"] = bp_p > self.alpha
            except:
                # Fallback to White test if BP fails
                try:
                    w_stat, w_p, _, _ = het_white(residuals, sm_model.model.exog)
                    assumptions["homoscedasticity"] = w_p > self.alpha
                except:
                    assumptions["homoscedasticity"] = None

            # 3. Independence (Durbin-Watson test)
            try:
                from statsmodels.stats.diagnostic import durbin_watson

                dw_stat = durbin_watson(residuals)
                # Rule of thumb: values between 1.5-2.5 suggest no autocorrelation
                assumptions["independence"] = 1.5 <= dw_stat <= 2.5
            except:
                assumptions["independence"] = None

        except Exception as e:
            logger.warning(f"Failed to check some model assumptions: {e}")

        return assumptions

    def get_result(self):
        """Get the comprehensive regression result."""
        check_is_fitted(self)
        return self.result_
