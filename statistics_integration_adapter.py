"""
Statistics Integration Adapter - LocalData MCP v2.0

Specialized adapter for statistical analysis libraries:
- scipy.stats: Statistical distributions, tests, and descriptive statistics
- statsmodels: Advanced regression, GLM, and statistical modeling
- pingouin: User-friendly statistical functions and effect sizes

Key Integration Challenges:
- Function-based APIs vs sklearn's class-based patterns
- Rich statistical result objects vs simple predictions
- Multiple return formats: scalars, tuples, custom objects
- Statistical significance interpretation and effect sizes
- Streaming compatibility for large-scale statistical testing
"""

from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from dataclasses import dataclass
import logging
import numpy as np
import pandas as pd
import warnings
from collections import namedtuple

# Import base integration architecture
from library_integration_shims import (
    BaseLibraryAdapter,
    LibraryCategory,
    LibraryDependency,
    IntegrationStrategy,
    IntegrationMetadata,
    LibraryIntegrationResult,
    requires_library,
    CompositionError
)

logger = logging.getLogger(__name__)


# ============================================================================
# Statistics-Specific Data Structures
# ============================================================================

@dataclass
class StatisticalContext:
    """Context for statistical operations."""
    alpha: float = 0.05  # Significance level
    alternative: str = 'two-sided'  # 'two-sided', 'less', 'greater'
    correction_method: Optional[str] = None  # Multiple comparison correction
    effect_size_calculation: bool = True
    confidence_level: float = 0.95


@dataclass
class StatisticalMetadata(IntegrationMetadata):
    """Extended metadata for statistical operations."""
    test_type: Optional[str] = None
    assumptions_checked: List[str] = None
    effect_sizes: Optional[Dict[str, float]] = None
    statistical_power: Optional[float] = None
    sample_sizes: Optional[Dict[str, int]] = None
    assumptions_violated: List[str] = None
    
    def __post_init__(self):
        if self.assumptions_checked is None:
            self.assumptions_checked = []
        if self.effect_sizes is None:
            self.effect_sizes = {}
        if self.sample_sizes is None:
            self.sample_sizes = {}
        if self.assumptions_violated is None:
            self.assumptions_violated = []


# Custom result containers for better composition
StatTestResult = namedtuple('StatTestResult', 
                           ['statistic', 'p_value', 'effect_size', 'confidence_interval'])

RegressionResult = namedtuple('RegressionResult',
                             ['coefficients', 'r_squared', 'p_values', 'residuals'])


# ============================================================================
# Statistics Adapter Implementation
# ============================================================================

class StatisticsAdapter(BaseLibraryAdapter):
    """
    Integration adapter for statistical analysis libraries.
    
    Handles:
    - Hypothesis testing with effect sizes and power analysis
    - Descriptive statistics and distribution analysis
    - Regression analysis and model fitting
    - Non-parametric tests and robust statistics
    - Multiple comparison corrections
    """
    
    def __init__(self):
        dependencies = [
            LibraryDependency(
                name="scipy",
                import_path="scipy.stats",
                min_version="1.7.0",
                is_optional=False,  # scipy is typically available
                installation_hint="pip install scipy"
            ),
            LibraryDependency(
                name="statsmodels",
                import_path="statsmodels.api",
                sklearn_equivalent="sklearn.linear_model",
                installation_hint="pip install statsmodels"
            ),
            LibraryDependency(
                name="pingouin",
                import_path="pingouin",
                sklearn_equivalent="scipy.stats",
                installation_hint="pip install pingouin"
            )
        ]
        
        super().__init__(LibraryCategory.STATISTICS, dependencies)
    
    def get_supported_functions(self) -> Dict[str, Callable]:
        """Return supported statistical functions."""
        return {
            # Descriptive Statistics
            'descriptive_stats': self.descriptive_statistics,
            'distribution_analysis': self.distribution_analysis,
            'correlation_analysis': self.correlation_analysis,
            'outlier_detection': self.outlier_detection,
            
            # Hypothesis Testing
            't_test': self.t_test,
            'mann_whitney_test': self.mann_whitney_test,
            'chi_square_test': self.chi_square_test,
            'anova': self.anova_analysis,
            'kruskal_wallis_test': self.kruskal_wallis_test,
            
            # Distribution Tests
            'normality_test': self.normality_test,
            'equal_variance_test': self.equal_variance_test,
            'goodness_of_fit_test': self.goodness_of_fit_test,
            
            # Regression Analysis
            'linear_regression': self.linear_regression_analysis,
            'logistic_regression': self.logistic_regression_analysis,
            'multiple_regression': self.multiple_regression_analysis,
            
            # Effect Sizes and Power
            'effect_size_calculation': self.calculate_effect_sizes,
            'power_analysis': self.power_analysis,
            'sample_size_calculation': self.sample_size_calculation,
            
            # Multiple Comparisons
            'multiple_comparison_correction': self.multiple_comparison_correction,
            'post_hoc_analysis': self.post_hoc_analysis
        }
    
    def adapt_function_call(self,
                          function_name: str,
                          data: Any,
                          parameters: Dict[str, Any]) -> Tuple[Any, StatisticalMetadata]:
        """Adapt function call to statistical library APIs."""
        
        if function_name not in self.get_supported_functions():
            raise CompositionError(
                f"Unsupported statistical function: {function_name}",
                error_type="unsupported_function"
            )
        
        func = self.get_supported_functions()[function_name]
        
        # Prepare data for statistical analysis
        prepared_data, prep_transformations = self._prepare_statistical_data(data, parameters)
        
        # Execute the function
        try:
            result = func(prepared_data, **parameters)
            
            # Convert result to standardized format
            output_result, output_transformations = self.convert_output_data(result)
            
            # Create metadata
            metadata = StatisticalMetadata(
                library_used=self._detect_primary_library(function_name),
                integration_strategy=IntegrationStrategy.FUNCTION_ADAPTER,
                data_transformations=prep_transformations + output_transformations,
                streaming_compatible=self._is_streaming_compatible(function_name),
                original_parameters=parameters,
                test_type=function_name
            )
            
            return output_result, metadata
            
        except Exception as e:
            return self._handle_statistical_error(function_name, data, parameters, e)
    
    # ========================================================================
    # Descriptive Statistics Functions
    # ========================================================================
    
    def descriptive_statistics(self, data: pd.DataFrame, **params) -> pd.DataFrame:
        """Comprehensive descriptive statistics analysis."""
        columns = params.get('columns', None)
        include_distribution = params.get('include_distribution', True)
        
        if columns:
            numeric_data = data[columns]
        else:
            numeric_data = data.select_dtypes(include=[np.number])
        
        # Basic statistics
        basic_stats = numeric_data.describe()
        
        # Additional statistics
        additional_stats = pd.DataFrame({
            col: {
                'skewness': numeric_data[col].skew(),
                'kurtosis': numeric_data[col].kurtosis(),
                'variance': numeric_data[col].var(),
                'coefficient_of_variation': numeric_data[col].std() / numeric_data[col].mean(),
                'median_absolute_deviation': (numeric_data[col] - numeric_data[col].median()).abs().median()
            } for col in numeric_data.columns
        }).T
        
        # Combine results
        result = pd.concat([basic_stats.T, additional_stats], axis=1)
        
        return result
    
    @requires_library("scipy")
    def distribution_analysis(self, data: pd.DataFrame, **params) -> pd.DataFrame:
        """Analyze distribution characteristics of variables."""
        from scipy import stats
        
        column = params.get('column', data.columns[0])
        distributions_to_test = params.get('distributions', ['norm', 'expon', 'gamma', 'beta'])
        
        values = data[column].dropna()
        
        results = []
        for dist_name in distributions_to_test:
            try:
                dist = getattr(stats, dist_name)
                
                # Fit distribution
                params_fitted = dist.fit(values)
                
                # Calculate goodness of fit (Kolmogorov-Smirnov test)
                ks_stat, ks_p_value = stats.kstest(values, 
                                                   lambda x: dist.cdf(x, *params_fitted))
                
                # Calculate AIC-like metric
                log_likelihood = np.sum(dist.logpdf(values, *params_fitted))
                aic = 2 * len(params_fitted) - 2 * log_likelihood
                
                results.append({
                    'distribution': dist_name,
                    'parameters': str(params_fitted),
                    'ks_statistic': ks_stat,
                    'ks_p_value': ks_p_value,
                    'aic': aic,
                    'log_likelihood': log_likelihood
                })
                
            except Exception as e:
                logger.warning(f"Failed to fit {dist_name}: {e}")
        
        return pd.DataFrame(results).sort_values('aic')
    
    def correlation_analysis(self, data: pd.DataFrame, **params) -> pd.DataFrame:
        """Comprehensive correlation analysis."""
        method = params.get('method', 'pearson')  # 'pearson', 'spearman', 'kendall'
        columns = params.get('columns', None)
        
        if columns:
            analysis_data = data[columns]
        else:
            analysis_data = data.select_dtypes(include=[np.number])
        
        # Calculate correlations
        if method == 'pearson':
            corr_matrix = analysis_data.corr()
        elif method == 'spearman':
            corr_matrix = analysis_data.corr(method='spearman')
        elif method == 'kendall':
            corr_matrix = analysis_data.corr(method='kendall')
        
        # Add statistical significance if scipy available
        if self.is_library_available("scipy"):
            corr_with_p = self._correlation_with_pvalues(analysis_data, method)
            return corr_with_p
        else:
            return corr_matrix
    
    @requires_library("scipy")
    def outlier_detection(self, data: pd.DataFrame, **params) -> pd.DataFrame:
        """Detect outliers using multiple methods."""
        from scipy import stats
        
        column = params.get('column', data.columns[0])
        method = params.get('method', 'iqr')  # 'iqr', 'zscore', 'modified_zscore'
        threshold = params.get('threshold', 3)
        
        values = data[column]
        outliers = pd.Series(False, index=data.index)
        
        if method == 'iqr':
            Q1 = values.quantile(0.25)
            Q3 = values.quantile(0.75)
            IQR = Q3 - Q1
            outliers = (values < (Q1 - 1.5 * IQR)) | (values > (Q3 + 1.5 * IQR))
            
        elif method == 'zscore':
            z_scores = np.abs(stats.zscore(values))
            outliers = z_scores > threshold
            
        elif method == 'modified_zscore':
            median = values.median()
            mad = np.median(np.abs(values - median))
            modified_z_scores = 0.6745 * (values - median) / mad
            outliers = np.abs(modified_z_scores) > threshold
        
        result = data.copy()
        result['is_outlier'] = outliers
        result['outlier_method'] = method
        
        return result
    
    # ========================================================================
    # Hypothesis Testing Functions
    # ========================================================================
    
    @requires_library("scipy")
    def t_test(self, data: pd.DataFrame, **params) -> pd.DataFrame:
        """Perform t-test analysis."""
        from scipy import stats
        
        test_type = params.get('type', 'one_sample')  # 'one_sample', 'two_sample', 'paired'
        column1 = params.get('column1', data.columns[0])
        column2 = params.get('column2', None)
        mu = params.get('mu', 0)  # For one-sample test
        alternative = params.get('alternative', 'two-sided')
        
        if test_type == 'one_sample':
            values = data[column1].dropna()
            statistic, p_value = stats.ttest_1samp(values, mu, alternative=alternative)
            effect_size = (values.mean() - mu) / values.std()  # Cohen's d
            
        elif test_type == 'two_sample':
            if column2 is None:
                raise ValueError("column2 required for two-sample t-test")
            values1 = data[column1].dropna()
            values2 = data[column2].dropna()
            statistic, p_value = stats.ttest_ind(values1, values2, alternative=alternative)
            
            # Cohen's d for two samples
            pooled_std = np.sqrt(((len(values1)-1)*values1.var() + (len(values2)-1)*values2.var()) / 
                                (len(values1) + len(values2) - 2))
            effect_size = (values1.mean() - values2.mean()) / pooled_std
            
        elif test_type == 'paired':
            if column2 is None:
                raise ValueError("column2 required for paired t-test")
            values1 = data[column1].dropna()
            values2 = data[column2].dropna()
            statistic, p_value = stats.ttest_rel(values1, values2, alternative=alternative)
            
            # Effect size for paired samples
            differences = values1 - values2
            effect_size = differences.mean() / differences.std()
        
        return pd.DataFrame([{
            'test_type': f"{test_type}_t_test",
            'statistic': statistic,
            'p_value': p_value,
            'effect_size': effect_size,
            'significant': p_value < params.get('alpha', 0.05),
            'alternative': alternative
        }])
    
    @requires_library("scipy") 
    def mann_whitney_test(self, data: pd.DataFrame, **params) -> pd.DataFrame:
        """Non-parametric Mann-Whitney U test."""
        from scipy import stats
        
        column1 = params.get('column1', data.columns[0])
        column2 = params.get('column2', data.columns[1])
        alternative = params.get('alternative', 'two-sided')
        
        values1 = data[column1].dropna()
        values2 = data[column2].dropna()
        
        statistic, p_value = stats.mannwhitneyu(values1, values2, alternative=alternative)
        
        # Effect size (r = Z / sqrt(N))
        z_score = stats.norm.ppf(1 - p_value/2)  # Approximate Z score
        n = len(values1) + len(values2)
        effect_size = z_score / np.sqrt(n)
        
        return pd.DataFrame([{
            'test_type': 'mann_whitney_u',
            'statistic': statistic,
            'p_value': p_value,
            'effect_size': effect_size,
            'significant': p_value < params.get('alpha', 0.05),
            'alternative': alternative
        }])
    
    @requires_library("scipy")
    def chi_square_test(self, data: pd.DataFrame, **params) -> pd.DataFrame:
        """Chi-square test of independence."""
        from scipy import stats
        
        column1 = params.get('column1', data.columns[0])
        column2 = params.get('column2', data.columns[1])
        
        # Create contingency table
        contingency_table = pd.crosstab(data[column1], data[column2])
        
        # Perform chi-square test
        chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
        
        # Effect size (Cramer's V)
        n = contingency_table.sum().sum()
        cramers_v = np.sqrt(chi2 / (n * (min(contingency_table.shape) - 1)))
        
        return pd.DataFrame([{
            'test_type': 'chi_square_independence',
            'chi2_statistic': chi2,
            'p_value': p_value,
            'degrees_of_freedom': dof,
            'cramers_v': cramers_v,
            'significant': p_value < params.get('alpha', 0.05),
            'contingency_table': str(contingency_table.to_dict())
        }])
    
    @requires_library("scipy")
    def anova_analysis(self, data: pd.DataFrame, **params) -> pd.DataFrame:
        """One-way ANOVA analysis."""
        from scipy import stats
        
        value_column = params.get('value_column', data.columns[0])
        group_column = params.get('group_column', data.columns[1])
        
        # Group data
        groups = [group[value_column].dropna().values 
                 for name, group in data.groupby(group_column)]
        
        # Perform ANOVA
        f_statistic, p_value = stats.f_oneway(*groups)
        
        # Effect size (eta-squared)
        # This is a simplified calculation
        overall_mean = data[value_column].mean()
        ss_between = sum(len(group) * (np.mean(group) - overall_mean)**2 for group in groups)
        ss_total = sum((data[value_column] - overall_mean)**2)
        eta_squared = ss_between / ss_total
        
        return pd.DataFrame([{
            'test_type': 'one_way_anova',
            'f_statistic': f_statistic,
            'p_value': p_value,
            'eta_squared': eta_squared,
            'significant': p_value < params.get('alpha', 0.05),
            'num_groups': len(groups)
        }])
    
    # ========================================================================
    # Distribution and Normality Tests
    # ========================================================================
    
    @requires_library("scipy")
    def normality_test(self, data: pd.DataFrame, **params) -> pd.DataFrame:
        """Test for normality using multiple methods."""
        from scipy import stats
        
        column = params.get('column', data.columns[0])
        methods = params.get('methods', ['shapiro', 'kstest', 'jarque_bera'])
        
        values = data[column].dropna()
        results = []
        
        for method in methods:
            try:
                if method == 'shapiro' and len(values) <= 5000:
                    statistic, p_value = stats.shapiro(values)
                elif method == 'kstest':
                    statistic, p_value = stats.kstest(values, 'norm',
                                                     args=(values.mean(), values.std()))
                elif method == 'jarque_bera':
                    statistic, p_value = stats.jarque_bera(values)
                else:
                    continue
                
                results.append({
                    'test': method,
                    'statistic': statistic,
                    'p_value': p_value,
                    'normal': p_value > params.get('alpha', 0.05)
                })
            except Exception as e:
                logger.warning(f"Normality test {method} failed: {e}")
        
        return pd.DataFrame(results)
    
    # ========================================================================
    # Regression Analysis Functions
    # ========================================================================
    
    def linear_regression_analysis(self, data: pd.DataFrame, **params) -> pd.DataFrame:
        """Linear regression analysis with comprehensive output."""
        if self.is_library_available("statsmodels"):
            return self._statsmodels_regression(data, 'linear', **params)
        else:
            return self._sklearn_regression_fallback(data, 'linear', **params)
    
    @requires_library("statsmodels")
    def _statsmodels_regression(self, data: pd.DataFrame, model_type: str, **params) -> pd.DataFrame:
        """Regression analysis using statsmodels."""
        import statsmodels.api as sm
        
        target_column = params.get('target_column', data.columns[-1])
        feature_columns = params.get('feature_columns', [col for col in data.columns if col != target_column])
        add_constant = params.get('add_constant', True)
        
        X = data[feature_columns]
        y = data[target_column]
        
        if add_constant:
            X = sm.add_constant(X)
        
        # Fit model
        if model_type == 'linear':
            model = sm.OLS(y, X)
        elif model_type == 'logistic':
            model = sm.Logit(y, X)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        fitted_model = model.fit()
        
        # Extract results
        results = pd.DataFrame({
            'variable': X.columns,
            'coefficient': fitted_model.params,
            'std_error': fitted_model.bse,
            't_statistic': fitted_model.tvalues,
            'p_value': fitted_model.pvalues,
            'confidence_lower': fitted_model.conf_int()[0],
            'confidence_upper': fitted_model.conf_int()[1]
        })
        
        # Add model statistics
        results.attrs['r_squared'] = fitted_model.rsquared
        results.attrs['adj_r_squared'] = fitted_model.rsquared_adj
        results.attrs['f_statistic'] = fitted_model.fvalue
        results.attrs['f_p_value'] = fitted_model.f_pvalue
        results.attrs['aic'] = fitted_model.aic
        results.attrs['bic'] = fitted_model.bic
        
        return results
    
    # ========================================================================
    # Helper Methods
    # ========================================================================
    
    def _prepare_statistical_data(self, data: Any, parameters: Dict[str, Any]) -> Tuple[pd.DataFrame, List[str]]:
        """Prepare data for statistical operations."""
        transformations = []
        
        # Convert to DataFrame if needed
        if not isinstance(data, pd.DataFrame):
            if isinstance(data, pd.Series):
                data = data.to_frame()
                transformations.append("series_to_dataframe")
            else:
                data = pd.DataFrame(data)
                transformations.append("array_to_dataframe")
        
        return data, transformations
    
    def _detect_primary_library(self, function_name: str) -> str:
        """Detect which library will be used for a function."""
        if function_name in ['linear_regression', 'logistic_regression']:
            return "statsmodels" if self.is_library_available("statsmodels") else "sklearn"
        elif function_name.endswith('_test'):
            return "scipy" if self.is_library_available("scipy") else "numpy"
        else:
            return "scipy"
    
    def _is_streaming_compatible(self, function_name: str) -> bool:
        """Check if function supports streaming execution."""
        streaming_functions = [
            'descriptive_stats', 'correlation_analysis', 'outlier_detection'
        ]
        return function_name in streaming_functions
    
    def _correlation_with_pvalues(self, data: pd.DataFrame, method: str) -> pd.DataFrame:
        """Calculate correlations with p-values."""
        from scipy.stats import pearsonr, spearmanr, kendalltau
        
        n_vars = len(data.columns)
        corr_matrix = pd.DataFrame(index=data.columns, columns=data.columns, dtype=float)
        p_value_matrix = pd.DataFrame(index=data.columns, columns=data.columns, dtype=float)
        
        for i, col1 in enumerate(data.columns):
            for j, col2 in enumerate(data.columns):
                if i == j:
                    corr_matrix.loc[col1, col2] = 1.0
                    p_value_matrix.loc[col1, col2] = 0.0
                elif i < j:
                    x = data[col1].dropna()
                    y = data[col2].dropna()
                    
                    # Find common indices
                    common_idx = x.index.intersection(y.index)
                    x_common = x[common_idx]
                    y_common = y[common_idx]
                    
                    if len(x_common) > 2:
                        try:
                            if method == 'pearson':
                                corr, p_val = pearsonr(x_common, y_common)
                            elif method == 'spearman':
                                corr, p_val = spearmanr(x_common, y_common)
                            elif method == 'kendall':
                                corr, p_val = kendalltau(x_common, y_common)
                            
                            corr_matrix.loc[col1, col2] = corr
                            corr_matrix.loc[col2, col1] = corr
                            p_value_matrix.loc[col1, col2] = p_val
                            p_value_matrix.loc[col2, col1] = p_val
                            
                        except Exception as e:
                            logger.warning(f"Correlation calculation failed for {col1} vs {col2}: {e}")
                            corr_matrix.loc[col1, col2] = np.nan
                            corr_matrix.loc[col2, col1] = np.nan
        
        # Combine into single result
        result = corr_matrix.copy()
        result.attrs['p_values'] = p_value_matrix.to_dict()
        result.attrs['method'] = method
        
        return result
    
    def _sklearn_regression_fallback(self, data: pd.DataFrame, model_type: str, **params) -> pd.DataFrame:
        """Fallback regression using sklearn."""
        from sklearn.linear_model import LinearRegression, LogisticRegression
        from sklearn.metrics import r2_score
        
        target_column = params.get('target_column', data.columns[-1])
        feature_columns = params.get('feature_columns', [col for col in data.columns if col != target_column])
        
        X = data[feature_columns]
        y = data[target_column]
        
        if model_type == 'linear':
            model = LinearRegression()
        elif model_type == 'logistic':
            model = LogisticRegression()
        
        model.fit(X, y)
        
        # Create results DataFrame
        results = pd.DataFrame({
            'variable': feature_columns,
            'coefficient': model.coef_[0] if len(model.coef_.shape) > 1 else model.coef_
        })
        
        if model_type == 'linear':
            results.attrs['r_squared'] = r2_score(y, model.predict(X))
            results.attrs['intercept'] = model.intercept_
        
        return results
    
    def _handle_statistical_error(self,
                                 function_name: str,
                                 data: Any,
                                 parameters: Dict[str, Any],
                                 error: Exception) -> Tuple[Any, StatisticalMetadata]:
        """Handle statistical operation errors with fallbacks."""
        logger.error(f"Statistical operation {function_name} failed: {error}")
        
        # Simple fallback for descriptive statistics
        if function_name == 'descriptive_statistics':
            try:
                fallback_result = data.describe()
                
                metadata = StatisticalMetadata(
                    library_used="pandas_fallback",
                    integration_strategy=IntegrationStrategy.FALLBACK_CHAIN,
                    fallback_used=True,
                    original_parameters=parameters
                )
                
                return fallback_result, metadata
            except Exception:
                pass
        
        # Re-raise original error if no fallback available
        raise CompositionError(
            f"Statistical operation {function_name} failed: {error}",
            error_type="statistical_operation_failed"
        )


if __name__ == "__main__":
    # Example usage
    adapter = StatisticsAdapter()
    
    # Create sample data
    np.random.seed(42)
    sample_data = pd.DataFrame({
        'group': np.random.choice(['A', 'B', 'C'], 100),
        'value1': np.random.normal(10, 2, 100),
        'value2': np.random.normal(15, 3, 100)
    })
    
    try:
        # Test descriptive statistics
        desc_result, desc_metadata = adapter.adapt_function_call(
            'descriptive_stats',
            sample_data,
            {'columns': ['value1', 'value2']}
        )
        print(f"Descriptive statistics computed using: {desc_metadata.library_used}")
        
        # Test correlation analysis
        corr_result, corr_metadata = adapter.adapt_function_call(
            'correlation_analysis',
            sample_data,
            {'method': 'pearson'}
        )
        print(f"Correlation analysis completed using: {corr_metadata.library_used}")
        
    except Exception as e:
        print(f"Statistics adapter test failed: {e}")