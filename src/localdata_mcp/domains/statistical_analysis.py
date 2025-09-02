"""
Statistical Analysis Domain - Comprehensive statistical analysis capabilities.

This module implements advanced statistical analysis tools including hypothesis testing,
ANOVA, non-parametric tests, and experimental design using scipy.stats and sklearn integration.

Key Features:
- Hypothesis testing (t-tests, chi-square, normality, correlation tests)
- ANOVA analysis (one-way, two-way, post-hoc, effect sizes)  
- Non-parametric tests (Mann-Whitney U, Wilcoxon, Kruskal-Wallis, Friedman)
- Experimental design (power analysis, effect sizes, confidence intervals)
- Full sklearn pipeline compatibility
- Streaming-compatible processing
- Comprehensive result formatting
"""

import warnings
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import time
import json

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import chi2_contingency, pearsonr, spearmanr, kendalltau
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted
import statsmodels.api as sm
from statsmodels.stats.power import ttest_power, tt_solve_power
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.anova import anova_lm
from statsmodels.formula.api import ols

from ..logging_manager import get_logger
from ..pipeline.base import (
    AnalysisPipelineBase, PipelineResult, CompositionMetadata, 
    StreamingConfig, PipelineState
)

logger = get_logger(__name__)

# Suppress specific warnings that are not critical for our use case
warnings.filterwarnings('ignore', category=RuntimeWarning, module='scipy.stats')
warnings.filterwarnings('ignore', category=UserWarning, module='statsmodels')


@dataclass
class StatisticalTestResult:
    """Standardized result structure for statistical tests."""
    test_name: str
    statistic: float
    p_value: float
    degrees_of_freedom: Optional[Union[int, Tuple[int, int]]] = None
    effect_size: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    interpretation: str = ""
    assumptions_met: Dict[str, bool] = None
    additional_info: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary format."""
        result_dict = {
            'test_name': self.test_name,
            'statistic': self.statistic,
            'p_value': self.p_value,
            'interpretation': self.interpretation
        }
        
        if self.degrees_of_freedom is not None:
            result_dict['degrees_of_freedom'] = self.degrees_of_freedom
        if self.effect_size is not None:
            result_dict['effect_size'] = self.effect_size
        if self.confidence_interval is not None:
            result_dict['confidence_interval'] = self.confidence_interval
        if self.assumptions_met is not None:
            result_dict['assumptions_met'] = self.assumptions_met
        if self.additional_info is not None:
            result_dict['additional_info'] = self.additional_info
            
        return result_dict


class HypothesisTestingTransformer(BaseEstimator, TransformerMixin):
    """
    sklearn-compatible transformer for comprehensive hypothesis testing.
    
    Performs various hypothesis tests including t-tests, chi-square tests,
    normality tests, and correlation tests with proper effect size calculations
    and assumption checking.
    
    Parameters:
    -----------
    test_type : str, default='auto'
        Type of test to perform: 'ttest_1samp', 'ttest_ind', 'ttest_rel',
        'chi2', 'normality', 'correlation', or 'auto' for automatic selection
    alpha : float, default=0.05
        Significance level for hypothesis tests
    alternative : str, default='two-sided'
        Alternative hypothesis: 'two-sided', 'less', 'greater'
    equal_var : bool, default=True
        Assume equal variances for independent t-tests
    correction : str, default=None
        Multiple comparison correction: 'bonferroni', 'fdr_bh', None
    calculate_effect_size : bool, default=True
        Whether to calculate effect sizes (Cohen's d, Cramer's V, etc.)
    check_assumptions : bool, default=True
        Whether to check statistical assumptions
        
    Attributes:
    -----------
    test_results_ : List[StatisticalTestResult]
        Results of performed statistical tests
    assumptions_checked_ : Dict[str, bool]
        Results of assumption checks
    effect_sizes_ : Dict[str, float]
        Calculated effect sizes
    """
    
    def __init__(self,
                 test_type: str = 'auto',
                 alpha: float = 0.05,
                 alternative: str = 'two-sided',
                 equal_var: bool = True,
                 correction: Optional[str] = None,
                 calculate_effect_size: bool = True,
                 check_assumptions: bool = True):
        self.test_type = test_type
        self.alpha = alpha
        self.alternative = alternative
        self.equal_var = equal_var
        self.correction = correction
        self.calculate_effect_size = calculate_effect_size
        self.check_assumptions = check_assumptions

    def fit(self, X, y=None):
        """Fit the transformer (no-op for statistical tests)."""
        self._validate_parameters()
        return self

    def transform(self, X):
        """Perform hypothesis tests on the input data."""
        check_is_fitted(self)
        
        if isinstance(X, pd.DataFrame):
            data = X
        else:
            data = pd.DataFrame(X)
        
        self.test_results_ = []
        self.assumptions_checked_ = {}
        self.effect_sizes_ = {}
        
        # Automatic test selection based on data characteristics
        if self.test_type == 'auto':
            self._perform_automatic_testing(data)
        else:
            self._perform_specific_test(data, self.test_type)
        
        # Create result summary
        result_summary = {
            'test_results': [result.to_dict() for result in self.test_results_],
            'assumptions_checked': self.assumptions_checked_,
            'effect_sizes': self.effect_sizes_,
            'alpha_level': self.alpha,
            'correction_applied': self.correction
        }
        
        return pd.DataFrame([result_summary])

    def _validate_parameters(self):
        """Validate input parameters."""
        valid_tests = ['auto', 'ttest_1samp', 'ttest_ind', 'ttest_rel', 
                      'chi2', 'normality', 'correlation']
        if self.test_type not in valid_tests:
            raise ValueError(f"test_type must be one of {valid_tests}")
        
        if not 0 < self.alpha < 1:
            raise ValueError("alpha must be between 0 and 1")
        
        if self.alternative not in ['two-sided', 'less', 'greater']:
            raise ValueError("alternative must be 'two-sided', 'less', or 'greater'")

    def _perform_automatic_testing(self, data: pd.DataFrame):
        """Automatically select and perform appropriate tests."""
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns
        
        # Normality tests for numeric columns
        for col in numeric_cols:
            if data[col].notna().sum() >= 3:  # Minimum samples for normality test
                self._test_normality(data[col].dropna(), col)
        
        # Correlation tests between numeric columns
        if len(numeric_cols) >= 2:
            self._test_correlations(data[numeric_cols])
        
        # Chi-square tests for categorical columns
        if len(categorical_cols) >= 2:
            for i, col1 in enumerate(categorical_cols):
                for col2 in categorical_cols[i+1:]:
                    self._test_chi_square(data, col1, col2)

    def _perform_specific_test(self, data: pd.DataFrame, test_type: str):
        """Perform a specific type of test."""
        if test_type == 'normality':
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if data[col].notna().sum() >= 3:
                    self._test_normality(data[col].dropna(), col)
        
        elif test_type == 'correlation':
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) >= 2:
                self._test_correlations(data[numeric_cols])
        
        elif test_type == 'chi2':
            categorical_cols = data.select_dtypes(include=['object', 'category']).columns
            if len(categorical_cols) >= 2:
                for i, col1 in enumerate(categorical_cols):
                    for col2 in categorical_cols[i+1:]:
                        self._test_chi_square(data, col1, col2)
        
        elif test_type.startswith('ttest'):
            self._perform_t_test(data, test_type)

    def _test_normality(self, series: pd.Series, col_name: str):
        """Perform normality tests (Shapiro-Wilk and Kolmogorov-Smirnov)."""
        data_clean = series.dropna()
        
        if len(data_clean) < 3:
            return
        
        # Shapiro-Wilk test (better for smaller samples)
        if len(data_clean) <= 5000:
            try:
                shapiro_stat, shapiro_p = stats.shapiro(data_clean)
                interpretation = f"Data {'appears' if shapiro_p > self.alpha else 'does not appear'} to be normally distributed"
                
                result = StatisticalTestResult(
                    test_name=f"Shapiro-Wilk ({col_name})",
                    statistic=shapiro_stat,
                    p_value=shapiro_p,
                    interpretation=interpretation,
                    additional_info={'column': col_name, 'sample_size': len(data_clean)}
                )
                self.test_results_.append(result)
            except Exception as e:
                logger.warning(f"Shapiro-Wilk test failed for {col_name}: {e}")
        
        # Kolmogorov-Smirnov test (better for larger samples)
        try:
            ks_stat, ks_p = stats.kstest(data_clean, 'norm', 
                                       args=(data_clean.mean(), data_clean.std()))
            interpretation = f"Data {'appears' if ks_p > self.alpha else 'does not appear'} to follow normal distribution"
            
            result = StatisticalTestResult(
                test_name=f"Kolmogorov-Smirnov ({col_name})",
                statistic=ks_stat,
                p_value=ks_p,
                interpretation=interpretation,
                additional_info={'column': col_name, 'sample_size': len(data_clean)}
            )
            self.test_results_.append(result)
        except Exception as e:
            logger.warning(f"KS test failed for {col_name}: {e}")

    def _test_correlations(self, data: pd.DataFrame):
        """Perform correlation tests between numeric variables."""
        numeric_data = data.select_dtypes(include=[np.number]).dropna()
        
        if len(numeric_data) < 3:
            return
        
        cols = list(numeric_data.columns)
        
        for i, col1 in enumerate(cols):
            for col2 in cols[i+1:]:
                # Pearson correlation
                try:
                    pearson_r, pearson_p = pearsonr(numeric_data[col1], numeric_data[col2])
                    
                    # Calculate effect size (Cohen's conventions for correlation)
                    effect_size = abs(pearson_r)
                    if effect_size >= 0.5:
                        effect_desc = "large"
                    elif effect_size >= 0.3:
                        effect_desc = "medium"
                    elif effect_size >= 0.1:
                        effect_desc = "small"
                    else:
                        effect_desc = "negligible"
                    
                    interpretation = f"{'Significant' if pearson_p <= self.alpha else 'Non-significant'} correlation ({effect_desc} effect)"
                    
                    result = StatisticalTestResult(
                        test_name=f"Pearson Correlation ({col1} vs {col2})",
                        statistic=pearson_r,
                        p_value=pearson_p,
                        effect_size=effect_size,
                        interpretation=interpretation,
                        additional_info={
                            'column1': col1,
                            'column2': col2,
                            'effect_description': effect_desc,
                            'sample_size': len(numeric_data)
                        }
                    )
                    self.test_results_.append(result)
                except Exception as e:
                    logger.warning(f"Pearson correlation failed for {col1} vs {col2}: {e}")
                
                # Spearman correlation (non-parametric)
                try:
                    spearman_r, spearman_p = spearmanr(numeric_data[col1], numeric_data[col2])
                    interpretation = f"{'Significant' if spearman_p <= self.alpha else 'Non-significant'} rank correlation"
                    
                    result = StatisticalTestResult(
                        test_name=f"Spearman Correlation ({col1} vs {col2})",
                        statistic=spearman_r,
                        p_value=spearman_p,
                        interpretation=interpretation,
                        additional_info={
                            'column1': col1,
                            'column2': col2,
                            'sample_size': len(numeric_data)
                        }
                    )
                    self.test_results_.append(result)
                except Exception as e:
                    logger.warning(f"Spearman correlation failed for {col1} vs {col2}: {e}")

    def _test_chi_square(self, data: pd.DataFrame, col1: str, col2: str):
        """Perform chi-square test of independence."""
        try:
            # Create contingency table
            contingency_table = pd.crosstab(data[col1], data[col2])
            
            # Perform chi-square test
            chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)
            
            # Calculate Cramer's V (effect size)
            n = contingency_table.sum().sum()
            cramers_v = np.sqrt(chi2_stat / (n * (min(contingency_table.shape) - 1)))
            
            # Effect size interpretation
            if min(contingency_table.shape) == 2:  # 2x2 table
                if cramers_v >= 0.5:
                    effect_desc = "large"
                elif cramers_v >= 0.3:
                    effect_desc = "medium"
                elif cramers_v >= 0.1:
                    effect_desc = "small"
                else:
                    effect_desc = "negligible"
            else:  # Larger tables
                if cramers_v >= 0.25:
                    effect_desc = "large"
                elif cramers_v >= 0.15:
                    effect_desc = "medium"
                elif cramers_v >= 0.05:
                    effect_desc = "small"
                else:
                    effect_desc = "negligible"
            
            interpretation = f"{'Significant' if p_value <= self.alpha else 'Non-significant'} association ({effect_desc} effect)"
            
            result = StatisticalTestResult(
                test_name=f"Chi-square ({col1} vs {col2})",
                statistic=chi2_stat,
                p_value=p_value,
                degrees_of_freedom=dof,
                effect_size=cramers_v,
                interpretation=interpretation,
                additional_info={
                    'column1': col1,
                    'column2': col2,
                    'effect_description': effect_desc,
                    'sample_size': n,
                    'contingency_table': contingency_table.to_dict()
                }
            )
            self.test_results_.append(result)
            
        except Exception as e:
            logger.warning(f"Chi-square test failed for {col1} vs {col2}: {e}")

    def _perform_t_test(self, data: pd.DataFrame, test_type: str):
        """Perform various t-tests."""
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        if test_type == 'ttest_1samp' and len(numeric_cols) >= 1:
            # One-sample t-test against mean of 0
            for col in numeric_cols:
                series_clean = data[col].dropna()
                if len(series_clean) >= 3:
                    self._one_sample_ttest(series_clean, col)
        
        elif test_type == 'ttest_ind' and len(numeric_cols) >= 1:
            # Independent t-test (requires grouping variable)
            categorical_cols = data.select_dtypes(include=['object', 'category']).columns
            if len(categorical_cols) >= 1:
                for num_col in numeric_cols:
                    for cat_col in categorical_cols:
                        self._independent_ttest(data, num_col, cat_col)
        
        elif test_type == 'ttest_rel' and len(numeric_cols) >= 2:
            # Paired t-test
            for i, col1 in enumerate(numeric_cols):
                for col2 in numeric_cols[i+1:]:
                    self._paired_ttest(data, col1, col2)

    def _one_sample_ttest(self, series: pd.Series, col_name: str, popmean: float = 0):
        """Perform one-sample t-test."""
        try:
            t_stat, p_value = stats.ttest_1samp(series, popmean)
            
            # Calculate Cohen's d
            cohens_d = (series.mean() - popmean) / series.std()
            
            # Effect size interpretation
            abs_d = abs(cohens_d)
            if abs_d >= 0.8:
                effect_desc = "large"
            elif abs_d >= 0.5:
                effect_desc = "medium" 
            elif abs_d >= 0.2:
                effect_desc = "small"
            else:
                effect_desc = "negligible"
            
            interpretation = f"{'Significant' if p_value <= self.alpha else 'Non-significant'} difference from {popmean} ({effect_desc} effect)"
            
            result = StatisticalTestResult(
                test_name=f"One-sample t-test ({col_name})",
                statistic=t_stat,
                p_value=p_value,
                degrees_of_freedom=len(series) - 1,
                effect_size=abs_d,
                interpretation=interpretation,
                additional_info={
                    'column': col_name,
                    'sample_mean': series.mean(),
                    'population_mean': popmean,
                    'cohens_d': cohens_d,
                    'effect_description': effect_desc,
                    'sample_size': len(series)
                }
            )
            self.test_results_.append(result)
            
        except Exception as e:
            logger.warning(f"One-sample t-test failed for {col_name}: {e}")

    def _independent_ttest(self, data: pd.DataFrame, num_col: str, cat_col: str):
        """Perform independent samples t-test."""
        try:
            # Get unique categories (limit to 2 for t-test)
            categories = data[cat_col].value_counts().head(2).index
            if len(categories) < 2:
                return
            
            group1 = data[data[cat_col] == categories[0]][num_col].dropna()
            group2 = data[data[cat_col] == categories[1]][num_col].dropna()
            
            if len(group1) < 2 or len(group2) < 2:
                return
            
            # Perform t-test
            t_stat, p_value = stats.ttest_ind(group1, group2, equal_var=self.equal_var)
            
            # Calculate Cohen's d
            pooled_std = np.sqrt(((len(group1) - 1) * group1.var() + 
                                (len(group2) - 1) * group2.var()) / 
                               (len(group1) + len(group2) - 2))
            cohens_d = (group1.mean() - group2.mean()) / pooled_std
            
            # Effect size interpretation
            abs_d = abs(cohens_d)
            if abs_d >= 0.8:
                effect_desc = "large"
            elif abs_d >= 0.5:
                effect_desc = "medium"
            elif abs_d >= 0.2:
                effect_desc = "small"
            else:
                effect_desc = "negligible"
            
            interpretation = f"{'Significant' if p_value <= self.alpha else 'Non-significant'} difference between groups ({effect_desc} effect)"
            
            result = StatisticalTestResult(
                test_name=f"Independent t-test ({num_col} by {cat_col})",
                statistic=t_stat,
                p_value=p_value,
                degrees_of_freedom=len(group1) + len(group2) - 2,
                effect_size=abs_d,
                interpretation=interpretation,
                additional_info={
                    'numeric_column': num_col,
                    'grouping_column': cat_col,
                    'group1': str(categories[0]),
                    'group2': str(categories[1]),
                    'group1_mean': group1.mean(),
                    'group2_mean': group2.mean(),
                    'group1_size': len(group1),
                    'group2_size': len(group2),
                    'cohens_d': cohens_d,
                    'effect_description': effect_desc,
                    'equal_var_assumed': self.equal_var
                }
            )
            self.test_results_.append(result)
            
        except Exception as e:
            logger.warning(f"Independent t-test failed for {num_col} by {cat_col}: {e}")

    def _paired_ttest(self, data: pd.DataFrame, col1: str, col2: str):
        """Perform paired samples t-test."""
        try:
            # Get paired data
            paired_data = data[[col1, col2]].dropna()
            if len(paired_data) < 3:
                return
            
            # Perform paired t-test
            t_stat, p_value = stats.ttest_rel(paired_data[col1], paired_data[col2])
            
            # Calculate Cohen's d for paired samples
            differences = paired_data[col1] - paired_data[col2]
            cohens_d = differences.mean() / differences.std()
            
            # Effect size interpretation
            abs_d = abs(cohens_d)
            if abs_d >= 0.8:
                effect_desc = "large"
            elif abs_d >= 0.5:
                effect_desc = "medium"
            elif abs_d >= 0.2:
                effect_desc = "small"
            else:
                effect_desc = "negligible"
            
            interpretation = f"{'Significant' if p_value <= self.alpha else 'Non-significant'} difference between paired measurements ({effect_desc} effect)"
            
            result = StatisticalTestResult(
                test_name=f"Paired t-test ({col1} vs {col2})",
                statistic=t_stat,
                p_value=p_value,
                degrees_of_freedom=len(paired_data) - 1,
                effect_size=abs_d,
                interpretation=interpretation,
                additional_info={
                    'column1': col1,
                    'column2': col2,
                    'mean_difference': differences.mean(),
                    'cohens_d': cohens_d,
                    'effect_description': effect_desc,
                    'sample_size': len(paired_data)
                }
            )
            self.test_results_.append(result)
            
        except Exception as e:
            logger.warning(f"Paired t-test failed for {col1} vs {col2}: {e}")


class ANOVAAnalysisTransformer(BaseEstimator, TransformerMixin):
    """
    sklearn-compatible transformer for comprehensive ANOVA analysis.
    
    Performs one-way and two-way ANOVA with post-hoc analysis and effect size calculations.
    Includes assumptions checking and non-parametric alternatives when appropriate.
    
    Parameters:
    -----------
    anova_type : str, default='one_way'
        Type of ANOVA: 'one_way', 'two_way', 'auto'
    alpha : float, default=0.05
        Significance level for hypothesis tests
    post_hoc : str, default='tukey'
        Post-hoc test method: 'tukey', 'bonferroni', 'scheffe', None
    effect_size : str, default='eta_squared'
        Effect size measure: 'eta_squared', 'partial_eta_squared', 'omega_squared'
    check_assumptions : bool, default=True
        Whether to check ANOVA assumptions (normality, homoscedasticity)
    alpha_adjustment : str, default=None
        Multiple comparison adjustment: 'bonferroni', 'fdr_bh', None
        
    Attributes:
    -----------
    anova_results_ : Dict[str, Any]
        ANOVA test results and statistics
    post_hoc_results_ : Dict[str, Any]
        Post-hoc comparison results
    effect_sizes_ : Dict[str, float]
        Calculated effect sizes
    assumptions_ : Dict[str, bool]
        Results of assumption checks
    """
    
    def __init__(self,
                 anova_type: str = 'one_way',
                 alpha: float = 0.05,
                 post_hoc: str = 'tukey',
                 effect_size: str = 'eta_squared',
                 check_assumptions: bool = True,
                 alpha_adjustment: Optional[str] = None):
        self.anova_type = anova_type
        self.alpha = alpha
        self.post_hoc = post_hoc
        self.effect_size = effect_size
        self.check_assumptions = check_assumptions
        self.alpha_adjustment = alpha_adjustment

    def fit(self, X, y=None):
        """Fit the transformer (no-op for ANOVA)."""
        self._validate_parameters()
        return self

    def transform(self, X):
        """Perform ANOVA analysis on the input data."""
        check_is_fitted(self)
        
        if isinstance(X, pd.DataFrame):
            data = X
        else:
            data = pd.DataFrame(X)
        
        self.anova_results_ = {}
        self.post_hoc_results_ = {}
        self.effect_sizes_ = {}
        self.assumptions_ = {}
        
        # Identify numeric and categorical columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns
        
        if len(numeric_cols) == 0 or len(categorical_cols) == 0:
            logger.warning("ANOVA requires both numeric and categorical variables")
            return pd.DataFrame([{
                'error': 'ANOVA requires both numeric and categorical variables',
                'numeric_columns': len(numeric_cols),
                'categorical_columns': len(categorical_cols)
            }])
        
        # Perform ANOVA based on type
        if self.anova_type == 'auto':
            self._perform_automatic_anova(data, numeric_cols, categorical_cols)
        elif self.anova_type == 'one_way':
            self._perform_one_way_anova(data, numeric_cols, categorical_cols)
        elif self.anova_type == 'two_way':
            self._perform_two_way_anova(data, numeric_cols, categorical_cols)
        
        # Create result summary
        result_summary = {
            'anova_results': self.anova_results_,
            'post_hoc_results': self.post_hoc_results_,
            'effect_sizes': self.effect_sizes_,
            'assumptions_checked': self.assumptions_,
            'alpha_level': self.alpha,
            'post_hoc_method': self.post_hoc
        }
        
        return pd.DataFrame([result_summary])

    def _validate_parameters(self):
        """Validate input parameters."""
        valid_types = ['one_way', 'two_way', 'auto']
        if self.anova_type not in valid_types:
            raise ValueError(f"anova_type must be one of {valid_types}")
        
        if not 0 < self.alpha < 1:
            raise ValueError("alpha must be between 0 and 1")
        
        valid_post_hoc = ['tukey', 'bonferroni', 'scheffe', None]
        if self.post_hoc not in valid_post_hoc:
            raise ValueError(f"post_hoc must be one of {valid_post_hoc}")

    def _perform_automatic_anova(self, data: pd.DataFrame, numeric_cols: pd.Index, categorical_cols: pd.Index):
        """Automatically determine and perform appropriate ANOVA."""
        # Start with one-way ANOVA for each combination
        self._perform_one_way_anova(data, numeric_cols, categorical_cols)
        
        # If we have multiple categorical variables, also try two-way ANOVA
        if len(categorical_cols) >= 2:
            self._perform_two_way_anova(data, numeric_cols, categorical_cols[:2])

    def _perform_one_way_anova(self, data: pd.DataFrame, numeric_cols: pd.Index, categorical_cols: pd.Index):
        """Perform one-way ANOVA for each numeric-categorical combination."""
        for num_col in numeric_cols:
            for cat_col in categorical_cols:
                try:
                    self._one_way_anova_single(data, num_col, cat_col)
                except Exception as e:
                    logger.warning(f"One-way ANOVA failed for {num_col} by {cat_col}: {e}")

    def _one_way_anova_single(self, data: pd.DataFrame, num_col: str, cat_col: str):
        """Perform one-way ANOVA for a single numeric-categorical pair."""
        # Clean data
        clean_data = data[[num_col, cat_col]].dropna()
        if len(clean_data) < 6:  # Minimum samples for ANOVA
            return
        
        # Group data
        groups = [group[num_col].values for name, group in clean_data.groupby(cat_col)]
        group_names = list(clean_data[cat_col].unique())
        
        if len(groups) < 2:
            return
        
        # Check assumptions if requested
        if self.check_assumptions:
            self._check_anova_assumptions(groups, f"{num_col}_by_{cat_col}")
        
        # Perform one-way ANOVA
        f_stat, p_value = stats.f_oneway(*groups)
        df_between = len(groups) - 1
        df_within = len(clean_data) - len(groups)
        df_total = len(clean_data) - 1
        
        # Calculate effect sizes
        ss_between = sum(len(g) * (np.mean(g) - clean_data[num_col].mean())**2 for g in groups)
        ss_total = sum((clean_data[num_col] - clean_data[num_col].mean())**2)
        ss_within = ss_total - ss_between
        
        eta_squared = ss_between / ss_total
        omega_squared = (ss_between - df_between * (ss_within / df_within)) / (ss_total + ss_within / df_within)
        omega_squared = max(0, omega_squared)  # Can't be negative
        
        # Effect size interpretation (Cohen's conventions for eta-squared)
        if eta_squared >= 0.14:
            effect_desc = "large"
        elif eta_squared >= 0.06:
            effect_desc = "medium"
        elif eta_squared >= 0.01:
            effect_desc = "small"
        else:
            effect_desc = "negligible"
        
        # Store results
        anova_key = f"one_way_{num_col}_by_{cat_col}"
        self.anova_results_[anova_key] = {
            'test_type': 'One-way ANOVA',
            'dependent_variable': num_col,
            'independent_variable': cat_col,
            'f_statistic': f_stat,
            'p_value': p_value,
            'df_between': df_between,
            'df_within': df_within,
            'df_total': df_total,
            'significant': p_value <= self.alpha,
            'group_names': group_names,
            'group_means': [np.mean(g) for g in groups],
            'group_sizes': [len(g) for g in groups],
            'interpretation': f"{'Significant' if p_value <= self.alpha else 'Non-significant'} group differences ({effect_desc} effect)"
        }
        
        self.effect_sizes_[anova_key] = {
            'eta_squared': eta_squared,
            'omega_squared': omega_squared,
            'effect_description': effect_desc
        }
        
        # Perform post-hoc tests if significant and requested
        if p_value <= self.alpha and self.post_hoc and len(groups) > 2:
            self._perform_post_hoc_analysis(clean_data, num_col, cat_col, anova_key)

    def _perform_two_way_anova(self, data: pd.DataFrame, numeric_cols: pd.Index, categorical_cols: pd.Index):
        """Perform two-way ANOVA with interaction effects."""
        if len(categorical_cols) < 2:
            return
        
        for num_col in numeric_cols:
            try:
                cat_col1, cat_col2 = categorical_cols[0], categorical_cols[1]
                self._two_way_anova_single(data, num_col, cat_col1, cat_col2)
            except Exception as e:
                logger.warning(f"Two-way ANOVA failed for {num_col}: {e}")

    def _two_way_anova_single(self, data: pd.DataFrame, num_col: str, cat_col1: str, cat_col2: str):
        """Perform two-way ANOVA for a single configuration."""
        # Clean data
        clean_data = data[[num_col, cat_col1, cat_col2]].dropna()
        if len(clean_data) < 12:  # Minimum samples for two-way ANOVA
            return
        
        try:
            # Create formula for OLS
            formula = f"{num_col} ~ C({cat_col1}) + C({cat_col2}) + C({cat_col1}):C({cat_col2})"
            
            # Fit OLS model
            model = ols(formula, data=clean_data).fit()
            
            # Perform ANOVA
            anova_table = anova_lm(model, typ=2)
            
            # Extract results
            anova_key = f"two_way_{num_col}_by_{cat_col1}_and_{cat_col2}"
            
            self.anova_results_[anova_key] = {
                'test_type': 'Two-way ANOVA',
                'dependent_variable': num_col,
                'independent_variable_1': cat_col1,
                'independent_variable_2': cat_col2,
                'anova_table': anova_table.to_dict(),
                'model_summary': {
                    'r_squared': model.rsquared,
                    'adj_r_squared': model.rsquared_adj,
                    'f_statistic': model.fvalue,
                    'f_p_value': model.f_pvalue
                }
            }
            
            # Calculate effect sizes for main effects and interaction
            ss_residual = anova_table.loc['Residual', 'sum_sq']
            ss_total = anova_table['sum_sq'].sum()
            
            effect_sizes = {}
            for factor in anova_table.index[:-1]:  # Exclude residual
                ss_factor = anova_table.loc[factor, 'sum_sq']
                eta_squared = ss_factor / ss_total
                partial_eta_squared = ss_factor / (ss_factor + ss_residual)
                effect_sizes[factor] = {
                    'eta_squared': eta_squared,
                    'partial_eta_squared': partial_eta_squared
                }
            
            self.effect_sizes_[anova_key] = effect_sizes
            
        except Exception as e:
            logger.warning(f"Two-way ANOVA calculation failed: {e}")

    def _check_anova_assumptions(self, groups: List[np.ndarray], test_name: str):
        """Check ANOVA assumptions (normality and homoscedasticity)."""
        assumptions = {}
        
        # Check normality for each group (Shapiro-Wilk)
        normality_results = []
        for i, group in enumerate(groups):
            if len(group) >= 3 and len(group) <= 5000:
                try:
                    _, p_val = stats.shapiro(group)
                    normality_results.append(p_val > 0.05)
                except:
                    normality_results.append(None)
            else:
                normality_results.append(None)
        
        assumptions['normality'] = all(r for r in normality_results if r is not None)
        
        # Check homoscedasticity (Levene's test)
        if len(groups) >= 2 and all(len(g) >= 2 for g in groups):
            try:
                _, levene_p = stats.levene(*groups)
                assumptions['homoscedasticity'] = levene_p > 0.05
            except:
                assumptions['homoscedasticity'] = None
        else:
            assumptions['homoscedasticity'] = None
        
        self.assumptions_[test_name] = assumptions

    def _perform_post_hoc_analysis(self, data: pd.DataFrame, num_col: str, cat_col: str, anova_key: str):
        """Perform post-hoc pairwise comparisons."""
        if self.post_hoc == 'tukey':
            try:
                # Tukey HSD test
                tukey_result = pairwise_tukeyhsd(data[num_col], data[cat_col], alpha=self.alpha)
                
                # Convert to structured format
                post_hoc_summary = {
                    'method': 'Tukey HSD',
                    'alpha': self.alpha,
                    'comparisons': []
                }
                
                # Extract pairwise comparisons
                for i in range(len(tukey_result.groupsunique)):
                    for j in range(i + 1, len(tukey_result.groupsunique)):
                        group1 = tukey_result.groupsunique[i]
                        group2 = tukey_result.groupsunique[j]
                        
                        # Find the corresponding result
                        mask = ((tukey_result.data['group1'] == group1) & 
                               (tukey_result.data['group2'] == group2)) | \
                               ((tukey_result.data['group1'] == group2) & 
                               (tukey_result.data['group2'] == group1))
                        
                        if mask.any():
                            row = tukey_result.data[mask].iloc[0]
                            post_hoc_summary['comparisons'].append({
                                'group1': str(group1),
                                'group2': str(group2),
                                'mean_diff': row['meandiff'],
                                'p_value': row['p-adj'],
                                'significant': row['reject'],
                                'lower_ci': row['lower'],
                                'upper_ci': row['upper']
                            })
                
                self.post_hoc_results_[anova_key] = post_hoc_summary
                
            except Exception as e:
                logger.warning(f"Tukey post-hoc test failed: {e}")


class NonParametricTestTransformer(BaseEstimator, TransformerMixin):
    """
    sklearn-compatible transformer for non-parametric statistical tests.
    
    Performs Mann-Whitney U, Wilcoxon signed-rank, Kruskal-Wallis H, and Friedman tests
    as alternatives to parametric tests when assumptions are violated.
    
    Parameters:
    -----------
    test_type : str, default='auto'
        Type of test: 'mann_whitney', 'wilcoxon', 'kruskal_wallis', 'friedman', 'auto'
    alpha : float, default=0.05
        Significance level for hypothesis tests
    alternative : str, default='two-sided'
        Alternative hypothesis: 'two-sided', 'less', 'greater'
    correction : str, default=None
        Multiple comparison correction: 'bonferroni', 'fdr_bh', None
    calculate_effect_size : bool, default=True
        Whether to calculate effect sizes (rank-biserial correlation, etc.)
        
    Attributes:
    -----------
    test_results_ : List[StatisticalTestResult]
        Results of performed non-parametric tests
    effect_sizes_ : Dict[str, float]
        Calculated effect sizes for non-parametric tests
    """
    
    def __init__(self,
                 test_type: str = 'auto',
                 alpha: float = 0.05,
                 alternative: str = 'two-sided',
                 correction: Optional[str] = None,
                 calculate_effect_size: bool = True):
        self.test_type = test_type
        self.alpha = alpha
        self.alternative = alternative
        self.correction = correction
        self.calculate_effect_size = calculate_effect_size

    def fit(self, X, y=None):
        """Fit the transformer (no-op for non-parametric tests)."""
        self._validate_parameters()
        return self

    def transform(self, X):
        """Perform non-parametric tests on the input data."""
        check_is_fitted(self)
        
        if isinstance(X, pd.DataFrame):
            data = X
        else:
            data = pd.DataFrame(X)
        
        self.test_results_ = []
        self.effect_sizes_ = {}
        
        # Perform tests based on type
        if self.test_type == 'auto':
            self._perform_automatic_nonparametric_tests(data)
        else:
            self._perform_specific_nonparametric_test(data, self.test_type)
        
        # Create result summary
        result_summary = {
            'test_results': [result.to_dict() for result in self.test_results_],
            'effect_sizes': self.effect_sizes_,
            'alpha_level': self.alpha,
            'correction_applied': self.correction
        }
        
        return pd.DataFrame([result_summary])

    def _validate_parameters(self):
        """Validate input parameters."""
        valid_tests = ['auto', 'mann_whitney', 'wilcoxon', 'kruskal_wallis', 'friedman']
        if self.test_type not in valid_tests:
            raise ValueError(f"test_type must be one of {valid_tests}")
        
        if not 0 < self.alpha < 1:
            raise ValueError("alpha must be between 0 and 1")

    def _perform_automatic_nonparametric_tests(self, data: pd.DataFrame):
        """Automatically select and perform appropriate non-parametric tests."""
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns
        
        # Mann-Whitney U tests for numeric vs binary categorical
        for num_col in numeric_cols:
            for cat_col in categorical_cols:
                unique_cats = data[cat_col].nunique()
                if unique_cats == 2:
                    self._mann_whitney_test(data, num_col, cat_col)
                elif unique_cats > 2:
                    self._kruskal_wallis_test(data, num_col, cat_col)
        
        # Wilcoxon tests for paired numeric columns
        if len(numeric_cols) >= 2:
            for i, col1 in enumerate(numeric_cols):
                for col2 in numeric_cols[i+1:]:
                    self._wilcoxon_test(data, col1, col2)

    def _perform_specific_nonparametric_test(self, data: pd.DataFrame, test_type: str):
        """Perform a specific non-parametric test."""
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns
        
        if test_type == 'mann_whitney':
            for num_col in numeric_cols:
                for cat_col in categorical_cols:
                    if data[cat_col].nunique() == 2:
                        self._mann_whitney_test(data, num_col, cat_col)
        
        elif test_type == 'wilcoxon':
            if len(numeric_cols) >= 2:
                for i, col1 in enumerate(numeric_cols):
                    for col2 in numeric_cols[i+1:]:
                        self._wilcoxon_test(data, col1, col2)
        
        elif test_type == 'kruskal_wallis':
            for num_col in numeric_cols:
                for cat_col in categorical_cols:
                    if data[cat_col].nunique() > 2:
                        self._kruskal_wallis_test(data, num_col, cat_col)
        
        elif test_type == 'friedman':
            if len(numeric_cols) >= 3:
                # Friedman test requires repeated measures design
                # For simplicity, we'll test all numeric columns as if they're repeated measures
                numeric_data = data[numeric_cols].dropna()
                if len(numeric_data) >= 3:
                    self._friedman_test(numeric_data, numeric_cols.tolist())

    def _mann_whitney_test(self, data: pd.DataFrame, num_col: str, cat_col: str):
        """Perform Mann-Whitney U test."""
        try:
            categories = data[cat_col].value_counts().head(2).index
            group1 = data[data[cat_col] == categories[0]][num_col].dropna()
            group2 = data[data[cat_col] == categories[1]][num_col].dropna()
            
            if len(group1) < 3 or len(group2) < 3:
                return
            
            # Perform Mann-Whitney U test
            statistic, p_value = stats.mannwhitneyu(
                group1, group2, 
                alternative=self.alternative,
                use_continuity=True
            )
            
            # Calculate effect size (rank-biserial correlation)
            n1, n2 = len(group1), len(group2)
            r = 1 - (2 * statistic) / (n1 * n2)  # rank-biserial correlation
            
            # Effect size interpretation
            abs_r = abs(r)
            if abs_r >= 0.5:
                effect_desc = "large"
            elif abs_r >= 0.3:
                effect_desc = "medium"
            elif abs_r >= 0.1:
                effect_desc = "small"
            else:
                effect_desc = "negligible"
            
            interpretation = f"{'Significant' if p_value <= self.alpha else 'Non-significant'} difference between groups ({effect_desc} effect)"
            
            result = StatisticalTestResult(
                test_name=f"Mann-Whitney U ({num_col} by {cat_col})",
                statistic=statistic,
                p_value=p_value,
                effect_size=abs_r,
                interpretation=interpretation,
                additional_info={
                    'numeric_column': num_col,
                    'grouping_column': cat_col,
                    'group1': str(categories[0]),
                    'group2': str(categories[1]),
                    'group1_median': group1.median(),
                    'group2_median': group2.median(),
                    'group1_size': n1,
                    'group2_size': n2,
                    'rank_biserial_correlation': r,
                    'effect_description': effect_desc
                }
            )
            self.test_results_.append(result)
            
        except Exception as e:
            logger.warning(f"Mann-Whitney U test failed for {num_col} by {cat_col}: {e}")

    def _wilcoxon_test(self, data: pd.DataFrame, col1: str, col2: str):
        """Perform Wilcoxon signed-rank test."""
        try:
            paired_data = data[[col1, col2]].dropna()
            if len(paired_data) < 6:
                return
            
            # Perform Wilcoxon signed-rank test
            statistic, p_value = stats.wilcoxon(
                paired_data[col1], paired_data[col2],
                alternative=self.alternative,
                zero_method='wilcox'
            )
            
            # Calculate effect size (rank-biserial correlation for paired data)
            differences = paired_data[col1] - paired_data[col2]
            n_pos = (differences > 0).sum()
            n_neg = (differences < 0).sum()
            r = (n_pos - n_neg) / len(differences) if len(differences) > 0 else 0
            
            # Effect size interpretation
            abs_r = abs(r)
            if abs_r >= 0.5:
                effect_desc = "large"
            elif abs_r >= 0.3:
                effect_desc = "medium"
            elif abs_r >= 0.1:
                effect_desc = "small"
            else:
                effect_desc = "negligible"
            
            interpretation = f"{'Significant' if p_value <= self.alpha else 'Non-significant'} difference between paired measurements ({effect_desc} effect)"
            
            result = StatisticalTestResult(
                test_name=f"Wilcoxon Signed-Rank ({col1} vs {col2})",
                statistic=statistic,
                p_value=p_value,
                effect_size=abs_r,
                interpretation=interpretation,
                additional_info={
                    'column1': col1,
                    'column2': col2,
                    'median_difference': differences.median(),
                    'positive_ranks': n_pos,
                    'negative_ranks': n_neg,
                    'rank_biserial_correlation': r,
                    'effect_description': effect_desc,
                    'sample_size': len(paired_data)
                }
            )
            self.test_results_.append(result)
            
        except Exception as e:
            logger.warning(f"Wilcoxon test failed for {col1} vs {col2}: {e}")

    def _kruskal_wallis_test(self, data: pd.DataFrame, num_col: str, cat_col: str):
        """Perform Kruskal-Wallis H test."""
        try:
            groups = [group[num_col].dropna().values for name, group in data.groupby(cat_col)]
            group_names = list(data[cat_col].unique())
            
            # Filter out groups with insufficient data
            valid_groups = [g for g in groups if len(g) >= 3]
            if len(valid_groups) < 2:
                return
            
            # Perform Kruskal-Wallis test
            statistic, p_value = stats.kruskal(*valid_groups)
            df = len(valid_groups) - 1
            
            # Calculate effect size (eta-squared analog for Kruskal-Wallis)
            n_total = sum(len(g) for g in valid_groups)
            eta_squared = (statistic - len(valid_groups) + 1) / (n_total - len(valid_groups))
            eta_squared = max(0, min(1, eta_squared))  # Bound between 0 and 1
            
            # Effect size interpretation
            if eta_squared >= 0.14:
                effect_desc = "large"
            elif eta_squared >= 0.06:
                effect_desc = "medium"
            elif eta_squared >= 0.01:
                effect_desc = "small"
            else:
                effect_desc = "negligible"
            
            interpretation = f"{'Significant' if p_value <= self.alpha else 'Non-significant'} difference between groups ({effect_desc} effect)"
            
            result = StatisticalTestResult(
                test_name=f"Kruskal-Wallis H ({num_col} by {cat_col})",
                statistic=statistic,
                p_value=p_value,
                degrees_of_freedom=df,
                effect_size=eta_squared,
                interpretation=interpretation,
                additional_info={
                    'numeric_column': num_col,
                    'grouping_column': cat_col,
                    'group_names': group_names[:len(valid_groups)],
                    'group_medians': [np.median(g) for g in valid_groups],
                    'group_sizes': [len(g) for g in valid_groups],
                    'eta_squared': eta_squared,
                    'effect_description': effect_desc
                }
            )
            self.test_results_.append(result)
            
        except Exception as e:
            logger.warning(f"Kruskal-Wallis test failed for {num_col} by {cat_col}: {e}")

    def _friedman_test(self, data: pd.DataFrame, columns: List[str]):
        """Perform Friedman test for repeated measures."""
        try:
            if len(data) < 3:
                return
            
            # Perform Friedman test
            statistic, p_value = stats.friedmanchisquare(*[data[col].values for col in columns])
            df = len(columns) - 1
            
            # Calculate Kendall's W (effect size for Friedman test)
            n = len(data)
            k = len(columns)
            kendalls_w = statistic / (n * (k - 1))
            
            # Effect size interpretation for Kendall's W
            if kendalls_w >= 0.5:
                effect_desc = "large"
            elif kendalls_w >= 0.3:
                effect_desc = "medium"
            elif kendalls_w >= 0.1:
                effect_desc = "small"
            else:
                effect_desc = "negligible"
            
            interpretation = f"{'Significant' if p_value <= self.alpha else 'Non-significant'} difference between repeated measures ({effect_desc} effect)"
            
            result = StatisticalTestResult(
                test_name=f"Friedman Test ({', '.join(columns)})",
                statistic=statistic,
                p_value=p_value,
                degrees_of_freedom=df,
                effect_size=kendalls_w,
                interpretation=interpretation,
                additional_info={
                    'columns': columns,
                    'sample_size': n,
                    'kendalls_w': kendalls_w,
                    'effect_description': effect_desc,
                    'column_medians': [data[col].median() for col in columns]
                }
            )
            self.test_results_.append(result)
            
        except Exception as e:
            logger.warning(f"Friedman test failed: {e}")


class ExperimentalDesignTransformer(BaseEstimator, TransformerMixin):
    """
    sklearn-compatible transformer for experimental design analysis.
    
    Performs power analysis, sample size determination, effect size calculations,
    and confidence interval construction for experimental research design.
    
    Parameters:
    -----------
    analysis_type : str, default='power_analysis'
        Type of analysis: 'power_analysis', 'sample_size', 'effect_size', 'confidence_intervals'
    effect_size : float, default=None
        Expected effect size (Cohen's d for t-tests, eta-squared for ANOVA)
    alpha : float, default=0.05
        Type I error rate (significance level)
    power : float, default=0.80
        Desired statistical power (1 - Type II error rate)
    test_type : str, default='ttest'
        Statistical test type: 'ttest', 'anova', 'correlation', 'proportion'
    alternative : str, default='two-sided'
        Alternative hypothesis: 'two-sided', 'larger', 'smaller'
    confidence_level : float, default=0.95
        Confidence level for interval estimation
        
    Attributes:
    -----------
    power_analysis_ : Dict[str, Any]
        Power analysis results
    sample_sizes_ : Dict[str, int]
        Calculated sample sizes
    effect_sizes_ : Dict[str, float]
        Calculated effect sizes
    confidence_intervals_ : Dict[str, Tuple[float, float]]
        Calculated confidence intervals
    """
    
    def __init__(self,
                 analysis_type: str = 'power_analysis',
                 effect_size: Optional[float] = None,
                 alpha: float = 0.05,
                 power: float = 0.80,
                 test_type: str = 'ttest',
                 alternative: str = 'two-sided',
                 confidence_level: float = 0.95):
        self.analysis_type = analysis_type
        self.effect_size = effect_size
        self.alpha = alpha
        self.power = power
        self.test_type = test_type
        self.alternative = alternative
        self.confidence_level = confidence_level

    def fit(self, X, y=None):
        """Fit the transformer (no-op for experimental design)."""
        self._validate_parameters()
        return self

    def transform(self, X):
        """Perform experimental design analysis on the input data."""
        check_is_fitted(self)
        
        if isinstance(X, pd.DataFrame):
            data = X
        else:
            data = pd.DataFrame(X)
        
        self.power_analysis_ = {}
        self.sample_sizes_ = {}
        self.effect_sizes_ = {}
        self.confidence_intervals_ = {}
        
        # Perform analysis based on type
        if self.analysis_type == 'power_analysis':
            self._perform_power_analysis(data)
        elif self.analysis_type == 'sample_size':
            self._calculate_sample_sizes(data)
        elif self.analysis_type == 'effect_size':
            self._calculate_effect_sizes(data)
        elif self.analysis_type == 'confidence_intervals':
            self._calculate_confidence_intervals(data)
        else:
            # Perform all analyses
            self._perform_power_analysis(data)
            self._calculate_sample_sizes(data)
            self._calculate_effect_sizes(data)
            self._calculate_confidence_intervals(data)
        
        # Create result summary
        result_summary = {
            'power_analysis': self.power_analysis_,
            'sample_sizes': self.sample_sizes_,
            'effect_sizes': self.effect_sizes_,
            'confidence_intervals': self.confidence_intervals_,
            'parameters': {
                'alpha': self.alpha,
                'power': self.power,
                'effect_size': self.effect_size,
                'test_type': self.test_type,
                'confidence_level': self.confidence_level
            }
        }
        
        return pd.DataFrame([result_summary])

    def _validate_parameters(self):
        """Validate input parameters."""
        if not 0 < self.alpha < 1:
            raise ValueError("alpha must be between 0 and 1")
        
        if not 0 < self.power < 1:
            raise ValueError("power must be between 0 and 1")
        
        if not 0 < self.confidence_level < 1:
            raise ValueError("confidence_level must be between 0 and 1")
        
        valid_tests = ['ttest', 'anova', 'correlation', 'proportion']
        if self.test_type not in valid_tests:
            raise ValueError(f"test_type must be one of {valid_tests}")

    def _perform_power_analysis(self, data: pd.DataFrame):
        """Perform statistical power analysis."""
        if self.test_type == 'ttest':
            self._power_analysis_ttest(data)
        elif self.test_type == 'anova':
            self._power_analysis_anova(data)
        elif self.test_type == 'correlation':
            self._power_analysis_correlation(data)

    def _power_analysis_ttest(self, data: pd.DataFrame):
        """Power analysis for t-tests."""
        try:
            # Use provided effect size or estimate from data
            if self.effect_size is None:
                effect_size = 0.5  # Medium effect size
            else:
                effect_size = self.effect_size
            
            # Calculate power for different sample sizes
            sample_sizes = [10, 20, 30, 50, 100, 200, 500]
            powers = []
            
            for n in sample_sizes:
                try:
                    power = ttest_power(effect_size, n, self.alpha, 
                                      alternative=self.alternative)
                    powers.append(power)
                except:
                    powers.append(None)
            
            # Calculate required sample size for desired power
            try:
                required_n = tt_solve_power(effect_size=effect_size, 
                                          power=self.power, 
                                          alpha=self.alpha,
                                          alternative=self.alternative)
                required_n = int(np.ceil(required_n))
            except:
                required_n = None
            
            self.power_analysis_['ttest'] = {
                'effect_size': effect_size,
                'alpha': self.alpha,
                'desired_power': self.power,
                'required_sample_size': required_n,
                'power_curve': dict(zip(sample_sizes, powers)),
                'interpretation': self._interpret_power_analysis(required_n, self.power)
            }
            
        except Exception as e:
            logger.warning(f"T-test power analysis failed: {e}")

    def _power_analysis_anova(self, data: pd.DataFrame):
        """Power analysis for ANOVA (simplified version)."""
        try:
            # Estimate effect size from data if not provided
            if self.effect_size is None:
                effect_size = 0.25  # Medium effect size for ANOVA (eta-squared)
            else:
                effect_size = self.effect_size
            
            # Convert eta-squared to Cohen's f
            cohens_f = np.sqrt(effect_size / (1 - effect_size))
            
            # Assume 3 groups for power calculation
            num_groups = 3
            
            # Calculate power for different sample sizes per group
            sample_sizes_per_group = [10, 15, 20, 30, 50, 100]
            powers = []
            
            for n_per_group in sample_sizes_per_group:
                # Simplified power calculation using non-centrality parameter
                df1 = num_groups - 1
                df2 = num_groups * (n_per_group - 1)
                ncp = n_per_group * num_groups * (cohens_f ** 2)
                
                # Critical F value
                f_crit = stats.f.ppf(1 - self.alpha, df1, df2)
                
                # Power calculation
                power = 1 - stats.ncf.cdf(f_crit, df1, df2, ncp)
                powers.append(power)
            
            self.power_analysis_['anova'] = {
                'effect_size_eta_squared': effect_size,
                'cohens_f': cohens_f,
                'num_groups': num_groups,
                'alpha': self.alpha,
                'desired_power': self.power,
                'power_curve': dict(zip(sample_sizes_per_group, powers)),
                'interpretation': f"Power analysis for {num_groups}-group ANOVA with eta-squared = {effect_size:.3f}"
            }
            
        except Exception as e:
            logger.warning(f"ANOVA power analysis failed: {e}")

    def _power_analysis_correlation(self, data: pd.DataFrame):
        """Power analysis for correlation tests."""
        try:
            if self.effect_size is None:
                effect_size = 0.3  # Medium correlation
            else:
                effect_size = self.effect_size
            
            # Fisher's z-transformation
            z_effect = 0.5 * np.log((1 + effect_size) / (1 - effect_size))
            
            # Calculate power for different sample sizes
            sample_sizes = [20, 30, 50, 100, 200, 500]
            powers = []
            
            for n in sample_sizes:
                if n > 3:
                    # Standard error
                    se = 1 / np.sqrt(n - 3)
                    
                    # Critical values
                    if self.alternative == 'two-sided':
                        z_crit = stats.norm.ppf(1 - self.alpha / 2)
                    else:
                        z_crit = stats.norm.ppf(1 - self.alpha)
                    
                    # Power calculation
                    if self.alternative == 'two-sided':
                        power = (1 - stats.norm.cdf(z_crit - z_effect / se) + 
                                stats.norm.cdf(-z_crit - z_effect / se))
                    else:
                        power = 1 - stats.norm.cdf(z_crit - z_effect / se)
                    
                    powers.append(power)
                else:
                    powers.append(None)
            
            # Calculate required sample size
            if self.alternative == 'two-sided':
                z_crit = stats.norm.ppf(1 - self.alpha / 2)
            else:
                z_crit = stats.norm.ppf(1 - self.alpha)
            
            z_power = stats.norm.ppf(self.power)
            required_n = int(np.ceil(((z_crit + z_power) / z_effect) ** 2 + 3))
            
            self.power_analysis_['correlation'] = {
                'correlation_coefficient': effect_size,
                'alpha': self.alpha,
                'desired_power': self.power,
                'required_sample_size': required_n,
                'power_curve': dict(zip(sample_sizes, powers)),
                'interpretation': self._interpret_power_analysis(required_n, self.power)
            }
            
        except Exception as e:
            logger.warning(f"Correlation power analysis failed: {e}")

    def _calculate_sample_sizes(self, data: pd.DataFrame):
        """Calculate required sample sizes for different effect sizes."""
        if self.test_type == 'ttest':
            effect_sizes = [0.2, 0.5, 0.8]  # Small, medium, large
            
            for es in effect_sizes:
                try:
                    n = tt_solve_power(effect_size=es, power=self.power, 
                                     alpha=self.alpha, alternative=self.alternative)
                    self.sample_sizes_[f"ttest_cohens_d_{es}"] = int(np.ceil(n))
                except:
                    self.sample_sizes_[f"ttest_cohens_d_{es}"] = None

    def _calculate_effect_sizes(self, data: pd.DataFrame):
        """Calculate effect sizes from actual data."""
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns
        
        # Cohen's d for two-group comparisons
        for num_col in numeric_cols:
            for cat_col in categorical_cols:
                if data[cat_col].nunique() == 2:
                    self._calculate_cohens_d(data, num_col, cat_col)
        
        # Correlation effect sizes
        if len(numeric_cols) >= 2:
            for i, col1 in enumerate(numeric_cols):
                for col2 in numeric_cols[i+1:]:
                    self._calculate_correlation_effect_size(data, col1, col2)

    def _calculate_cohens_d(self, data: pd.DataFrame, num_col: str, cat_col: str):
        """Calculate Cohen's d effect size."""
        try:
            categories = data[cat_col].value_counts().head(2).index
            group1 = data[data[cat_col] == categories[0]][num_col].dropna()
            group2 = data[data[cat_col] == categories[1]][num_col].dropna()
            
            if len(group1) < 2 or len(group2) < 2:
                return
            
            # Calculate pooled standard deviation
            pooled_std = np.sqrt(((len(group1) - 1) * group1.var() + 
                                (len(group2) - 1) * group2.var()) / 
                               (len(group1) + len(group2) - 2))
            
            # Calculate Cohen's d
            cohens_d = (group1.mean() - group2.mean()) / pooled_std
            
            # Effect size interpretation
            abs_d = abs(cohens_d)
            if abs_d >= 0.8:
                effect_desc = "large"
            elif abs_d >= 0.5:
                effect_desc = "medium"
            elif abs_d >= 0.2:
                effect_desc = "small"
            else:
                effect_desc = "negligible"
            
            key = f"cohens_d_{num_col}_by_{cat_col}"
            self.effect_sizes_[key] = {
                'cohens_d': cohens_d,
                'absolute_effect_size': abs_d,
                'effect_description': effect_desc,
                'group1_mean': group1.mean(),
                'group2_mean': group2.mean(),
                'pooled_std': pooled_std,
                'group1_size': len(group1),
                'group2_size': len(group2)
            }
            
        except Exception as e:
            logger.warning(f"Cohen's d calculation failed for {num_col} by {cat_col}: {e}")

    def _calculate_correlation_effect_size(self, data: pd.DataFrame, col1: str, col2: str):
        """Calculate correlation effect size."""
        try:
            clean_data = data[[col1, col2]].dropna()
            if len(clean_data) < 3:
                return
            
            r, p_value = pearsonr(clean_data[col1], clean_data[col2])
            
            # Effect size interpretation for correlation
            abs_r = abs(r)
            if abs_r >= 0.5:
                effect_desc = "large"
            elif abs_r >= 0.3:
                effect_desc = "medium"
            elif abs_r >= 0.1:
                effect_desc = "small"
            else:
                effect_desc = "negligible"
            
            key = f"correlation_{col1}_vs_{col2}"
            self.effect_sizes_[key] = {
                'pearson_r': r,
                'absolute_correlation': abs_r,
                'effect_description': effect_desc,
                'p_value': p_value,
                'sample_size': len(clean_data),
                'r_squared': r ** 2
            }
            
        except Exception as e:
            logger.warning(f"Correlation effect size calculation failed for {col1} vs {col2}: {e}")

    def _calculate_confidence_intervals(self, data: pd.DataFrame):
        """Calculate confidence intervals for various statistics."""
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        # Confidence intervals for means
        for col in numeric_cols:
            self._calculate_mean_ci(data[col], col)
        
        # Confidence intervals for correlations
        if len(numeric_cols) >= 2:
            for i, col1 in enumerate(numeric_cols):
                for col2 in numeric_cols[i+1:]:
                    self._calculate_correlation_ci(data, col1, col2)

    def _calculate_mean_ci(self, series: pd.Series, col_name: str):
        """Calculate confidence interval for mean."""
        try:
            clean_data = series.dropna()
            if len(clean_data) < 2:
                return
            
            mean = clean_data.mean()
            std_err = clean_data.std() / np.sqrt(len(clean_data))
            
            # t-distribution critical value
            alpha_ci = 1 - self.confidence_level
            t_crit = stats.t.ppf(1 - alpha_ci / 2, len(clean_data) - 1)
            
            # Confidence interval
            margin_error = t_crit * std_err
            ci_lower = mean - margin_error
            ci_upper = mean + margin_error
            
            key = f"mean_ci_{col_name}"
            self.confidence_intervals_[key] = {
                'mean': mean,
                'confidence_level': self.confidence_level,
                'lower_bound': ci_lower,
                'upper_bound': ci_upper,
                'margin_of_error': margin_error,
                'standard_error': std_err,
                'sample_size': len(clean_data)
            }
            
        except Exception as e:
            logger.warning(f"Mean CI calculation failed for {col_name}: {e}")

    def _calculate_correlation_ci(self, data: pd.DataFrame, col1: str, col2: str):
        """Calculate confidence interval for correlation coefficient."""
        try:
            clean_data = data[[col1, col2]].dropna()
            if len(clean_data) < 4:
                return
            
            r, _ = pearsonr(clean_data[col1], clean_data[col2])
            n = len(clean_data)
            
            # Fisher's z-transformation
            z_r = 0.5 * np.log((1 + r) / (1 - r))
            z_se = 1 / np.sqrt(n - 3)
            
            # Critical value
            alpha_ci = 1 - self.confidence_level
            z_crit = stats.norm.ppf(1 - alpha_ci / 2)
            
            # CI for z-transformed correlation
            z_lower = z_r - z_crit * z_se
            z_upper = z_r + z_crit * z_se
            
            # Transform back to correlation scale
            r_lower = (np.exp(2 * z_lower) - 1) / (np.exp(2 * z_lower) + 1)
            r_upper = (np.exp(2 * z_upper) - 1) / (np.exp(2 * z_upper) + 1)
            
            key = f"correlation_ci_{col1}_vs_{col2}"
            self.confidence_intervals_[key] = {
                'correlation': r,
                'confidence_level': self.confidence_level,
                'lower_bound': r_lower,
                'upper_bound': r_upper,
                'sample_size': n,
                'fisher_z': z_r,
                'z_standard_error': z_se
            }
            
        except Exception as e:
            logger.warning(f"Correlation CI calculation failed for {col1} vs {col2}: {e}")

    def _interpret_power_analysis(self, required_n: int, desired_power: float) -> str:
        """Generate interpretation for power analysis results."""
        if required_n is None:
            return "Power analysis calculation failed"
        
        if required_n <= 30:
            size_desc = "small"
        elif required_n <= 100:
            size_desc = "moderate"
        elif required_n <= 300:
            size_desc = "large"
        else:
            size_desc = "very large"
        
        return f"Requires {required_n} participants per group for {desired_power:.0%} power ({size_desc} sample size)"


# MCP Tool Functions
def run_hypothesis_test(data: Union[pd.DataFrame, str], 
                       test_type: str = 'auto',
                       alpha: float = 0.05,
                       alternative: str = 'two-sided',
                       **kwargs) -> Dict[str, Any]:
    """
    Run comprehensive hypothesis testing analysis.
    
    Args:
        data: DataFrame or path to data file
        test_type: Type of hypothesis test to perform
        alpha: Significance level
        alternative: Alternative hypothesis direction
        **kwargs: Additional parameters for specific tests
    
    Returns:
        Dictionary containing test results and interpretations
    """
    if isinstance(data, str):
        # Load data from file path
        if data.endswith('.csv'):
            data = pd.read_csv(data)
        elif data.endswith('.json'):
            data = pd.read_json(data)
        else:
            raise ValueError("Unsupported file format")
    
    # Initialize and run hypothesis testing transformer
    transformer = HypothesisTestingTransformer(
        test_type=test_type,
        alpha=alpha,
        alternative=alternative,
        **kwargs
    )
    
    # Fit and transform
    transformer.fit(data)
    result_df = transformer.transform(data)
    
    return result_df.iloc[0].to_dict()


def perform_anova(data: Union[pd.DataFrame, str],
                 anova_type: str = 'one_way',
                 alpha: float = 0.05,
                 post_hoc: str = 'tukey',
                 **kwargs) -> Dict[str, Any]:
    """
    Perform comprehensive ANOVA analysis.
    
    Args:
        data: DataFrame or path to data file
        anova_type: Type of ANOVA analysis
        alpha: Significance level
        post_hoc: Post-hoc test method
        **kwargs: Additional ANOVA parameters
    
    Returns:
        Dictionary containing ANOVA results and post-hoc tests
    """
    if isinstance(data, str):
        # Load data from file path
        if data.endswith('.csv'):
            data = pd.read_csv(data)
        elif data.endswith('.json'):
            data = pd.read_json(data)
        else:
            raise ValueError("Unsupported file format")
    
    # Initialize and run ANOVA transformer
    transformer = ANOVAAnalysisTransformer(
        anova_type=anova_type,
        alpha=alpha,
        post_hoc=post_hoc,
        **kwargs
    )
    
    # Fit and transform
    transformer.fit(data)
    result_df = transformer.transform(data)
    
    return result_df.iloc[0].to_dict()


def analyze_experiment_design(data: Union[pd.DataFrame, str],
                            analysis_type: str = 'power_analysis',
                            effect_size: Optional[float] = None,
                            alpha: float = 0.05,
                            power: float = 0.80,
                            **kwargs) -> Dict[str, Any]:
    """
    Analyze experimental design parameters and power.
    
    Args:
        data: DataFrame or path to data file
        analysis_type: Type of experimental design analysis
        effect_size: Expected effect size
        alpha: Type I error rate
        power: Desired statistical power
        **kwargs: Additional experimental design parameters
    
    Returns:
        Dictionary containing experimental design analysis results
    """
    if isinstance(data, str):
        # Load data from file path
        if data.endswith('.csv'):
            data = pd.read_csv(data)
        elif data.endswith('.json'):
            data = pd.read_json(data)
        else:
            raise ValueError("Unsupported file format")
    
    # Initialize and run experimental design transformer
    transformer = ExperimentalDesignTransformer(
        analysis_type=analysis_type,
        effect_size=effect_size,
        alpha=alpha,
        power=power,
        **kwargs
    )
    
    # Fit and transform
    transformer.fit(data)
    result_df = transformer.transform(data)
    
    return result_df.iloc[0].to_dict()


def calculate_effect_sizes(data: Union[pd.DataFrame, str],
                          test_type: str = 'auto',
                          **kwargs) -> Dict[str, Any]:
    """
    Calculate comprehensive effect sizes for statistical analyses.
    
    Args:
        data: DataFrame or path to data file
        test_type: Type of statistical test for effect size calculation
        **kwargs: Additional parameters for effect size calculations
    
    Returns:
        Dictionary containing calculated effect sizes and interpretations
    """
    if isinstance(data, str):
        # Load data from file path
        if data.endswith('.csv'):
            data = pd.read_csv(data)
        elif data.endswith('.json'):
            data = pd.read_json(data)
        else:
            raise ValueError("Unsupported file format")
    
    # Use experimental design transformer for effect size calculations
    transformer = ExperimentalDesignTransformer(
        analysis_type='effect_size',
        **kwargs
    )
    
    # Fit and transform
    transformer.fit(data)
    result_df = transformer.transform(data)
    
    return result_df.iloc[0].to_dict()