"""
Comprehensive tests for the Statistical Analysis Domain.

Tests validate statistical computations against known reference datasets,
edge cases, and cross-validation with established statistical methods.
"""

import pytest
import numpy as np
import pandas as pd
from scipy import stats
import warnings
from unittest.mock import patch
from typing import Dict, Any

from src.localdata_mcp.domains.statistical_analysis import (
    HypothesisTestingTransformer,
    ANOVAAnalysisTransformer, 
    NonParametricTestTransformer,
    ExperimentalDesignTransformer,
    StatisticalTestResult,
    run_hypothesis_test,
    perform_anova,
    analyze_experiment_design,
    calculate_effect_sizes
)

# Suppress warnings during testing
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)


class TestStatisticalTestResult:
    """Test the StatisticalTestResult dataclass."""
    
    def test_basic_result_creation(self):
        """Test basic result creation and to_dict conversion."""
        result = StatisticalTestResult(
            test_name="Test t-test",
            statistic=2.45,
            p_value=0.014,
            interpretation="Significant result"
        )
        
        assert result.test_name == "Test t-test"
        assert result.statistic == 2.45
        assert result.p_value == 0.014
        assert result.interpretation == "Significant result"
        
        result_dict = result.to_dict()
        assert result_dict['test_name'] == "Test t-test"
        assert result_dict['statistic'] == 2.45
        assert result_dict['p_value'] == 0.014
        
    def test_complete_result_creation(self):
        """Test result creation with all fields."""
        result = StatisticalTestResult(
            test_name="Complete test",
            statistic=1.96,
            p_value=0.05,
            degrees_of_freedom=15,
            effect_size=0.5,
            confidence_interval=(0.1, 0.9),
            interpretation="Medium effect",
            assumptions_met={"normality": True, "homoscedasticity": False},
            additional_info={"sample_size": 30}
        )
        
        result_dict = result.to_dict()
        assert result_dict['degrees_of_freedom'] == 15
        assert result_dict['effect_size'] == 0.5
        assert result_dict['confidence_interval'] == (0.1, 0.9)
        assert result_dict['assumptions_met']['normality'] is True
        assert result_dict['additional_info']['sample_size'] == 30


class TestHypothesisTestingTransformer:
    """Test the HypothesisTestingTransformer class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        data = {
            'numeric1': np.random.normal(0, 1, 100),
            'numeric2': np.random.normal(0.5, 1.2, 100),  # Different distribution
            'numeric3': np.random.normal(0, 1, 100),
            'categorical1': np.random.choice(['A', 'B'], 100),
            'categorical2': np.random.choice(['X', 'Y', 'Z'], 100)
        }
        return pd.DataFrame(data)
    
    @pytest.fixture
    def small_sample_data(self):
        """Create small sample data for edge case testing."""
        return pd.DataFrame({
            'numeric1': [1, 2, 3],
            'numeric2': [4, 5, 6],
            'categorical1': ['A', 'A', 'B']
        })
    
    def test_transformer_initialization(self):
        """Test transformer initialization with various parameters."""
        transformer = HypothesisTestingTransformer()
        assert transformer.test_type == 'auto'
        assert transformer.alpha == 0.05
        assert transformer.alternative == 'two-sided'
        
        transformer = HypothesisTestingTransformer(
            test_type='ttest_ind',
            alpha=0.01,
            alternative='greater',
            equal_var=False
        )
        assert transformer.test_type == 'ttest_ind'
        assert transformer.alpha == 0.01
        assert transformer.alternative == 'greater'
        assert transformer.equal_var is False
    
    def test_parameter_validation(self):
        """Test parameter validation."""
        with pytest.raises(ValueError, match="test_type must be one of"):
            HypothesisTestingTransformer(test_type='invalid_test')
        
        with pytest.raises(ValueError, match="alpha must be between 0 and 1"):
            HypothesisTestingTransformer(alpha=1.5)
        
        with pytest.raises(ValueError, match="alternative must be"):
            HypothesisTestingTransformer(alternative='invalid')
    
    def test_automatic_testing_mode(self, sample_data):
        """Test automatic test selection and execution."""
        transformer = HypothesisTestingTransformer(test_type='auto')
        transformer.fit(sample_data)
        result_df = transformer.transform(sample_data)
        
        assert not result_df.empty
        results = result_df.iloc[0]
        
        # Should have performed multiple tests
        assert len(results['test_results']) > 0
        
        # Check that normality tests were performed
        normality_tests = [r for r in results['test_results'] 
                          if 'Shapiro-Wilk' in r['test_name'] or 'Kolmogorov-Smirnov' in r['test_name']]
        assert len(normality_tests) > 0
        
        # Check that correlation tests were performed
        correlation_tests = [r for r in results['test_results'] 
                           if 'Correlation' in r['test_name']]
        assert len(correlation_tests) > 0
    
    def test_normality_testing(self, sample_data):
        """Test normality testing specifically."""
        transformer = HypothesisTestingTransformer(test_type='normality')
        transformer.fit(sample_data)
        result_df = transformer.transform(sample_data)
        
        results = result_df.iloc[0]
        normality_tests = [r for r in results['test_results'] 
                          if 'Shapiro-Wilk' in r['test_name'] or 'Kolmogorov-Smirnov' in r['test_name']]
        
        # Should have normality tests for each numeric column
        assert len(normality_tests) >= 3  # At least one test per numeric column
        
        # Verify test structure
        for test in normality_tests:
            assert 'test_name' in test
            assert 'statistic' in test
            assert 'p_value' in test
            assert 'interpretation' in test
            assert test['p_value'] >= 0 and test['p_value'] <= 1
    
    def test_correlation_testing(self, sample_data):
        """Test correlation testing."""
        transformer = HypothesisTestingTransformer(test_type='correlation')
        transformer.fit(sample_data)
        result_df = transformer.transform(sample_data)
        
        results = result_df.iloc[0]
        correlation_tests = [r for r in results['test_results'] 
                           if 'Correlation' in r['test_name']]
        
        # Should have correlation tests between numeric columns
        assert len(correlation_tests) > 0
        
        # Verify correlation test structure
        for test in correlation_tests:
            assert test['p_value'] >= 0 and test['p_value'] <= 1
            assert 'effect_size' in test
            assert test['effect_size'] >= 0 and test['effect_size'] <= 1
            assert 'additional_info' in test
            assert 'column1' in test['additional_info']
            assert 'column2' in test['additional_info']
    
    def test_chi_square_testing(self, sample_data):
        """Test chi-square testing."""
        transformer = HypothesisTestingTransformer(test_type='chi2')
        transformer.fit(sample_data)
        result_df = transformer.transform(sample_data)
        
        results = result_df.iloc[0]
        chi2_tests = [r for r in results['test_results'] 
                     if 'Chi-square' in r['test_name']]
        
        # Should have chi-square tests between categorical columns
        assert len(chi2_tests) > 0
        
        # Verify chi-square test structure
        for test in chi2_tests:
            assert test['p_value'] >= 0 and test['p_value'] <= 1
            assert 'degrees_of_freedom' in test
            assert 'effect_size' in test  # Cramer's V
            assert test['effect_size'] >= 0 and test['effect_size'] <= 1
    
    def test_one_sample_ttest(self):
        """Test one-sample t-test with known data."""
        # Create data with known mean different from 0
        np.random.seed(42)
        data = pd.DataFrame({'values': np.random.normal(2.0, 1.0, 50)})
        
        transformer = HypothesisTestingTransformer(test_type='ttest_1samp')
        transformer.fit(data)
        result_df = transformer.transform(data)
        
        results = result_df.iloc[0]
        ttest_results = [r for r in results['test_results'] 
                        if 'One-sample t-test' in r['test_name']]
        
        assert len(ttest_results) > 0
        
        # Verify t-test results
        ttest = ttest_results[0]
        assert ttest['p_value'] < 0.05  # Should be significant (mean != 0)
        assert 'effect_size' in ttest
        assert 'degrees_of_freedom' in ttest
        assert ttest['degrees_of_freedom'] == 49  # n-1
    
    def test_independent_ttest(self):
        """Test independent samples t-test."""
        np.random.seed(42)
        data = pd.DataFrame({
            'values': np.concatenate([
                np.random.normal(0, 1, 25),    # Group A
                np.random.normal(1, 1, 25)     # Group B (different mean)
            ]),
            'group': ['A'] * 25 + ['B'] * 25
        })
        
        transformer = HypothesisTestingTransformer(test_type='ttest_ind')
        transformer.fit(data)
        result_df = transformer.transform(data)
        
        results = result_df.iloc[0]
        ttest_results = [r for r in results['test_results'] 
                        if 'Independent t-test' in r['test_name']]
        
        assert len(ttest_results) > 0
        
        # Verify independent t-test results
        ttest = ttest_results[0]
        assert 'effect_size' in ttest
        assert 'additional_info' in ttest
        assert 'group1_mean' in ttest['additional_info']
        assert 'group2_mean' in ttest['additional_info']
        assert 'cohens_d' in ttest['additional_info']
    
    def test_paired_ttest(self, sample_data):
        """Test paired samples t-test."""
        transformer = HypothesisTestingTransformer(test_type='ttest_rel')
        transformer.fit(sample_data)
        result_df = transformer.transform(sample_data)
        
        results = result_df.iloc[0]
        paired_tests = [r for r in results['test_results'] 
                       if 'Paired t-test' in r['test_name']]
        
        assert len(paired_tests) > 0
        
        # Verify paired t-test structure
        for test in paired_tests:
            assert 'effect_size' in test
            assert 'additional_info' in test
            assert 'mean_difference' in test['additional_info']
            assert 'cohens_d' in test['additional_info']
    
    def test_small_sample_handling(self, small_sample_data):
        """Test handling of small sample sizes."""
        transformer = HypothesisTestingTransformer(test_type='auto')
        transformer.fit(small_sample_data)
        result_df = transformer.transform(small_sample_data)
        
        # Should not crash with small samples
        assert not result_df.empty
        results = result_df.iloc[0]
        
        # May have fewer or no tests due to sample size constraints
        assert isinstance(results['test_results'], list)
    
    def test_effect_size_interpretations(self, sample_data):
        """Test that effect sizes are properly interpreted."""
        transformer = HypothesisTestingTransformer(test_type='auto')
        transformer.fit(sample_data)
        result_df = transformer.transform(sample_data)
        
        results = result_df.iloc[0]
        
        # Check that effect size interpretations are provided
        for test_result in results['test_results']:
            if 'effect_size' in test_result:
                assert 'additional_info' in test_result
                if 'effect_description' in test_result['additional_info']:
                    effect_desc = test_result['additional_info']['effect_description']
                    assert effect_desc in ['negligible', 'small', 'medium', 'large']


class TestANOVAAnalysisTransformer:
    """Test the ANOVAAnalysisTransformer class."""
    
    @pytest.fixture
    def anova_data(self):
        """Create sample data suitable for ANOVA."""
        np.random.seed(42)
        
        # Create data with group differences
        group_a = np.random.normal(0, 1, 30)
        group_b = np.random.normal(1, 1, 30) 
        group_c = np.random.normal(2, 1, 30)
        
        data = pd.DataFrame({
            'values': np.concatenate([group_a, group_b, group_c]),
            'group': ['A'] * 30 + ['B'] * 30 + ['C'] * 30,
            'factor2': np.random.choice(['X', 'Y'], 90)
        })
        
        return data
    
    def test_transformer_initialization(self):
        """Test ANOVA transformer initialization."""
        transformer = ANOVAAnalysisTransformer()
        assert transformer.anova_type == 'one_way'
        assert transformer.alpha == 0.05
        assert transformer.post_hoc == 'tukey'
        
        transformer = ANOVAAnalysisTransformer(
            anova_type='two_way',
            alpha=0.01,
            post_hoc='bonferroni'
        )
        assert transformer.anova_type == 'two_way'
        assert transformer.alpha == 0.01
        assert transformer.post_hoc == 'bonferroni'
    
    def test_parameter_validation(self):
        """Test ANOVA parameter validation."""
        with pytest.raises(ValueError, match="anova_type must be one of"):
            ANOVAAnalysisTransformer(anova_type='invalid_type')
        
        with pytest.raises(ValueError, match="alpha must be between 0 and 1"):
            ANOVAAnalysisTransformer(alpha=1.5)
        
        with pytest.raises(ValueError, match="post_hoc must be one of"):
            ANOVAAnalysisTransformer(post_hoc='invalid_method')
    
    def test_one_way_anova(self, anova_data):
        """Test one-way ANOVA analysis."""
        transformer = ANOVAAnalysisTransformer(anova_type='one_way')
        transformer.fit(anova_data)
        result_df = transformer.transform(anova_data)
        
        results = result_df.iloc[0]
        anova_results = results['anova_results']
        
        # Should have one-way ANOVA results
        assert len(anova_results) > 0
        
        # Check structure of ANOVA results
        for key, result in anova_results.items():
            if 'one_way' in key:
                assert result['test_type'] == 'One-way ANOVA'
                assert 'f_statistic' in result
                assert 'p_value' in result
                assert 'df_between' in result
                assert 'df_within' in result
                assert 'group_means' in result
                assert 'group_sizes' in result
                assert isinstance(result['significant'], bool)
    
    def test_two_way_anova(self, anova_data):
        """Test two-way ANOVA analysis."""
        transformer = ANOVAAnalysisTransformer(anova_type='two_way')
        transformer.fit(anova_data)
        result_df = transformer.transform(anova_data)
        
        results = result_df.iloc[0]
        anova_results = results['anova_results']
        
        # Should have two-way ANOVA results
        two_way_results = [k for k in anova_results.keys() if 'two_way' in k]
        assert len(two_way_results) > 0
        
        # Check structure of two-way ANOVA results
        for key in two_way_results:
            result = anova_results[key]
            assert result['test_type'] == 'Two-way ANOVA'
            assert 'anova_table' in result
            assert 'model_summary' in result
            assert 'r_squared' in result['model_summary']
    
    def test_effect_size_calculations(self, anova_data):
        """Test effect size calculations for ANOVA."""
        transformer = ANOVAAnalysisTransformer(anova_type='one_way')
        transformer.fit(anova_data)
        result_df = transformer.transform(anova_data)
        
        results = result_df.iloc[0]
        effect_sizes = results['effect_sizes']
        
        # Should have effect size calculations
        assert len(effect_sizes) > 0
        
        for key, effects in effect_sizes.items():
            assert 'eta_squared' in effects
            assert 'omega_squared' in effects
            assert 'effect_description' in effects
            
            # Effect sizes should be between 0 and 1
            assert 0 <= effects['eta_squared'] <= 1
            assert 0 <= effects['omega_squared'] <= 1
            
            # Effect description should be valid
            assert effects['effect_description'] in ['negligible', 'small', 'medium', 'large']
    
    def test_post_hoc_analysis(self, anova_data):
        """Test post-hoc analysis when ANOVA is significant."""
        # Ensure we have significant group differences
        np.random.seed(42)
        group_a = np.random.normal(0, 0.5, 20)
        group_b = np.random.normal(2, 0.5, 20) 
        group_c = np.random.normal(4, 0.5, 20)
        
        significant_data = pd.DataFrame({
            'values': np.concatenate([group_a, group_b, group_c]),
            'group': ['A'] * 20 + ['B'] * 20 + ['C'] * 20
        })
        
        transformer = ANOVAAnalysisTransformer(anova_type='one_way', post_hoc='tukey')
        transformer.fit(significant_data)
        result_df = transformer.transform(significant_data)
        
        results = result_df.iloc[0]
        post_hoc_results = results['post_hoc_results']
        
        if len(post_hoc_results) > 0:
            # Check post-hoc analysis structure
            for key, post_hoc in post_hoc_results.items():
                assert post_hoc['method'] == 'Tukey HSD'
                assert 'comparisons' in post_hoc
                assert isinstance(post_hoc['comparisons'], list)
                
                for comparison in post_hoc['comparisons']:
                    assert 'group1' in comparison
                    assert 'group2' in comparison
                    assert 'mean_diff' in comparison
                    assert 'p_value' in comparison
                    assert 'significant' in comparison
    
    def test_assumption_checking(self, anova_data):
        """Test ANOVA assumption checking."""
        transformer = ANOVAAnalysisTransformer(anova_type='one_way', check_assumptions=True)
        transformer.fit(anova_data)
        result_df = transformer.transform(anova_data)
        
        results = result_df.iloc[0]
        assumptions = results['assumptions_checked']
        
        # Should have assumption checks
        assert len(assumptions) > 0
        
        for test_name, assumptions_dict in assumptions.items():
            if assumptions_dict:
                assert 'normality' in assumptions_dict
                assert 'homoscedasticity' in assumptions_dict
                
                # Assumption results should be boolean or None
                for assumption, result in assumptions_dict.items():
                    assert result is None or isinstance(result, bool)
    
    def test_insufficient_data_handling(self):
        """Test handling of insufficient data for ANOVA."""
        small_data = pd.DataFrame({
            'values': [1, 2, 3, 4],
            'group': ['A', 'A', 'B', 'B']
        })
        
        transformer = ANOVAAnalysisTransformer(anova_type='one_way')
        transformer.fit(small_data)
        result_df = transformer.transform(small_data)
        
        # Should not crash with insufficient data
        assert not result_df.empty
        results = result_df.iloc[0]
        
        # May have no ANOVA results due to insufficient data
        assert isinstance(results['anova_results'], dict)


class TestNonParametricTestTransformer:
    """Test the NonParametricTestTransformer class."""
    
    @pytest.fixture
    def nonparametric_data(self):
        """Create sample data for non-parametric testing."""
        np.random.seed(42)
        
        # Create non-normal data
        data = pd.DataFrame({
            'skewed1': np.random.exponential(2, 100),  # Right-skewed
            'skewed2': np.random.exponential(1.5, 100),
            'ordinal': np.random.choice([1, 2, 3, 4, 5], 100),
            'binary_group': np.random.choice(['Group1', 'Group2'], 100),
            'multi_group': np.random.choice(['A', 'B', 'C'], 100)
        })
        
        return data
    
    def test_transformer_initialization(self):
        """Test non-parametric transformer initialization."""
        transformer = NonParametricTestTransformer()
        assert transformer.test_type == 'auto'
        assert transformer.alpha == 0.05
        assert transformer.alternative == 'two-sided'
        
        transformer = NonParametricTestTransformer(
            test_type='mann_whitney',
            alpha=0.01,
            alternative='greater'
        )
        assert transformer.test_type == 'mann_whitney'
        assert transformer.alpha == 0.01
        assert transformer.alternative == 'greater'
    
    def test_parameter_validation(self):
        """Test non-parametric parameter validation."""
        with pytest.raises(ValueError, match="test_type must be one of"):
            NonParametricTestTransformer(test_type='invalid_test')
        
        with pytest.raises(ValueError, match="alpha must be between 0 and 1"):
            NonParametricTestTransformer(alpha=2.0)
    
    def test_mann_whitney_test(self, nonparametric_data):
        """Test Mann-Whitney U test."""
        transformer = NonParametricTestTransformer(test_type='mann_whitney')
        transformer.fit(nonparametric_data)
        result_df = transformer.transform(nonparametric_data)
        
        results = result_df.iloc[0]
        test_results = results['test_results']
        
        # Should have Mann-Whitney tests
        mann_whitney_tests = [r for r in test_results 
                             if 'Mann-Whitney U' in r['test_name']]
        assert len(mann_whitney_tests) > 0
        
        # Verify test structure
        for test in mann_whitney_tests:
            assert 'statistic' in test
            assert 'p_value' in test
            assert 'effect_size' in test  # rank-biserial correlation
            assert 0 <= test['effect_size'] <= 1
            assert 'additional_info' in test
            assert 'rank_biserial_correlation' in test['additional_info']
    
    def test_wilcoxon_test(self, nonparametric_data):
        """Test Wilcoxon signed-rank test."""
        transformer = NonParametricTestTransformer(test_type='wilcoxon')
        transformer.fit(nonparametric_data)
        result_df = transformer.transform(nonparametric_data)
        
        results = result_df.iloc[0]
        test_results = results['test_results']
        
        # Should have Wilcoxon tests
        wilcoxon_tests = [r for r in test_results 
                         if 'Wilcoxon Signed-Rank' in r['test_name']]
        assert len(wilcoxon_tests) > 0
        
        # Verify test structure
        for test in wilcoxon_tests:
            assert 'statistic' in test
            assert 'p_value' in test
            assert 'effect_size' in test
            assert 'additional_info' in test
            assert 'positive_ranks' in test['additional_info']
            assert 'negative_ranks' in test['additional_info']
    
    def test_kruskal_wallis_test(self, nonparametric_data):
        """Test Kruskal-Wallis H test."""
        transformer = NonParametricTestTransformer(test_type='kruskal_wallis')
        transformer.fit(nonparametric_data)
        result_df = transformer.transform(nonparametric_data)
        
        results = result_df.iloc[0]
        test_results = results['test_results']
        
        # Should have Kruskal-Wallis tests
        kw_tests = [r for r in test_results 
                   if 'Kruskal-Wallis H' in r['test_name']]
        assert len(kw_tests) > 0
        
        # Verify test structure
        for test in kw_tests:
            assert 'statistic' in test
            assert 'p_value' in test
            assert 'degrees_of_freedom' in test
            assert 'effect_size' in test  # eta-squared analog
            assert test['effect_size'] >= 0
    
    def test_friedman_test(self):
        """Test Friedman test for repeated measures."""
        # Create repeated measures data
        np.random.seed(42)
        n_subjects = 20
        data = pd.DataFrame({
            'measure1': np.random.exponential(1, n_subjects),
            'measure2': np.random.exponential(1.5, n_subjects),
            'measure3': np.random.exponential(2, n_subjects),
            'measure4': np.random.exponential(1.2, n_subjects)
        })
        
        transformer = NonParametricTestTransformer(test_type='friedman')
        transformer.fit(data)
        result_df = transformer.transform(data)
        
        results = result_df.iloc[0]
        test_results = results['test_results']
        
        # Should have Friedman test
        friedman_tests = [r for r in test_results 
                         if 'Friedman Test' in r['test_name']]
        assert len(friedman_tests) > 0
        
        # Verify test structure
        for test in friedman_tests:
            assert 'statistic' in test
            assert 'p_value' in test
            assert 'degrees_of_freedom' in test
            assert 'effect_size' in test  # Kendall's W
            assert 0 <= test['effect_size'] <= 1
            assert 'additional_info' in test
            assert 'kendalls_w' in test['additional_info']
    
    def test_automatic_test_selection(self, nonparametric_data):
        """Test automatic test selection."""
        transformer = NonParametricTestTransformer(test_type='auto')
        transformer.fit(nonparametric_data)
        result_df = transformer.transform(nonparametric_data)
        
        results = result_df.iloc[0]
        test_results = results['test_results']
        
        # Should perform multiple appropriate tests
        assert len(test_results) > 0
        
        # Should include Mann-Whitney for binary groups
        mann_whitney_tests = [r for r in test_results 
                             if 'Mann-Whitney U' in r['test_name']]
        assert len(mann_whitney_tests) > 0
        
        # Should include Kruskal-Wallis for multi-group
        kw_tests = [r for r in test_results 
                   if 'Kruskal-Wallis H' in r['test_name']]
        assert len(kw_tests) > 0
    
    def test_effect_size_calculations(self, nonparametric_data):
        """Test effect size calculations for non-parametric tests."""
        transformer = NonParametricTestTransformer(test_type='auto', calculate_effect_size=True)
        transformer.fit(nonparametric_data)
        result_df = transformer.transform(nonparametric_data)
        
        results = result_df.iloc[0]
        effect_sizes = results['effect_sizes']
        test_results = results['test_results']
        
        # Effect sizes should be calculated for tests
        for test in test_results:
            if 'effect_size' in test:
                assert test['effect_size'] >= 0
                if 'additional_info' in test and 'effect_description' in test['additional_info']:
                    effect_desc = test['additional_info']['effect_description']
                    assert effect_desc in ['negligible', 'small', 'medium', 'large']


class TestExperimentalDesignTransformer:
    """Test the ExperimentalDesignTransformer class."""
    
    @pytest.fixture
    def design_data(self):
        """Create sample data for experimental design analysis."""
        np.random.seed(42)
        return pd.DataFrame({
            'outcome': np.random.normal(0, 1, 100),
            'group': np.random.choice(['Control', 'Treatment'], 100),
            'covariate': np.random.normal(0, 1, 100)
        })
    
    def test_transformer_initialization(self):
        """Test experimental design transformer initialization."""
        transformer = ExperimentalDesignTransformer()
        assert transformer.analysis_type == 'power_analysis'
        assert transformer.alpha == 0.05
        assert transformer.power == 0.80
        assert transformer.test_type == 'ttest'
        
        transformer = ExperimentalDesignTransformer(
            analysis_type='sample_size',
            effect_size=0.5,
            alpha=0.01,
            power=0.90,
            test_type='anova'
        )
        assert transformer.analysis_type == 'sample_size'
        assert transformer.effect_size == 0.5
        assert transformer.alpha == 0.01
        assert transformer.power == 0.90
        assert transformer.test_type == 'anova'
    
    def test_parameter_validation(self):
        """Test experimental design parameter validation."""
        with pytest.raises(ValueError, match="alpha must be between 0 and 1"):
            ExperimentalDesignTransformer(alpha=1.5)
        
        with pytest.raises(ValueError, match="power must be between 0 and 1"):
            ExperimentalDesignTransformer(power=1.2)
        
        with pytest.raises(ValueError, match="confidence_level must be between 0 and 1"):
            ExperimentalDesignTransformer(confidence_level=1.5)
        
        with pytest.raises(ValueError, match="test_type must be one of"):
            ExperimentalDesignTransformer(test_type='invalid_test')
    
    def test_power_analysis_ttest(self, design_data):
        """Test power analysis for t-tests."""
        transformer = ExperimentalDesignTransformer(
            analysis_type='power_analysis', 
            test_type='ttest',
            effect_size=0.5
        )
        transformer.fit(design_data)
        result_df = transformer.transform(design_data)
        
        results = result_df.iloc[0]
        power_analysis = results['power_analysis']
        
        # Should have t-test power analysis
        assert 'ttest' in power_analysis
        ttest_power = power_analysis['ttest']
        
        assert 'effect_size' in ttest_power
        assert 'required_sample_size' in ttest_power
        assert 'power_curve' in ttest_power
        assert 'interpretation' in ttest_power
        
        # Power curve should have multiple sample sizes
        assert len(ttest_power['power_curve']) > 1
        
        # Sample size should be reasonable
        if ttest_power['required_sample_size']:
            assert ttest_power['required_sample_size'] > 0
    
    def test_power_analysis_anova(self, design_data):
        """Test power analysis for ANOVA."""
        transformer = ExperimentalDesignTransformer(
            analysis_type='power_analysis', 
            test_type='anova',
            effect_size=0.25
        )
        transformer.fit(design_data)
        result_df = transformer.transform(design_data)
        
        results = result_df.iloc[0]
        power_analysis = results['power_analysis']
        
        # Should have ANOVA power analysis
        assert 'anova' in power_analysis
        anova_power = power_analysis['anova']
        
        assert 'effect_size_eta_squared' in anova_power
        assert 'cohens_f' in anova_power
        assert 'num_groups' in anova_power
        assert 'power_curve' in anova_power
        
        # Cohen's f should be calculated correctly
        assert anova_power['cohens_f'] > 0
    
    def test_power_analysis_correlation(self, design_data):
        """Test power analysis for correlations."""
        transformer = ExperimentalDesignTransformer(
            analysis_type='power_analysis', 
            test_type='correlation',
            effect_size=0.3
        )
        transformer.fit(design_data)
        result_df = transformer.transform(design_data)
        
        results = result_df.iloc[0]
        power_analysis = results['power_analysis']
        
        # Should have correlation power analysis
        assert 'correlation' in power_analysis
        corr_power = power_analysis['correlation']
        
        assert 'correlation_coefficient' in corr_power
        assert 'required_sample_size' in corr_power
        assert 'power_curve' in corr_power
        
        # Required sample size should be reasonable
        if corr_power['required_sample_size']:
            assert corr_power['required_sample_size'] > 3
    
    def test_sample_size_calculation(self, design_data):
        """Test sample size calculations."""
        transformer = ExperimentalDesignTransformer(
            analysis_type='sample_size',
            test_type='ttest'
        )
        transformer.fit(design_data)
        result_df = transformer.transform(design_data)
        
        results = result_df.iloc[0]
        sample_sizes = results['sample_sizes']
        
        # Should have sample sizes for different effect sizes
        assert len(sample_sizes) > 0
        
        for key, n in sample_sizes.items():
            if n is not None:
                assert n > 0
                assert isinstance(n, int)
    
    def test_effect_size_calculation(self, design_data):
        """Test effect size calculations from data."""
        transformer = ExperimentalDesignTransformer(analysis_type='effect_size')
        transformer.fit(design_data)
        result_df = transformer.transform(design_data)
        
        results = result_df.iloc[0]
        effect_sizes = results['effect_sizes']
        
        # Should calculate effect sizes from data
        if len(effect_sizes) > 0:
            for key, effect in effect_sizes.items():
                if isinstance(effect, dict):
                    # Check effect size structure
                    if 'effect_description' in effect:
                        assert effect['effect_description'] in ['negligible', 'small', 'medium', 'large']
    
    def test_confidence_intervals(self, design_data):
        """Test confidence interval calculations."""
        transformer = ExperimentalDesignTransformer(
            analysis_type='confidence_intervals',
            confidence_level=0.95
        )
        transformer.fit(design_data)
        result_df = transformer.transform(design_data)
        
        results = result_df.iloc[0]
        confidence_intervals = results['confidence_intervals']
        
        # Should have confidence intervals
        if len(confidence_intervals) > 0:
            for key, ci in confidence_intervals.items():
                assert 'confidence_level' in ci
                assert ci['confidence_level'] == 0.95
                assert 'lower_bound' in ci
                assert 'upper_bound' in ci
                assert ci['lower_bound'] < ci['upper_bound']
    
    def test_comprehensive_analysis(self, design_data):
        """Test comprehensive analysis (all types)."""
        transformer = ExperimentalDesignTransformer(
            analysis_type='comprehensive',  # Invalid type triggers all analyses
            test_type='ttest'
        )
        transformer.fit(design_data)
        result_df = transformer.transform(design_data)
        
        results = result_df.iloc[0]
        
        # Should have results from multiple analysis types
        assert 'power_analysis' in results
        assert 'sample_sizes' in results
        assert 'effect_sizes' in results
        assert 'confidence_intervals' in results
        assert 'parameters' in results


class TestMCPToolFunctions:
    """Test the MCP tool functions."""
    
    @pytest.fixture
    def sample_data_df(self):
        """Create sample DataFrame for MCP tool testing."""
        np.random.seed(42)
        return pd.DataFrame({
            'numeric1': np.random.normal(0, 1, 50),
            'numeric2': np.random.normal(0.5, 1, 50),
            'categorical': np.random.choice(['A', 'B', 'C'], 50),
            'binary': np.random.choice(['X', 'Y'], 50)
        })
    
    def test_run_hypothesis_test_function(self, sample_data_df, tmp_path):
        """Test run_hypothesis_test MCP tool function."""
        # Test with DataFrame
        result = run_hypothesis_test(sample_data_df, test_type='auto', alpha=0.05)
        
        assert isinstance(result, dict)
        assert 'test_results' in result
        assert 'alpha_level' in result
        assert len(result['test_results']) > 0
        
        # Test with file path
        csv_file = tmp_path / "test_data.csv"
        sample_data_df.to_csv(csv_file, index=False)
        
        result = run_hypothesis_test(str(csv_file), test_type='normality')
        assert isinstance(result, dict)
        assert 'test_results' in result
    
    def test_perform_anova_function(self, sample_data_df, tmp_path):
        """Test perform_anova MCP tool function."""
        # Test with DataFrame
        result = perform_anova(sample_data_df, anova_type='one_way', alpha=0.05)
        
        assert isinstance(result, dict)
        assert 'anova_results' in result
        assert 'effect_sizes' in result
        assert 'alpha_level' in result
        
        # Test with JSON file
        json_file = tmp_path / "test_data.json"
        sample_data_df.to_json(json_file, orient='records')
        
        result = perform_anova(str(json_file), anova_type='auto')
        assert isinstance(result, dict)
    
    def test_analyze_experiment_design_function(self, sample_data_df):
        """Test analyze_experiment_design MCP tool function."""
        result = analyze_experiment_design(
            sample_data_df, 
            analysis_type='power_analysis',
            effect_size=0.5,
            power=0.8
        )
        
        assert isinstance(result, dict)
        assert 'power_analysis' in result
        assert 'parameters' in result
        assert result['parameters']['power'] == 0.8
        assert result['parameters']['effect_size'] == 0.5
    
    def test_calculate_effect_sizes_function(self, sample_data_df):
        """Test calculate_effect_sizes MCP tool function."""
        result = calculate_effect_sizes(sample_data_df, test_type='auto')
        
        assert isinstance(result, dict)
        assert 'effect_sizes' in result
    
    def test_unsupported_file_format(self, tmp_path):
        """Test handling of unsupported file formats."""
        unsupported_file = tmp_path / "test_data.txt"
        unsupported_file.write_text("some text")
        
        with pytest.raises(ValueError, match="Unsupported file format"):
            run_hypothesis_test(str(unsupported_file))


class TestIntegrationAndPerformance:
    """Integration tests and performance benchmarks."""
    
    def test_large_dataset_performance(self):
        """Test performance with larger datasets."""
        np.random.seed(42)
        
        # Create larger dataset
        large_data = pd.DataFrame({
            'numeric1': np.random.normal(0, 1, 1000),
            'numeric2': np.random.normal(0.3, 1.2, 1000),
            'categorical': np.random.choice(['A', 'B', 'C', 'D'], 1000),
            'binary': np.random.choice(['X', 'Y'], 1000)
        })
        
        # Test hypothesis testing performance
        import time
        start_time = time.time()
        
        transformer = HypothesisTestingTransformer(test_type='auto')
        transformer.fit(large_data)
        result = transformer.transform(large_data)
        
        execution_time = time.time() - start_time
        
        # Should complete within reasonable time (< 30 seconds)
        assert execution_time < 30
        assert not result.empty
    
    def test_statistical_accuracy_validation(self):
        """Validate statistical accuracy against known results."""
        # Create data with known statistical properties
        np.random.seed(42)
        
        # Two groups with known difference (Cohen's d ≈ 1.0)
        group1 = np.random.normal(0, 1, 50)
        group2 = np.random.normal(1, 1, 50)  # 1 SD difference
        
        data = pd.DataFrame({
            'values': np.concatenate([group1, group2]),
            'group': ['A'] * 50 + ['B'] * 50
        })
        
        # Test independent t-test
        transformer = HypothesisTestingTransformer(test_type='ttest_ind')
        transformer.fit(data)
        result = transformer.transform(data)
        
        results = result.iloc[0]
        ttest_results = [r for r in results['test_results'] 
                        if 'Independent t-test' in r['test_name']]
        
        assert len(ttest_results) > 0
        
        # Check that effect size is approximately 1.0 (large effect)
        ttest = ttest_results[0]
        effect_size = ttest['effect_size']
        
        # Should detect large effect size
        assert effect_size > 0.7  # Should be close to 1.0
        assert 'large' in ttest['additional_info']['effect_description']
    
    def test_edge_case_robustness(self):
        """Test robustness with edge cases."""
        # Test with missing values
        data_with_na = pd.DataFrame({
            'numeric1': [1, 2, np.nan, 4, 5],
            'numeric2': [np.nan, 2, 3, 4, np.nan],
            'categorical': ['A', 'B', None, 'A', 'B']
        })
        
        transformer = HypothesisTestingTransformer(test_type='auto')
        transformer.fit(data_with_na)
        result = transformer.transform(data_with_na)
        
        # Should handle missing values gracefully
        assert not result.empty
        
        # Test with single-valued columns
        constant_data = pd.DataFrame({
            'constant': [1, 1, 1, 1, 1],
            'variable': [1, 2, 3, 4, 5],
            'group': ['A', 'A', 'B', 'B', 'B']
        })
        
        transformer = HypothesisTestingTransformer(test_type='auto')
        transformer.fit(constant_data)
        result = transformer.transform(constant_data)
        
        # Should handle constant data without crashing
        assert not result.empty
    
    def test_memory_efficiency(self):
        """Test memory efficiency with streaming-like processing."""
        # This test ensures the transformers don't hold excessive memory
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Process multiple datasets
        for i in range(10):
            np.random.seed(i)
            data = pd.DataFrame({
                'numeric1': np.random.normal(0, 1, 500),
                'numeric2': np.random.normal(0, 1, 500),
                'categorical': np.random.choice(['A', 'B', 'C'], 500)
            })
            
            transformer = HypothesisTestingTransformer(test_type='auto')
            transformer.fit(data)
            result = transformer.transform(data)
            
            # Force garbage collection
            del transformer, result, data
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (< 100 MB for this test)
        assert memory_increase < 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])