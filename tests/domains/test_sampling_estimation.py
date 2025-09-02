"""
Tests for Sampling & Estimation Domain.

This module contains comprehensive tests for all sampling techniques,
bootstrap methods, Monte Carlo simulations, and Bayesian estimation
capabilities in the sampling_estimation domain.
"""

import pytest
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.utils.validation import check_is_fitted

from src.localdata_mcp.domains.sampling_estimation import (
    SamplingTransformer,
    BootstrapTransformer, 
    MonteCarloTransformer,
    BayesianEstimationTransformer,
    SamplingResult,
    BootstrapResult,
    MonteCarloResult,
    BayesianResult,
    generate_sample,
    bootstrap_statistic,
    monte_carlo_simulate,
    bayesian_estimate
)


class TestSamplingResult:
    """Test SamplingResult dataclass."""
    
    def test_sampling_result_creation(self):
        """Test SamplingResult creation and to_dict conversion."""
        result = SamplingResult(
            sampling_method='simple_random',
            sample_size=100,
            population_size=1000,
            sample_indices=np.array([1, 2, 3, 4, 5]),
            representativeness_score=0.95,
            coverage_metrics={'mean_diff': 0.02},
            strata_info={'A': {'size': 50}, 'B': {'size': 50}}
        )
        
        result_dict = result.to_dict()
        
        assert result_dict['sampling_method'] == 'simple_random'
        assert result_dict['sample_size'] == 100
        assert result_dict['population_size'] == 1000
        assert result_dict['sample_indices'] == [1, 2, 3, 4, 5]
        assert 'quality_metrics' in result_dict
        assert 'strata_info' in result_dict


class TestBootstrapResult:
    """Test BootstrapResult dataclass."""
    
    def test_bootstrap_result_creation(self):
        """Test BootstrapResult creation and to_dict conversion."""
        bootstrap_dist = np.random.normal(5, 1, 1000)
        
        result = BootstrapResult(
            statistic_name='mean_test',
            original_statistic=5.0,
            bootstrap_method='percentile',
            n_bootstrap=1000,
            bootstrap_distribution=bootstrap_dist,
            confidence_intervals={'95%': (4.8, 5.2)},
            bias_estimate=0.01,
            variance_estimate=0.05
        )
        
        result_dict = result.to_dict()
        
        assert result_dict['statistic_name'] == 'mean_test'
        assert result_dict['original_statistic'] == 5.0
        assert result_dict['bootstrap_method'] == 'percentile'
        assert result_dict['bias_estimate'] == 0.01
        assert '95%' in result_dict['confidence_intervals']


class TestMonteCarloResult:
    """Test MonteCarloResult dataclass."""
    
    def test_monte_carlo_result_creation(self):
        """Test MonteCarloResult creation and to_dict conversion."""
        simulation_results = np.random.normal(0, 1, 10000)
        
        result = MonteCarloResult(
            simulation_type='integration',
            n_simulations=10000,
            simulation_results=simulation_results,
            estimated_value=0.68,
            confidence_interval=(0.65, 0.71),
            standard_error=0.01,
            integration_bounds=(-1, 1),
            convergence_diagnostic={'relative_error': 0.001}
        )
        
        result_dict = result.to_dict()
        
        assert result_dict['simulation_type'] == 'integration'
        assert result_dict['n_simulations'] == 10000
        assert result_dict['estimated_value'] == 0.68
        assert result_dict['integration_bounds'] == (-1, 1)
        assert 'convergence_diagnostic' in result_dict


class TestBayesianResult:
    """Test BayesianResult dataclass."""
    
    def test_bayesian_result_creation(self):
        """Test BayesianResult creation and to_dict conversion."""
        posterior_samples = np.random.normal(5, 1, 1000)
        
        result = BayesianResult(
            parameter_name='mean_param',
            estimation_method='conjugate',
            posterior_samples=posterior_samples,
            posterior_mean=5.0,
            posterior_median=4.98,
            credible_intervals={'95%': (3.2, 6.8)},
            bayes_factor=2.5,
            prior_info={'distribution': 'normal', 'mu': 0, 'sigma': 10}
        )
        
        result_dict = result.to_dict()
        
        assert result_dict['parameter_name'] == 'mean_param'
        assert result_dict['estimation_method'] == 'conjugate'
        assert result_dict['posterior_mean'] == 5.0
        assert result_dict['bayes_factor'] == 2.5
        assert 'prior_info' in result_dict


class TestSamplingTransformer:
    """Test SamplingTransformer class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        return pd.DataFrame({
            'numeric_col': np.random.normal(100, 15, 1000),
            'category_col': np.random.choice(['A', 'B', 'C'], 1000),
            'weight_col': np.random.exponential(1, 1000),
            'cluster_col': np.random.choice([1, 2, 3, 4, 5], 1000)
        })
    
    def test_simple_random_sampling(self, sample_data):
        """Test simple random sampling."""
        transformer = SamplingTransformer(
            sampling_method='simple_random',
            sample_size=100,
            random_state=42
        )
        
        transformer.fit(sample_data)
        sample = transformer.transform(sample_data)
        
        assert len(sample) == 100
        assert hasattr(transformer, 'sampling_result_')
        assert transformer.sampling_result_.sampling_method == 'simple_random'
        assert transformer.sampling_result_.sample_size == 100
        assert transformer.sampling_result_.population_size == 1000
    
    def test_stratified_sampling(self, sample_data):
        """Test stratified sampling."""
        transformer = SamplingTransformer(
            sampling_method='stratified',
            sample_size=150,
            stratify_column='category_col',
            random_state=42
        )
        
        transformer.fit(sample_data)
        sample = transformer.transform(sample_data)
        
        assert len(sample) <= 150  # May be slightly less due to allocation
        assert hasattr(transformer, 'sampling_result_')
        assert transformer.sampling_result_.sampling_method == 'stratified'
        assert len(transformer.sampling_result_.strata_info) > 0
        
        # Check that all categories are represented
        sample_categories = set(sample['category_col'].unique())
        original_categories = set(sample_data['category_col'].unique())
        assert len(sample_categories.intersection(original_categories)) > 0
    
    def test_cluster_sampling(self, sample_data):
        """Test cluster sampling."""
        transformer = SamplingTransformer(
            sampling_method='cluster',
            sample_size=200,
            cluster_column='cluster_col',
            random_state=42
        )
        
        transformer.fit(sample_data)
        sample = transformer.transform(sample_data)
        
        assert len(sample) <= 200
        assert hasattr(transformer, 'sampling_result_')
        assert transformer.sampling_result_.sampling_method == 'cluster'
        assert len(transformer.sampling_result_.cluster_info) > 0
    
    def test_systematic_sampling(self, sample_data):
        """Test systematic sampling."""
        transformer = SamplingTransformer(
            sampling_method='systematic',
            sample_size=100,
            random_state=42
        )
        
        transformer.fit(sample_data)
        sample = transformer.transform(sample_data)
        
        assert len(sample) == 100
        assert hasattr(transformer, 'sampling_result_')
        assert transformer.sampling_result_.sampling_method == 'systematic'
        
        # Check that indices follow systematic pattern
        indices = transformer.sampling_result_.sample_indices
        if len(indices) > 1:
            intervals = np.diff(indices)
            # All intervals should be approximately equal (allowing for end effects)
            assert np.std(intervals[:-1]) < 2  # Small variation allowed
    
    def test_weighted_sampling(self, sample_data):
        """Test weighted sampling."""
        transformer = SamplingTransformer(
            sampling_method='weighted',
            sample_size=100,
            weights_column='weight_col',
            random_state=42
        )
        
        transformer.fit(sample_data)
        sample = transformer.transform(sample_data)
        
        assert len(sample) == 100
        assert hasattr(transformer, 'sampling_result_')
        assert transformer.sampling_result_.sampling_method == 'weighted'
    
    def test_fractional_sample_size(self, sample_data):
        """Test fractional sample size."""
        transformer = SamplingTransformer(
            sampling_method='simple_random',
            sample_size=0.1,  # 10% of data
            random_state=42
        )
        
        transformer.fit(sample_data)
        sample = transformer.transform(sample_data)
        
        assert len(sample) == 100  # 10% of 1000
    
    def test_quality_metrics_calculation(self, sample_data):
        """Test quality metrics calculation."""
        transformer = SamplingTransformer(
            sampling_method='simple_random',
            sample_size=200,
            random_state=42
        )
        
        transformer.fit(sample_data)
        sample = transformer.transform(sample_data)
        
        result = transformer.sampling_result_
        assert result.representativeness_score is not None
        assert 0 <= result.representativeness_score <= 1
        assert len(result.coverage_metrics) > 0
    
    def test_invalid_parameters(self):
        """Test parameter validation."""
        with pytest.raises(ValueError):
            SamplingTransformer(sampling_method='invalid_method')
        
        with pytest.raises(ValueError):
            SamplingTransformer(sample_size=-1)
        
        with pytest.raises(ValueError):
            SamplingTransformer(sample_size=1.5)  # > 1 for float
    
    def test_sklearn_compatibility(self, sample_data):
        """Test sklearn compatibility."""
        transformer = SamplingTransformer(random_state=42)
        
        # Should work with sklearn pipeline patterns
        transformer.fit(sample_data)
        check_is_fitted(transformer)
        
        sample = transformer.transform(sample_data)
        assert isinstance(sample, pd.DataFrame)


class TestBootstrapTransformer:
    """Test BootstrapTransformer class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        return pd.DataFrame({
            'normal_data': np.random.normal(10, 2, 100),
            'skewed_data': np.random.exponential(3, 100),
            'categorical': ['A'] * 50 + ['B'] * 50
        })
    
    def test_bootstrap_mean(self, sample_data):
        """Test bootstrap for mean statistic."""
        transformer = BootstrapTransformer(
            statistic_func='mean',
            n_bootstrap=1000,
            random_state=42
        )
        
        transformer.fit(sample_data)
        result = transformer.transform(sample_data)
        
        assert len(transformer.bootstrap_results_) >= 1  # At least one numeric column
        
        for bootstrap_result in transformer.bootstrap_results_:
            assert bootstrap_result.n_bootstrap == 1000
            assert len(bootstrap_result.bootstrap_distribution) == 1000
            assert bootstrap_result.original_statistic is not None
            assert bootstrap_result.standard_error > 0
    
    def test_bootstrap_median(self, sample_data):
        """Test bootstrap for median statistic."""
        transformer = BootstrapTransformer(
            statistic_func='median',
            n_bootstrap=500,
            random_state=42
        )
        
        transformer.fit(sample_data)
        result = transformer.transform(sample_data)
        
        assert len(transformer.bootstrap_results_) >= 1
        
        for bootstrap_result in transformer.bootstrap_results_:
            assert bootstrap_result.statistic_name.startswith('median_')
    
    def test_bootstrap_custom_function(self, sample_data):
        """Test bootstrap with custom statistic function."""
        def custom_stat(x):
            return np.percentile(x, 90)  # 90th percentile
        
        transformer = BootstrapTransformer(
            statistic_func=custom_stat,
            n_bootstrap=200,
            random_state=42
        )
        
        transformer.fit(sample_data)
        result = transformer.transform(sample_data)
        
        assert len(transformer.bootstrap_results_) >= 1
    
    def test_confidence_intervals(self, sample_data):
        """Test confidence interval calculation."""
        transformer = BootstrapTransformer(
            statistic_func='mean',
            confidence_level=0.95,
            method='percentile',
            n_bootstrap=1000,
            random_state=42
        )
        
        transformer.fit(sample_data)
        result = transformer.transform(sample_data)
        
        for bootstrap_result in transformer.bootstrap_results_:
            assert len(bootstrap_result.confidence_intervals) > 0
            
            for ci_name, (lower, upper) in bootstrap_result.confidence_intervals.items():
                assert lower < upper
                assert lower <= bootstrap_result.original_statistic <= upper or \
                       abs(bootstrap_result.original_statistic - (lower + upper) / 2) < 3 * bootstrap_result.standard_error
    
    def test_bca_method(self, sample_data):
        """Test BCa bootstrap method."""
        transformer = BootstrapTransformer(
            statistic_func='mean',
            method='bca',
            n_bootstrap=500,
            random_state=42
        )
        
        transformer.fit(sample_data)
        result = transformer.transform(sample_data)
        
        for bootstrap_result in transformer.bootstrap_results_:
            assert bootstrap_result.bootstrap_method == 'bca'
            if 'bca' in bootstrap_result.confidence_intervals:
                lower, upper = bootstrap_result.confidence_intervals['bca']
                assert lower < upper
    
    def test_bias_correction(self, sample_data):
        """Test bias estimation and correction."""
        transformer = BootstrapTransformer(
            statistic_func='mean',
            n_bootstrap=1000,
            random_state=42
        )
        
        transformer.fit(sample_data)
        result = transformer.transform(sample_data)
        
        for bootstrap_result in transformer.bootstrap_results_:
            assert bootstrap_result.bias_estimate is not None
            assert bootstrap_result.bias_corrected_estimate is not None
            assert bootstrap_result.variance_estimate > 0


class TestMonteCarloTransformer:
    """Test MonteCarloTransformer class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        return pd.DataFrame({
            'normal_data': np.random.normal(0, 1, 1000),
            'uniform_data': np.random.uniform(0, 1, 1000),
            'positive_data': np.abs(np.random.normal(2, 1, 1000))
        })
    
    def test_monte_carlo_integration(self, sample_data):
        """Test Monte Carlo integration."""
        # Test integration of normal PDF from -1 to 1
        def normal_pdf(x):
            return (1/np.sqrt(2*np.pi)) * np.exp(-0.5 * x**2)
        
        transformer = MonteCarloTransformer(
            simulation_type='integration',
            n_simulations=10000,
            target_function=normal_pdf,
            integration_bounds=(-1, 1),
            random_state=42
        )
        
        transformer.fit(sample_data)
        result = transformer.transform(sample_data)
        
        assert len(transformer.monte_carlo_results_) > 0
        
        mc_result = transformer.monte_carlo_results_[0]
        assert mc_result.simulation_type == 'integration'
        assert mc_result.n_simulations == 10000
        assert mc_result.integration_bounds == (-1, 1)
        
        # The integral should be approximately 0.68 (within 1 std of mean for normal)
        true_value = stats.norm.cdf(1) - stats.norm.cdf(-1)  # ≈ 0.68
        assert abs(mc_result.estimated_value - true_value) < 0.05  # 5% tolerance
    
    def test_uncertainty_quantification(self, sample_data):
        """Test uncertainty quantification."""
        transformer = MonteCarloTransformer(
            simulation_type='uncertainty',
            n_simulations=5000,
            random_state=42
        )
        
        transformer.fit(sample_data)
        result = transformer.transform(sample_data)
        
        assert len(transformer.monte_carlo_results_) >= 1  # At least one numeric column
        
        for mc_result in transformer.monte_carlo_results_:
            assert mc_result.simulation_type == 'uncertainty'
            assert len(mc_result.simulation_results) == 5000
            assert mc_result.confidence_interval[0] < mc_result.confidence_interval[1]
            assert mc_result.standard_error > 0
    
    def test_importance_sampling(self, sample_data):
        """Test importance sampling for rare events."""
        transformer = MonteCarloTransformer(
            simulation_type='importance',
            n_simulations=10000,
            random_state=42
        )
        
        transformer.fit(sample_data)
        result = transformer.transform(sample_data)
        
        assert len(transformer.monte_carlo_results_) >= 1
        
        for mc_result in transformer.monte_carlo_results_:
            assert mc_result.simulation_type == 'importance_sampling'
            # Should estimate probability of rare event (should be small but > 0)
            assert 0 <= mc_result.estimated_value <= 1
    
    def test_mcmc_sampling(self, sample_data):
        """Test basic MCMC sampling."""
        transformer = MonteCarloTransformer(
            simulation_type='mcmc',
            n_simulations=5000,
            random_state=42
        )
        
        transformer.fit(sample_data)
        result = transformer.transform(sample_data)
        
        assert len(transformer.monte_carlo_results_) >= 1
        
        for mc_result in transformer.monte_carlo_results_:
            assert mc_result.simulation_type == 'mcmc'
            assert 'acceptance_rate' in mc_result.convergence_diagnostic
            
            # Acceptance rate should be reasonable (0.2 - 0.7)
            acceptance_rate = mc_result.convergence_diagnostic['acceptance_rate']
            assert 0.1 <= acceptance_rate <= 0.9
    
    def test_convergence_diagnostics(self, sample_data):
        """Test convergence diagnostics."""
        transformer = MonteCarloTransformer(
            simulation_type='integration',
            n_simulations=1000,
            random_state=42
        )
        
        transformer.fit(sample_data)
        result = transformer.transform(sample_data)
        
        for mc_result in transformer.monte_carlo_results_:
            assert len(mc_result.convergence_diagnostic) > 0
            if 'relative_std_error' in mc_result.convergence_diagnostic:
                assert mc_result.convergence_diagnostic['relative_std_error'] >= 0


class TestBayesianEstimationTransformer:
    """Test BayesianEstimationTransformer class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        return pd.DataFrame({
            'normal_data': np.random.normal(5, 2, 100),
            'positive_data': np.abs(np.random.normal(3, 1, 100)),
            'large_variance': np.random.normal(0, 5, 100)
        })
    
    def test_normal_conjugate_prior(self, sample_data):
        """Test Bayesian estimation with normal conjugate prior."""
        transformer = BayesianEstimationTransformer(
            estimation_type='posterior',
            prior_distribution='normal',
            prior_params={'mu': 0, 'sigma2': 100},
            random_state=42
        )
        
        transformer.fit(sample_data)
        result = transformer.transform(sample_data)
        
        assert len(transformer.bayesian_results_) >= 1
        
        for bayesian_result in transformer.bayesian_results_:
            assert bayesian_result.estimation_method in ['normal_conjugate', 'approximate']
            assert bayesian_result.posterior_mean is not None
            assert len(bayesian_result.credible_intervals) > 0
            assert 'distribution' in bayesian_result.prior_info
    
    def test_gamma_conjugate_prior(self, sample_data):
        """Test Bayesian estimation with gamma conjugate prior."""
        transformer = BayesianEstimationTransformer(
            estimation_type='posterior',
            prior_distribution='gamma',
            prior_params={'alpha': 1, 'beta': 1},
            random_state=42
        )
        
        transformer.fit(sample_data)
        result = transformer.transform(sample_data)
        
        assert len(transformer.bayesian_results_) >= 1
        
        for bayesian_result in transformer.bayesian_results_:
            assert bayesian_result.posterior_mean > 0  # Variance should be positive
    
    def test_credible_intervals(self, sample_data):
        """Test credible interval calculation."""
        transformer = BayesianEstimationTransformer(
            estimation_type='credible_interval',
            confidence_level=0.95,
            random_state=42
        )
        
        transformer.fit(sample_data)
        result = transformer.transform(sample_data)
        
        for bayesian_result in transformer.bayesian_results_:
            assert len(bayesian_result.credible_intervals) > 1  # Multiple levels
            
            for level, (lower, upper) in bayesian_result.credible_intervals.items():
                assert lower < upper
                # Posterior mean should often be within credible interval
                if '95%' in level:
                    # Allow some tolerance for edge cases
                    interval_width = upper - lower
                    assert (lower - interval_width * 0.1 <= bayesian_result.posterior_mean <= 
                           upper + interval_width * 0.1)
    
    def test_model_comparison(self, sample_data):
        """Test Bayesian model comparison."""
        transformer = BayesianEstimationTransformer(
            estimation_type='model_comparison',
            random_state=42
        )
        
        transformer.fit(sample_data)
        result = transformer.transform(sample_data)
        
        for bayesian_result in transformer.bayesian_results_:
            if bayesian_result.bayes_factor is not None:
                assert bayesian_result.bayes_factor > 0
                assert 'compared_models' in bayesian_result.prior_info
                assert 'selected_model' in bayesian_result.prior_info
    
    def test_posterior_samples_quality(self, sample_data):
        """Test quality of posterior samples."""
        transformer = BayesianEstimationTransformer(
            estimation_type='posterior',
            n_samples=10000,
            random_state=42
        )
        
        transformer.fit(sample_data)
        result = transformer.transform(sample_data)
        
        for bayesian_result in transformer.bayesian_results_:
            if bayesian_result.posterior_samples is not None:
                samples = bayesian_result.posterior_samples
                assert len(samples) == 10000
                
                # Sample mean should be close to reported posterior mean
                sample_mean = np.mean(samples)
                assert abs(sample_mean - bayesian_result.posterior_mean) < 0.1
                
                # Samples should have reasonable variance
                assert np.var(samples) > 0


class TestMCPToolFunctions:
    """Test MCP tool functions."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        return pd.DataFrame({
            'values': np.random.normal(100, 15, 500),
            'category': np.random.choice(['A', 'B', 'C'], 500),
            'weights': np.random.exponential(1, 500)
        })
    
    def test_generate_sample_function(self, sample_data):
        """Test generate_sample MCP function."""
        result = generate_sample(
            sample_data,
            sampling_method='simple_random',
            sample_size=50,
            random_state=42
        )
        
        assert 'sample_data' in result
        assert 'sampling_results' in result
        assert len(result['sample_data']) == 50
        assert result['sampling_results']['sampling_method'] == 'simple_random'
    
    def test_bootstrap_statistic_function(self, sample_data):
        """Test bootstrap_statistic MCP function."""
        result = bootstrap_statistic(
            sample_data,
            statistic_func='mean',
            n_bootstrap=1000,
            confidence_level=0.95,
            random_state=42
        )
        
        assert 'bootstrap_results' in result
        assert len(result['bootstrap_results']) >= 1
        assert result['n_bootstrap'] == 1000
        assert result['confidence_level'] == 0.95
    
    def test_monte_carlo_simulate_function(self, sample_data):
        """Test monte_carlo_simulate MCP function."""
        result = monte_carlo_simulate(
            sample_data,
            simulation_type='uncertainty',
            n_simulations=5000,
            random_state=42
        )
        
        assert 'monte_carlo_results' in result
        assert len(result['monte_carlo_results']) >= 1
        assert result['simulation_type'] == 'uncertainty'
        assert result['n_simulations'] == 5000
    
    def test_bayesian_estimate_function(self, sample_data):
        """Test bayesian_estimate MCP function."""
        result = bayesian_estimate(
            sample_data,
            estimation_type='posterior',
            prior_distribution='normal',
            confidence_level=0.95,
            random_state=42
        )
        
        assert 'bayesian_results' in result
        assert len(result['bayesian_results']) >= 1
        assert result['estimation_type'] == 'posterior'
        assert result['prior_distribution'] == 'normal'
    
    def test_file_input_support(self, sample_data, tmp_path):
        """Test that MCP functions support file input."""
        # Create temporary CSV file
        csv_path = tmp_path / "test_data.csv"
        sample_data.to_csv(csv_path, index=False)
        
        # Test with file path
        result = generate_sample(
            str(csv_path),
            sampling_method='simple_random',
            sample_size=30,
            random_state=42
        )
        
        assert 'sample_data' in result
        assert len(result['sample_data']) == 30


class TestIntegrationAndEdgeCases:
    """Test integration scenarios and edge cases."""
    
    def test_small_dataset_handling(self):
        """Test behavior with very small datasets."""
        small_data = pd.DataFrame({
            'values': [1, 2, 3, 4, 5],
            'category': ['A', 'A', 'B', 'B', 'C']
        })
        
        # Sampling with small data
        sampler = SamplingTransformer(sample_size=3, random_state=42)
        sampler.fit(small_data)
        sample = sampler.transform(small_data)
        assert len(sample) == 3
        
        # Bootstrap with small data
        bootstrapper = BootstrapTransformer(n_bootstrap=100, random_state=42)
        bootstrapper.fit(small_data)
        result = bootstrapper.transform(small_data)
        assert len(bootstrapper.bootstrap_results_) >= 1
    
    def test_missing_data_handling(self):
        """Test handling of missing data."""
        data_with_na = pd.DataFrame({
            'values': [1, 2, np.nan, 4, 5, np.nan, 7, 8, 9, 10],
            'category': ['A', 'B', 'A', 'B', 'A', 'B', 'A', 'B', 'A', 'B']
        })
        
        # Sampling should work with missing data
        sampler = SamplingTransformer(sample_size=5, random_state=42)
        sampler.fit(data_with_na)
        sample = sampler.transform(data_with_na)
        assert len(sample) == 5
        
        # Bootstrap should handle missing data by dropping NaN values
        bootstrapper = BootstrapTransformer(n_bootstrap=100, random_state=42)
        bootstrapper.fit(data_with_na)
        result = bootstrapper.transform(data_with_na)
        # Should still produce results for the numeric column with non-NaN values
        assert len(bootstrapper.bootstrap_results_) >= 1
    
    def test_single_column_data(self):
        """Test with single column data."""
        single_col_data = pd.DataFrame({
            'values': np.random.normal(50, 10, 100)
        })
        
        # All transformers should handle single column data
        sampler = SamplingTransformer(sample_size=20, random_state=42)
        sampler.fit(single_col_data)
        sample = sampler.transform(single_col_data)
        assert len(sample) == 20
        assert 'values' in sample.columns
    
    def test_extreme_sample_sizes(self):
        """Test extreme sample size scenarios."""
        data = pd.DataFrame({
            'values': np.random.normal(0, 1, 100)
        })
        
        # Sample size larger than population
        large_sampler = SamplingTransformer(sample_size=150, random_state=42)
        large_sampler.fit(data)
        sample = large_sampler.transform(data)
        assert len(sample) <= 100  # Should be capped at population size
        
        # Very small fractional sample size
        small_sampler = SamplingTransformer(sample_size=0.01, random_state=42)  # 1%
        small_sampler.fit(data)
        sample = small_sampler.transform(data)
        assert len(sample) >= 1  # Should have at least 1 sample
    
    def test_reproducibility(self):
        """Test that results are reproducible with same random_state."""
        data = pd.DataFrame({
            'values': np.random.normal(10, 2, 200)
        })
        
        # Two identical samplers should produce identical results
        sampler1 = SamplingTransformer(sample_size=50, random_state=42)
        sampler2 = SamplingTransformer(sample_size=50, random_state=42)
        
        sampler1.fit(data)
        sample1 = sampler1.transform(data)
        
        sampler2.fit(data)
        sample2 = sampler2.transform(data)
        
        # Samples should be identical
        pd.testing.assert_frame_equal(sample1, sample2)
        
        # Sample indices should be identical
        np.testing.assert_array_equal(
            sampler1.sampling_result_.sample_indices,
            sampler2.sampling_result_.sample_indices
        )
    
    def test_performance_with_large_bootstrap(self):
        """Test performance with large number of bootstrap samples."""
        data = pd.DataFrame({
            'values': np.random.normal(0, 1, 1000)
        })
        
        # Large number of bootstrap samples
        bootstrapper = BootstrapTransformer(
            statistic_func='mean',
            n_bootstrap=10000,
            random_state=42
        )
        
        import time
        start_time = time.time()
        
        bootstrapper.fit(data)
        result = bootstrapper.transform(data)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Should complete in reasonable time (less than 30 seconds)
        assert execution_time < 30
        
        # Should still produce valid results
        assert len(bootstrapper.bootstrap_results_) >= 1
        bootstrap_result = bootstrapper.bootstrap_results_[0]
        assert len(bootstrap_result.bootstrap_distribution) == 10000


if __name__ == '__main__':
    pytest.main([__file__])