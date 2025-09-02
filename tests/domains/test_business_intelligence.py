"""
Tests for Business Intelligence Domain functionality.

This module tests the comprehensive business intelligence capabilities including
customer analytics, A/B testing, attribution modeling, and funnel analysis.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from src.localdata_mcp.domains.business_intelligence import (
    # Core transformers
    RFMAnalysisTransformer,
    CohortAnalysisTransformer,
    CLVCalculator,
    ABTestAnalyzer,
    PowerAnalysisTransformer,
    AttributionAnalyzer,
    FunnelAnalyzer,
    BusinessIntelligencePipeline,
    
    # High-level functions
    analyze_rfm,
    perform_cohort_analysis,
    calculate_clv,
    perform_ab_test,
    analyze_attribution,
    analyze_funnel,
    enhanced_ab_test,
    
    # Result classes
    RFMResult,
    CohortAnalysisResult,
    CLVResult,
    ABTestResult,
    AttributionResult,
    FunnelAnalysisResult,
    
    # Enums
    AttributionModel,
    ExperimentStatus
)


@pytest.fixture
def sample_transaction_data():
    """Create sample transaction data for testing."""
    np.random.seed(42)
    
    # Generate 100 customers with varying transaction patterns
    customers = [f"cust_{i}" for i in range(1, 101)]
    transactions = []
    
    start_date = datetime(2023, 1, 1)
    
    for customer in customers:
        # Varying number of transactions per customer (1-20)
        num_transactions = np.random.randint(1, 21)
        
        for _ in range(num_transactions):
            # Random date within last year
            days_back = np.random.randint(0, 365)
            transaction_date = start_date + timedelta(days=days_back)
            
            # Random transaction amount ($10-$1000)
            amount = np.random.uniform(10, 1000)
            
            transactions.append({
                'customer_id': customer,
                'date': transaction_date,
                'amount': amount
            })
    
    return pd.DataFrame(transactions)


@pytest.fixture
def sample_ab_test_data():
    """Create sample A/B test data for testing."""
    np.random.seed(42)
    
    # Generate A/B test data with different conversion rates
    n_control = 1000
    n_treatment = 1000
    
    # Control group: 10% conversion rate
    control_conversions = np.random.binomial(1, 0.10, n_control)
    control_data = pd.DataFrame({
        'group': ['control'] * n_control,
        'converted': control_conversions
    })
    
    # Treatment group: 12% conversion rate
    treatment_conversions = np.random.binomial(1, 0.12, n_treatment)
    treatment_data = pd.DataFrame({
        'group': ['treatment'] * n_treatment,
        'converted': treatment_conversions
    })
    
    return pd.concat([control_data, treatment_data], ignore_index=True)


@pytest.fixture
def sample_touchpoint_data():
    """Create sample marketing touchpoint data for testing."""
    np.random.seed(42)
    
    channels = ['email', 'social', 'search', 'display', 'direct']
    customers = [f"cust_{i}" for i in range(1, 51)]
    
    touchpoints = []
    start_date = datetime(2023, 1, 1)
    
    for customer in customers:
        # Random customer journey length (1-5 touchpoints)
        journey_length = np.random.randint(1, 6)
        converted = np.random.choice([True, False], p=[0.15, 0.85])
        
        for i in range(journey_length):
            touchpoint_date = start_date + timedelta(days=np.random.randint(0, 30))
            channel = np.random.choice(channels)
            
            touchpoints.append({
                'customer_id': customer,
                'channel': channel,
                'timestamp': touchpoint_date,
                'converted': converted if i == journey_length - 1 else False
            })
    
    return pd.DataFrame(touchpoints)


@pytest.fixture
def sample_funnel_data():
    """Create sample funnel data for testing."""
    np.random.seed(42)
    
    # Simulate funnel with decreasing conversion rates
    n_users = 10000
    
    funnel_data = pd.DataFrame({
        'user_id': range(n_users),
        'awareness': [True] * n_users,
        'interest': np.random.binomial(1, 0.7, n_users).astype(bool),
        'consideration': np.random.binomial(1, 0.4, n_users).astype(bool),
        'purchase': np.random.binomial(1, 0.1, n_users).astype(bool)
    })
    
    return funnel_data


class TestRFMAnalysisTransformer:
    """Test RFM Analysis transformer."""
    
    def test_rfm_analysis_basic(self, sample_transaction_data):
        """Test basic RFM analysis functionality."""
        transformer = RFMAnalysisTransformer()
        transformer.fit(sample_transaction_data)
        result = transformer.transform(sample_transaction_data)
        
        assert isinstance(result, RFMResult)
        assert len(result.rfm_scores) > 0
        assert len(result.segments) > 0
        assert 'customer_id' in result.rfm_scores.columns
        assert 'RFM_Score' in result.rfm_scores.columns
        assert 'Segment' in result.segments.columns
        
    def test_rfm_analysis_custom_columns(self, sample_transaction_data):
        """Test RFM analysis with custom column names."""
        # Rename columns
        data = sample_transaction_data.rename(columns={
            'customer_id': 'cust_id',
            'date': 'transaction_date',
            'amount': 'purchase_amount'
        })
        
        transformer = RFMAnalysisTransformer(
            customer_column='cust_id',
            date_column='transaction_date',
            amount_column='purchase_amount'
        )
        
        transformer.fit(data)
        result = transformer.transform(data)
        
        assert isinstance(result, RFMResult)
        assert 'cust_id' in result.rfm_scores.columns
        
    def test_rfm_quartile_boundaries(self, sample_transaction_data):
        """Test RFM quartile boundary calculation."""
        transformer = RFMAnalysisTransformer()
        transformer.fit(sample_transaction_data)
        result = transformer.transform(sample_transaction_data)
        
        boundaries = result.quartile_boundaries
        assert 'recency' in boundaries
        assert 'frequency' in boundaries
        assert 'monetary' in boundaries
        assert len(boundaries['recency']) == 3
        
    def test_analyze_rfm_function(self, sample_transaction_data):
        """Test high-level analyze_rfm function."""
        result = analyze_rfm(sample_transaction_data)
        
        assert isinstance(result, RFMResult)
        assert len(result.segments) > 0
        
        # Check that segments make business sense
        segments = result.segments['Segment'].unique()
        expected_segments = ['Champions', 'Loyal Customers', 'At Risk', 'Lost']
        assert any(seg in segments for seg in expected_segments)


class TestCohortAnalysisTransformer:
    """Test Cohort Analysis transformer."""
    
    def test_cohort_analysis_monthly(self, sample_transaction_data):
        """Test monthly cohort analysis."""
        transformer = CohortAnalysisTransformer(period_type='monthly')
        transformer.fit(sample_transaction_data)
        result = transformer.transform(sample_transaction_data)
        
        assert isinstance(result, CohortAnalysisResult)
        assert not result.cohort_table.empty
        assert not result.cohort_sizes.empty
        assert not result.retention_rates.empty
        
    def test_cohort_analysis_weekly(self, sample_transaction_data):
        """Test weekly cohort analysis."""
        result = perform_cohort_analysis(sample_transaction_data, period_type='weekly')
        
        assert isinstance(result, CohortAnalysisResult)
        assert result.period_summary['analysis_period_type'] == 'weekly'
        
    def test_cohort_retention_rates(self, sample_transaction_data):
        """Test cohort retention rate calculations."""
        result = perform_cohort_analysis(sample_transaction_data)
        
        # Retention rates should be between 0 and 1
        retention_values = result.retention_rates.values.flatten()
        retention_values = retention_values[retention_values > 0]  # Exclude 0s
        
        assert all(0 <= rate <= 1 for rate in retention_values)


class TestCLVCalculator:
    """Test Customer Lifetime Value calculator."""
    
    def test_clv_historical_method(self, sample_transaction_data):
        """Test historical CLV calculation method."""
        transformer = CLVCalculator(method='historical')
        transformer.fit(sample_transaction_data)
        result = transformer.transform(sample_transaction_data)
        
        assert isinstance(result, CLVResult)
        assert len(result.clv_scores) > 0
        assert 'clv_estimate' in result.clv_scores.columns
        assert result.model_metrics['method'] == 'historical'
        
    def test_clv_distribution_stats(self, sample_transaction_data):
        """Test CLV distribution statistics."""
        result = calculate_clv(sample_transaction_data)
        
        distribution = result.clv_distribution
        assert 'mean' in distribution
        assert 'median' in distribution
        assert 'std' in distribution
        assert distribution['mean'] >= 0
        assert distribution['median'] >= 0


class TestABTestAnalyzer:
    """Test A/B Test analyzer."""
    
    def test_ab_test_proportion_analysis(self, sample_ab_test_data):
        """Test A/B test analysis for proportions."""
        transformer = ABTestAnalyzer(test_type='proportion')
        transformer.fit(sample_ab_test_data)
        result = transformer.transform(sample_ab_test_data)
        
        assert isinstance(result, ABTestResult)
        assert result.p_value >= 0
        assert result.p_value <= 1
        assert len(result.sample_sizes) == 2
        assert len(result.conversion_rates) == 2
        
    def test_ab_test_confidence_interval(self, sample_ab_test_data):
        """Test confidence interval calculation."""
        result = perform_ab_test(sample_ab_test_data)
        
        ci_lower, ci_upper = result.confidence_interval
        assert ci_lower <= ci_upper
        
    def test_ab_test_power_calculation(self, sample_ab_test_data):
        """Test statistical power calculation."""
        result = perform_ab_test(sample_ab_test_data)
        
        assert 0 <= result.power <= 1
        assert isinstance(result.conclusion, str)
        assert len(result.conclusion) > 0


class TestPowerAnalysisTransformer:
    """Test Power Analysis transformer."""
    
    def test_power_analysis_sample_size(self):
        """Test sample size calculation for different effect sizes."""
        transformer = PowerAnalysisTransformer(power=0.8, alpha=0.05)
        transformer.fit(pd.DataFrame())
        result = transformer.transform(pd.DataFrame())
        
        assert isinstance(result, pd.DataFrame)
        assert 'effect_size' in result.columns
        assert 'required_n_per_group' in result.columns
        assert len(result) > 0
        
    def test_power_analysis_specific_effect_size(self):
        """Test power analysis for specific effect size."""
        transformer = PowerAnalysisTransformer(effect_size=0.3)
        transformer.fit(pd.DataFrame())
        result = transformer.transform(pd.DataFrame())
        
        assert len(result) == 1
        assert result.iloc[0]['effect_size'] == 0.3


class TestAttributionAnalyzer:
    """Test Attribution analyzer."""
    
    def test_first_touch_attribution(self, sample_touchpoint_data):
        """Test first-touch attribution model."""
        transformer = AttributionAnalyzer(attribution_model='first_touch')
        transformer.fit(sample_touchpoint_data)
        result = transformer.transform(sample_touchpoint_data)
        
        assert isinstance(result, AttributionResult)
        assert not result.attribution_weights.empty
        assert not result.channel_attribution.empty
        
    def test_last_touch_attribution(self, sample_touchpoint_data):
        """Test last-touch attribution model."""
        result = analyze_attribution(sample_touchpoint_data, model='last_touch')
        
        assert isinstance(result, AttributionResult)
        
        # Check that attribution weights sum appropriately
        total_attribution = result.channel_attribution['total_attribution'].sum()
        assert total_attribution > 0
        
    def test_linear_attribution(self, sample_touchpoint_data):
        """Test linear attribution model."""
        result = analyze_attribution(sample_touchpoint_data, model='linear')
        
        assert isinstance(result, AttributionResult)
        
        # Linear attribution should distribute weights evenly
        if not result.attribution_weights.empty:
            assert 'attribution_weight' in result.attribution_weights.columns
            
    def test_attribution_model_comparison(self, sample_touchpoint_data):
        """Test attribution model comparison."""
        transformer = AttributionAnalyzer()
        transformer.fit(sample_touchpoint_data)
        result = transformer.transform(sample_touchpoint_data)
        
        comparison = result.model_comparison
        assert isinstance(comparison, dict)
        assert len(comparison) >= 3  # At least 3 models compared


class TestFunnelAnalyzer:
    """Test Funnel analyzer."""
    
    def test_funnel_analysis_basic(self, sample_funnel_data):
        """Test basic funnel analysis."""
        steps = ['awareness', 'interest', 'consideration', 'purchase']
        transformer = FunnelAnalyzer(steps=steps)
        transformer.fit(sample_funnel_data)
        result = transformer.transform(sample_funnel_data)
        
        assert isinstance(result, FunnelAnalysisResult)
        assert len(result.funnel_steps) == len(steps)
        assert len(result.conversion_rates) > 0
        assert len(result.drop_off_rates) > 0
        
    def test_funnel_conversion_rates(self, sample_funnel_data):
        """Test funnel conversion rate calculations."""
        result = analyze_funnel(sample_funnel_data)
        
        # Conversion rates should be between 0 and 1
        for rate in result.conversion_rates.values():
            assert 0 <= rate <= 1
            
    def test_funnel_bottleneck_identification(self, sample_funnel_data):
        """Test bottleneck identification."""
        result = analyze_funnel(sample_funnel_data)
        
        assert isinstance(result.bottlenecks, list)
        assert isinstance(result.optimization_recommendations, list)
        assert len(result.optimization_recommendations) > 0


class TestBusinessIntelligencePipeline:
    """Test Business Intelligence pipeline."""
    
    def test_pipeline_initialization(self):
        """Test pipeline initialization with different configurations."""
        pipeline = BusinessIntelligencePipeline()
        
        assert pipeline.customer_analytics
        assert pipeline.ab_testing
        assert pipeline.attribution_modeling
        assert pipeline.funnel_analysis
        assert len(pipeline.transformers) > 0
        
    def test_pipeline_selective_components(self):
        """Test pipeline with selective components enabled."""
        pipeline = BusinessIntelligencePipeline(
            customer_analytics=True,
            ab_testing=False,
            attribution_modeling=False,
            funnel_analysis=False
        )
        
        assert 'rfm' in pipeline.transformers
        assert 'ab_test' not in pipeline.transformers
        
    def test_pipeline_fit_transform(self, sample_transaction_data):
        """Test pipeline fit and transform process."""
        pipeline = BusinessIntelligencePipeline(
            ab_testing=False,  # Disable components that need different data
            attribution_modeling=False,
            funnel_analysis=False
        )
        
        pipeline.fit(sample_transaction_data)
        result = pipeline.transform(sample_transaction_data)
        
        assert hasattr(result, 'data')
        assert isinstance(result.data, dict)
        
    def test_customer_journey_analysis(self, sample_transaction_data, 
                                     sample_touchpoint_data, sample_funnel_data):
        """Test comprehensive customer journey analysis."""
        pipeline = BusinessIntelligencePipeline()
        
        journey_analysis = pipeline.analyze_customer_journey(
            transaction_data=sample_transaction_data,
            touchpoint_data=sample_touchpoint_data,
            funnel_data=sample_funnel_data
        )
        
        assert isinstance(journey_analysis, dict)
        assert 'insights' in journey_analysis
        assert isinstance(journey_analysis['insights'], list)


class TestEnhancedABTest:
    """Test enhanced A/B testing with statistical domain integration."""
    
    def test_enhanced_ab_test_basic(self, sample_ab_test_data):
        """Test enhanced A/B test analysis."""
        with patch('src.localdata_mcp.domains.business_intelligence.run_hypothesis_test') as mock_stat:
            mock_stat.return_value = {'p_value': 0.045, 'statistic': 2.1}
            
            result = enhanced_ab_test(sample_ab_test_data)
            
            assert isinstance(result, dict)
            assert 'business_intelligence' in result
            
    def test_enhanced_ab_test_without_statistical_domain(self, sample_ab_test_data):
        """Test enhanced A/B test when statistical domain is not available."""
        result = enhanced_ab_test(sample_ab_test_data, use_statistical_domain=False)
        
        assert isinstance(result, dict)
        assert 'business_intelligence' in result
        assert 'statistical_analysis' not in result


class TestResultClasses:
    """Test result class serialization and methods."""
    
    def test_rfm_result_to_dict(self, sample_transaction_data):
        """Test RFM result serialization to dictionary."""
        result = analyze_rfm(sample_transaction_data)
        result_dict = result.to_dict()
        
        assert isinstance(result_dict, dict)
        assert 'rfm_scores' in result_dict
        assert 'segments' in result_dict
        assert isinstance(result_dict['rfm_scores'], list)
        
    def test_ab_test_result_to_dict(self, sample_ab_test_data):
        """Test A/B test result serialization to dictionary."""
        result = perform_ab_test(sample_ab_test_data)
        result_dict = result.to_dict()
        
        assert isinstance(result_dict, dict)
        assert 'p_value' in result_dict
        assert 'confidence_interval' in result_dict


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_missing_columns_error(self):
        """Test error handling for missing required columns."""
        invalid_data = pd.DataFrame({'wrong_col': [1, 2, 3]})
        
        with pytest.raises(ValueError, match="Missing required columns"):
            transformer = RFMAnalysisTransformer()
            transformer.fit(invalid_data)
            
    def test_empty_data_handling(self):
        """Test handling of empty datasets."""
        empty_data = pd.DataFrame(columns=['customer_id', 'date', 'amount'])
        
        transformer = RFMAnalysisTransformer()
        transformer.fit(empty_data)
        
        # Should not raise an error, but return empty results
        result = transformer.transform(empty_data)
        assert isinstance(result, RFMResult)
        
    def test_attribution_model_enum(self):
        """Test attribution model enum validation."""
        valid_models = ['first_touch', 'last_touch', 'linear', 'time_decay', 'position_based']
        
        for model in valid_models:
            # Should not raise an error
            AttributionModel(model)
            
    def test_invalid_ab_test_groups(self):
        """Test A/B test with invalid number of groups."""
        invalid_data = pd.DataFrame({
            'group': ['A', 'B', 'C'],  # 3 groups instead of 2
            'converted': [1, 0, 1]
        })
        
        with pytest.raises(ValueError, match="Currently supports only 2-group"):
            transformer = ABTestAnalyzer()
            transformer.fit(invalid_data)
            transformer.transform(invalid_data)


class TestIntegrationScenarios:
    """Test real-world integration scenarios."""
    
    def test_customer_segmentation_to_targeting(self, sample_transaction_data):
        """Test customer segmentation for marketing targeting."""
        # Perform RFM analysis
        rfm_result = analyze_rfm(sample_transaction_data)
        
        # Extract champion customers for targeting
        champions = rfm_result.segments[
            rfm_result.segments['Segment'] == 'Champions'
        ]
        
        if not champions.empty:
            assert len(champions) > 0
            assert 'customer_id' in champions.columns
            
    def test_cohort_retention_insights(self, sample_transaction_data):
        """Test cohort analysis for retention insights."""
        cohort_result = perform_cohort_analysis(sample_transaction_data)
        
        # Check that retention analysis provides actionable insights
        period_summary = cohort_result.period_summary
        assert 'overall_average_retention' in period_summary
        assert period_summary['overall_average_retention'] >= 0
        
    def test_attribution_optimization_workflow(self, sample_touchpoint_data):
        """Test attribution analysis for channel optimization."""
        attribution_result = analyze_attribution(sample_touchpoint_data)
        
        if not attribution_result.channel_attribution.empty:
            # Identify top performing channels
            top_channel = attribution_result.channel_attribution.iloc[0]
            assert 'channel' in top_channel
            assert top_channel['total_attribution'] > 0
            
    def test_funnel_optimization_recommendations(self, sample_funnel_data):
        """Test funnel analysis for conversion optimization."""
        funnel_result = analyze_funnel(sample_funnel_data)
        
        # Should provide actionable recommendations
        recommendations = funnel_result.optimization_recommendations
        assert len(recommendations) > 0
        assert all(isinstance(rec, str) for rec in recommendations)


if __name__ == "__main__":
    pytest.main([__file__])