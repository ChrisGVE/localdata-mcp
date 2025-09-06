"""
Integration Example: Cross-Domain Data Science Workflows with Domain Shims

This example demonstrates how the pre-built domain shims enable seamless
integration between statistical, regression, time series, and pattern recognition
domains in LocalData MCP v2.0.

Key Scenarios:
1. Statistical Analysis → Regression Modeling → Time Series Forecasting
2. Time Series Analysis → Pattern Recognition → Statistical Validation
3. Regression Results → Pattern Recognition → Business Intelligence
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any

# Import LocalData MCP domain shims
from src.localdata_mcp.pipeline.integration import (
    # Domain shims
    create_all_domain_shims,
    StatisticalShim, RegressionShim, TimeSeriesShim, PatternRecognitionShim,
    
    # Data structures  
    DataFormat, ConversionRequest, ConversionContext,
    
    # Registry
    ShimRegistry, create_shim_registry
)


def create_sample_datasets():
    """Create sample datasets for demonstration."""
    np.random.seed(42)
    
    # 1. Time series data with trend and seasonality
    dates = pd.date_range('2023-01-01', periods=365, freq='D')
    trend = np.linspace(100, 150, 365)
    seasonal = 10 * np.sin(2 * np.pi * np.arange(365) / 365.25 * 4)  # Quarterly seasonality
    noise = np.random.normal(0, 5, 365)
    ts_values = trend + seasonal + noise
    
    time_series_data = pd.DataFrame({
        'sales': ts_values,
        'date': dates
    }).set_index('date')
    
    # 2. Statistical correlation data
    feature_data = pd.DataFrame({
        'marketing_spend': ts_values[:50] * 0.01 + np.random.normal(0, 0.1, 50),
        'season_index': seasonal[:50],
        'competitor_price': 100 - ts_values[:50] * 0.05 + np.random.normal(0, 2, 50),
        'sales_target': ts_values[:50]
    })
    
    correlation_matrix = feature_data.corr()
    
    # 3. Regression model results
    regression_results = {
        'coefficients': [0.8, 0.4, -0.3, 0.1],
        'feature_names': ['marketing_spend', 'season_index', 'competitor_price', 'intercept'],
        'fitted_values': ts_values[:50] + np.random.normal(0, 2, 50),
        'residuals': np.random.normal(0, 3, 50),
        'r2_score': 0.87,
        'p_values': [0.001, 0.05, 0.02, 0.3],
        'f_statistic': 45.2,
        'f_pvalue': 0.001
    }
    
    # 4. Clustering results for pattern recognition
    n_samples = 100
    cluster_centers = np.array([[2, 2], [6, 6], [2, 6]])
    cluster_labels = np.random.choice([0, 1, 2], n_samples)
    
    clustering_results = {
        'cluster_labels': cluster_labels,
        'centroids': cluster_centers,
        'silhouette_score': 0.75,
        'inertia': 45.8,
        'feature_names': ['feature_1', 'feature_2']
    }
    
    return {
        'time_series': time_series_data,
        'correlation_matrix': correlation_matrix,
        'regression_results': regression_results,
        'clustering_results': clustering_results
    }


def demonstrate_statistical_to_regression_to_timeseries_workflow():
    """
    Scenario 1: Statistical Analysis → Regression Modeling → Time Series Forecasting
    
    Use Case: Start with correlation analysis, use results for feature selection
    in regression, then convert regression model for time series forecasting.
    """
    print("=" * 80)
    print("SCENARIO 1: Statistical → Regression → Time Series Workflow")
    print("=" * 80)
    
    # Initialize domain shims
    shims = create_all_domain_shims(auto_register=False)
    
    # Initialize all shims
    for shim_name, shim in shims.items():
        shim.initialize()
        shim.activate()
        print(f"✓ Initialized {shim_name} shim")
    
    # Get sample data
    datasets = create_sample_datasets()
    
    # Step 1: Statistical Analysis (Correlation Matrix)
    print("\n--- Step 1: Statistical Analysis ---")
    statistical_data = {
        'correlation_matrix': datasets['correlation_matrix'],
        'p_values': np.random.rand(4, 4) * 0.1  # Simulated p-values
    }
    
    print(f"Correlation matrix shape: {statistical_data['correlation_matrix'].shape}")
    print("Top correlations:")
    corr_matrix = statistical_data['correlation_matrix']
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            print(f"  {corr_matrix.columns[i]} ↔ {corr_matrix.columns[j]}: {corr_val:.3f}")
    
    # Step 2: Convert Statistical → Regression
    print("\n--- Step 2: Statistical → Regression Conversion ---")
    stat_to_reg_request = ConversionRequest(
        source_data=statistical_data,
        source_format=DataFormat.STATISTICAL_RESULT,
        target_format=DataFormat.REGRESSION_MODEL,
        context=ConversionContext(
            source_domain='statistical',
            target_domain='regression',
            user_intention='feature_selection_for_regression'
        )
    )
    
    reg_result = shims['statistical'].convert(stat_to_reg_request)
    print(f"✓ Statistical → Regression conversion success: {reg_result.success}")
    print(f"  Quality score: {reg_result.quality_score:.3f}")
    print(f"  Execution time: {reg_result.execution_time:.3f}s")
    
    if 'feature_correlation_matrix' in reg_result.converted_data:
        print("  → Generated regression-ready feature correlation matrix")
    if 'feature_selection_hints' in reg_result.converted_data:
        print("  → Provided feature selection recommendations")
    
    # Step 3: Enhance with actual regression results
    enhanced_reg_data = reg_result.converted_data.copy()
    enhanced_reg_data.update(datasets['regression_results'])
    
    # Step 4: Convert Regression → Time Series
    print("\n--- Step 3: Regression → Time Series Conversion ---")
    reg_to_ts_request = ConversionRequest(
        source_data=enhanced_reg_data,
        source_format=DataFormat.REGRESSION_MODEL,
        target_format=DataFormat.TIME_SERIES,
        context=ConversionContext(
            source_domain='regression',
            target_domain='time_series',
            user_intention='trend_forecasting'
        )
    )
    
    ts_result = shims['regression'].convert(reg_to_ts_request)
    print(f"✓ Regression → Time Series conversion success: {ts_result.success}")
    print(f"  Quality score: {ts_result.quality_score:.3f}")
    print(f"  Execution time: {ts_result.execution_time:.3f}s")
    
    if 'trend_model' in ts_result.converted_data:
        trend_model = ts_result.converted_data['trend_model']
        print(f"  → Generated trend model with {len(trend_model.get('trend_parameters', []))} parameters")
    if 'forecast_info' in ts_result.converted_data:
        forecast_info = ts_result.converted_data['forecast_info']
        print(f"  → Forecast reliability: {forecast_info.get('forecast_reliability', 0):.3f}")
    
    print("\n🎯 Workflow Result: Successfully converted statistical correlations")
    print("   through regression modeling to time series forecasting components!")
    
    return {
        'statistical_result': reg_result,
        'time_series_result': ts_result
    }


def demonstrate_timeseries_to_pattern_recognition_to_statistical_workflow():
    """
    Scenario 2: Time Series Analysis → Pattern Recognition → Statistical Validation
    
    Use Case: Analyze time series patterns, use pattern recognition for clustering
    temporal segments, then validate clusters with statistical analysis.
    """
    print("\n\n" + "=" * 80)
    print("SCENARIO 2: Time Series → Pattern Recognition → Statistical Workflow")
    print("=" * 80)
    
    # Initialize domain shims
    shims = create_all_domain_shims(auto_register=False)
    
    # Initialize all shims
    for shim_name, shim in shims.items():
        shim.initialize()
        shim.activate()
    
    # Get sample data
    datasets = create_sample_datasets()
    
    # Step 1: Time Series Data
    print("\n--- Step 1: Time Series Analysis ---")
    ts_data = datasets['time_series']
    print(f"Time series shape: {ts_data.shape}")
    print(f"Date range: {ts_data.index.min()} to {ts_data.index.max()}")
    print(f"Sales range: {ts_data['sales'].min():.1f} to {ts_data['sales'].max():.1f}")
    
    # Step 2: Convert Time Series → Pattern Recognition
    print("\n--- Step 2: Time Series → Pattern Recognition Conversion ---")
    ts_to_pr_request = ConversionRequest(
        source_data=ts_data,
        source_format=DataFormat.TIME_SERIES,
        target_format=DataFormat.PATTERN_RECOGNITION_RESULT,
        context=ConversionContext(
            source_domain='time_series',
            target_domain='pattern_recognition',
            user_intention='temporal_pattern_mining'
        )
    )
    
    pr_result = shims['time_series'].convert(ts_to_pr_request)
    print(f"✓ Time Series → Pattern Recognition conversion success: {pr_result.success}")
    print(f"  Quality score: {pr_result.quality_score:.3f}")
    print(f"  Execution time: {pr_result.execution_time:.3f}s")
    
    if 'temporal_features' in pr_result.converted_data:
        temp_features = pr_result.converted_data['temporal_features']
        print(f"  → Extracted {len(temp_features)} temporal features")
        print(f"    - Trend slope: {temp_features.get('trend_slope', 0):.4f}")
        print(f"    - Volatility: {temp_features.get('std', 0):.2f}")
    
    if 'pattern_detection' in pr_result.converted_data:
        pattern_info = pr_result.converted_data['pattern_detection']
        print(f"  → Pattern analysis:")
        print(f"    - Seasonal detected: {pattern_info.get('seasonal_detected', False)}")
        print(f"    - Trend detected: {pattern_info.get('trend_detected', False)}")
    
    # Step 3: Enhance with clustering results
    enhanced_pr_data = pr_result.converted_data.copy()
    enhanced_pr_data.update(datasets['clustering_results'])
    
    # Step 4: Convert Pattern Recognition → Statistical
    print("\n--- Step 3: Pattern Recognition → Statistical Conversion ---")
    pr_to_stat_request = ConversionRequest(
        source_data=enhanced_pr_data,
        source_format=DataFormat.CLUSTERING_RESULT,
        target_format=DataFormat.STATISTICAL_RESULT,
        context=ConversionContext(
            source_domain='pattern_recognition',
            target_domain='statistical',
            user_intention='cluster_validation'
        )
    )
    
    stat_result = shims['pattern_recognition'].convert(pr_to_stat_request)
    print(f"✓ Pattern Recognition → Statistical conversion success: {stat_result.success}")
    print(f"  Quality score: {stat_result.quality_score:.3f}")
    print(f"  Execution time: {stat_result.execution_time:.3f}s")
    
    if 'group_statistics' in stat_result.converted_data:
        group_stats = stat_result.converted_data['group_statistics']
        print(f"  → Generated statistics for {len(group_stats)} groups")
        for group_name, stats in group_stats.items():
            print(f"    - {group_name}: size={stats['size']}, proportion={stats['proportion']:.3f}")
    
    if 'clustering_validity' in stat_result.converted_data:
        validity = stat_result.converted_data['clustering_validity']
        print(f"  → Clustering validation:")
        print(f"    - Number of clusters: {validity['n_clusters']}")
        print(f"    - Silhouette score: {validity.get('silhouette_score', 0):.3f}")
    
    print("\n🎯 Workflow Result: Successfully analyzed time series patterns,")
    print("   clustered temporal segments, and validated clusters statistically!")
    
    return {
        'pattern_recognition_result': pr_result,
        'statistical_result': stat_result
    }


def demonstrate_registry_based_workflow():
    """
    Scenario 3: Using ShimRegistry for automated domain shim management
    
    Use Case: Register all domain shims in a registry and demonstrate
    automatic shim discovery and health monitoring.
    """
    print("\n\n" + "=" * 80)
    print("SCENARIO 3: Registry-Based Domain Shim Management")
    print("=" * 80)
    
    # Create and configure registry
    print("\n--- Step 1: Registry Setup ---")
    registry = create_shim_registry(enable_auto_discovery=True)
    
    # Create all domain shims and register them
    shims = create_all_domain_shims(registry=registry, auto_register=True)
    
    print(f"✓ Created registry with {len(shims)} domain shims")
    print(f"  Registered adapters: {registry.list_adapters()}")
    
    # Initialize and activate all adapters
    print("\n--- Step 2: Lifecycle Management ---")
    init_results = registry.initialize_all_adapters()
    activation_results = registry.activate_all_adapters()
    
    successful_inits = sum(1 for success in init_results.values() if success)
    successful_activations = sum(1 for success in activation_results.values() if success)
    
    print(f"✓ Initialized {successful_inits}/{len(init_results)} adapters")
    print(f"✓ Activated {successful_activations}/{len(activation_results)} adapters")
    
    # Perform health checks
    print("\n--- Step 3: Health Monitoring ---")
    health_results = registry.perform_health_checks()
    
    healthy_adapters = sum(1 for result in health_results.values() if result.is_healthy)
    print(f"✓ Health check completed: {healthy_adapters}/{len(health_results)} adapters healthy")
    
    for adapter_id, health_result in health_results.items():
        status_emoji = "🟢" if health_result.is_healthy else "🔴"
        print(f"  {status_emoji} {adapter_id}: {health_result.status}")
        if health_result.warnings:
            for warning in health_result.warnings:
                print(f"    ⚠️  {warning}")
    
    # Demonstrate automatic adapter discovery
    print("\n--- Step 4: Adapter Discovery ---")
    datasets = create_sample_datasets()
    
    # Test conversion request with automatic adapter selection
    test_request = ConversionRequest(
        source_data=datasets['correlation_matrix'],
        source_format=DataFormat.STATISTICAL_RESULT,
        target_format=DataFormat.REGRESSION_MODEL
    )
    
    # Find compatible adapters
    compatible_adapters = registry.get_compatible_adapters(test_request)
    print(f"✓ Found {len(compatible_adapters)} compatible adapters for conversion")
    
    for adapter, confidence in compatible_adapters:
        print(f"  - {adapter.adapter_id}: confidence={confidence:.3f}")
    
    # Use best adapter for conversion
    if compatible_adapters:
        best_adapter, best_confidence = compatible_adapters[0]
        print(f"\n--- Step 5: Automatic Conversion ---")
        print(f"Using best adapter: {best_adapter.adapter_id} (confidence: {best_confidence:.3f})")
        
        result = best_adapter.convert(test_request)
        print(f"✓ Conversion success: {result.success}")
        print(f"  Quality score: {result.quality_score:.3f}")
        print(f"  Execution time: {result.execution_time:.3f}s")
    
    # Get registry statistics
    print("\n--- Step 6: Registry Statistics ---")
    stats = registry.get_registry_stats()
    print(f"📊 Registry Statistics:")
    print(f"  - Total adapters: {stats['total_adapters']}")
    print(f"  - Total conversions: {stats['total_conversions']}")
    print(f"  - Error rate: {stats['error_rate']:.1%}")
    print(f"  - Adapter states: {stats['adapter_states']}")
    
    # Cleanup
    print("\n--- Step 7: Cleanup ---")
    shutdown_results = registry.shutdown_all_adapters()
    successful_shutdowns = sum(1 for success in shutdown_results.values() if success)
    print(f"✓ Shutdown {successful_shutdowns}/{len(shutdown_results)} adapters")
    
    print("\n🎯 Registry Result: Successfully demonstrated automated domain shim")
    print("   management with lifecycle control and health monitoring!")
    
    return {
        'registry': registry,
        'health_results': health_results,
        'statistics': stats
    }


def demonstrate_semantic_context_preservation():
    """
    Scenario 4: Semantic Context Preservation Across Domains
    
    Use Case: Show how semantic context and analytical intentions are
    preserved and enhanced across domain boundaries.
    """
    print("\n\n" + "=" * 80)
    print("SCENARIO 4: Semantic Context Preservation")
    print("=" * 80)
    
    # Initialize domain shims
    shims = create_all_domain_shims(auto_register=False)
    for shim in shims.values():
        shim.initialize()
        shim.activate()
    
    # Get sample data
    datasets = create_sample_datasets()
    
    # Create request with rich semantic context
    print("\n--- Step 1: Rich Semantic Context ---")
    context = ConversionContext(
        source_domain='statistical',
        target_domain='regression',
        user_intention='feature_selection_for_predictive_modeling',
        pipeline_context={
            'business_goal': 'sales_forecasting',
            'model_type': 'linear_regression',
            'performance_target': 'high_accuracy',
            'interpretation_required': True
        },
        performance_hints={
            'memory_efficient': True,
            'real_time_inference': False,
            'explainable_features': True
        }
    )
    
    print(f"User intention: {context.user_intention}")
    print(f"Business goal: {context.pipeline_context.get('business_goal')}")
    print(f"Performance hints: {list(context.performance_hints.keys())}")
    
    # Perform conversion with semantic context
    print("\n--- Step 2: Context-Aware Conversion ---")
    request = ConversionRequest(
        source_data=datasets['regression_results'],
        source_format=DataFormat.REGRESSION_MODEL,
        target_format=DataFormat.TIME_SERIES,
        context=context
    )
    
    result = shims['regression'].convert(request)
    
    print(f"✓ Conversion success: {result.success}")
    print(f"  Quality score: {result.quality_score:.3f}")
    
    # Examine preserved semantic context
    print("\n--- Step 3: Semantic Context Analysis ---")
    if 'domain_shim' in result.metadata:
        domain_metadata = result.metadata['domain_shim']
        semantic_context = domain_metadata.get('semantic_context', {})
        
        print("📋 Preserved Semantic Context:")
        print(f"  - Analytical goal: {semantic_context.get('analytical_goal')}")
        print(f"  - Domain context: {semantic_context.get('domain_context')}")
        print(f"  - Target use case: {semantic_context.get('target_use_case')}")
        
        mapping_info = domain_metadata.get('mapping_used', {})
        print(f"  - Quality preservation: {mapping_info.get('quality_preservation'):.3f}")
    
    # Show domain-specific enhancements
    if 'domain_conversion' in result.converted_data:
        conversion_info = result.converted_data['domain_conversion']
        print("🔄 Domain Conversion Info:")
        print(f"  - Source domain: {conversion_info['source']}")
        print(f"  - Target domain: {conversion_info['target']}")
        print(f"  - Semantic goal: {conversion_info['semantic_goal']}")
    
    # Performance metrics with context
    performance_metrics = result.performance_metrics
    print("📊 Performance Metrics:")
    print(f"  - Execution time: {performance_metrics.get('execution_time', 0):.3f}s")
    print(f"  - Domain mapping: {performance_metrics.get('domain_mapping', 'none')}")
    print(f"  - Semantic context: {performance_metrics.get('semantic_context', 'none')}")
    
    print("\n🎯 Semantic Result: Successfully preserved and enhanced semantic")
    print("   context across domain boundaries with quality tracking!")
    
    return result


def main():
    """Run all integration examples."""
    print("LocalData MCP v2.0 - Domain Shims Integration Examples")
    print("Demonstrating cross-domain data science workflows with semantic preservation")
    
    try:
        # Scenario 1: Statistical → Regression → Time Series
        scenario1_results = demonstrate_statistical_to_regression_to_timeseries_workflow()
        
        # Scenario 2: Time Series → Pattern Recognition → Statistical  
        scenario2_results = demonstrate_timeseries_to_pattern_recognition_to_statistical_workflow()
        
        # Scenario 3: Registry-based management
        scenario3_results = demonstrate_registry_based_workflow()
        
        # Scenario 4: Semantic context preservation
        scenario4_results = demonstrate_semantic_context_preservation()
        
        # Summary
        print("\n\n" + "=" * 80)
        print("INTEGRATION EXAMPLES SUMMARY")
        print("=" * 80)
        print("✅ All scenarios completed successfully!")
        print()
        print("Key Demonstrations:")
        print("1. 📊 Statistical → Regression → Time Series workflow")
        print("2. 📈 Time Series → Pattern Recognition → Statistical workflow")
        print("3. 🔧 Registry-based domain shim management")
        print("4. 🧠 Semantic context preservation across domains")
        print()
        print("Domain Shims Enable:")
        print("• Seamless cross-domain data science workflows")
        print("• Intelligent parameter mapping and result normalization")
        print("• Semantic context preservation and enhancement")
        print("• Automated quality preservation and monitoring")
        print("• Lifecycle management and health monitoring")
        print()
        print("🎯 LocalData MCP v2.0 Domain Shims: Bridging Data Science Domains!")
        
    except Exception as e:
        print(f"\n❌ Error during integration examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()