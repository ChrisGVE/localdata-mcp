"""
Integration Examples for Automatic Shim Insertion Logic.

This module demonstrates how to use the automatic shim insertion system
for seamless cross-domain data science workflows in LocalData MCP v2.0.

Examples include:
- Basic pipeline analysis and shim injection
- Multi-domain pipeline composition
- Custom optimization strategies
- Performance monitoring and validation
"""

import time
import pandas as pd
import numpy as np
from typing import List, Dict, Any

from src.localdata_mcp.pipeline.integration import (
    # Core components
    DataFormat, PipelineAnalyzer, ShimInjector, PipelineValidator,
    
    # Data structures
    PipelineStep, OptimizationCriteria, AnalysisType, InjectionStrategy,
    
    # Factory functions
    create_pipeline_analyzer, create_shim_injector, create_pipeline_validator,
    create_optimization_criteria, create_pipeline_step,
    
    # Infrastructure
    PipelineCompatibilityMatrix, ShimRegistry, EnhancedShimAdapter,
    create_compatibility_matrix, create_shim_registry,
    
    # Utility function
    analyze_and_fix_pipeline
)


def example1_basic_pipeline_analysis():
    """
    Example 1: Basic pipeline analysis with automatic shim injection.
    
    Demonstrates analyzing a simple cross-domain pipeline and automatically
    inserting shims to resolve format incompatibilities.
    """
    print("=== Example 1: Basic Pipeline Analysis ===")
    
    # Create infrastructure components
    compatibility_matrix = create_compatibility_matrix(enable_caching=True)
    shim_registry = create_shim_registry(compatibility_matrix=compatibility_matrix)
    
    # Define a simple pipeline with format incompatibilities
    pipeline_steps = [
        create_pipeline_step(
            step_id="load_data",
            domain="data_loading",
            operation="load_csv",
            input_format=DataFormat.CSV,
            output_format=DataFormat.PANDAS_DATAFRAME,
            metadata={"file_path": "data.csv"}
        ),
        create_pipeline_step(
            step_id="preprocess",
            domain="preprocessing", 
            operation="normalize_features",
            input_format=DataFormat.PANDAS_DATAFRAME,
            output_format=DataFormat.NUMPY_ARRAY,
            metadata={"method": "standard_scaling"}
        ),
        create_pipeline_step(
            step_id="analyze",
            domain="statistical_analysis",
            operation="correlation_analysis",
            input_format=DataFormat.PANDAS_DATAFRAME,  # Incompatible! Expects DataFrame but gets NumPy array
            output_format=DataFormat.STATISTICAL_RESULT,
            metadata={"method": "pearson"}
        )
    ]
    
    print(f"Original pipeline has {len(pipeline_steps)} steps")
    print("Identified format incompatibility: NumPy array -> DataFrame")
    
    # Use high-level utility function to analyze and fix
    result = analyze_and_fix_pipeline(
        pipeline_steps=pipeline_steps,
        compatibility_matrix=compatibility_matrix,
        shim_registry=shim_registry,
        auto_fix=True
    )
    
    print(f"\nAnalysis Results:")
    print(f"  - Pipeline ID: {result.get('pipeline_id', 'N/A')}")
    print(f"  - Is Valid: {result.get('is_valid', False)}")
    print(f"  - Validation Score: {result.get('validation_score', 0.0):.2f}")
    print(f"  - Fixes Applied: {len(result.get('fixes_applied', []))}")
    print(f"  - Final Pipeline Steps: {len(result.get('final_pipeline', []))}")
    
    if result.get('fixes_applied'):
        print("\nShims Inserted:")
        for fix in result['fixes_applied']:
            shim_step = fix.get('shim_step')
            if shim_step:
                print(f"  - {shim_step.step_id}: {shim_step.operation}")
    
    return result


def example2_multi_domain_pipeline():
    """
    Example 2: Multi-domain pipeline with complex format transitions.
    
    Demonstrates handling a complex pipeline spanning multiple data science
    domains with various format requirements and optimization strategies.
    """
    print("\n=== Example 2: Multi-Domain Pipeline ===")
    
    # Create infrastructure with custom optimization
    compatibility_matrix = create_compatibility_matrix()
    shim_registry = create_shim_registry(compatibility_matrix=compatibility_matrix)
    
    # Create analyzer with performance focus
    analyzer = create_pipeline_analyzer(
        compatibility_matrix=compatibility_matrix,
        shim_registry=shim_registry,
        enable_caching=True,
        max_analysis_threads=4
    )
    
    # Define complex multi-domain pipeline
    pipeline_steps = [
        # Data loading domain
        create_pipeline_step(
            "load", "data_loading", "load_parquet",
            DataFormat.PARQUET, DataFormat.PANDAS_DATAFRAME,
            metadata={"chunks": True, "memory_map": True}
        ),
        
        # Preprocessing domain
        create_pipeline_step(
            "clean", "preprocessing", "clean_missing_values",
            DataFormat.PANDAS_DATAFRAME, DataFormat.PANDAS_DATAFRAME,
            metadata={"strategy": "drop", "threshold": 0.1}
        ),
        
        # Feature engineering domain
        create_pipeline_step(
            "engineer", "feature_engineering", "create_polynomial_features",
            DataFormat.PANDAS_DATAFRAME, DataFormat.NUMPY_ARRAY,
            metadata={"degree": 2, "include_bias": False}
        ),
        
        # Statistical analysis domain (format mismatch!)
        create_pipeline_step(
            "stats", "statistical_analysis", "hypothesis_testing",
            DataFormat.PANDAS_DATAFRAME, DataFormat.STATISTICAL_RESULT,
            metadata={"test": "t_test", "alpha": 0.05}
        ),
        
        # Regression modeling domain 
        create_pipeline_step(
            "model", "regression_modeling", "linear_regression",
            DataFormat.PANDAS_DATAFRAME, DataFormat.REGRESSION_MODEL,  # Another format mismatch!
            metadata={"fit_intercept": True, "normalize": False}
        ),
        
        # Time series domain (significant format mismatch!)
        create_pipeline_step(
            "forecast", "time_series", "arima_forecast",
            DataFormat.TIME_SERIES, DataFormat.FORECAST_RESULT,
            metadata={"order": (1, 1, 1), "seasonal": False}
        ),
        
        # Pattern recognition domain
        create_pipeline_step(
            "cluster", "pattern_recognition", "kmeans_clustering",
            DataFormat.NUMPY_ARRAY, DataFormat.CLUSTERING_RESULT,
            metadata={"n_clusters": 5, "random_state": 42}
        )
    ]
    
    print(f"Complex pipeline has {len(pipeline_steps)} steps across multiple domains")
    
    # Perform comprehensive analysis
    analysis_result = analyzer.analyze_pipeline(
        pipeline_steps=pipeline_steps,
        analysis_type=AnalysisType.COMPLETE,
        pipeline_id="multi_domain_example"
    )
    
    print(f"\nComprehensive Analysis:")
    print(f"  - Compatible: {analysis_result.is_compatible}")
    print(f"  - Compatibility Score: {analysis_result.compatibility_score:.2f}")
    print(f"  - Incompatible Connections: {len(analysis_result.incompatible_connections)}")
    print(f"  - Issues Identified: {len(analysis_result.identified_issues)}")
    print(f"  - Shim Recommendations: {len(analysis_result.shim_recommendations)}")
    print(f"  - Analysis Time: {analysis_result.execution_time:.3f}s")
    
    # Show detailed issues
    if analysis_result.identified_issues:
        print(f"\nIdentified Issues:")
        for i, issue in enumerate(analysis_result.identified_issues[:3], 1):  # Show first 3
            print(f"  {i}. {issue.severity.upper()}: {issue.description}")
            if issue.suggested_solutions:
                print(f"     Solution: {issue.suggested_solutions[0]}")
    
    return analysis_result


def example3_custom_optimization_strategies():
    """
    Example 3: Custom optimization strategies for shim selection.
    
    Demonstrates different optimization strategies and their impact on
    shim selection and pipeline performance.
    """
    print("\n=== Example 3: Custom Optimization Strategies ===")
    
    # Create infrastructure
    compatibility_matrix = create_compatibility_matrix()
    shim_registry = create_shim_registry(compatibility_matrix=compatibility_matrix)
    
    # Define optimization criteria variants
    optimization_strategies = {
        "performance_focused": create_optimization_criteria(
            prioritize_performance=True,
            prioritize_quality=False,
            prioritize_memory=False,
            performance_weight=0.7,
            quality_weight=0.2,
            cost_weight=0.1
        ),
        "quality_focused": create_optimization_criteria(
            prioritize_performance=False,
            prioritize_quality=True,
            prioritize_memory=False,
            performance_weight=0.1,
            quality_weight=0.8,
            cost_weight=0.1
        ),
        "memory_efficient": create_optimization_criteria(
            prioritize_performance=False,
            prioritize_quality=False,
            prioritize_memory=True,
            performance_weight=0.2,
            quality_weight=0.3,
            cost_weight=0.5
        ),
        "balanced": create_optimization_criteria(
            prioritize_performance=True,
            prioritize_quality=True,
            prioritize_memory=True,
            performance_weight=0.33,
            quality_weight=0.33,
            cost_weight=0.34
        )
    }
    
    # Simple pipeline for testing strategies
    test_pipeline = [
        create_pipeline_step(
            "input", "data_loading", "load_data",
            DataFormat.CSV, DataFormat.PANDAS_DATAFRAME
        ),
        create_pipeline_step(
            "process", "preprocessing", "sparse_transform",
            DataFormat.SCIPY_SPARSE, DataFormat.NUMPY_ARRAY  # Format mismatch
        )
    ]
    
    print(f"Testing {len(optimization_strategies)} optimization strategies on simple pipeline")
    
    # Test each strategy
    results = {}
    for strategy_name, criteria in optimization_strategies.items():
        print(f"\n--- Testing {strategy_name} strategy ---")
        
        # Create injector with specific criteria
        injector = create_shim_injector(
            shim_registry=shim_registry,
            compatibility_matrix=compatibility_matrix,
            optimization_criteria=criteria
        )
        
        # Analyze pipeline first
        analyzer = create_pipeline_analyzer(compatibility_matrix, shim_registry)
        analysis_result = analyzer.analyze_pipeline(test_pipeline)
        
        # Apply injection strategy
        injection_strategies = [
            InjectionStrategy.MINIMAL,
            InjectionStrategy.BALANCED,
            InjectionStrategy.OPTIMAL,
            InjectionStrategy.SAFE
        ]
        
        strategy_results = {}
        for injection_strategy in injection_strategies:
            try:
                modified_pipeline, injection_metadata = injector.inject_shims_for_pipeline(
                    pipeline_steps=test_pipeline,
                    analysis_result=analysis_result,
                    strategy=injection_strategy
                )
                
                strategy_results[injection_strategy.value] = {
                    'shims_inserted': len(injection_metadata.get('injections', [])),
                    'errors': len(injection_metadata.get('errors', [])),
                    'execution_time': injection_metadata.get('execution_time', 0.0)
                }
                
            except Exception as e:
                strategy_results[injection_strategy.value] = {
                    'error': str(e)
                }
        
        results[strategy_name] = strategy_results
        
        # Show results for this optimization strategy
        for inj_strategy, result in strategy_results.items():
            if 'error' not in result:
                print(f"  {inj_strategy}: {result['shims_inserted']} shims, "
                      f"{result['execution_time']:.3f}s, {result['errors']} errors")
            else:
                print(f"  {inj_strategy}: ERROR - {result['error']}")
    
    return results


def example4_performance_monitoring():
    """
    Example 4: Performance monitoring and validation with execution plans.
    
    Demonstrates comprehensive pipeline validation with performance monitoring,
    execution plan generation, and optimization suggestions.
    """
    print("\n=== Example 4: Performance Monitoring ===")
    
    # Create infrastructure
    compatibility_matrix = create_compatibility_matrix()
    shim_registry = create_shim_registry(compatibility_matrix=compatibility_matrix)
    
    # Create comprehensive validator
    validator = create_pipeline_validator(
        compatibility_matrix=compatibility_matrix,
        shim_registry=shim_registry
    )
    
    # Define performance-critical pipeline
    performance_pipeline = [
        create_pipeline_step(
            "load", "data_loading", "load_large_dataset",
            DataFormat.HDF5, DataFormat.PANDAS_DATAFRAME,
            metadata={"size_gb": 2.5, "chunk_size": 10000}
        ),
        create_pipeline_step(
            "sparse", "preprocessing", "create_sparse_matrix", 
            DataFormat.PANDAS_DATAFRAME, DataFormat.SCIPY_SPARSE,
            metadata={"density_threshold": 0.05}
        ),
        create_pipeline_step(
            "dense", "feature_engineering", "dense_features",
            DataFormat.NUMPY_ARRAY, DataFormat.PANDAS_DATAFRAME,  # Format mismatch
            metadata={"feature_count": 1000}
        ),
        create_pipeline_step(
            "model", "regression_modeling", "large_scale_regression",
            DataFormat.PANDAS_DATAFRAME, DataFormat.REGRESSION_MODEL,
            metadata={"solver": "saga", "max_iter": 1000}
        ),
        create_pipeline_step(
            "optimize", "optimization", "hyperparameter_search",
            DataFormat.REGRESSION_MODEL, DataFormat.OPTIMIZATION_RESULT,
            metadata={"search_space": "large", "cv_folds": 5}
        )
    ]
    
    print(f"Performance-critical pipeline with {len(performance_pipeline)} steps")
    
    # Comprehensive validation with auto-fix
    validation_result = validator.validate_and_fix_pipeline(
        pipeline_steps=performance_pipeline,
        auto_fix=True,
        validation_level="strict"
    )
    
    print(f"\nValidation Results:")
    print(f"  - Valid: {validation_result['is_valid']}")
    print(f"  - Score: {validation_result['validation_score']:.2f}")
    print(f"  - Structural Issues: {len(validation_result.get('structural_issues', []))}")
    print(f"  - Performance Issues: {len(validation_result.get('performance_issues', []))}")
    print(f"  - Fixes Applied: {len(validation_result.get('fixes_applied', []))}")
    print(f"  - Validation Time: {validation_result['execution_time']:.3f}s")
    
    # Show execution plan if available
    execution_plan = validation_result.get('execution_plan')
    if execution_plan:
        print(f"\nExecution Plan:")
        print(f"  - Total Estimated Time: {execution_plan['estimated_total_time']:.1f}s")
        print(f"  - Peak Memory Usage: {execution_plan['estimated_total_memory']:.0f}MB")
        print(f"  - Parallel Opportunities: {len(execution_plan['parallel_opportunities'])}")
        
        if execution_plan['optimization_suggestions']:
            print(f"  - Optimization Suggestions:")
            for suggestion in execution_plan['optimization_suggestions'][:2]:
                print(f"    * {suggestion}")
    
    # Show performance issues
    if validation_result.get('performance_issues'):
        print(f"\nPerformance Issues Detected:")
        for issue in validation_result['performance_issues'][:3]:
            print(f"  - {issue}")
    
    return validation_result


def example5_real_world_scenario():
    """
    Example 5: Real-world data science pipeline scenario.
    
    Simulates a complete end-to-end data science workflow with multiple
    domains, format transitions, and automatic optimization.
    """
    print("\n=== Example 5: Real-World Data Science Pipeline ===")
    
    # Create infrastructure with realistic settings
    compatibility_matrix = create_compatibility_matrix(enable_caching=True, cache_size=1000)
    shim_registry = create_shim_registry(
        compatibility_matrix=compatibility_matrix,
        enable_auto_discovery=True,
        max_concurrent_health_checks=8,
        health_check_interval_seconds=300
    )
    
    # Comprehensive pipeline simulating real ML workflow
    ml_pipeline = [
        # Data ingestion
        create_pipeline_step(
            "ingest", "data_loading", "multi_source_ingestion",
            DataFormat.JSON, DataFormat.PANDAS_DATAFRAME,
            metadata={
                "sources": ["api", "database", "files"],
                "parallel_load": True
            }
        ),
        
        # Data validation and cleaning
        create_pipeline_step(
            "validate", "data_validation", "schema_validation", 
            DataFormat.PANDAS_DATAFRAME, DataFormat.PANDAS_DATAFRAME,
            metadata={"strict_mode": True, "auto_fix": False}
        ),
        
        # Exploratory data analysis
        create_pipeline_step(
            "eda", "statistical_analysis", "comprehensive_eda",
            DataFormat.PANDAS_DATAFRAME, DataFormat.STATISTICAL_RESULT,
            metadata={"include_plots": True, "correlation_analysis": True}
        ),
        
        # Feature engineering
        create_pipeline_step(
            "features", "feature_engineering", "automated_feature_engineering",
            DataFormat.PANDAS_DATAFRAME, DataFormat.NUMPY_ARRAY,  # Format transition
            metadata={
                "create_interactions": True,
                "polynomial_features": 2,
                "scaling": "robust"
            }
        ),
        
        # Dimensionality reduction 
        create_pipeline_step(
            "reduce", "pattern_recognition", "pca_reduction",
            DataFormat.NUMPY_ARRAY, DataFormat.NUMPY_ARRAY,
            metadata={"n_components": 0.95, "whiten": True}
        ),
        
        # Time series preparation (format mismatch!)
        create_pipeline_step(
            "timeseries", "time_series", "prepare_sequences",
            DataFormat.TIME_SERIES, DataFormat.TIME_SERIES,
            metadata={"sequence_length": 24, "forecast_horizon": 7}
        ),
        
        # Model training
        create_pipeline_step(
            "train", "regression_modeling", "ensemble_training",
            DataFormat.NUMPY_ARRAY, DataFormat.REGRESSION_MODEL,  # Another format mismatch!
            metadata={
                "models": ["random_forest", "gradient_boosting", "neural_network"],
                "cross_validation": 5
            }
        ),
        
        # Model evaluation
        create_pipeline_step(
            "evaluate", "model_evaluation", "comprehensive_evaluation",
            DataFormat.REGRESSION_MODEL, DataFormat.STATISTICAL_RESULT,
            metadata={
                "metrics": ["rmse", "mae", "r2", "mape"],
                "bootstrap_confidence": True
            }
        ),
        
        # Hyperparameter optimization
        create_pipeline_step(
            "optimize", "optimization", "bayesian_optimization",
            DataFormat.REGRESSION_MODEL, DataFormat.OPTIMIZATION_RESULT,
            metadata={
                "n_trials": 100,
                "optimization_metric": "rmse",
                "parallel_jobs": 4
            }
        ),
        
        # Final model deployment preparation
        create_pipeline_step(
            "deploy", "model_deployment", "prepare_for_deployment",
            DataFormat.REGRESSION_MODEL, DataFormat.PYTHON_DICT,  # Final format transition
            metadata={
                "serialization_format": "pickle",
                "include_preprocessors": True,
                "api_wrapper": True
            }
        )
    ]
    
    print(f"Real-world ML pipeline with {len(ml_pipeline)} steps")
    print("Spans 8 different domains with multiple format transitions")
    
    # Use high-level utility for complete analysis and fixing
    start_time = time.time()
    
    final_result = analyze_and_fix_pipeline(
        pipeline_steps=ml_pipeline,
        compatibility_matrix=compatibility_matrix,
        shim_registry=shim_registry,
        auto_fix=True
    )
    
    total_time = time.time() - start_time
    
    print(f"\nComplete Pipeline Analysis & Fix:")
    print(f"  - Processing Time: {total_time:.3f}s")
    print(f"  - Final Pipeline Valid: {final_result.get('is_valid', False)}")
    print(f"  - Compatibility Score: {final_result.get('validation_score', 0.0):.2f}")
    print(f"  - Original Steps: {final_result.get('original_steps_count', 0)}")
    print(f"  - Final Steps: {len(final_result.get('final_pipeline', []))}")
    print(f"  - Shims Inserted: {len(final_result.get('fixes_applied', []))}")
    
    # Show statistics
    compatibility_stats = compatibility_matrix.get_statistics()
    registry_stats = shim_registry.get_registry_stats()
    
    print(f"\nInfrastructure Statistics:")
    print(f"  - Compatibility Assessments: {compatibility_stats.get('total_assessments', 0)}")
    print(f"  - Cache Hit Rate: {compatibility_stats.get('cache_hit_rate', 0.0):.2%}")
    print(f"  - Registry Adapters: {registry_stats.get('total_adapters', 0)}")
    print(f"  - Total Conversions: {registry_stats.get('total_conversions', 0)}")
    
    # Show inserted shims
    if final_result.get('fixes_applied'):
        print(f"\nAutomatically Inserted Shims:")
        for i, fix in enumerate(final_result['fixes_applied'][:5], 1):  # Show first 5
            shim_step = fix.get('shim_step')
            if shim_step:
                print(f"  {i}. {shim_step.operation}")
                print(f"     Input: {shim_step.input_format.value}")
                print(f"     Output: {shim_step.output_format.value}")
    
    return final_result


def run_all_examples():
    """Run all examples in sequence."""
    print("Running Automatic Shim Insertion Examples for LocalData MCP v2.0")
    print("=" * 70)
    
    examples = [
        example1_basic_pipeline_analysis,
        example2_multi_domain_pipeline,
        example3_custom_optimization_strategies,
        example4_performance_monitoring,
        example5_real_world_scenario
    ]
    
    results = {}
    total_start_time = time.time()
    
    for i, example_func in enumerate(examples, 1):
        try:
            print(f"\n{'='*20} Running Example {i} {'='*20}")
            start_time = time.time()
            result = example_func()
            execution_time = time.time() - start_time
            
            results[example_func.__name__] = {
                'result': result,
                'execution_time': execution_time,
                'success': True
            }
            
            print(f"\nExample {i} completed successfully in {execution_time:.3f}s")
            
        except Exception as e:
            execution_time = time.time() - start_time
            results[example_func.__name__] = {
                'error': str(e),
                'execution_time': execution_time,
                'success': False
            }
            
            print(f"\nExample {i} failed after {execution_time:.3f}s: {e}")
    
    total_time = time.time() - total_start_time
    
    print(f"\n{'='*20} Summary {'='*20}")
    print(f"Total execution time: {total_time:.3f}s")
    print(f"Examples completed: {sum(1 for r in results.values() if r['success'])}/{len(examples)}")
    
    for name, result in results.items():
        status = "SUCCESS" if result['success'] else "FAILED"
        print(f"  {name}: {status} ({result['execution_time']:.3f}s)")
    
    return results


if __name__ == "__main__":
    # Run all examples when script is executed directly
    results = run_all_examples()
    
    print(f"\nAutomatic Shim Insertion Examples completed!")
    print(f"Check the results above to see the system in action.")