"""
Domain-specific analysis modules for the LocalData MCP pipeline.

This package contains specialized analysis domains that integrate with the
core pipeline framework, each implementing specific analytical capabilities
using sklearn-compatible transformers.

Available domains:
- statistical_analysis: Comprehensive statistical analysis tools
- regression_modeling: Comprehensive regression analysis and modeling capabilities
- pattern_recognition: Advanced clustering, dimensionality reduction, and anomaly detection
- optimization: Optimization and operations research capabilities
"""

from .statistical_analysis import (
    HypothesisTestingTransformer,
    ANOVAAnalysisTransformer,
    NonParametricTestTransformer,
    ExperimentalDesignTransformer,
    run_hypothesis_test,
    perform_anova,
    analyze_experiment_design,
    calculate_effect_sizes,
)

from .regression_modeling import (
    # Core transformers
    LinearRegressionTransformer,
    RegularizedRegressionTransformer,
    LogisticRegressionTransformer,
    PolynomialRegressionTransformer,
    ResidualAnalysisTransformer,
    FeatureSelectionTransformer,
    
    # Pipeline
    RegressionModelingPipeline,
    
    # High-level functions
    fit_regression_model,
    evaluate_model_performance,
    analyze_residuals,
    select_features,
    
    # Result classes
    RegressionModelResult,
    ResidualAnalysisResult
)

from .pattern_recognition import (
    # Core transformers
    ClusteringTransformer,
    DimensionalityReductionTransformer,
    AnomalyDetectionTransformer,
    PatternEvaluationTransformer,
    
    # High-level functions
    perform_clustering,
    reduce_dimensions,
    detect_anomalies,
    evaluate_patterns,
    
    # Result classes
    ClusteringResult,
    DimensionalityReductionResult,
    AnomalyDetectionResult,
    PatternEvaluationResult
)

from .optimization import (
    # Core transformers
    LinearProgrammingSolver,
    ConstrainedOptimizer,
    NetworkAnalyzer,
    AssignmentSolver,
    
    # High-level functions
    solve_linear_program,
    optimize_constrained,
    analyze_network,
    solve_assignment_problem,
    
    # Result classes
    OptimizationResult,
    LinearProgramResult,
    ConstrainedOptResult,
    NetworkAnalysisResult,
    AssignmentResult
)

from .geospatial_analysis import (
    # Core transformers
    GeospatialDependencyChecker,
    SpatialCoordinateTransformer,
    SpatialDistanceTransformer,
    SpatialGeometryTransformer,
    SpatialAutocorrelationTransformer,
    SpatialInterpolationTransformer,
    SpatialJoinTransformer,
    SpatialOverlayTransformer,
    SpatialNetworkTransformer,
    
    # Spatial data structures
    SpatialPoint,
    SpatialDataFrame,
    
    # Analysis engines
    SpatialJoinEngine,
    SpatialOverlayEngine,
    SpatialAggregator,
    SpatialNetwork,
    NetworkRouter,
    AccessibilityAnalyzer,
    IsochroneGenerator,
    
    # High-level functions
    analyze_spatial_autocorrelation,
    perform_spatial_clustering,
    calculate_spatial_distance,
    optimize_route,
    optimize_routes,
    analyze_accessibility,
    generate_service_isochrones,
    perform_spatial_join,
    perform_spatial_overlay,
    aggregate_points_in_polygons,
    
    # Result classes
    SpatialJoinResult,
    OverlayResult,
    RouteResult,
    AccessibilityResult,
    IsochroneResult,
    
    # Pipeline
    GeospatialAnalysisPipeline,
    
    # Enums
    GeospatialLibrary,
    SpatialJoinType,
    OverlayOperation,
    NetworkAnalysisType
)

__all__ = [
    # Statistical Analysis Domain
    "HypothesisTestingTransformer",
    "ANOVAAnalysisTransformer", 
    "NonParametricTestTransformer",
    "ExperimentalDesignTransformer",
    "run_hypothesis_test",
    "perform_anova",
    "analyze_experiment_design",
    "calculate_effect_sizes",
    
    # Regression Modeling Domain
    "LinearRegressionTransformer",
    "RegularizedRegressionTransformer",
    "LogisticRegressionTransformer",
    "PolynomialRegressionTransformer",
    "ResidualAnalysisTransformer",
    "FeatureSelectionTransformer",
    "RegressionModelingPipeline",
    "fit_regression_model",
    "evaluate_model_performance",
    "analyze_residuals",
    "select_features",
    "RegressionModelResult",
    "ResidualAnalysisResult",
    
    # Pattern Recognition Domain
    "ClusteringTransformer",
    "DimensionalityReductionTransformer", 
    "AnomalyDetectionTransformer",
    "PatternEvaluationTransformer",
    "perform_clustering",
    "reduce_dimensions",
    "detect_anomalies",
    "evaluate_patterns",
    "ClusteringResult",
    "DimensionalityReductionResult",
    "AnomalyDetectionResult",
    "PatternEvaluationResult",
    
    # Optimization Domain
    "LinearProgrammingSolver",
    "ConstrainedOptimizer",
    "NetworkAnalyzer",
    "AssignmentSolver",
    "solve_linear_program",
    "optimize_constrained",
    "analyze_network",
    "solve_assignment_problem",
    "OptimizationResult",
    "LinearProgramResult",
    "ConstrainedOptResult",
    "NetworkAnalysisResult",
    "AssignmentResult",
    
    # Geospatial Analysis Domain
    "GeospatialDependencyChecker",
    "SpatialCoordinateTransformer",
    "SpatialDistanceTransformer",
    "SpatialGeometryTransformer",
    "SpatialAutocorrelationTransformer",
    "SpatialInterpolationTransformer",
    "SpatialJoinTransformer",
    "SpatialOverlayTransformer",
    "SpatialNetworkTransformer",
    "SpatialPoint",
    "SpatialDataFrame",
    "SpatialJoinEngine",
    "SpatialOverlayEngine",
    "SpatialAggregator",
    "SpatialNetwork",
    "NetworkRouter",
    "AccessibilityAnalyzer",
    "IsochroneGenerator",
    "analyze_spatial_autocorrelation",
    "perform_spatial_clustering",
    "calculate_spatial_distance",
    "optimize_route",
    "optimize_routes",
    "analyze_accessibility",
    "generate_service_isochrones",
    "perform_spatial_join",
    "perform_spatial_overlay",
    "aggregate_points_in_polygons",
    "SpatialJoinResult",
    "OverlayResult",
    "RouteResult",
    "AccessibilityResult",
    "IsochroneResult",
    "GeospatialAnalysisPipeline",
    "GeospatialLibrary",
    "SpatialJoinType",
    "OverlayOperation",
    "NetworkAnalysisType"
]