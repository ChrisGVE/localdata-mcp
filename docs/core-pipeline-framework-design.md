# Core Pipeline Framework Design
## LocalData MCP v2.0 - Comprehensive Data Science Platform

**Design Phase: Architecture Specification**  
**Status: Complete Design - Ready for Implementation**  
**Author: Claude Code**  
**Date: 2025-01-01**

---

## Executive Summary

This document specifies the concrete Core Pipeline Framework that implements the First Principles Architecture for LocalData MCP v2.0. The framework provides the foundational infrastructure to support 15+ data science domains through intention-driven interfaces, streaming-first processing, and progressive disclosure patterns.

### Key Design Achievements

1. **Unified Pipeline Base Classes** - sklearn-compatible transformers with LLM-friendly interfaces
2. **Streaming-First Architecture** - Memory-bounded processing for 16-64GB memory targets
3. **Context-Aware Composition** - Enriched results designed for downstream tool chaining
4. **Progressive Disclosure** - Simple defaults with powerful advanced capabilities
5. **Modular Domain Integration** - Extensible framework for cross-domain workflows

---

## Core Architecture

### 1. Input Stage Architecture

The input stage handles SQL/CSV/JSON → DataFrame conversion with streaming compatibility:

```python
class DataInputPipeline(BaseEstimator, TransformerMixin):
    """
    Input stage pipeline with streaming compatibility and LLM-friendly interface.
    
    First Principle: Intention-Driven Interface
    - LLM agents express what data they want, not how to get it
    - Automatic format detection and conversion
    - Intelligent streaming decisions based on data characteristics
    """
    
    def __init__(self, 
                 intention: str,  # "analyze sales data", "explore customer patterns"
                 data_source: Union[str, Dict[str, Any]],  # SQL, file path, or connection info
                 streaming_threshold_mb: int = 100,
                 chunk_size_adaptive: bool = True):
        """
        Initialize input pipeline with intention-driven parameters.
        
        Args:
            intention: Natural language description of analytical intent
            data_source: SQL query, file path, or database connection
            streaming_threshold_mb: Auto-enable streaming above this data size
            chunk_size_adaptive: Enable adaptive chunk sizing
        """
        self.intention = intention
        self.data_source = data_source
        self.streaming_threshold_mb = streaming_threshold_mb
        self.chunk_size_adaptive = chunk_size_adaptive
        
        # Internal state for streaming
        self._streaming_source: Optional[StreamingDataSource] = None
        self._data_characteristics: Optional[Dict[str, Any]] = None
        self._memory_profile: Optional[MemoryStatus] = None
    
    def fit(self, X=None, y=None):
        """
        Analyze data source and prepare streaming configuration.
        
        Returns self for sklearn compatibility while building streaming context.
        """
        # Analyze data source characteristics
        self._data_characteristics = self._analyze_data_source()
        
        # Configure streaming based on data size and system memory
        self._configure_streaming()
        
        return self
    
    def transform(self, X=None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Transform data source into DataFrame with enriched metadata.
        
        Returns:
            Tuple[pd.DataFrame, Dict[str, Any]]: (data, enriched_metadata)
            
        The metadata includes:
        - data_characteristics: Schema, size, complexity
        - streaming_info: Pagination, chunking recommendations  
        - composition_context: For downstream tool chaining
        - intention_context: Original analytical intent preserved
        """
        if self._streaming_source is None:
            raise ValueError("Pipeline not fitted. Call fit() first.")
        
        # Execute streaming transformation
        executor = StreamingQueryExecutor()
        query_id = f"input_pipeline_{int(time.time() * 1000)}"
        
        first_chunk, streaming_metadata = executor.execute_streaming(
            self._streaming_source, query_id
        )
        
        # Build enriched metadata for composition
        enriched_metadata = self._build_composition_metadata(
            first_chunk, streaming_metadata
        )
        
        return first_chunk, enriched_metadata
```

### 2. Preprocessing Stage Design

Data cleaning, normalization, and feature engineering pipeline:

```python
class DataPreprocessingPipeline(BaseEstimator, TransformerMixin):
    """
    Preprocessing pipeline with progressive disclosure and streaming support.
    
    First Principle: Progressive Disclosure Architecture
    - Simple by default: automatic data cleaning and type inference
    - Powerful when needed: custom transformations and advanced preprocessing
    """
    
    def __init__(self,
                 preprocessing_intent: str = "auto",  # "auto", "minimal", "comprehensive"
                 custom_transformations: Optional[List[Callable]] = None,
                 streaming_compatible: bool = True,
                 memory_efficient: bool = True):
        """
        Initialize preprocessing with progressive complexity.
        
        Args:
            preprocessing_intent: Level of preprocessing complexity
            custom_transformations: Optional custom transformation functions
            streaming_compatible: Enable chunk-by-chunk processing
            memory_efficient: Optimize for memory usage over speed
        """
        self.preprocessing_intent = preprocessing_intent
        self.custom_transformations = custom_transformations or []
        self.streaming_compatible = streaming_compatible
        self.memory_efficient = memory_efficient
        
        # Progressive disclosure: build transformation pipeline based on intent
        self._transformation_pipeline = self._build_transformation_pipeline()
        
    def _build_transformation_pipeline(self) -> List[Callable]:
        """Build transformation pipeline based on preprocessing intent."""
        pipeline = []
        
        if self.preprocessing_intent == "auto":
            pipeline.extend([
                self._handle_missing_values,
                self._infer_and_convert_types,
                self._detect_and_handle_outliers,
                self._normalize_text_columns,
                self._encode_categorical_variables
            ])
        elif self.preprocessing_intent == "minimal":
            pipeline.extend([
                self._handle_missing_values,
                self._infer_and_convert_types
            ])
        elif self.preprocessing_intent == "comprehensive":
            pipeline.extend([
                self._handle_missing_values,
                self._infer_and_convert_types,
                self._detect_and_handle_outliers,
                self._normalize_text_columns,
                self._encode_categorical_variables,
                self._feature_scaling,
                self._dimensionality_assessment,
                self._correlation_analysis
            ])
        
        # Add custom transformations
        pipeline.extend(self.custom_transformations)
        
        return pipeline
    
    def fit(self, X: pd.DataFrame, y=None):
        """
        Fit preprocessing transformations to the data.
        
        Learns data characteristics for streaming-compatible transformations.
        """
        # Learn data characteristics for each transformation
        self._transformation_states = {}
        
        for transform_func in self._transformation_pipeline:
            transform_name = transform_func.__name__
            self._transformation_states[transform_name] = self._fit_transform_state(
                transform_func, X
            )
        
        return self
    
    def transform(self, X: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Apply preprocessing transformations with streaming support.
        
        Returns:
            Tuple[pd.DataFrame, Dict[str, Any]]: (preprocessed_data, preprocessing_metadata)
        """
        processed_data = X.copy()
        preprocessing_log = []
        
        # Apply each transformation in pipeline
        for transform_func in self._transformation_pipeline:
            transform_name = transform_func.__name__
            transform_state = self._transformation_states.get(transform_name, {})
            
            try:
                processed_data, transform_metadata = transform_func(
                    processed_data, **transform_state
                )
                preprocessing_log.append({
                    "transformation": transform_name,
                    "status": "success",
                    "metadata": transform_metadata
                })
            except Exception as e:
                preprocessing_log.append({
                    "transformation": transform_name,
                    "status": "error",
                    "error": str(e)
                })
                logger.warning(f"Preprocessing transformation {transform_name} failed: {e}")
        
        # Build enriched metadata
        enriched_metadata = {
            "preprocessing_log": preprocessing_log,
            "data_quality_score": self._calculate_data_quality_score(processed_data),
            "transformation_summary": self._summarize_transformations(preprocessing_log),
            "streaming_compatibility": {
                "chunk_safe": self.streaming_compatible,
                "memory_efficient": self.memory_efficient
            },
            "composition_context": {
                "ready_for_analysis": True,
                "suggested_next_steps": self._suggest_next_steps(processed_data),
                "data_characteristics": self._analyze_processed_data(processed_data)
            }
        }
        
        return processed_data, enriched_metadata
```

### 3. Output Standardization Patterns

Consistent result formats across all domains:

```python
class AnalysisOutputStandardizer:
    """
    Standardizes analysis outputs across all data science domains.
    
    First Principle: Context-Aware Composition
    - All outputs include composition metadata for downstream tools
    - Standardized result schema across domains
    - LLM-friendly result interpretation guidance
    """
    
    @staticmethod
    def standardize_analysis_result(
        domain: str,  # "time_series", "ml", "statistical", etc.
        analysis_type: str,  # "forecast", "classification", "correlation", etc.  
        raw_result: Any,
        input_context: Dict[str, Any],
        computation_metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Standardize analysis results into consistent format.
        
        Returns standardized result with:
        - interpretation: LLM-friendly explanation of results
        - visualizations: Chart specifications for display
        - composition_metadata: For chaining with other tools
        - quality_metrics: Confidence and reliability indicators
        - next_steps: Recommended follow-up analyses
        """
        
        base_result = {
            "domain": domain,
            "analysis_type": analysis_type,
            "timestamp": time.time(),
            "success": True,
            "result": {
                "primary": raw_result,  # Main analysis result
                "interpretation": AnalysisInterpreter.interpret_result(
                    domain, analysis_type, raw_result
                ),
                "quality_metrics": QualityMetrics.calculate_metrics(
                    domain, analysis_type, raw_result, input_context
                ),
                "visualizations": VisualizationSpecGenerator.generate_specs(
                    domain, analysis_type, raw_result
                )
            },
            "metadata": {
                "input_context": input_context,
                "computation": computation_metadata,
                "streaming_info": computation_metadata.get("streaming_info", {}),
                "memory_usage": computation_metadata.get("memory_usage", {})
            },
            "composition_context": {
                "downstream_compatible_tools": ToolCompatibility.find_compatible_tools(
                    domain, analysis_type, raw_result
                ),
                "suggested_compositions": CompositionSuggester.suggest_combinations(
                    domain, analysis_type, raw_result, input_context
                ),
                "data_artifacts": DataArtifactExtractor.extract_artifacts(
                    raw_result, input_context
                )
            },
            "guidance": {
                "interpretation_guide": f"This {analysis_type} analysis from the {domain} domain shows...",
                "confidence_level": QualityMetrics.get_confidence_level(raw_result),
                "limitations": AnalysisLimitations.identify_limitations(
                    domain, analysis_type, raw_result, input_context
                ),
                "next_steps": NextStepRecommender.recommend_next_analyses(
                    domain, analysis_type, raw_result, input_context
                )
            }
        }
        
        return base_result
```

### 4. Error Handling Framework

Graceful degradation and error propagation:

```python
class PipelineErrorHandler:
    """
    Comprehensive error handling for pipeline operations.
    
    First Principle: Streaming-First Data Science  
    - Error recovery that preserves streaming context
    - Partial results when full computation fails
    - Memory-safe error handling
    """
    
    def __init__(self):
        self.error_history: List[Dict[str, Any]] = []
        self.recovery_strategies: Dict[str, Callable] = {
            "memory_overflow": self._handle_memory_overflow,
            "computation_timeout": self._handle_computation_timeout,
            "data_quality_failure": self._handle_data_quality_failure,
            "streaming_interruption": self._handle_streaming_interruption
        }
    
    def handle_pipeline_error(self, 
                            error: Exception, 
                            pipeline_context: Dict[str, Any],
                            partial_results: Optional[Any] = None) -> Dict[str, Any]:
        """
        Handle pipeline errors with graceful degradation.
        
        Returns:
            Dict with error details, recovery actions, and partial results
        """
        error_classification = self._classify_error(error, pipeline_context)
        
        recovery_result = {
            "error": {
                "type": type(error).__name__,
                "message": str(error),
                "classification": error_classification,
                "pipeline_stage": pipeline_context.get("current_stage", "unknown"),
                "timestamp": time.time()
            },
            "recovery": {
                "attempted": False,
                "successful": False,
                "strategy": None,
                "partial_results_available": partial_results is not None
            },
            "partial_results": partial_results,
            "fallback_options": self._generate_fallback_options(
                error_classification, pipeline_context
            )
        }
        
        # Attempt recovery if strategy exists
        if error_classification in self.recovery_strategies:
            recovery_strategy = self.recovery_strategies[error_classification]
            try:
                recovery_data = recovery_strategy(error, pipeline_context, partial_results)
                recovery_result["recovery"]["attempted"] = True
                recovery_result["recovery"]["successful"] = True
                recovery_result["recovery"]["strategy"] = error_classification
                recovery_result["recovered_data"] = recovery_data
                
            except Exception as recovery_error:
                recovery_result["recovery"]["attempted"] = True
                recovery_result["recovery"]["successful"] = False
                recovery_result["recovery"]["recovery_error"] = str(recovery_error)
        
        # Log error for pattern analysis
        self._log_error_pattern(error_classification, pipeline_context, recovery_result)
        
        return recovery_result
```

### 5. Pipeline Base Classes

Abstract base classes following sklearn patterns:

```python
class AnalysisPipelineBase(BaseEstimator, TransformerMixin):
    """
    Abstract base class for all analysis pipelines.
    
    Implements the five first principles:
    1. Intention-Driven Interface - Natural language configuration
    2. Context-Aware Composition - Enriched metadata for tool chaining  
    3. Progressive Disclosure - Simple defaults, powerful options
    4. Streaming-First - Memory-bounded processing
    5. Modular Domain Integration - Extensible architecture
    """
    
    def __init__(self, 
                 analytical_intention: str,
                 streaming_threshold_mb: int = 100,
                 progressive_complexity: str = "auto",
                 composition_aware: bool = True):
        """
        Initialize analysis pipeline with intention-driven configuration.
        
        Args:
            analytical_intention: Natural language description of analysis goal
            streaming_threshold_mb: Enable streaming above this data size
            progressive_complexity: "minimal", "auto", "comprehensive", "custom"
            composition_aware: Include metadata for downstream tool composition
        """
        self.analytical_intention = analytical_intention
        self.streaming_threshold_mb = streaming_threshold_mb
        self.progressive_complexity = progressive_complexity
        self.composition_aware = composition_aware
        
        # Initialize pipeline state
        self._pipeline_state = "initialized"
        self._streaming_config: Optional[Dict[str, Any]] = None
        self._composition_metadata: Optional[Dict[str, Any]] = None
        
    @abstractmethod
    def _configure_analysis_pipeline(self) -> List[Callable]:
        """Configure analysis steps based on intention and complexity level."""
        pass
    
    @abstractmethod  
    def _execute_analysis_step(self, step: Callable, data: pd.DataFrame, 
                              context: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        """Execute individual analysis step with error handling and metadata."""
        pass
    
    def fit(self, X: pd.DataFrame, y=None):
        """
        Fit the analysis pipeline to the data.
        
        Analyzes data characteristics and configures streaming/processing options.
        """
        # Analyze data characteristics
        data_profile = self._profile_data_characteristics(X)
        
        # Configure streaming if needed
        if self._should_enable_streaming(X, data_profile):
            self._streaming_config = self._configure_streaming(X, data_profile)
        
        # Configure analysis pipeline based on intention and data characteristics
        self._analysis_pipeline = self._configure_analysis_pipeline()
        
        # Build composition metadata
        if self.composition_aware:
            self._composition_metadata = self._build_composition_context(X, data_profile)
        
        self._pipeline_state = "fitted"
        return self
    
    def transform(self, X: pd.DataFrame) -> Tuple[Any, Dict[str, Any]]:
        """
        Execute analysis pipeline with streaming support and enriched metadata.
        
        Returns:
            Tuple[Any, Dict[str, Any]]: (analysis_results, enriched_metadata)
        """
        if self._pipeline_state != "fitted":
            raise ValueError("Pipeline not fitted. Call fit() first.")
        
        # Execute with streaming if configured
        if self._streaming_config:
            return self._execute_streaming_analysis(X)
        else:
            return self._execute_standard_analysis(X)
    
    def _execute_streaming_analysis(self, X: pd.DataFrame) -> Tuple[Any, Dict[str, Any]]:
        """Execute analysis with streaming support for large datasets."""
        # Implementation details for streaming execution
        pass
    
    def _execute_standard_analysis(self, X: pd.DataFrame) -> Tuple[Any, Dict[str, Any]]:
        """Execute analysis on full dataset in memory."""
        # Implementation details for standard execution  
        pass
```

---

## Implementation Patterns

### Streaming Integration Pattern

All pipeline stages integrate with the existing streaming architecture:

```python
# Example: Time Series Analysis Pipeline with Streaming
class TimeSeriesAnalysisPipeline(AnalysisPipelineBase):
    """
    Time series analysis with streaming support and intention-driven interface.
    """
    
    def _configure_analysis_pipeline(self) -> List[Callable]:
        if self.analytical_intention.lower().contains("forecast"):
            return [
                self._detect_time_patterns,
                self._analyze_seasonality, 
                self._build_forecast_model,
                self._generate_predictions
            ]
        elif self.analytical_intention.lower().contains("anomaly"):
            return [
                self._detect_time_patterns,
                self._analyze_baseline_behavior,
                self._detect_anomalies
            ]
        # ... other intention-based configurations
```

### Progressive Disclosure Pattern

Simple defaults with advanced capabilities:

```python
# Simple usage - intention-driven
pipeline = TimeSeriesAnalysisPipeline(
    analytical_intention="forecast next 30 days of sales"
)

# Advanced usage - custom configuration
pipeline = TimeSeriesAnalysisPipeline(
    analytical_intention="forecast sales with seasonal adjustment",
    progressive_complexity="comprehensive",
    custom_parameters={
        "seasonality_components": ["weekly", "monthly", "yearly"],
        "forecast_intervals": [0.8, 0.95],
        "model_selection": "auto_ensemble"
    }
)
```

### Composition-Aware Results Pattern

All results include metadata for downstream composition:

```python
# Analysis result includes composition guidance
result, metadata = pipeline.transform(sales_data)

# Metadata includes suggested next steps
print(metadata["composition_context"]["suggested_compositions"])
# Output: [
#   "statistical_analysis.correlation_analysis",
#   "visualization.time_series_chart", 
#   "ml.feature_importance_analysis"
# ]
```

---

## Technical Specifications

### Memory Management

- **Streaming Threshold**: Auto-enable streaming for datasets > 100MB
- **Chunk Size Adaptation**: Dynamic sizing based on available memory (16-64GB targets)
- **Buffer Management**: Automatic cleanup of intermediate results
- **Memory Monitoring**: Real-time memory usage tracking with alerts

### Performance Targets

- **Tool Discovery**: Sub-100ms response time for pipeline configuration
- **Streaming Initiation**: < 2 seconds to first chunk for any data source  
- **Memory Efficiency**: Support analysis of datasets 10x larger than available RAM
- **Composition Speed**: < 500ms to identify compatible downstream tools

### Error Handling

- **Graceful Degradation**: Partial results when full computation fails
- **Recovery Strategies**: Automatic retry with reduced complexity
- **Error Classification**: Structured error types with recovery recommendations
- **Context Preservation**: Maintain streaming state during error recovery

---

## Integration Architecture

### Domain Integration Points

Each of the 15+ data science domains integrates through standardized interfaces:

```python
# Domain registry for dynamic tool discovery
DOMAIN_PIPELINES = {
    "time_series": TimeSeriesAnalysisPipeline,
    "statistical": StatisticalAnalysisPipeline, 
    "ml": MachineLearningPipeline,
    "nlp": NaturalLanguageProcessingPipeline,
    "geospatial": GeospatialAnalysisPipeline,
    # ... other domains
}

# Dynamic pipeline creation based on analytical intention
def create_analysis_pipeline(intention: str, data_characteristics: Dict[str, Any]) -> AnalysisPipelineBase:
    """
    Factory function to create appropriate analysis pipeline based on intention.
    """
    domain = IntentionClassifier.classify_domain(intention, data_characteristics)
    pipeline_class = DOMAIN_PIPELINES[domain]
    return pipeline_class(analytical_intention=intention)
```

### MCP Tool Integration

Integration with existing MCP tool discovery system:

```python
# Enhanced MCP tool registration with pipeline awareness
@mcp.tool
def analyze_data_with_pipeline(
    data_source: str,
    analytical_intention: str,
    complexity_level: str = "auto"
) -> Dict[str, Any]:
    """
    Analyze data using intention-driven pipeline framework.
    
    Args:
        data_source: SQL query, file path, or database connection
        analytical_intention: Natural language description of analysis goal
        complexity_level: "minimal", "auto", "comprehensive"
    
    Returns:
        Standardized analysis results with composition metadata
    """
    # Create input pipeline
    input_pipeline = DataInputPipeline(analytical_intention, data_source)
    data, input_metadata = input_pipeline.fit_transform()
    
    # Create analysis pipeline based on intention
    analysis_pipeline = create_analysis_pipeline(analytical_intention, input_metadata)
    results, analysis_metadata = analysis_pipeline.fit_transform(data)
    
    # Standardize output
    return AnalysisOutputStandardizer.standardize_analysis_result(
        domain=analysis_metadata["domain"],
        analysis_type=analysis_metadata["analysis_type"],
        raw_result=results,
        input_context=input_metadata,
        computation_metadata=analysis_metadata
    )
```

---

## Validation and Testing

### Pipeline Testing Framework

```python
class PipelineTestSuite:
    """
    Comprehensive testing framework for pipeline validation.
    """
    
    def test_streaming_compatibility(self, pipeline: AnalysisPipelineBase):
        """Test pipeline works correctly with streaming data sources."""
        pass
    
    def test_memory_bounds(self, pipeline: AnalysisPipelineBase):
        """Test pipeline respects memory constraints."""
        pass
    
    def test_composition_metadata(self, pipeline: AnalysisPipelineBase):
        """Test pipeline generates valid composition metadata."""
        pass
    
    def test_progressive_disclosure(self, pipeline: AnalysisPipelineBase):
        """Test pipeline supports all complexity levels."""
        pass
    
    def test_error_recovery(self, pipeline: AnalysisPipelineBase):
        """Test pipeline handles errors gracefully.""" 
        pass
```

---

## Implementation Roadmap

### Phase 1: Core Framework (Weeks 1-2)
1. Implement base classes and interfaces
2. Create input/preprocessing pipelines
3. Build output standardization system
4. Integrate with existing streaming architecture

### Phase 2: Domain Integration (Weeks 3-4)
1. Implement first 3 domain pipelines (time_series, statistical, ml)
2. Create dynamic pipeline factory
3. Build composition metadata system
4. Integrate with MCP tool discovery

### Phase 3: Advanced Features (Weeks 5-6)
1. Implement error handling and recovery
2. Create comprehensive testing framework
3. Add performance monitoring and optimization
4. Build pipeline composition engine

### Phase 4: Production Readiness (Weeks 7-8)
1. Complete documentation and examples
2. Performance tuning and optimization
3. Security review and hardening
4. Deployment and monitoring setup

---

## Success Metrics

### Technical Metrics
- **Tool Discovery Speed**: < 100ms for pipeline configuration
- **Memory Efficiency**: Handle datasets 10x larger than available RAM
- **Error Recovery**: > 90% successful graceful degradation
- **Streaming Performance**: < 2s to first chunk for any data source

### User Experience Metrics  
- **Intention Recognition**: > 95% accuracy for common analytical intentions
- **Composition Success**: > 80% of suggested tool compositions work correctly
- **Progressive Disclosure**: All complexity levels produce valid results
- **Documentation Completeness**: 100% API coverage with examples

---

## Conclusion

The Core Pipeline Framework provides the foundational architecture to transform LocalData MCP from a database tool into a comprehensive data science platform. By implementing the five first principles through concrete pipeline classes, streaming integration, and composition-aware results, the framework enables:

1. **Natural Language Analytics** - LLM agents can express analytical intentions rather than technical procedures
2. **Unlimited Scale Processing** - Streaming architecture handles datasets far exceeding memory limits
3. **Seamless Tool Composition** - Rich metadata enables chaining 5+ tools for complex workflows  
4. **Progressive Complexity** - Simple defaults for basic use, powerful customization for advanced needs
5. **Domain Extensibility** - Clean architecture for integrating 15+ specialized data science domains

The framework is designed for immediate implementation, building on proven streaming architecture while introducing the abstraction layers needed for comprehensive data science capabilities.

**Status: Design Complete - Ready for Implementation**