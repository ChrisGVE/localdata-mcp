# Pipeline Composition System Architecture

## Design Status: ARCHITECTURAL DESIGN PHASE
**Task**: 32.4 - Pipeline Composition System Design  
**Context**: Building on First Principles Architecture Framework (32.1) and Core Pipeline Framework (32.2)

## Sequential Thinking: Component-by-Component Design

### Component 1: Composition Validation System

**Purpose**: Validate tool chain compatibility before execution to prevent runtime failures

**Key Design Patterns**:
```python
class CompositionValidator:
    """
    Validates pipeline compositions before execution.
    Ensures data type compatibility, semantic coherence, and performance feasibility.
    """
    
    def __init__(self, registry: 'ToolRegistry'):
        self.registry = registry
        self.compatibility_rules = self._load_compatibility_rules()
    
    def validate_composition(self, composition: 'AnalysisComposition') -> ValidationResult:
        """
        Multi-layer validation:
        1. Data Type Compatibility - Can outputs flow to inputs?
        2. Semantic Compatibility - Does the analytical flow make sense?
        3. Performance Feasibility - Will this composition exceed resource limits?
        4. Dependency Resolution - Are all required tools available?
        """
        results = []
        
        # Layer 1: Data Type Validation
        results.append(self._validate_data_types(composition))
        
        # Layer 2: Semantic Validation  
        results.append(self._validate_semantic_flow(composition))
        
        # Layer 3: Performance Validation
        results.append(self._validate_performance_limits(composition))
        
        # Layer 4: Dependency Validation
        results.append(self._validate_dependencies(composition))
        
        return self._aggregate_validation_results(results)
    
    def _validate_data_types(self, composition) -> ValidationResult:
        """
        Check data type compatibility between pipeline stages.
        
        Examples:
        - pandas.DataFrame → sklearn.LinearRegression ✅
        - str → numpy.ndarray ❌ (needs conversion)
        - scipy.optimize.Result → pandas.DataFrame ✅ (with metadata extraction)
        """
        incompatibilities = []
        
        for i in range(len(composition.stages) - 1):
            current_stage = composition.stages[i]
            next_stage = composition.stages[i + 1]
            
            output_types = self.registry.get_output_types(current_stage.tool_name)
            input_types = self.registry.get_input_types(next_stage.tool_name)
            
            if not self._types_compatible(output_types, input_types):
                conversion_path = self._find_conversion_path(output_types, input_types)
                
                if conversion_path:
                    # Auto-conversion possible
                    next_stage.add_conversion(conversion_path)
                else:
                    # Incompatible - record error
                    incompatibilities.append({
                        'stage_from': i,
                        'stage_to': i + 1,
                        'output_types': output_types,
                        'input_types': input_types,
                        'error': 'no_conversion_path'
                    })
        
        return ValidationResult(
            valid=len(incompatibilities) == 0,
            category='data_types',
            errors=incompatibilities
        )
```

**Compatibility Rule System**:
```python
class CompatibilityRules:
    """
    Defines compatibility rules between different data science tools and data types.
    """
    
    # Core compatibility matrix
    TYPE_COMPATIBILITY = {
        'pandas.DataFrame': {
            'sklearn.base.BaseEstimator': True,  # Can feed DataFrames to sklearn
            'scipy.stats': True,                 # Can use DataFrames with scipy.stats  
            'numpy.ndarray': True,               # Can convert to arrays
            'plotly.graph_objects.Figure': True  # Can visualize DataFrames
        },
        'numpy.ndarray': {
            'sklearn.base.BaseEstimator': True,
            'scipy.optimize': True,
            'pandas.DataFrame': True,            # Can create DataFrames from arrays
            'matplotlib.axes.Axes': True
        },
        'sklearn.base.BaseEstimator': {
            'pandas.DataFrame': False,           # Models don't become DataFrames
            'sklearn.metrics': True,             # Can evaluate models
            'dict': True                         # Can extract model parameters
        }
    }
    
    # Semantic compatibility - which analytical flows make sense
    SEMANTIC_COMPATIBILITY = {
        'data_ingestion': {
            'next_valid': ['data_cleaning', 'exploratory_analysis', 'transformation'],
            'next_invalid': ['modeling', 'evaluation']  # Can't model raw data
        },
        'data_cleaning': {
            'next_valid': ['exploratory_analysis', 'transformation', 'modeling'],
            'next_invalid': ['evaluation']  # Need model first
        },
        'modeling': {
            'next_valid': ['evaluation', 'prediction', 'interpretation'],
            'next_invalid': ['data_ingestion']  # Can't go backwards in pipeline
        },
        'evaluation': {
            'next_valid': ['interpretation', 'model_selection', 'reporting'],
            'next_invalid': ['data_ingestion', 'data_cleaning']
        }
    }
```

### Component 2: Data Flow Architecture

**Purpose**: Manage seamless data movement between pipeline stages with automatic type conversion

**Core Design**:
```python
class PipelineDataFlow:
    """
    Manages data flow between pipeline stages with automatic conversion and streaming support.
    
    Key Features:
    - Automatic type conversion between incompatible stages
    - Memory-efficient streaming for large datasets
    - Rich metadata preservation through compositions
    - Performance optimization with intelligent caching
    """
    
    def __init__(self, memory_manager: 'MemoryManager'):
        self.memory_manager = memory_manager
        self.converters = self._initialize_converters()
        self.cache = CompositionCache()
    
    def execute_composition(self, composition: 'AnalysisComposition') -> CompositionResult:
        """
        Execute a validated composition with streaming data flow.
        """
        context = CompositionContext()
        
        for stage_index, stage in enumerate(composition.stages):
            # Get input data from previous stage or initial data
            if stage_index == 0:
                input_data = composition.initial_data
            else:
                input_data = context.get_stage_output(stage_index - 1)
            
            # Apply automatic type conversion if needed
            if stage.requires_conversion:
                input_data = self._apply_conversion(input_data, stage.conversion_path)
            
            # Execute stage with streaming support
            stage_result = self._execute_stage_streaming(stage, input_data, context)
            
            # Store result for next stage
            context.set_stage_output(stage_index, stage_result)
            
            # Memory management
            if self.memory_manager.should_cleanup(context):
                context.cleanup_intermediate_results()
        
        return CompositionResult(
            final_result=context.get_final_output(),
            metadata=context.get_composition_metadata(),
            performance_stats=context.get_performance_stats()
        )
```

**Type Conversion Engine**:
```python
class TypeConversionEngine:
    """
    Handles automatic conversion between incompatible data types in pipeline compositions.
    """
    
    def __init__(self):
        self.conversion_registry = {
            ('pandas.DataFrame', 'numpy.ndarray'): self._dataframe_to_array,
            ('numpy.ndarray', 'pandas.DataFrame'): self._array_to_dataframe,
            ('sklearn.base.BaseEstimator', 'dict'): self._model_to_dict,
            ('scipy.optimize.OptimizeResult', 'pandas.DataFrame'): self._optimize_result_to_dataframe,
            ('dict', 'pandas.DataFrame'): self._dict_to_dataframe
        }
    
    def convert(self, data, from_type: str, to_type: str) -> ConversionResult:
        """
        Convert data from one type to another with metadata preservation.
        """
        conversion_key = (from_type, to_type)
        
        if conversion_key not in self.conversion_registry:
            return ConversionResult(
                success=False,
                error=f"No conversion path from {from_type} to {to_type}"
            )
        
        converter_func = self.conversion_registry[conversion_key]
        
        try:
            converted_data = converter_func(data)
            return ConversionResult(
                success=True,
                data=converted_data,
                metadata=self._extract_conversion_metadata(data, converted_data)
            )
        except Exception as e:
            return ConversionResult(
                success=False,
                error=f"Conversion failed: {str(e)}"
            )
    
    def _dataframe_to_array(self, df: pd.DataFrame) -> np.ndarray:
        """Convert DataFrame to numpy array for sklearn/scipy tools."""
        # Handle mixed types by separating numeric and categorical
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == len(df.columns):
            # All numeric - direct conversion
            return df.values
        else:
            # Mixed types - need encoding strategy
            from sklearn.preprocessing import LabelEncoder
            encoded_df = df.copy()
            
            for col in df.select_dtypes(include=['object']).columns:
                le = LabelEncoder()
                encoded_df[col] = le.fit_transform(df[col].astype(str))
            
            return encoded_df.values
    
    def _optimize_result_to_dataframe(self, result: 'scipy.optimize.OptimizeResult') -> pd.DataFrame:
        """Convert scipy optimization result to DataFrame for downstream analysis."""
        result_dict = {
            'optimization_success': [result.success],
            'optimization_message': [result.message],
            'function_value': [result.fun],
            'iterations': [result.nit],
            'function_evaluations': [result.nfev]
        }
        
        # Add parameter values if available
        if hasattr(result, 'x'):
            for i, param_value in enumerate(result.x):
                result_dict[f'parameter_{i}'] = [param_value]
        
        return pd.DataFrame(result_dict)
```

### Component 3: Error Handling & Recovery

**Purpose**: Provide graceful degradation when composition stages fail

**Design Pattern**:
```python
class CompositionErrorHandler:
    """
    Handles errors and implements graceful degradation for pipeline compositions.
    
    Error Recovery Strategies:
    1. Retry with different parameters
    2. Skip failed stage and continue with partial results
    3. Substitute failed stage with alternative tool
    4. Provide diagnostic information for manual intervention
    """
    
    def __init__(self):
        self.recovery_strategies = {
            'memory_error': self._handle_memory_error,
            'type_conversion_error': self._handle_conversion_error,
            'tool_execution_error': self._handle_tool_error,
            'dependency_missing': self._handle_missing_dependency
        }
    
    def handle_composition_error(self, error: CompositionError, context: CompositionContext) -> ErrorHandlingResult:
        """
        Handle composition errors with appropriate recovery strategy.
        """
        error_type = self._classify_error(error)
        
        if error_type in self.recovery_strategies:
            recovery_func = self.recovery_strategies[error_type]
            return recovery_func(error, context)
        else:
            return ErrorHandlingResult(
                recovery_possible=False,
                error_message=f"Unhandled error type: {error_type}",
                diagnostic_info=self._generate_diagnostic_info(error, context)
            )
    
    def _handle_memory_error(self, error: CompositionError, context: CompositionContext) -> ErrorHandlingResult:
        """
        Handle memory errors by switching to streaming or reducing data size.
        """
        # Strategy 1: Switch to streaming execution
        if not context.current_stage.is_streaming:
            modified_stage = context.current_stage.copy()
            modified_stage.enable_streaming()
            
            return ErrorHandlingResult(
                recovery_possible=True,
                recovery_action='switch_to_streaming',
                modified_stage=modified_stage,
                message="Switched to streaming execution to reduce memory usage"
            )
        
        # Strategy 2: Reduce data chunk size
        if context.current_stage.chunk_size > 100:
            modified_stage = context.current_stage.copy()
            modified_stage.chunk_size = max(100, context.current_stage.chunk_size // 2)
            
            return ErrorHandlingResult(
                recovery_possible=True,
                recovery_action='reduce_chunk_size',
                modified_stage=modified_stage,
                message=f"Reduced chunk size to {modified_stage.chunk_size}"
            )
        
        return ErrorHandlingResult(
            recovery_possible=False,
            error_message="Memory error: unable to recover with current data size"
        )
    
    def _handle_tool_error(self, error: CompositionError, context: CompositionContext) -> ErrorHandlingResult:
        """
        Handle tool execution errors by trying alternative tools or parameters.
        """
        failed_stage = context.current_stage
        
        # Strategy 1: Try alternative tool for same function
        alternatives = self._find_alternative_tools(failed_stage.tool_name, failed_stage.function)
        
        if alternatives:
            alternative_tool = alternatives[0]  # Try first alternative
            modified_stage = failed_stage.copy()
            modified_stage.tool_name = alternative_tool['name']
            modified_stage.parameters = alternative_tool['default_parameters']
            
            return ErrorHandlingResult(
                recovery_possible=True,
                recovery_action='substitute_tool',
                modified_stage=modified_stage,
                message=f"Substituted {failed_stage.tool_name} with {alternative_tool['name']}"
            )
        
        # Strategy 2: Skip stage and provide partial results
        if failed_stage.is_optional:
            return ErrorHandlingResult(
                recovery_possible=True,
                recovery_action='skip_stage',
                message=f"Skipped optional stage {failed_stage.tool_name}, continuing with partial results"
            )
        
        return ErrorHandlingResult(
            recovery_possible=False,
            error_message=f"Critical stage {failed_stage.tool_name} failed with no alternatives available"
        )
```

### Component 4: Workflow Optimization Engine

**Purpose**: Intelligent pipeline optimization, caching, and parallelization

**Core Architecture**:
```python
class WorkflowOptimizationEngine:
    """
    Optimizes pipeline compositions for performance through:
    1. Intelligent caching of intermediate results
    2. Parallel execution of independent stages
    3. Query optimization for database-heavy pipelines
    4. Memory-efficient data flow patterns
    """
    
    def __init__(self, cache: CompositionCache, scheduler: ParallelScheduler):
        self.cache = cache
        self.scheduler = scheduler
        self.optimization_rules = self._initialize_optimization_rules()
    
    def optimize_composition(self, composition: 'AnalysisComposition') -> OptimizedComposition:
        """
        Apply optimization strategies to improve composition performance.
        """
        optimized_stages = []
        optimization_log = []
        
        # Analysis Phase: Identify optimization opportunities
        dependency_graph = self._build_dependency_graph(composition)
        parallelizable_groups = self._identify_parallel_groups(dependency_graph)
        cacheable_stages = self._identify_cacheable_stages(composition)
        
        # Optimization Phase: Apply optimizations
        for optimization_rule in self.optimization_rules:
            result = optimization_rule.apply(composition, {
                'dependency_graph': dependency_graph,
                'parallel_groups': parallelizable_groups,
                'cacheable_stages': cacheable_stages
            })
            
            if result.applied:
                optimized_stages.extend(result.optimized_stages)
                optimization_log.append(result.optimization_description)
        
        return OptimizedComposition(
            stages=optimized_stages,
            parallel_groups=parallelizable_groups,
            caching_strategy=cacheable_stages,
            optimization_log=optimization_log,
            estimated_performance_improvement=self._estimate_performance_gain(
                composition, optimized_stages
            )
        )
    
    def _identify_parallel_groups(self, dependency_graph) -> List[ParallelGroup]:
        """
        Identify stages that can be executed in parallel.
        
        Example:
        Query → [Feature Engineering, Statistical Analysis] → Modeling
        
        Feature Engineering and Statistical Analysis can run in parallel
        since they don't depend on each other.
        """
        parallel_groups = []
        
        # Find nodes with no dependencies between them
        for level in self._topological_levels(dependency_graph):
            if len(level) > 1:
                # Multiple independent stages at same level
                parallel_groups.append(ParallelGroup(
                    stages=level,
                    max_concurrency=min(len(level), 4),  # Limit to 4 concurrent stages
                    synchronization_point=True  # Wait for all to complete before next level
                ))
        
        return parallel_groups
```

**Caching Strategy**:
```python
class CompositionCache:
    """
    Intelligent caching system for pipeline composition intermediate results.
    """
    
    def __init__(self, storage_backend: CacheStorage):
        self.storage = storage_backend
        self.cache_policies = {
            'expensive_computation': ExpirationPolicy(ttl_hours=24),
            'database_query': ExpirationPolicy(ttl_hours=1),
            'model_training': ExpirationPolicy(ttl_hours=168),  # 1 week
            'data_transformation': ExpirationPolicy(ttl_hours=6)
        }
    
    def get_cache_key(self, stage: CompositionStage, input_data_hash: str) -> str:
        """
        Generate cache key based on stage configuration and input data.
        """
        stage_signature = {
            'tool_name': stage.tool_name,
            'function': stage.function,
            'parameters': stage.parameters,
            'version': stage.tool_version
        }
        
        import hashlib
        signature_str = json.dumps(stage_signature, sort_keys=True)
        signature_hash = hashlib.sha256(signature_str.encode()).hexdigest()
        
        return f"composition:{stage.tool_name}:{signature_hash}:{input_data_hash}"
    
    def should_cache_stage(self, stage: CompositionStage) -> bool:
        """
        Determine if stage results should be cached based on computation cost.
        """
        # Cache expensive operations
        expensive_operations = {
            'machine_learning_training',
            'large_dataset_aggregation',
            'complex_statistical_analysis',
            'optimization_algorithms'
        }
        
        if stage.operation_category in expensive_operations:
            return True
        
        # Cache long-running operations (> 10 seconds estimated)
        if stage.estimated_duration_seconds > 10:
            return True
        
        # Cache deterministic operations with stable inputs
        if stage.is_deterministic and not stage.has_time_dependency:
            return True
        
        return False
```

### Component 5: LLM Interface Design

**Purpose**: Natural interface for LLM agents to request and configure complex compositions

**Design Pattern**:
```python
class LLMCompositionInterface:
    """
    Natural language interface for LLM agents to create and execute complex analytical pipelines.
    
    Key Features:
    - Intent-based pipeline specification
    - Automatic tool selection based on analytical goals
    - Progressive disclosure of complexity
    - Rich result interpretation for downstream tools
    """
    
    def __init__(self, tool_registry: ToolRegistry, validator: CompositionValidator):
        self.registry = tool_registry
        self.validator = validator
        self.intent_parser = AnalyticalIntentParser()
    
    def create_composition_from_intent(self, intent: str, data_context: Dict) -> CompositionBuilder:
        """
        Create pipeline composition from analytical intent expressed in natural language.
        
        Example intents:
        - "Analyze customer churn with feature importance and model performance"
        - "Time series forecasting with trend decomposition and seasonal adjustment"
        - "Geospatial clustering with visualization and statistical validation"
        """
        # Parse analytical intent
        parsed_intent = self.intent_parser.parse(intent)
        
        # Map intent to tool sequence
        composition_builder = CompositionBuilder()
        
        for analytical_step in parsed_intent.steps:
            # Find appropriate tools for each step
            suitable_tools = self.registry.find_tools_for_intent(
                analytical_step.intent_category,
                analytical_step.required_capabilities,
                data_context
            )
            
            if suitable_tools:
                best_tool = self._select_best_tool(suitable_tools, analytical_step, data_context)
                composition_builder.add_stage(
                    tool_name=best_tool.name,
                    function=analytical_step.function,
                    parameters=self._infer_parameters(best_tool, analytical_step, data_context)
                )
        
        # Validate and optimize
        composition = composition_builder.build()
        validation_result = self.validator.validate_composition(composition)
        
        if not validation_result.valid:
            # Auto-fix common issues
            composition = self._auto_fix_composition(composition, validation_result)
        
        return composition
    
    def _infer_parameters(self, tool: Tool, step: AnalyticalStep, data_context: Dict) -> Dict:
        """
        Intelligently infer tool parameters based on data characteristics and analytical intent.
        """
        inferred_params = tool.default_parameters.copy()
        
        # Data-driven parameter inference
        if 'data_size' in data_context:
            data_size = data_context['data_size']
            
            # Adjust chunk sizes for large datasets
            if data_size > 1000000:  # > 1M rows
                inferred_params['chunk_size'] = 10000
                inferred_params['streaming'] = True
            
            # Adjust sampling for very large datasets
            if data_size > 10000000:  # > 10M rows
                inferred_params['sample_size'] = 100000
        
        # Intent-driven parameter inference
        if step.intent_category == 'statistical_modeling':
            if 'high_accuracy' in step.requirements:
                inferred_params['cross_validation'] = True
                inferred_params['hyperparameter_tuning'] = True
            
            if 'interpretability' in step.requirements:
                inferred_params['feature_importance'] = True
                inferred_params['model_explanation'] = True
        
        return inferred_params
```

**Example LLM Integration**:
```python
@mcp.tool
def create_analytical_pipeline(
    analytical_intent: str,
    data_source: str,
    performance_requirements: Optional[Dict] = None
) -> Dict:
    """
    Create and execute a complex analytical pipeline based on natural language intent.
    
    Args:
        analytical_intent: Natural language description of desired analysis
        data_source: Database connection name or file path
        performance_requirements: Optional performance constraints
    
    Returns:
        Pipeline execution results with metadata and composition details
    
    Example:
        create_analytical_pipeline(
            analytical_intent="Predict customer lifetime value with feature importance analysis",
            data_source="customers_db",
            performance_requirements={"max_execution_time": 300, "max_memory_mb": 2048}
        )
    """
    interface = LLMCompositionInterface(tool_registry, composition_validator)
    
    # Get data context
    data_context = _analyze_data_source(data_source)
    
    # Create composition from intent
    composition = interface.create_composition_from_intent(analytical_intent, data_context)
    
    # Apply performance constraints
    if performance_requirements:
        composition = _apply_performance_constraints(composition, performance_requirements)
    
    # Execute pipeline
    executor = PipelineExecutor(memory_manager, error_handler)
    result = executor.execute_composition(composition)
    
    return {
        'pipeline_executed': True,
        'composition_stages': len(composition.stages),
        'execution_time_seconds': result.execution_time,
        'results': result.final_result,
        'metadata': result.metadata,
        'performance_stats': result.performance_stats,
        'next_recommended_actions': _suggest_next_steps(result)
    }
```

## Integration Architecture

### Pipeline Composition Flow
```
LLM Intent → Intent Parser → Tool Selection → Composition Building → Validation → Optimization → Execution → Results

Example Complex Workflow:
"Analyze customer churn with geospatial clustering and predictive modeling"

↓ Intent Parsing
Steps: [data_ingestion, geospatial_analysis, clustering, feature_engineering, modeling, evaluation]

↓ Tool Selection  
Tools: [query_database, geospatial_cluster, kmeans_clustering, feature_importance, logistic_regression, model_evaluation]

↓ Composition Building
Stage 1: query_database(customers_table) → DataFrame
Stage 2: geospatial_cluster(DataFrame, location_columns) → DataFrame + cluster_labels  
Stage 3: feature_importance(DataFrame) → feature_scores + DataFrame
Stage 4: logistic_regression(DataFrame, target=churn) → trained_model
Stage 5: model_evaluation(model, test_data) → evaluation_metrics

↓ Validation
✅ Data types compatible
✅ Semantic flow valid  
✅ Performance within limits
✅ All dependencies available

↓ Optimization
- Parallel execution: geospatial_cluster + feature_importance (independent)
- Caching: feature_importance results (expensive computation)
- Streaming: query_database (large dataset)

↓ Execution
Results: Churn prediction model + geospatial insights + feature importance rankings
```

## Key Architectural Decisions

1. **Validation-First Design**: All compositions validated before execution prevents runtime failures
2. **Streaming-Compatible**: All composition stages support streaming for memory efficiency  
3. **Error Recovery Built-In**: Graceful degradation with multiple recovery strategies
4. **LLM-Native Interface**: Natural language intent mapping to technical pipelines
5. **Performance-Optimized**: Automatic caching, parallelization, and resource management

This architecture enables LocalData MCP v2.0 to support complex multi-stage analytical workflows that LLM agents can request and execute naturally, with robust error handling and performance optimization.