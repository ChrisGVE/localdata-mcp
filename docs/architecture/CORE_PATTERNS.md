# LocalData MCP v2.0 - Core Architectural Patterns

## Overview

This document defines the fundamental architectural patterns that implement our [First Principles](./FIRST_PRINCIPLES.md). These patterns provide concrete guidance for implementing features that align with our architectural philosophy while being evaluated through our [Design Decision Framework](./DESIGN_DECISIONS.md).

## Foundational Patterns

### Pattern 1: Analytical Intent Resolution

**Purpose**: Transform LLM analytical intentions into specific tool operations.

**Problem**: LLMs express analytical goals semantically ("find strong relationships") but tools traditionally require technical parameters (correlation method, significance thresholds, etc.).

**Solution**: Intent resolution layer that maps semantic intentions to appropriate technical implementations.

#### Implementation Structure
```python
class IntentResolver:
    def resolve_intent(self, intent: str, data_context: DataContext) -> ToolConfiguration:
        """
        Maps analytical intentions to specific tool configurations
        """
        semantic_params = self.parse_intent(intent)
        data_characteristics = self.analyze_context(data_context)
        return self.configure_tools(semantic_params, data_characteristics)
        
    def parse_intent(self, intent: str) -> SemanticParameters:
        """Extract semantic meaning from natural language"""
        # "strong correlations" -> strength="high", type="correlations"
        # "recent patterns" -> time_focus="recent", analysis="patterns"
        pass
        
    def configure_tools(self, semantic: SemanticParameters, context: DataContext) -> ToolConfiguration:
        """Convert semantic parameters to technical configuration"""
        # strength="high" + numeric_data -> correlation_threshold=0.7
        # strength="high" + categorical_data -> chi_square_threshold=0.05
        pass
```

#### Usage Examples
```python
# LLM says: "Find strong relationships in this dataset"
intent = "find strong relationships"
config = resolver.resolve_intent(intent, data_context)
# Results in: correlation_analysis(threshold=0.7, methods=["pearson", "spearman"])

# LLM says: "Detect unusual patterns in recent data"  
intent = "detect unusual patterns in recent data"
config = resolver.resolve_intent(intent, data_context)
# Results in: anomaly_detection(focus="temporal", sensitivity="moderate", window="recent")
```

#### Key Characteristics
- **Context-sensitive**: Same intent maps to different tools based on data characteristics
- **Extensible**: New intents can be added without changing existing mappings
- **Composable**: Resolved configurations can be chained together
- **Transparent**: Resolution process can be exposed for debugging/learning

### Pattern 2: Context-Enriched Results

**Purpose**: Structure tool results to maximize downstream composition and LLM interpretation.

**Problem**: Traditional analysis results are designed for human consumption, not tool composition or LLM reasoning.

**Solution**: Standardized result structure that includes primary results, metadata, composition hooks, and interpretation guidance.

#### Implementation Structure
```python
@dataclass
class AnalysisResult:
    """Standardized result structure for all analysis tools"""
    
    # Primary analysis results
    primary_result: Dict[str, Any]
    
    # Metadata for tool composition and system optimization
    metadata: ResultMetadata
    
    # Hooks for specific downstream tools
    composition_hooks: Dict[str, Any]
    
    # Natural language interpretation for LLMs
    interpretation: ResultInterpretation
    
    # Context preservation for workflow continuity
    context: AnalysisContext

@dataclass 
class ResultMetadata:
    method_used: str
    data_characteristics: DataCharacteristics
    confidence_level: float
    memory_usage: int
    processing_time: float
    suggested_next_steps: List[str]
    warnings: List[str]

@dataclass
class ResultInterpretation:
    summary: str  # Natural language summary
    key_insights: List[str]  # Bullet points of main findings
    business_implications: List[str]  # Practical implications
    statistical_notes: List[str]  # Technical details for advanced users
```

#### Usage Examples
```python
# Correlation analysis result
correlation_result = AnalysisResult(
    primary_result={
        "correlation_matrix": correlation_matrix,
        "significant_pairs": significant_correlations
    },
    metadata=ResultMetadata(
        method_used="pearson_with_bonferroni_correction",
        confidence_level=0.95,
        suggested_next_steps=[
            "model_relationships", 
            "explore_causality", 
            "visualize_correlations"
        ]
    ),
    composition_hooks={
        "regression_input": {
            "target_variable": "strongest_target",
            "feature_candidates": "high_correlation_features"
        },
        "clustering_input": {
            "distance_metric": "correlation_based",
            "feature_weights": "correlation_strength"
        }
    },
    interpretation=ResultInterpretation(
        summary="Found 8 strong correlations (>0.7) among 15 variables",
        key_insights=[
            "Sales and Marketing_Spend show strongest relationship (r=0.89)",
            "Customer_Satisfaction correlates with multiple outcomes",
            "Geographic variables show regional clustering patterns"
        ]
    )
)
```

#### Key Characteristics
- **Self-describing**: Results explain their own creation and meaning
- **Composable**: Hooks enable automatic configuration of downstream tools
- **Interpretable**: Natural language summaries for LLM consumption
- **Traceable**: Metadata enables understanding of how results were produced

### Pattern 3: Domain-Bridge Workflows

**Purpose**: Enable seamless integration between different data science domains.

**Problem**: Traditional tools operate in domain silos - time series tools don't naturally connect to regression tools, statistical tools don't inform machine learning workflows.

**Solution**: Domain bridge pattern that translates context and results between domains while preserving analytical intent.

#### Implementation Structure
```python
class DomainBridge:
    """Manages cross-domain context translation and workflow coordination"""
    
    def __init__(self):
        self.domain_registry = {}
        self.bridge_mappings = {}
        
    def register_domain(self, domain: AnalysisDomain):
        """Register a data science domain with its capabilities"""
        self.domain_registry[domain.name] = domain
        
    def create_bridge(self, from_domain: str, to_domain: str, mapping: BridgeMapping):
        """Define how context translates between domains"""
        self.bridge_mappings[(from_domain, to_domain)] = mapping
        
    def execute_workflow(self, steps: List[WorkflowStep]) -> WorkflowResult:
        """Execute multi-domain workflow with automatic context bridging"""
        context = AnalysisContext()
        results = []
        
        for step in steps:
            # Translate context from previous domains
            if results:
                context = self.bridge_context(context, step.domain, results[-1])
                
            # Execute step with enriched context
            result = self.execute_step(step, context)
            results.append(result)
            
            # Update context for next step
            context = self.update_context(context, result)
            
        return WorkflowResult(results, context)
```

#### Usage Examples
```python
# Time series -> Regression -> Business Intelligence workflow
workflow = [
    WorkflowStep(
        domain="time_series",
        operation="analyze_trends",
        intent="understand temporal patterns"
    ),
    WorkflowStep(
        domain="regression", 
        operation="model_relationships",
        intent="predict future values"
    ),
    WorkflowStep(
        domain="business_intelligence",
        operation="generate_insights", 
        intent="create actionable recommendations"
    )
]

# Automatic context bridging:
# - Time series detrending informs regression feature engineering
# - Seasonal patterns become model variables
# - Forecast confidence intervals inform business risk assessment
result = domain_bridge.execute_workflow(workflow)
```

#### Key Characteristics
- **Transparent**: Domain boundaries are invisible to LLM users
- **Contextual**: Information flows naturally between analysis steps
- **Extensible**: New domains can be added with bridge mappings to existing ones
- **Intelligent**: Context translation adapts to specific analytical goals

### Pattern 4: Adaptive Processing Strategy

**Purpose**: Automatically select processing approaches based on data characteristics and memory constraints.

**Problem**: Traditional tools require users to choose between exact vs. approximate algorithms, batch vs. streaming processing, full vs. sample analysis.

**Solution**: Adaptive processing that automatically selects the best approach while maintaining result quality within constraints.

#### Implementation Structure
```python
class AdaptiveProcessor:
    """Manages processing strategy selection based on constraints and requirements"""
    
    def __init__(self, memory_budget: int, accuracy_target: float):
        self.memory_budget = memory_budget
        self.accuracy_target = accuracy_target
        self.strategy_registry = {}
        
    def select_strategy(self, data_characteristics: DataCharacteristics, 
                       operation: str) -> ProcessingStrategy:
        """Choose optimal processing strategy for operation and data"""
        
        estimated_memory = self.estimate_memory_usage(data_characteristics, operation)
        required_accuracy = self.estimate_accuracy_requirements(operation)
        
        if estimated_memory < self.memory_budget * 0.5:
            return ExactBatchStrategy()
        elif estimated_memory < self.memory_budget * 0.8:
            return ApproximateStrategy(self.accuracy_target)
        else:
            return StreamingStrategy(self.memory_budget)
            
    def execute_with_adaptation(self, operation: str, data: DataSource) -> AnalysisResult:
        """Execute operation with automatic strategy adaptation"""
        
        try:
            strategy = self.select_strategy(data.characteristics, operation)
            return strategy.execute(operation, data)
        except MemoryError:
            # Automatic fallback to more memory-efficient strategy
            fallback_strategy = self.get_fallback_strategy(operation)
            return fallback_strategy.execute(operation, data)
```

#### Usage Examples
```python
# Automatic adaptation based on data size
processor = AdaptiveProcessor(memory_budget=8_000_000_000)  # 8GB

# Small dataset -> exact algorithms
small_data = DataSource(rows=10_000, columns=20)
result1 = processor.execute_with_adaptation("correlation_analysis", small_data)
# Uses: Full correlation matrix with exact p-values

# Medium dataset -> approximate algorithms  
medium_data = DataSource(rows=1_000_000, columns=50)
result2 = processor.execute_with_adaptation("correlation_analysis", medium_data)
# Uses: Incremental correlation with sampling-based p-values

# Large dataset -> streaming algorithms
large_data = DataSource(rows=100_000_000, columns=100) 
result3 = processor.execute_with_adaptation("correlation_analysis", large_data)
# Uses: Streaming correlation with bootstrap confidence intervals
```

#### Key Characteristics
- **Transparent**: Users don't need to understand algorithm trade-offs
- **Adaptive**: Automatically adjusts to data characteristics and constraints
- **Graceful**: Degrades quality predictably when necessary
- **Recoverable**: Can fall back to more conservative approaches when needed

### Pattern 5: Modular Domain Architecture

**Purpose**: Structure data science domains as composable, interoperable modules while maintaining domain-specific optimizations.

**Problem**: Data science encompasses many specialized domains (statistics, machine learning, time series, etc.) that need both deep specialization and cross-domain integration.

**Solution**: Modular architecture where each domain is a self-contained module with standardized interfaces for discovery, analysis, and composition.

#### Implementation Structure
```python
class AnalysisDomain(ABC):
    """Base class for all data science domains"""
    
    @property
    @abstractmethod
    def domain_name(self) -> str:
        """Unique identifier for this domain"""
        pass
        
    @property
    @abstractmethod
    def capabilities(self) -> List[Capability]:
        """What this domain can analyze"""
        pass
        
    @abstractmethod
    def discover_tools(self, intent: str, data_context: DataContext) -> List[Tool]:
        """Find tools that match analytical intent"""
        pass
        
    @abstractmethod
    def execute_analysis(self, tool: str, data: DataSource, 
                        context: AnalysisContext) -> AnalysisResult:
        """Perform domain-specific analysis"""
        pass
        
    @abstractmethod
    def interpret_results(self, result: AnalysisResult, 
                         target_audience: str) -> Interpretation:
        """Provide domain-specific interpretation"""
        pass

class StatisticalDomain(AnalysisDomain):
    """Statistical analysis domain implementation"""
    
    domain_name = "statistical_analysis"
    
    capabilities = [
        Capability("descriptive_statistics", ["numerical", "categorical"]),
        Capability("hypothesis_testing", ["numerical", "categorical"]),
        Capability("correlation_analysis", ["numerical"]),
        Capability("distribution_analysis", ["numerical"])
    ]
    
    def discover_tools(self, intent: str, data_context: DataContext) -> List[Tool]:
        """Map intentions to statistical tools"""
        # "find relationships" -> correlation tools
        # "compare groups" -> hypothesis testing tools
        # "understand distribution" -> descriptive statistics tools
        pass
```

#### Usage Examples
```python
# Domain registration and discovery
domain_manager = DomainManager()
domain_manager.register_domain(StatisticalDomain())
domain_manager.register_domain(RegressionDomain())  
domain_manager.register_domain(TimeSeriesDomain())

# Cross-domain tool discovery
intent = "find predictive patterns in time-based data"
applicable_domains = domain_manager.discover_domains(intent, data_context)
# Returns: [TimeSeriesDomain, RegressionDomain, StatisticalDomain]

# Coordinated analysis across domains
workflow = domain_manager.create_workflow(intent, data_context)
# Automatically sequences: time series analysis -> feature engineering -> regression modeling
```

#### Key Characteristics
- **Self-contained**: Each domain manages its own tools and algorithms
- **Interoperable**: Common interfaces enable cross-domain workflows  
- **Discoverable**: Tools can be found by analytical intent, not just domain knowledge
- **Extensible**: New domains integrate automatically with existing ones

## Composite Patterns

### Pattern: Intent-Driven Analysis Pipeline

**Purpose**: Combine intent resolution, adaptive processing, and domain bridging into seamless analytical workflows.

```python
class AnalysisPipeline:
    """High-level pipeline that combines all core patterns"""
    
    def __init__(self):
        self.intent_resolver = IntentResolver()
        self.domain_bridge = DomainBridge()
        self.adaptive_processor = AdaptiveProcessor()
        
    def analyze(self, intent: str, data: DataSource) -> PipelineResult:
        """Execute complete analysis pipeline from intent to results"""
        
        # 1. Resolve intent to specific operations
        operations = self.intent_resolver.resolve(intent, data.context)
        
        # 2. Create cross-domain workflow if needed
        workflow = self.domain_bridge.create_workflow(operations)
        
        # 3. Execute with adaptive processing
        results = []
        context = AnalysisContext()
        
        for step in workflow.steps:
            result = self.adaptive_processor.execute_with_adaptation(
                step.operation, data, context
            )
            results.append(result)
            context = self.domain_bridge.update_context(context, result)
            
        # 4. Generate comprehensive interpretation
        interpretation = self.generate_interpretation(intent, results)
        
        return PipelineResult(results, interpretation, context)
```

## Implementation Guidelines

### For New Tools
1. **Start with intent**: What analytical intention does this serve?
2. **Design results first**: What should downstream tools receive?
3. **Plan for adaptation**: How does this work with different data sizes?
4. **Consider composition**: How does this connect to other domains?
5. **Add interpretation**: How should LLMs understand these results?

### For New Domains
1. **Define capabilities clearly**: What analytical questions can this domain answer?
2. **Design bridge mappings**: How does context translate to/from other domains?
3. **Implement adaptive strategies**: How does processing adapt to constraints?
4. **Plan tool discovery**: How do intents map to domain tools?
5. **Design interpretations**: How should results be explained?

### For Workflow Integration
1. **Preserve context**: Ensure information flows between steps
2. **Enable backtracking**: Allow workflow modification based on intermediate results
3. **Plan for failure**: Graceful degradation when steps fail
4. **Optimize transitions**: Minimize data transformation between domains
5. **Maintain traceability**: Track how results were derived

## Pattern Interactions

### Positive Interactions
- **Intent Resolution + Adaptive Processing**: Intentions can guide processing strategy selection
- **Context-Enriched Results + Domain Bridging**: Rich results enable better cross-domain context
- **Modular Domains + Intent Resolution**: Domain-specific intent interpretation improves accuracy

### Potential Conflicts
- **Adaptive Processing + Result Consistency**: Different strategies may produce different result formats
- **Domain Bridging + Processing Efficiency**: Context translation overhead may impact performance
- **Intent Resolution + Deterministic Behavior**: Same intent may resolve differently based on context

### Resolution Strategies
- **Standardized Result Contracts**: All processing strategies must produce compatible result formats
- **Lazy Context Translation**: Only translate context when actually needed by downstream tools
- **Intent Disambiguation**: Provide mechanisms for users to clarify ambiguous intentions

---

*These core patterns provide the foundation for implementing LocalData MCP v2.0 features that align with our architectural principles while delivering natural LLM-agent experiences.*