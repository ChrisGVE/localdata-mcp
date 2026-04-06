# Domain Integration Layer Architecture

## Overview

The Domain Integration Layer serves as the universal interface between LLM agents and 15+ data science domains, enabling seamless analytical workflows while preserving each domain's specialized capabilities. This architecture implements our established first principles to create consistent, intention-driven interfaces across diverse analytical capabilities.

## Architectural Principles

Building on the First Principles Architecture Framework (32.1) and Core Pipeline Framework (32.2):

1. **Intention-Driven Interface** - LLM agents express analytical intentions, not statistical procedures
2. **Context-Aware Composition** - Results designed for downstream tool composition with metadata
3. **Progressive Disclosure Architecture** - Simple by default, powerful when needed
4. **Streaming-First Data Science** - Memory constraints as architectural requirements
5. **Modular Domain Integration** - Seamless cross-domain analytical workflows

## Core Architecture Components

### 1. Universal Domain Interface

All domains implement a standardized interface that translates LLM intentions into domain-specific operations:

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from .core_pipeline import AnalysisResult, DataStream, CompositionMetadata

@dataclass
class DomainCapabilities:
    """Registry of domain analytical capabilities"""
    domain_name: str
    intentions: List[str]           # ["test correlation", "predict values", "find patterns"]
    input_types: List[str]          # ["tabular", "time_series", "text", "geospatial"]
    output_types: List[str]         # ["statistical_test", "predictions", "clusters"]
    dependencies: List[str]         # Other domains this relies on
    streaming_support: bool         # Supports chunked data processing
    phase: int                      # 1=core, 2=extended, 3=advanced
    library_requirements: List[str] # ["scipy", "sklearn", "statsmodels"]

class DomainInterface(ABC):
    """Universal interface all domains must implement"""
    
    @abstractmethod
    def get_capabilities(self) -> DomainCapabilities:
        """Return domain's analytical capabilities"""
        pass
    
    @abstractmethod  
    def analyze(self, intention: str, data: DataStream, context: Optional[List[AnalysisResult]] = None, **params) -> AnalysisResult:
        """Execute analysis based on LLM intention"""
        pass
    
    @abstractmethod
    def compose_with(self, other_result: AnalysisResult) -> CompositionMetadata:
        """Define how this domain's results compose with others"""
        pass
    
    @abstractmethod
    def validate_input(self, data: DataStream, intention: str) -> ValidationResult:
        """Validate data compatibility with intention"""
        pass
```

### 2. Domain Capabilities Registry

Centralized catalog system enabling domain discovery and automatic workflow composition:

```python
class DomainRegistry:
    """Centralized registry of all analytical domains"""
    
    def __init__(self):
        self.domains: Dict[str, DomainInterface] = {}
        self.capabilities: Dict[str, DomainCapabilities] = {}
        self.dependency_graph = DependencyGraph()
        self.phase_manager = PhaseManager()
    
    def register_domain(self, domain: DomainInterface) -> None:
        """Register new domain with capability validation"""
        capabilities = domain.get_capabilities()
        self._validate_domain_capabilities(capabilities)
        
        self.domains[capabilities.domain_name] = domain
        self.capabilities[capabilities.domain_name] = capabilities
        self.dependency_graph.add_domain(capabilities)
    
    def find_domains_for_intention(self, intention: str) -> List[str]:
        """Find domains capable of handling specific intention"""
        matching_domains = []
        for domain_name, caps in self.capabilities.items():
            if self._matches_intention(intention, caps.intentions):
                matching_domains.append(domain_name)
        return matching_domains
    
    def get_workflow_sequence(self, intention: str, available_domains: List[str] = None) -> List[str]:
        """Determine optimal domain sequence for complex intentions"""
        return self.dependency_graph.resolve_execution_order(intention, available_domains)
```

### 3. Cross-Domain Workflow Orchestrator

Enables automatic composition of multi-domain analytical pipelines:

```python
class DomainWorkflow:
    """Orchestrates multi-domain analytical workflows"""
    
    def __init__(self, registry: DomainRegistry):
        self.registry = registry
        self.execution_engine = WorkflowEngine()
    
    def execute_intention(self, intention: str, data: DataStream, **params) -> WorkflowResult:
        """Execute complex intention across multiple domains"""
        
        # Parse complex intention into domain-specific sub-intentions
        sub_intentions = self._parse_complex_intention(intention)
        
        # Determine optimal domain execution sequence
        domain_sequence = self._resolve_domain_sequence(sub_intentions)
        
        # Execute pipeline with automatic handoffs
        results = []
        current_data = data
        
        for domain_name, sub_intention in domain_sequence:
            domain = self.registry.domains[domain_name]
            
            # Execute analysis with context from previous results
            result = domain.analyze(
                intention=sub_intention,
                data=current_data, 
                context=results,
                **params
            )
            
            results.append(result)
            
            # Update data stream for next domain (if applicable)
            current_data = self._prepare_data_for_next_domain(result, current_data)
        
        return self._compose_final_result(results)
    
    def _parse_complex_intention(self, intention: str) -> List[Tuple[str, str]]:
        """Parse complex intentions like 'analyze trends and predict future values'"""
        # Implementation would use NLP to identify sub-intentions
        # Returns list of (domain_name, sub_intention) pairs
        pass
```

## Domain-Specific Integration Patterns

### Phase 1 Domains (Core Capabilities)

#### Statistical Analysis Domain
```python
class StatisticalAnalysisDomain(DomainInterface):
    """Core statistical analysis capabilities"""
    
    def get_capabilities(self) -> DomainCapabilities:
        return DomainCapabilities(
            domain_name="statistical_analysis",
            intentions=[
                "test correlation", "check normality", "compare groups",
                "test independence", "calculate effect size", "validate assumptions"
            ],
            input_types=["tabular", "numeric_array"],
            output_types=["statistical_test", "descriptive_stats", "assumption_validation"],
            dependencies=[],  # Foundation domain
            streaming_support=True,
            phase=1,
            library_requirements=["scipy", "pingouin", "statsmodels"]
        )
    
    def analyze(self, intention: str, data: DataStream, **params) -> AnalysisResult:
        """Map intentions to statistical tests"""
        if "correlation" in intention:
            return self._analyze_correlation(data, **params)
        elif "normality" in intention:
            return self._test_normality(data, **params)
        elif "compare groups" in intention:
            return self._compare_groups(data, **params)
        # ... additional intention mapping
```

#### Regression & Modeling Domain
```python
class RegressionDomain(DomainInterface):
    """Predictive modeling and regression analysis"""
    
    def get_capabilities(self) -> DomainCapabilities:
        return DomainCapabilities(
            domain_name="regression_modeling",
            intentions=[
                "predict values", "model relationship", "forecast target",
                "build model", "validate model", "feature importance"
            ],
            input_types=["tabular", "feature_target_pairs"],
            output_types=["predictions", "model_diagnostics", "feature_importance"],
            dependencies=["statistical_analysis"],  # For assumption validation
            streaming_support=True,
            phase=1,
            library_requirements=["sklearn", "statsmodels"]
        )
```

#### Pattern Recognition Domain
```python
class PatternRecognitionDomain(DomainInterface):
    """Advanced pattern matching and anomaly detection"""
    
    def get_capabilities(self) -> DomainCapabilities:
        return DomainCapabilities(
            domain_name="pattern_recognition",
            intentions=[
                "find similar records", "match fuzzy strings", "detect anomalies",
                "cluster data", "find duplicates", "entity resolution"
            ],
            input_types=["tabular", "text", "mixed"],
            output_types=["similarity_matches", "clusters", "anomalies"],
            dependencies=[],  # Independent domain
            streaming_support=True,
            phase=1,
            library_requirements=["rapidfuzz", "fuzzywuzzy", "sklearn"]
        )
```

### Phase 2 Domains (High Business Value)

#### Time Series Analysis Domain
```python
class TimeSeriesDomain(DomainInterface):
    """Time series analysis and forecasting"""
    
    def get_capabilities(self) -> DomainCapabilities:
        return DomainCapabilities(
            domain_name="time_series",
            intentions=[
                "analyze trends", "seasonal decomposition", "forecast future",
                "detect changepoints", "time series clustering"
            ],
            input_types=["time_series", "temporal_tabular"],
            output_types=["forecasts", "trend_analysis", "seasonal_components"],
            dependencies=["statistical_analysis", "regression_modeling"],
            streaming_support=True,
            phase=2,
            library_requirements=["statsmodels", "prophet", "sktime"]
        )
```

#### Business Intelligence Domain
```python
class BusinessIntelligenceDomain(DomainInterface):
    """Business metrics and KPI analysis"""
    
    def get_capabilities(self) -> DomainCapabilities:
        return DomainCapabilities(
            domain_name="business_intelligence",
            intentions=[
                "calculate kpis", "analyze business metrics", "cohort analysis",
                "funnel analysis", "retention analysis", "growth metrics"
            ],
            input_types=["tabular", "event_data", "transactional"],
            output_types=["kpi_dashboard", "business_metrics", "cohort_tables"],
            dependencies=["statistical_analysis", "time_series"],
            streaming_support=True,
            phase=2,
            library_requirements=["pandas", "numpy"]
        )
```

## Library Integration Architecture

### Adapter Pattern Implementation

```python
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

class LibraryAdapter(ABC):
    """Base adapter for external library integration"""
    
    @abstractmethod
    def wrap_library_function(self, func_name: str, *args, **kwargs) -> AnalysisResult:
        """Wrap external library function into our domain interface"""
        pass
    
    @abstractmethod
    def map_parameters(self, semantic_params: Dict[str, Any]) -> Dict[str, Any]:
        """Convert semantic parameters to library-specific parameters"""
        pass
    
    @abstractmethod
    def wrap_result(self, library_result: Any, metadata: Dict[str, Any]) -> AnalysisResult:
        """Convert library result to our AnalysisResult format"""
        pass

class ScipyStatsAdapter(LibraryAdapter):
    """Adapter for scipy.stats functions"""
    
    def wrap_library_function(self, func_name: str, *args, **kwargs) -> AnalysisResult:
        import scipy.stats as stats
        
        # Get the scipy function
        scipy_func = getattr(stats, func_name)
        
        # Map semantic parameters to scipy parameters
        scipy_params = self.map_parameters(kwargs)
        
        # Execute scipy function
        result = scipy_func(*args, **scipy_params)
        
        # Wrap in our standard format
        return self.wrap_result(result, {"function": func_name, "library": "scipy.stats"})
    
    def map_parameters(self, semantic_params: Dict[str, Any]) -> Dict[str, Any]:
        """Convert semantic parameter names to scipy equivalents"""
        mapping = {
            "confidence_level": "alpha",
            "alternative_hypothesis": "alternative",
            "equal_variances": "equal_var"
        }
        
        scipy_params = {}
        for key, value in semantic_params.items():
            scipy_key = mapping.get(key, key)
            scipy_params[scipy_key] = value
        
        return scipy_params

class StatsmodelsAdapter(LibraryAdapter):
    """Adapter for statsmodels functions"""
    # Similar implementation for statsmodels integration

class SklearnAdapter(LibraryAdapter):
    """Adapter for sklearn estimators"""
    # Similar implementation for sklearn integration
```

### Library Registry and Discovery

```python
class LibraryRegistry:
    """Registry for available library adapters with graceful degradation"""
    
    def __init__(self):
        self.adapters: Dict[str, LibraryAdapter] = {}
        self.available_libraries = self._discover_available_libraries()
        self.capability_fallbacks = self._build_fallback_map()
    
    def _discover_available_libraries(self) -> Dict[str, bool]:
        """Discover which libraries are actually installed"""
        libraries = {}
        
        try:
            import scipy.stats
            libraries["scipy"] = True
        except ImportError:
            libraries["scipy"] = False
        
        try:
            import sklearn
            libraries["sklearn"] = True
        except ImportError:
            libraries["sklearn"] = False
        
        try:
            import statsmodels
            libraries["statsmodels"] = True
        except ImportError:
            libraries["statsmodels"] = False
        
        return libraries
    
    def get_best_adapter(self, capability: str) -> Optional[LibraryAdapter]:
        """Get best available adapter for requested capability"""
        preferred_adapters = self.capability_fallbacks.get(capability, [])
        
        for adapter_name in preferred_adapters:
            if adapter_name in self.adapters and self.available_libraries.get(adapter_name, False):
                return self.adapters[adapter_name]
        
        return None
    
    def _build_fallback_map(self) -> Dict[str, List[str]]:
        """Define fallback preferences for capabilities"""
        return {
            "correlation_test": ["scipy", "statsmodels", "pandas"],
            "regression_modeling": ["sklearn", "statsmodels"],
            "time_series_forecast": ["statsmodels", "prophet", "sklearn"],
            "fuzzy_matching": ["rapidfuzz", "fuzzywuzzy"],
        }
```

## Phase-Based Integration Strategy

### Phase Management System

```python
class PhaseManager:
    """Manages incremental domain integration phases"""
    
    def __init__(self):
        self.current_phase = self._detect_current_phase()
        self.phase_definitions = self._load_phase_definitions()
        self.available_domains = self._load_phase_domains(self.current_phase)
    
    def _load_phase_definitions(self) -> Dict[int, Dict[str, Any]]:
        """Define what capabilities are available in each phase"""
        return {
            1: {
                "name": "Core Capabilities",
                "domains": ["statistical_analysis", "regression_modeling", "pattern_recognition"],
                "priority": "Production-ready foundation",
                "timeline": "Immediate implementation"
            },
            2: {
                "name": "High Business Value",
                "domains": ["time_series", "business_intelligence", "sampling_estimation", "non_parametric"],
                "priority": "Business-critical analytics",
                "timeline": "Next implementation cycle"
            },
            3: {
                "name": "Advanced Analytics",
                "domains": ["geospatial", "multivariate_stats", "survival_analysis"],
                "priority": "Specialized analytical capabilities",
                "timeline": "Future expansion"
            },
            4: {
                "name": "Emerging Domains",
                "domains": ["experimental_design", "causal_inference", "bayesian_methods"],
                "priority": "Cutting-edge methodologies",
                "timeline": "Research and development"
            }
        }
    
    def upgrade_to_phase(self, target_phase: int) -> UpgradeResult:
        """Upgrade system to support additional phase capabilities"""
        if target_phase <= self.current_phase:
            return UpgradeResult(success=False, message="Already at or beyond target phase")
        
        # Check dependencies and requirements
        missing_requirements = self._check_phase_requirements(target_phase)
        if missing_requirements:
            return UpgradeResult(success=False, missing_requirements=missing_requirements)
        
        # Load new domain capabilities
        new_domains = self._load_phase_domains(target_phase)
        
        # Update registry
        for domain in new_domains:
            self.available_domains[domain.get_capabilities().domain_name] = domain
        
        self.current_phase = target_phase
        return UpgradeResult(success=True, new_capabilities=len(new_domains))
```

## Domain Extension Framework

### Plugin Architecture

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class DomainPlugin(Protocol):
    """Protocol for domain plugins"""
    
    @staticmethod
    def register_domain() -> 'DomainRegistration':
        """Return domain registration information"""
        ...
    
    def create_domain_instance(self) -> DomainInterface:
        """Create instance of the domain implementation"""
        ...

@dataclass
class DomainRegistration:
    """Registration information for domain plugins"""
    name: str
    capabilities: DomainCapabilities
    implementation_class: type
    dependencies: List[str]
    phase: int
    version: str
    author: str
    description: str

class DomainPluginManager:
    """Manages domain plugin lifecycle"""
    
    def __init__(self, registry: DomainRegistry):
        self.registry = registry
        self.plugins: Dict[str, DomainPlugin] = {}
        self.validator = DomainValidator()
    
    def discover_plugins(self, plugin_directory: str = "plugins/") -> List[DomainPlugin]:
        """Discover available domain plugins"""
        plugins = []
        for plugin_file in Path(plugin_directory).glob("*.py"):
            plugin = self._load_plugin(plugin_file)
            if plugin and self.validator.validate_plugin(plugin):
                plugins.append(plugin)
        return plugins
    
    def register_plugin(self, plugin: DomainPlugin) -> RegistrationResult:
        """Register new domain plugin"""
        registration = plugin.register_domain()
        
        # Validate plugin compatibility
        validation_result = self.validator.validate_plugin(plugin)
        if not validation_result.is_valid:
            return RegistrationResult(success=False, errors=validation_result.errors)
        
        # Check dependency satisfaction
        if not self._check_dependencies(registration.dependencies):
            return RegistrationResult(success=False, errors=["Missing dependencies"])
        
        # Create domain instance and register
        domain_instance = plugin.create_domain_instance()
        self.registry.register_domain(domain_instance)
        
        self.plugins[registration.name] = plugin
        return RegistrationResult(success=True, domain_name=registration.name)

class DomainValidator:
    """Validates domain plugins for compatibility and correctness"""
    
    def validate_plugin(self, plugin: DomainPlugin) -> ValidationResult:
        """Comprehensive plugin validation"""
        errors = []
        warnings = []
        
        try:
            # Check protocol compliance
            if not isinstance(plugin, DomainPlugin):
                errors.append("Plugin does not implement DomainPlugin protocol")
            
            # Test domain instance creation
            domain_instance = plugin.create_domain_instance()
            if not isinstance(domain_instance, DomainInterface):
                errors.append("Plugin does not create valid DomainInterface instance")
            
            # Validate capabilities
            capabilities = domain_instance.get_capabilities()
            self._validate_capabilities(capabilities, errors, warnings)
            
            # Test interface methods
            self._validate_interface_methods(domain_instance, errors, warnings)
            
        except Exception as e:
            errors.append(f"Plugin validation failed: {str(e)}")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )
    
    def _validate_capabilities(self, capabilities: DomainCapabilities, errors: List[str], warnings: List[str]) -> None:
        """Validate domain capabilities specification"""
        if not capabilities.intentions:
            errors.append("Domain must specify at least one intention")
        
        if not capabilities.input_types:
            errors.append("Domain must specify supported input types")
        
        if capabilities.phase < 1 or capabilities.phase > 4:
            errors.append("Domain phase must be between 1 and 4")
        
        # Check for streaming support if required
        if not capabilities.streaming_support:
            warnings.append("Domain does not support streaming - may impact performance")
```

### Example Custom Domain Plugin

```python
# plugins/custom_nlp_domain.py
from typing import List, Optional
from ..core import DomainInterface, DomainCapabilities, AnalysisResult, DataStream
from ..integration import DomainPlugin, DomainRegistration

class NLPDomain(DomainInterface):
    """Custom NLP analysis domain"""
    
    def get_capabilities(self) -> DomainCapabilities:
        return DomainCapabilities(
            domain_name="nlp_analysis",
            intentions=[
                "sentiment analysis", "topic modeling", "text classification",
                "named entity recognition", "text similarity"
            ],
            input_types=["text", "text_array"],
            output_types=["sentiment_scores", "topics", "entities", "classifications"],
            dependencies=["statistical_analysis", "pattern_recognition"],
            streaming_support=True,
            phase=3,
            library_requirements=["nltk", "spacy", "transformers"]
        )
    
    def analyze(self, intention: str, data: DataStream, context: Optional[List[AnalysisResult]] = None, **params) -> AnalysisResult:
        """Execute NLP analysis based on intention"""
        if "sentiment" in intention:
            return self._analyze_sentiment(data, **params)
        elif "topic" in intention:
            return self._model_topics(data, **params)
        # ... additional NLP capabilities
    
    # ... implement remaining interface methods

class NLPDomainPlugin:
    """Plugin wrapper for NLP domain"""
    
    @staticmethod
    def register_domain() -> DomainRegistration:
        return DomainRegistration(
            name="nlp_analysis",
            capabilities=NLPDomain().get_capabilities(),
            implementation_class=NLPDomain,
            dependencies=["statistical_analysis", "pattern_recognition"],
            phase=3,
            version="1.0.0",
            author="Custom Plugin Developer",
            description="Natural Language Processing analysis capabilities"
        )
    
    def create_domain_instance(self) -> DomainInterface:
        return NLPDomain()

# Plugin registration
def get_plugin() -> DomainPlugin:
    return NLPDomainPlugin()
```

## Integration Benefits

### For LLM Agents
- **Consistent Interface**: Same intention-based pattern across all 15 domains
- **Automatic Discovery**: Query available capabilities without domain-specific knowledge
- **Workflow Composition**: Complex multi-domain analyses compose automatically
- **Rich Context**: Comprehensive metadata enables intelligent pipeline decisions

### For System Architecture
- **Incremental Growth**: Phase-based integration allows controlled capability expansion
- **Library Independence**: Graceful degradation when optional libraries unavailable
- **Plugin Extensibility**: New domains integrate without core system changes
- **Streaming Compatibility**: All domains support memory-efficient chunked processing

### For Domain Specialists
- **Specialized Preservation**: Each domain maintains its unique analytical strengths
- **Library Freedom**: Use best-of-breed libraries within consistent interface
- **Context Awareness**: Access to results from other domains for informed decisions
- **Standard Integration**: Predictable integration patterns reduce development overhead

## Implementation Roadmap

### Phase 1: Foundation (Immediate)
1. Implement Universal Domain Interface
2. Create Domain Registry system
3. Build Statistical Analysis, Regression, and Pattern Recognition domains
4. Establish Library Integration Architecture

### Phase 2: Extension (Next Cycle)
1. Implement Cross-Domain Workflow Orchestrator
2. Add Time Series, Business Intelligence, and Non-Parametric domains
3. Create Phase Management System
4. Develop comprehensive testing framework

### Phase 3: Advanced Capabilities (Future)
1. Implement Domain Extension Framework
2. Add Geospatial, Multivariate Statistics, and Survival Analysis domains
3. Create plugin validation and management tools
4. Establish domain marketplace ecosystem

This Domain Integration Layer Architecture provides the foundation for LocalData MCP v2.0 to seamlessly support 15+ data science domains while maintaining consistency, extensibility, and the intention-driven interface that makes complex analytics accessible to LLM agents.