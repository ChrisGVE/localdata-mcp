# Claude Code Instructions

## LocalData MCP v2.0 - Architectural Constitution

### First Principles (MANDATORY REFERENCE FOR ALL DECISIONS)

All development decisions MUST align with our five foundational architectural principles:

**1. Intention-Driven Interface**
- LLM agents think in analytical intentions, not statistical procedures
- Tools accept semantic parameters: "find strong correlations" vs "threshold=0.7"
- Tool discovery by analytical goal, not statistical method name
- Results include interpretation guidance for downstream reasoning

**2. Context-Aware Composition** 
- Data science workflows are naturally compositional
- Every tool result includes metadata for downstream composition
- Tools automatically adapt behavior based on upstream context
- Workflow state preserved and enriched through analysis chain

**3. Progressive Disclosure Architecture**
- Simple by default, powerful when needed
- Basic calls provide high-level insights with sensible defaults
- Advanced parameters available for fine-grained control when needed
- Graceful degradation - works with minimal input, optimizes with rich input

**4. Streaming-First Data Science**
- Memory constraints (16-64GB) are architectural requirements, not implementation details
- All operations designed for streaming/chunked processing by default
- Tools automatically switch processing strategies based on data size
- Sub-100ms tool discovery regardless of dataset size

**5. Modular Domain Integration**
- Each domain (statistical, regression, time series, etc.) is a composable module
- Cross-domain workflows are first-class citizens
- Domain knowledge embedded in tools, not required from users
- New domains integrate with existing ones automatically

### Design Decision Framework

Every design decision must pass these tests:
1. **LLM Naturalness Test**: Would an LLM agent naturally use this API?
2. **Composition Test**: Does this result feed naturally into other tools?
3. **Streaming Compatibility Test**: Can this work with chunked/streaming data?
4. **Progressive Complexity Test**: Simple for basic use, detailed for advanced?
5. **Domain Integration Test**: Does this fit naturally with other data science domains?

**Trade-off Resolution Hierarchy**: 
1. LLM Experience (top priority)
2. Composition & Workflow 
3. Memory Constraints (hard limits)
4. Simplicity 
5. Performance (within constraints)

### Key Architecture References

**Core Documentation**: 
- `/docs/architecture/FIRST_PRINCIPLES.md` - Constitutional foundation
- `/docs/architecture/DESIGN_DECISIONS.md` - Decision evaluation framework
- `/docs/architecture/CORE_PATTERNS.md` - Implementation patterns
- `/docs/architecture/LIBRARY_STRATEGY.md` - External library integration

**Before implementing ANY feature**:
1. Reference First Principles for alignment
2. Apply Design Decision Framework tests
3. Use Core Patterns for implementation guidance
4. Follow Library Strategy for external dependencies

### Current Project Status

**Mission**: Evolve LocalData MCP from v1.4.0 database tool to v2.0 comprehensive data science platform optimized for LLM-agent workflows.

**Phase 1 Domains**: Statistical Analysis, Regression & Modeling, Advanced Pattern Recognition
**Phase 2 Domains**: Time Series, Sampling & Estimation, Business Intelligence, Non-Parametric Methods

**Architecture Status**: First Principles Framework ESTABLISHED ✅
**Next**: Domain implementation following architectural guidelines

## Task Master AI Instructions
**Import Task Master's development workflow commands and guidelines, treat as if import is in the main CLAUDE.md file.**
@./.taskmaster/CLAUDE.md
