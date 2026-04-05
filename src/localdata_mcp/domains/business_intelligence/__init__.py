"""
Business Intelligence Domain - Comprehensive business analytics and KPI analysis.

This package implements advanced business intelligence tools including customer analytics,
A/B testing, attribution modeling, and marketing metrics using sklearn integration and
specialized business analytics libraries.

Key Features:
- Customer Analytics (RFM analysis, cohort analysis, CLV calculation, churn prediction)
- A/B Testing & Experimental Design (power analysis, significance testing, multi-armed bandit)
- Attribution Modeling (first-touch, last-touch, multi-touch, Markov chains)
- Marketing Metrics (funnel analysis, CAC, ROAS, market basket analysis)
- Full sklearn pipeline compatibility
- Streaming-compatible processing
- Business-focused KPI calculations
"""

# Enums
from .models import AttributionModel, ExperimentStatus

# Result classes
from .models import (
    RFMResult,
    CohortAnalysisResult,
    CLVResult,
    ABTestResult,
    AttributionResult,
    FunnelAnalysisResult,
)

# Core transformers
from .customer_analytics import RFMAnalysisTransformer
from .cohort_clv import CohortAnalysisTransformer, CLVCalculator
from .ab_testing import ABTestAnalyzer, PowerAnalysisTransformer
from .attribution import AttributionAnalyzer
from .funnel import FunnelAnalyzer

# Pipeline
from .pipeline import BusinessIntelligencePipeline

# High-level functions
from .convenience import (
    analyze_rfm,
    perform_cohort_analysis,
    calculate_clv,
    perform_ab_test,
    analyze_attribution,
    analyze_funnel,
    enhanced_ab_test,
)

__all__ = [
    # Enums
    "AttributionModel",
    "ExperimentStatus",
    # Result classes
    "RFMResult",
    "CohortAnalysisResult",
    "CLVResult",
    "ABTestResult",
    "AttributionResult",
    "FunnelAnalysisResult",
    # Core transformers
    "RFMAnalysisTransformer",
    "CohortAnalysisTransformer",
    "CLVCalculator",
    "ABTestAnalyzer",
    "PowerAnalysisTransformer",
    "AttributionAnalyzer",
    "FunnelAnalyzer",
    # Pipeline
    "BusinessIntelligencePipeline",
    # High-level functions
    "analyze_rfm",
    "perform_cohort_analysis",
    "calculate_clv",
    "perform_ab_test",
    "analyze_attribution",
    "analyze_funnel",
    "enhanced_ab_test",
]
