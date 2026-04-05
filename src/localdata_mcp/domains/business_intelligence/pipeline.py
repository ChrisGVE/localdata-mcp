"""
Business Intelligence Domain - Pipeline orchestration.

This module implements the BusinessIntelligencePipeline that combines customer
analytics, A/B testing, attribution modeling, and funnel analysis into a
comprehensive end-to-end business intelligence workflow.
"""

from typing import Any, Dict, List

import pandas as pd

from ...logging_manager import get_logger
from ...pipeline.base import (
    AnalysisPipelineBase,
    PipelineResult,
    CompositionMetadata,
    StreamingConfig,
    PipelineState,
)
from .customer_analytics import RFMAnalysisTransformer
from .cohort_clv import CohortAnalysisTransformer, CLVCalculator
from .ab_testing import ABTestAnalyzer, PowerAnalysisTransformer
from .attribution import AttributionAnalyzer
from .funnel import FunnelAnalyzer
from .convenience import (
    analyze_rfm,
    perform_cohort_analysis,
    calculate_clv,
    analyze_attribution,
    analyze_funnel,
)

logger = get_logger(__name__)


class BusinessIntelligencePipeline(AnalysisPipelineBase):
    """
    Comprehensive Business Intelligence pipeline combining customer analytics,
    A/B testing, attribution modeling, and funnel analysis.

    This pipeline orchestrates multiple BI transformers to provide end-to-end
    business intelligence capabilities for marketing and customer analytics.

    Parameters:
    -----------
    customer_analytics : bool, default=True
        Whether to include customer analytics (RFM, cohort, CLV)
    ab_testing : bool, default=True
        Whether to include A/B testing capabilities
    attribution_modeling : bool, default=True
        Whether to include marketing attribution analysis
    funnel_analysis : bool, default=True
        Whether to include conversion funnel analysis
    """

    def __init__(
        self,
        customer_analytics=True,
        ab_testing=True,
        attribution_modeling=True,
        funnel_analysis=True,
        streaming_config=None,
        **kwargs,
    ):
        super().__init__(
            analytical_intention="business intelligence analysis",
            streaming_config=streaming_config,
            **kwargs,
        )

        self.customer_analytics = customer_analytics
        self.ab_testing = ab_testing
        self.attribution_modeling = attribution_modeling
        self.funnel_analysis = funnel_analysis

        # Initialize component transformers
        self._init_transformers()

    def _init_transformers(self):
        """Initialize component transformers based on configuration."""
        self.transformers = {}

        if self.customer_analytics:
            self.transformers["rfm"] = RFMAnalysisTransformer()
            self.transformers["cohort"] = CohortAnalysisTransformer()
            self.transformers["clv"] = CLVCalculator()

        if self.ab_testing:
            self.transformers["ab_test"] = ABTestAnalyzer()
            self.transformers["power_analysis"] = PowerAnalysisTransformer()

        if self.attribution_modeling:
            self.transformers["attribution"] = AttributionAnalyzer()

        if self.funnel_analysis:
            self.transformers["funnel"] = FunnelAnalyzer()

    def fit(self, X, y=None):
        """Fit all enabled BI transformers."""
        logger.info("Fitting Business Intelligence pipeline components")

        for name, transformer in self.transformers.items():
            try:
                transformer.fit(X, y)
                logger.debug(f"Successfully fitted {name} transformer")
            except Exception as e:
                logger.warning(f"Failed to fit {name} transformer: {str(e)}")

        self.is_fitted_ = True
        return self

    def transform(self, X):
        """Transform data using all fitted BI transformers."""
        logger.info("Executing Business Intelligence pipeline analysis")

        results = {}
        metadata = CompositionMetadata(
            domain="business_intelligence",
            analysis_type="business_intelligence",
            result_type="multi_component_analysis",
        )

        for name, transformer in self.transformers.items():
            try:
                if hasattr(transformer, "is_fitted_") and transformer.is_fitted_:
                    result = transformer.transform(X)
                    results[name] = result
                    logger.debug(f"Successfully executed {name} analysis")
                else:
                    logger.warning(f"Transformer {name} not fitted, skipping")
            except Exception as e:
                logger.error(f"Error executing {name} analysis: {str(e)}")
                results[name] = f"Error: {str(e)}"

        # Create pipeline result
        pipeline_result = PipelineResult(
            success=True,
            data=results,
            metadata={},
            execution_time_seconds=0.0,
            memory_used_mb=0.0,
            pipeline_stage="transform",
            composition_metadata=metadata,
        )

        return pipeline_result

    def get_analysis_type(self) -> str:
        """Get the specific analysis type this pipeline performs."""
        return "business_intelligence"

    def _configure_analysis_pipeline(self):
        """Configure analysis steps based on intention and complexity level."""
        steps = []
        if self.customer_analytics:
            steps.append(self._run_customer_analytics)
        if self.ab_testing:
            steps.append(self._run_ab_testing)
        if self.attribution_modeling:
            steps.append(self._run_attribution)
        if self.funnel_analysis:
            steps.append(self._run_funnel_analysis)
        return steps

    def _execute_analysis_step(self, step, data, context):
        """Execute individual analysis step with error handling and metadata."""
        result = step(data) if callable(step) else step
        return result, {}

    def _execute_streaming_analysis(self, data):
        """Execute analysis with streaming support for large datasets."""
        return self._execute_standard_analysis(data)

    def _execute_standard_analysis(self, data):
        """Execute analysis on full dataset in memory."""
        results = {}
        for name, transformer in self.transformers.items():
            try:
                if hasattr(transformer, "is_fitted_") and transformer.is_fitted_:
                    results[name] = transformer.transform(data)
            except Exception as e:
                logger.error(f"Error in {name}: {str(e)}")
                results[name] = f"Error: {str(e)}"
        return results, {}

    def _run_customer_analytics(self, data):
        """Run customer analytics transformers."""
        results = {}
        for name in ("rfm", "cohort", "clv"):
            t = self.transformers.get(name)
            if t and hasattr(t, "is_fitted_") and t.is_fitted_:
                results[name] = t.transform(data)
        return results

    def _run_ab_testing(self, data):
        """Run A/B testing transformers."""
        results = {}
        for name in ("ab_test", "power_analysis"):
            t = self.transformers.get(name)
            if t and hasattr(t, "is_fitted_") and t.is_fitted_:
                results[name] = t.transform(data)
        return results

    def _run_attribution(self, data):
        """Run attribution analysis transformer."""
        t = self.transformers.get("attribution")
        if t and hasattr(t, "is_fitted_") and t.is_fitted_:
            return t.transform(data)
        return {}

    def _run_funnel_analysis(self, data):
        """Run funnel analysis transformer."""
        t = self.transformers.get("funnel")
        if t and hasattr(t, "is_fitted_") and t.is_fitted_:
            return t.transform(data)
        return {}

    def analyze_customer_journey(
        self,
        transaction_data: pd.DataFrame,
        touchpoint_data: pd.DataFrame = None,
        funnel_data: pd.DataFrame = None,
    ) -> Dict[str, Any]:
        """
        Comprehensive customer journey analysis combining all BI components.

        Parameters:
        -----------
        transaction_data : pd.DataFrame
            Customer transaction data for RFM, cohort, and CLV analysis
        touchpoint_data : pd.DataFrame, optional
            Marketing touchpoint data for attribution analysis
        funnel_data : pd.DataFrame, optional
            Funnel step data for conversion analysis

        Returns:
        --------
        analysis : Dict[str, Any]
            Complete customer journey analysis results
        """
        logger.info("Performing comprehensive customer journey analysis")

        journey_analysis = {}

        # Customer Analytics
        if self.customer_analytics and transaction_data is not None:
            journey_analysis["customer_segments"] = analyze_rfm(transaction_data)
            journey_analysis["retention_analysis"] = perform_cohort_analysis(
                transaction_data
            )
            journey_analysis["lifetime_value"] = calculate_clv(transaction_data)

        # Attribution Analysis
        if self.attribution_modeling and touchpoint_data is not None:
            journey_analysis["attribution"] = analyze_attribution(touchpoint_data)

        # Funnel Analysis
        if self.funnel_analysis and funnel_data is not None:
            journey_analysis["funnel_optimization"] = analyze_funnel(funnel_data)

        # Cross-analysis insights
        journey_analysis["insights"] = self._generate_cross_analysis_insights(
            journey_analysis
        )

        return journey_analysis

    def _generate_cross_analysis_insights(
        self, analysis_results: Dict[str, Any]
    ) -> List[str]:
        """Generate insights from cross-analysis of BI components."""
        insights = []

        # RFM + CLV insights
        if (
            "customer_segments" in analysis_results
            and "lifetime_value" in analysis_results
        ):
            insights.append(
                "Cross-reference RFM segments with CLV to identify high-value customer characteristics"
            )

        # Cohort + Attribution insights
        if (
            "retention_analysis" in analysis_results
            and "attribution" in analysis_results
        ):
            insights.append(
                "Analyze which acquisition channels produce customers with better retention rates"
            )

        # Funnel + Attribution insights
        if (
            "funnel_optimization" in analysis_results
            and "attribution" in analysis_results
        ):
            insights.append(
                "Optimize attribution models based on funnel performance at each touchpoint"
            )

        if not insights:
            insights.append(
                "Enable multiple analysis components for cross-analysis insights"
            )

        return insights
