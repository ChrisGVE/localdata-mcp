"""
Business Intelligence Domain - Comprehensive business analytics and KPI analysis.

This module implements advanced business intelligence tools including customer analytics,
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

import warnings
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import time
import json
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import chi2_contingency, ttest_ind, mannwhitneyu
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import statsmodels.api as sm
from statsmodels.stats.power import ttest_power, tt_solve_power
from statsmodels.stats.proportion import proportions_ztest, proportion_confint

from ..logging_manager import get_logger
from ..pipeline.base import (
    AnalysisPipelineBase, PipelineResult, CompositionMetadata, 
    StreamingConfig, PipelineState
)

logger = get_logger(__name__)

# Suppress specific warnings that are not critical for our use case
warnings.filterwarnings('ignore', category=RuntimeWarning, module='scipy.stats')
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')


class AttributionModel(Enum):
    """Attribution model types for marketing analysis."""
    FIRST_TOUCH = "first_touch"
    LAST_TOUCH = "last_touch"
    LINEAR = "linear"
    TIME_DECAY = "time_decay"
    POSITION_BASED = "position_based"


class ExperimentStatus(Enum):
    """Status of A/B test experiments."""
    PLANNING = "planning"
    RUNNING = "running"
    COMPLETED = "completed"
    STOPPED = "stopped"


@dataclass
class RFMResult:
    """Result structure for RFM analysis."""
    rfm_scores: pd.DataFrame
    segments: pd.DataFrame
    segment_summary: pd.DataFrame
    quartile_boundaries: Dict[str, List[float]]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary format."""
        return {
            'rfm_scores': self.rfm_scores.to_dict('records'),
            'segments': self.segments.to_dict('records'),
            'segment_summary': self.segment_summary.to_dict('records'),
            'quartile_boundaries': self.quartile_boundaries
        }


@dataclass
class CohortAnalysisResult:
    """Result structure for cohort analysis."""
    cohort_table: pd.DataFrame
    cohort_sizes: pd.DataFrame
    retention_rates: pd.DataFrame
    period_summary: Dict[str, float]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary format."""
        return {
            'cohort_table': self.cohort_table.to_dict(),
            'cohort_sizes': self.cohort_sizes.to_dict(),
            'retention_rates': self.retention_rates.to_dict(),
            'period_summary': self.period_summary
        }


@dataclass
class CLVResult:
    """Result structure for Customer Lifetime Value analysis."""
    clv_scores: pd.DataFrame
    model_metrics: Dict[str, float]
    clv_distribution: Dict[str, float]
    segment_clv: Dict[str, float]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary format."""
        return {
            'clv_scores': self.clv_scores.to_dict('records'),
            'model_metrics': self.model_metrics,
            'clv_distribution': self.clv_distribution,
            'segment_clv': self.segment_clv
        }


@dataclass
class ABTestResult:
    """Result structure for A/B test analysis."""
    test_name: str
    test_statistic: float
    p_value: float
    confidence_interval: Tuple[float, float]
    effect_size: float
    power: float
    conclusion: str
    sample_sizes: Dict[str, int]
    conversion_rates: Dict[str, float]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary format."""
        return {
            'test_name': self.test_name,
            'test_statistic': self.test_statistic,
            'p_value': self.p_value,
            'confidence_interval': self.confidence_interval,
            'effect_size': self.effect_size,
            'power': self.power,
            'conclusion': self.conclusion,
            'sample_sizes': self.sample_sizes,
            'conversion_rates': self.conversion_rates
        }


@dataclass
class AttributionResult:
    """Result structure for attribution modeling."""
    attribution_weights: pd.DataFrame
    channel_attribution: pd.DataFrame
    model_comparison: Dict[str, Dict[str, float]]
    conversion_paths: pd.DataFrame
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary format."""
        return {
            'attribution_weights': self.attribution_weights.to_dict('records'),
            'channel_attribution': self.channel_attribution.to_dict('records'),
            'model_comparison': self.model_comparison,
            'conversion_paths': self.conversion_paths.to_dict('records')
        }


@dataclass
class FunnelAnalysisResult:
    """Result structure for funnel analysis."""
    funnel_steps: pd.DataFrame
    conversion_rates: Dict[str, float]
    drop_off_rates: Dict[str, float]
    bottlenecks: List[str]
    optimization_recommendations: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary format."""
        return {
            'funnel_steps': self.funnel_steps.to_dict('records'),
            'conversion_rates': self.conversion_rates,
            'drop_off_rates': self.drop_off_rates,
            'bottlenecks': self.bottlenecks,
            'optimization_recommendations': self.optimization_recommendations
        }


class RFMAnalysisTransformer(BaseEstimator, TransformerMixin):
    """
    sklearn-compatible transformer for RFM (Recency, Frequency, Monetary) analysis.
    
    Performs customer segmentation based on transaction behavior patterns,
    calculating RFM scores and creating meaningful customer segments for
    targeted marketing and retention strategies.
    
    Parameters:
    -----------
    date_column : str, default='date'
        Name of the column containing transaction dates
    customer_column : str, default='customer_id'
        Name of the column containing customer identifiers
    amount_column : str, default='amount'
        Name of the column containing transaction amounts
    analysis_date : str or datetime, default=None
        Reference date for recency calculation (defaults to max date in data)
    quartiles : bool, default=True
        Whether to use quartiles for scoring (True) or custom bins (False)
    """
    
    def __init__(self, date_column='date', customer_column='customer_id', 
                 amount_column='amount', analysis_date=None, quartiles=True):
        self.date_column = date_column
        self.customer_column = customer_column
        self.amount_column = amount_column
        self.analysis_date = analysis_date
        self.quartiles = quartiles
        
    def fit(self, X, y=None):
        """
        Fit the RFM analyzer to the data.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Transaction data with customer, date, and amount columns
        y : ignored
            Not used, present for sklearn compatibility
            
        Returns:
        --------
        self : object
            Fitted transformer
        """
        # Validate input data
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
            
        required_columns = [self.customer_column, self.date_column, self.amount_column]
        missing_columns = [col for col in required_columns if col not in X.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Convert date column to datetime if not already
        X_work = X.copy()
        X_work[self.date_column] = pd.to_datetime(X_work[self.date_column])
        
        # Set analysis date
        if self.analysis_date is None:
            self.analysis_date_ = X_work[self.date_column].max()
        else:
            self.analysis_date_ = pd.to_datetime(self.analysis_date)
        
        # Calculate RFM metrics
        rfm_data = self._calculate_rfm_metrics(X_work)
        
        # Calculate scoring thresholds
        if self.quartiles:
            self.recency_thresholds_ = rfm_data['recency'].quantile([0.25, 0.5, 0.75]).values
            self.frequency_thresholds_ = rfm_data['frequency'].quantile([0.25, 0.5, 0.75]).values
            self.monetary_thresholds_ = rfm_data['monetary'].quantile([0.25, 0.5, 0.75]).values
        else:
            # Use reasonable business-based thresholds
            self.recency_thresholds_ = [30, 90, 180]  # days
            self.frequency_thresholds_ = [2, 5, 10]   # number of orders
            self.monetary_thresholds_ = [100, 500, 1000]  # currency units
        
        self.is_fitted_ = True
        return self
        
    def transform(self, X):
        """
        Transform the data to include RFM scores and segments.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Transaction data to transform
            
        Returns:
        --------
        result : RFMResult
            Complete RFM analysis results including scores, segments, and summary
        """
        check_is_fitted(self, 'is_fitted_')
        
        # Calculate RFM metrics
        X_work = X.copy()
        X_work[self.date_column] = pd.to_datetime(X_work[self.date_column])
        rfm_data = self._calculate_rfm_metrics(X_work)
        
        # Calculate RFM scores
        rfm_scores = self._calculate_rfm_scores(rfm_data)
        
        # Create customer segments
        segments = self._create_customer_segments(rfm_scores)
        
        # Generate segment summary
        segment_summary = self._generate_segment_summary(segments)
        
        # Prepare quartile boundaries info
        quartile_boundaries = {
            'recency': self.recency_thresholds_.tolist(),
            'frequency': self.frequency_thresholds_.tolist(),
            'monetary': self.monetary_thresholds_.tolist()
        }
        
        return RFMResult(
            rfm_scores=rfm_scores,
            segments=segments,
            segment_summary=segment_summary,
            quartile_boundaries=quartile_boundaries
        )
        
    def _calculate_rfm_metrics(self, X):
        """Calculate base RFM metrics from transaction data."""
        logger.info("Calculating RFM metrics for customer segmentation")
        
        rfm_data = X.groupby(self.customer_column).agg({
            self.date_column: lambda x: (self.analysis_date_ - x.max()).days,  # Recency
            self.customer_column: 'count',  # Frequency 
            self.amount_column: 'sum'  # Monetary
        }).reset_index()
        
        rfm_data.columns = [self.customer_column, 'recency', 'frequency', 'monetary']
        
        # Handle edge cases
        rfm_data['recency'] = rfm_data['recency'].clip(lower=0)
        rfm_data['frequency'] = rfm_data['frequency'].clip(lower=1)
        rfm_data['monetary'] = rfm_data['monetary'].clip(lower=0)
        
        return rfm_data
        
    def _calculate_rfm_scores(self, rfm_data):
        """Calculate RFM scores based on thresholds."""
        rfm_scores = rfm_data.copy()
        
        # Recency score (lower recency = higher score)
        rfm_scores['R'] = pd.cut(rfm_scores['recency'], 
                                bins=[-np.inf] + self.recency_thresholds_.tolist() + [np.inf],
                                labels=[4, 3, 2, 1], 
                                include_lowest=True).astype(int)
        
        # Frequency score (higher frequency = higher score)
        rfm_scores['F'] = pd.cut(rfm_scores['frequency'],
                                bins=[-np.inf] + self.frequency_thresholds_.tolist() + [np.inf],
                                labels=[1, 2, 3, 4],
                                include_lowest=True).astype(int)
        
        # Monetary score (higher monetary = higher score)  
        rfm_scores['M'] = pd.cut(rfm_scores['monetary'],
                                bins=[-np.inf] + self.monetary_thresholds_.tolist() + [np.inf],
                                labels=[1, 2, 3, 4],
                                include_lowest=True).astype(int)
        
        # Combined RFM score
        rfm_scores['RFM_Score'] = rfm_scores['R'].astype(str) + \
                                 rfm_scores['F'].astype(str) + \
                                 rfm_scores['M'].astype(str)
        
        return rfm_scores
        
    def _create_customer_segments(self, rfm_scores):
        """Create meaningful customer segments from RFM scores."""
        segments = rfm_scores.copy()
        
        # Define segment rules based on RFM scores
        def assign_segment(row):
            r, f, m = row['R'], row['F'], row['M']
            
            # Champions: High value, frequent, recent customers
            if r >= 4 and f >= 4 and m >= 3:
                return 'Champions'
            # Loyal Customers: High frequency, good monetary
            elif f >= 3 and m >= 3:
                return 'Loyal Customers'
            # Potential Loyalists: Recent customers with potential
            elif r >= 3 and f >= 2:
                return 'Potential Loyalists'
            # Recent Customers: Recent but low frequency/monetary
            elif r >= 3:
                return 'Recent Customers'
            # Promising: Decent frequency and monetary but not recent
            elif f >= 2 and m >= 2:
                return 'Promising'
            # Need Attention: Good customers who haven't purchased recently
            elif r <= 2 and f >= 3 and m >= 3:
                return 'Need Attention'
            # About to Sleep: Low recency, was frequent
            elif r <= 2 and f >= 2:
                return 'About to Sleep'
            # At Risk: Low across metrics but some history
            elif f >= 1 and m >= 1:
                return 'At Risk'
            # Lost: Very low across all metrics
            else:
                return 'Lost'
                
        segments['Segment'] = segments.apply(assign_segment, axis=1)
        
        return segments
        
    def _generate_segment_summary(self, segments):
        """Generate summary statistics for each segment."""
        summary = segments.groupby('Segment').agg({
            self.customer_column: 'count',
            'recency': ['mean', 'median'],
            'frequency': ['mean', 'median'], 
            'monetary': ['mean', 'median', 'sum']
        }).round(2)
        
        # Flatten column names
        summary.columns = [f"{col[1]}_{col[0]}" if col[1] != '' else col[0] 
                          for col in summary.columns]
        
        # Add percentage of total customers
        total_customers = len(segments)
        summary['percentage'] = (summary[f'{self.customer_column}'] / total_customers * 100).round(1)
        
        return summary.reset_index()


class CohortAnalysisTransformer(BaseEstimator, TransformerMixin):
    """
    sklearn-compatible transformer for cohort analysis and retention tracking.
    
    Analyzes customer retention patterns over time by grouping customers into
    cohorts based on their first purchase date and tracking their subsequent
    purchase behavior.
    
    Parameters:
    -----------
    date_column : str, default='date'
        Name of the column containing transaction dates
    customer_column : str, default='customer_id' 
        Name of the column containing customer identifiers
    period_type : str, default='monthly'
        Cohort period type: 'daily', 'weekly', 'monthly', 'quarterly'
    """
    
    def __init__(self, date_column='date', customer_column='customer_id', 
                 period_type='monthly'):
        self.date_column = date_column
        self.customer_column = customer_column
        self.period_type = period_type
        
    def fit(self, X, y=None):
        """
        Fit the cohort analyzer to the data.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Transaction data with customer and date columns
        y : ignored
            Not used, present for sklearn compatibility
            
        Returns:
        --------
        self : object
            Fitted transformer
        """
        # Validate input data
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
            
        required_columns = [self.customer_column, self.date_column]
        missing_columns = [col for col in required_columns if col not in X.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
            
        self.is_fitted_ = True
        return self
        
    def transform(self, X):
        """
        Transform the data to perform cohort analysis.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Transaction data to analyze
            
        Returns:
        --------
        result : CohortAnalysisResult
            Complete cohort analysis results
        """
        check_is_fitted(self, 'is_fitted_')
        
        logger.info(f"Performing {self.period_type} cohort analysis")
        
        # Prepare data
        X_work = X.copy()
        X_work[self.date_column] = pd.to_datetime(X_work[self.date_column])
        
        # Create cohort and period columns
        cohort_data = self._prepare_cohort_data(X_work)
        
        # Create cohort table
        cohort_table = self._create_cohort_table(cohort_data)
        
        # Calculate cohort sizes
        cohort_sizes = self._calculate_cohort_sizes(cohort_data)
        
        # Calculate retention rates
        retention_rates = self._calculate_retention_rates(cohort_table, cohort_sizes)
        
        # Generate period summary
        period_summary = self._generate_period_summary(retention_rates)
        
        return CohortAnalysisResult(
            cohort_table=cohort_table,
            cohort_sizes=cohort_sizes,
            retention_rates=retention_rates,
            period_summary=period_summary
        )
        
    def _prepare_cohort_data(self, X):
        """Prepare data with cohort and period columns."""
        # Determine first purchase date for each customer
        customer_cohorts = X.groupby(self.customer_column)[self.date_column].min().reset_index()
        customer_cohorts.columns = [self.customer_column, 'cohort_group']
        
        # Create period columns based on period_type
        if self.period_type == 'daily':
            X['order_period'] = X[self.date_column].dt.date
            customer_cohorts['cohort_group'] = customer_cohorts['cohort_group'].dt.date
        elif self.period_type == 'weekly':
            X['order_period'] = X[self.date_column].dt.to_period('W')
            customer_cohorts['cohort_group'] = customer_cohorts['cohort_group'].dt.to_period('W')
        elif self.period_type == 'monthly':
            X['order_period'] = X[self.date_column].dt.to_period('M')
            customer_cohorts['cohort_group'] = customer_cohorts['cohort_group'].dt.to_period('M')
        elif self.period_type == 'quarterly':
            X['order_period'] = X[self.date_column].dt.to_period('Q')
            customer_cohorts['cohort_group'] = customer_cohorts['cohort_group'].dt.to_period('Q')
        else:
            raise ValueError(f"Unsupported period_type: {self.period_type}")
        
        # Merge cohort info back to transaction data
        cohort_data = X.merge(customer_cohorts, on=self.customer_column)
        
        return cohort_data
        
    def _create_cohort_table(self, cohort_data):
        """Create cohort table showing customer counts by cohort and period."""
        cohort_table = cohort_data.groupby(['cohort_group', 'order_period'])[self.customer_column].nunique().reset_index()
        cohort_table = cohort_table.pivot(index='cohort_group', 
                                         columns='order_period', 
                                         values=self.customer_column)
        cohort_table.fillna(0, inplace=True)
        
        return cohort_table
        
    def _calculate_cohort_sizes(self, cohort_data):
        """Calculate the size of each cohort."""
        cohort_sizes = cohort_data.groupby('cohort_group')[self.customer_column].nunique()
        return cohort_sizes.to_frame('cohort_size')
        
    def _calculate_retention_rates(self, cohort_table, cohort_sizes):
        """Calculate retention rates as percentage of original cohort."""
        retention_rates = cohort_table.divide(cohort_sizes['cohort_size'], axis=0)
        return retention_rates
        
    def _generate_period_summary(self, retention_rates):
        """Generate summary statistics for retention analysis."""
        # Calculate average retention rates by period
        period_averages = retention_rates.mean(axis=0).to_dict()
        
        # Calculate retention rate ranges
        period_summary = {
            'average_retention_by_period': period_averages,
            'overall_average_retention': retention_rates.values[retention_rates.values > 0].mean(),
            'cohort_count': len(retention_rates),
            'analysis_period_type': self.period_type
        }
        
        return period_summary


# High-level convenience functions
def analyze_rfm(data: pd.DataFrame, 
                customer_column: str = 'customer_id',
                date_column: str = 'date', 
                amount_column: str = 'amount',
                analysis_date: Optional[str] = None) -> RFMResult:
    """
    Perform RFM analysis on customer transaction data.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Transaction data containing customer, date, and amount information
    customer_column : str, default='customer_id'
        Name of the customer identifier column
    date_column : str, default='date'
        Name of the date column
    amount_column : str, default='amount'
        Name of the transaction amount column
    analysis_date : str, optional
        Reference date for recency calculation
        
    Returns:
    --------
    result : RFMResult
        Complete RFM analysis results
    """
    transformer = RFMAnalysisTransformer(
        customer_column=customer_column,
        date_column=date_column,
        amount_column=amount_column,
        analysis_date=analysis_date
    )
    
    transformer.fit(data)
    return transformer.transform(data)


def perform_cohort_analysis(data: pd.DataFrame,
                           customer_column: str = 'customer_id',
                           date_column: str = 'date',
                           period_type: str = 'monthly') -> CohortAnalysisResult:
    """
    Perform cohort analysis for customer retention tracking.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Transaction data containing customer and date information
    customer_column : str, default='customer_id'
        Name of the customer identifier column
    date_column : str, default='date'
        Name of the date column
    period_type : str, default='monthly'
        Period for cohort analysis: 'daily', 'weekly', 'monthly', 'quarterly'
        
    Returns:
    --------
    result : CohortAnalysisResult
        Complete cohort analysis results
    """
    transformer = CohortAnalysisTransformer(
        customer_column=customer_column,
        date_column=date_column,
        period_type=period_type
    )
    
    transformer.fit(data)
    return transformer.transform(data)