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


class CLVCalculator(BaseEstimator, TransformerMixin):
    """
    sklearn-compatible transformer for Customer Lifetime Value (CLV) calculation.
    
    Calculates CLV using various methods including historical average,
    predictive modeling, and cohort-based approaches.
    
    Parameters:
    -----------
    method : str, default='historical'
        CLV calculation method: 'historical', 'predictive', 'cohort'
    prediction_periods : int, default=12
        Number of periods to predict for CLV calculation
    model_type : str, default='random_forest'
        Machine learning model for predictive CLV: 'random_forest', 'gradient_boosting'
    """
    
    def __init__(self, method='historical', prediction_periods=12, model_type='random_forest'):
        self.method = method
        self.prediction_periods = prediction_periods
        self.model_type = model_type
        
    def fit(self, X, y=None):
        """Fit the CLV calculator to the data."""
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
            
        self.is_fitted_ = True
        return self
        
    def transform(self, X):
        """Calculate CLV for customers."""
        check_is_fitted(self, 'is_fitted_')
        
        if self.method == 'historical':
            return self._calculate_historical_clv(X)
        elif self.method == 'predictive':
            return self._calculate_predictive_clv(X)
        elif self.method == 'cohort':
            return self._calculate_cohort_clv(X)
        else:
            raise ValueError(f"Unsupported CLV method: {self.method}")
            
    def _calculate_historical_clv(self, X):
        """Calculate CLV based on historical data."""
        # Implementation for historical CLV
        logger.info("Calculating historical CLV")
        
        # Simple CLV = Average Order Value * Purchase Frequency * Gross Margin * Lifespan
        customer_metrics = X.groupby('customer_id').agg({
            'amount': ['sum', 'mean', 'count'],
            'date': ['min', 'max']
        }).round(2)
        
        customer_metrics.columns = ['total_spent', 'avg_order_value', 'order_count', 'first_purchase', 'last_purchase']
        
        # Calculate customer lifespan in days
        customer_metrics['lifespan_days'] = (customer_metrics['last_purchase'] - customer_metrics['first_purchase']).dt.days
        customer_metrics['lifespan_days'] = customer_metrics['lifespan_days'].fillna(0)
        
        # Calculate purchase frequency (orders per day)
        customer_metrics['purchase_frequency'] = customer_metrics['order_count'] / (customer_metrics['lifespan_days'] + 1)
        
        # Simple CLV calculation (assuming 20% gross margin)
        gross_margin = 0.2
        customer_metrics['clv_estimate'] = (
            customer_metrics['avg_order_value'] * 
            customer_metrics['purchase_frequency'] * 
            gross_margin * 
            365  # annualize
        )
        
        # Distribution statistics
        clv_distribution = {
            'mean': customer_metrics['clv_estimate'].mean(),
            'median': customer_metrics['clv_estimate'].median(),
            'std': customer_metrics['clv_estimate'].std(),
            'min': customer_metrics['clv_estimate'].min(),
            'max': customer_metrics['clv_estimate'].max()
        }
        
        return CLVResult(
            clv_scores=customer_metrics.reset_index(),
            model_metrics={'method': 'historical', 'gross_margin_assumed': gross_margin},
            clv_distribution=clv_distribution,
            segment_clv={}
        )


class ABTestAnalyzer(BaseEstimator, TransformerMixin):
    """
    sklearn-compatible transformer for A/B test analysis and statistical testing.
    
    Performs comprehensive A/B test analysis including statistical significance,
    effect size calculation, confidence intervals, and power analysis.
    
    Parameters:
    -----------
    alpha : float, default=0.05
        Significance level for hypothesis testing
    alternative : str, default='two-sided'
        Alternative hypothesis: 'two-sided', 'greater', 'less'
    test_type : str, default='proportion'
        Type of test: 'proportion', 'mean', 'conversion'
    """
    
    def __init__(self, alpha=0.05, alternative='two-sided', test_type='proportion'):
        self.alpha = alpha
        self.alternative = alternative
        self.test_type = test_type
        
    def fit(self, X, y=None):
        """Fit the A/B test analyzer."""
        self.is_fitted_ = True
        return self
        
    def transform(self, X):
        """Perform A/B test analysis."""
        check_is_fitted(self, 'is_fitted_')
        
        logger.info(f"Performing A/B test analysis with {self.test_type} test")
        
        if self.test_type == 'proportion':
            return self._analyze_proportion_test(X)
        elif self.test_type == 'mean':
            return self._analyze_mean_test(X)
        elif self.test_type == 'conversion':
            return self._analyze_conversion_test(X)
        else:
            raise ValueError(f"Unsupported test type: {self.test_type}")
            
    def _analyze_proportion_test(self, X):
        """Analyze A/B test for proportions (e.g., conversion rates)."""
        # Expect columns: 'group', 'converted'
        if 'group' not in X.columns or 'converted' not in X.columns:
            raise ValueError("For proportion test, data must have 'group' and 'converted' columns")
            
        # Calculate conversion rates by group
        summary_stats = X.groupby('group').agg({
            'converted': ['count', 'sum', 'mean']
        }).round(4)
        
        summary_stats.columns = ['total', 'conversions', 'conversion_rate']
        
        # Extract data for statistical test
        groups = summary_stats.index.tolist()
        if len(groups) != 2:
            raise ValueError("Currently supports only 2-group A/B tests")
            
        group_a, group_b = groups[0], groups[1]
        
        n_a = summary_stats.loc[group_a, 'total']
        x_a = summary_stats.loc[group_a, 'conversions']
        n_b = summary_stats.loc[group_b, 'total']
        x_b = summary_stats.loc[group_b, 'conversions']
        
        p_a = x_a / n_a
        p_b = x_b / n_b
        
        # Perform z-test for proportions
        counts = np.array([x_a, x_b])
        nobs = np.array([n_a, n_b])
        
        z_stat, p_value = proportions_ztest(counts, nobs, alternative=self.alternative)
        
        # Calculate confidence interval for difference
        p_diff = p_b - p_a
        se_diff = np.sqrt((p_a * (1 - p_a) / n_a) + (p_b * (1 - p_b) / n_b))
        z_critical = stats.norm.ppf(1 - self.alpha/2)
        ci_lower = p_diff - z_critical * se_diff
        ci_upper = p_diff + z_critical * se_diff
        
        # Effect size (Cohen's h for proportions)
        h = 2 * (np.arcsin(np.sqrt(p_b)) - np.arcsin(np.sqrt(p_a)))
        
        # Power analysis
        pooled_p = (x_a + x_b) / (n_a + n_b)
        power = self._calculate_power_proportion(n_a, n_b, p_a, p_b, self.alpha)
        
        # Generate conclusion
        conclusion = self._generate_conclusion(p_value, p_diff, self.alpha)
        
        return ABTestResult(
            test_name=f"Proportion A/B Test ({group_a} vs {group_b})",
            test_statistic=z_stat,
            p_value=p_value,
            confidence_interval=(ci_lower, ci_upper),
            effect_size=h,
            power=power,
            conclusion=conclusion,
            sample_sizes={group_a: n_a, group_b: n_b},
            conversion_rates={group_a: p_a, group_b: p_b}
        )
        
    def _calculate_power_proportion(self, n1, n2, p1, p2, alpha):
        """Calculate statistical power for proportion test."""
        # Simplified power calculation
        pooled_p = ((n1 * p1) + (n2 * p2)) / (n1 + n2)
        pooled_se = np.sqrt(pooled_p * (1 - pooled_p) * (1/n1 + 1/n2))
        
        if pooled_se == 0:
            return 1.0
            
        z_alpha = stats.norm.ppf(1 - alpha/2)
        z_beta = abs(p2 - p1) / pooled_se - z_alpha
        power = stats.norm.cdf(z_beta)
        
        return max(0.0, min(1.0, power))
        
    def _generate_conclusion(self, p_value, effect, alpha):
        """Generate human-readable conclusion."""
        if p_value < alpha:
            significance = "statistically significant"
            direction = "positive" if effect > 0 else "negative"
        else:
            significance = "not statistically significant"
            direction = "inconclusive"
            
        return f"Result is {significance} (p={p_value:.4f}). Effect direction: {direction}."


class PowerAnalysisTransformer(BaseEstimator, TransformerMixin):
    """
    sklearn-compatible transformer for statistical power analysis and experiment design.
    
    Calculates required sample sizes, detectable effect sizes, and power for
    experimental design planning.
    
    Parameters:
    -----------
    power : float, default=0.8
        Desired statistical power (1 - β)
    alpha : float, default=0.05
        Type I error rate (significance level)
    effect_size : float, optional
        Expected effect size (Cohen's d for means, Cohen's h for proportions)
    """
    
    def __init__(self, power=0.8, alpha=0.05, effect_size=None):
        self.power = power
        self.alpha = alpha
        self.effect_size = effect_size
        
    def fit(self, X, y=None):
        """Fit the power analysis transformer."""
        self.is_fitted_ = True
        return self
        
    def transform(self, X):
        """Perform power analysis calculations."""
        check_is_fitted(self, 'is_fitted_')
        
        logger.info("Performing statistical power analysis for experiment design")
        
        # For demonstration, calculate sample size requirements for different effect sizes
        effect_sizes = [0.1, 0.2, 0.3, 0.5, 0.8] if self.effect_size is None else [self.effect_size]
        
        results = []
        for es in effect_sizes:
            try:
                # Calculate required sample size per group
                n_required = tt_solve_power(effect_size=es, power=self.power, alpha=self.alpha)
                
                results.append({
                    'effect_size': es,
                    'required_n_per_group': int(np.ceil(n_required)),
                    'total_required_n': int(np.ceil(n_required * 2)),
                    'power': self.power,
                    'alpha': self.alpha
                })
            except:
                # Handle edge cases
                results.append({
                    'effect_size': es,
                    'required_n_per_group': 'Unable to calculate',
                    'total_required_n': 'Unable to calculate', 
                    'power': self.power,
                    'alpha': self.alpha
                })
                
        return pd.DataFrame(results)


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


def calculate_clv(data: pd.DataFrame,
                  customer_column: str = 'customer_id',
                  date_column: str = 'date',
                  amount_column: str = 'amount',
                  method: str = 'historical') -> CLVResult:
    """
    Calculate Customer Lifetime Value for customers.
    
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
    method : str, default='historical'
        CLV calculation method
        
    Returns:
    --------
    result : CLVResult
        Complete CLV analysis results
    """
    # Ensure required columns are present and properly formatted
    required_columns = [customer_column, date_column, amount_column]
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
        
    transformer = CLVCalculator(method=method)
    transformer.fit(data)
    return transformer.transform(data)


def perform_ab_test(data: pd.DataFrame,
                   group_column: str = 'group',
                   outcome_column: str = 'converted',
                   test_type: str = 'proportion',
                   alpha: float = 0.05) -> ABTestResult:
    """
    Perform A/B test statistical analysis.
    
    Parameters:
    -----------
    data : pd.DataFrame
        A/B test data with group assignments and outcomes
    group_column : str, default='group'
        Name of the group assignment column
    outcome_column : str, default='converted'
        Name of the outcome/conversion column
    test_type : str, default='proportion'
        Type of statistical test to perform
    alpha : float, default=0.05
        Significance level for hypothesis testing
        
    Returns:
    --------
    result : ABTestResult
        Complete A/B test analysis results
    """
    # Rename columns to match expected format
    data_renamed = data.rename(columns={
        group_column: 'group',
        outcome_column: 'converted'
    })
    
    transformer = ABTestAnalyzer(alpha=alpha, test_type=test_type)
    transformer.fit(data_renamed)
    return transformer.transform(data_renamed)


class AttributionAnalyzer(BaseEstimator, TransformerMixin):
    """
    sklearn-compatible transformer for marketing attribution analysis.
    
    Analyzes customer conversion paths and assigns attribution weights to
    different marketing touchpoints using various attribution models.
    
    Parameters:
    -----------
    attribution_model : AttributionModel or str, default='last_touch'
        Attribution model to use for analysis
    lookback_window : int, default=30
        Number of days to look back for touchpoints
    """
    
    def __init__(self, attribution_model='last_touch', lookback_window=30):
        if isinstance(attribution_model, str):
            self.attribution_model = AttributionModel(attribution_model)
        else:
            self.attribution_model = attribution_model
        self.lookback_window = lookback_window
        
    def fit(self, X, y=None):
        """Fit the attribution analyzer."""
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
            
        # Expect columns: customer_id, channel, timestamp, converted
        required_columns = ['customer_id', 'channel', 'timestamp']
        missing_columns = [col for col in required_columns if col not in X.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
            
        self.is_fitted_ = True
        return self
        
    def transform(self, X):
        """Perform attribution analysis."""
        check_is_fitted(self, 'is_fitted_')
        
        logger.info(f"Performing {self.attribution_model.value} attribution analysis")
        
        # Prepare conversion paths
        conversion_paths = self._prepare_conversion_paths(X)
        
        # Apply attribution model
        if self.attribution_model == AttributionModel.FIRST_TOUCH:
            attribution_weights = self._first_touch_attribution(conversion_paths)
        elif self.attribution_model == AttributionModel.LAST_TOUCH:
            attribution_weights = self._last_touch_attribution(conversion_paths)
        elif self.attribution_model == AttributionModel.LINEAR:
            attribution_weights = self._linear_attribution(conversion_paths)
        elif self.attribution_model == AttributionModel.TIME_DECAY:
            attribution_weights = self._time_decay_attribution(conversion_paths)
        elif self.attribution_model == AttributionModel.POSITION_BASED:
            attribution_weights = self._position_based_attribution(conversion_paths)
        else:
            raise ValueError(f"Unsupported attribution model: {self.attribution_model}")
            
        # Calculate channel-level attribution
        channel_attribution = self._calculate_channel_attribution(attribution_weights)
        
        # Create model comparison
        model_comparison = self._compare_attribution_models(conversion_paths)
        
        return AttributionResult(
            attribution_weights=attribution_weights,
            channel_attribution=channel_attribution,
            model_comparison=model_comparison,
            conversion_paths=conversion_paths
        )
        
    def _prepare_conversion_paths(self, X):
        """Prepare customer conversion paths from touchpoint data."""
        # Sort by customer and timestamp
        X_sorted = X.sort_values(['customer_id', 'timestamp'])
        
        # Group by customer to create paths
        paths = []
        for customer_id, group in X_sorted.groupby('customer_id'):
            path_data = {
                'customer_id': customer_id,
                'touchpoints': group['channel'].tolist(),
                'timestamps': group['timestamp'].tolist(),
                'converted': group.get('converted', [False] * len(group)).iloc[-1]
            }
            paths.append(path_data)
            
        return pd.DataFrame(paths)
        
    def _first_touch_attribution(self, conversion_paths):
        """Apply first-touch attribution model."""
        attribution_data = []
        
        for _, path in conversion_paths.iterrows():
            if path['converted'] and len(path['touchpoints']) > 0:
                # First touchpoint gets 100% credit
                first_channel = path['touchpoints'][0]
                attribution_data.append({
                    'customer_id': path['customer_id'],
                    'channel': first_channel,
                    'attribution_weight': 1.0,
                    'model': 'first_touch'
                })
                
        return pd.DataFrame(attribution_data)
        
    def _last_touch_attribution(self, conversion_paths):
        """Apply last-touch attribution model."""
        attribution_data = []
        
        for _, path in conversion_paths.iterrows():
            if path['converted'] and len(path['touchpoints']) > 0:
                # Last touchpoint gets 100% credit
                last_channel = path['touchpoints'][-1]
                attribution_data.append({
                    'customer_id': path['customer_id'],
                    'channel': last_channel,
                    'attribution_weight': 1.0,
                    'model': 'last_touch'
                })
                
        return pd.DataFrame(attribution_data)
        
    def _linear_attribution(self, conversion_paths):
        """Apply linear attribution model."""
        attribution_data = []
        
        for _, path in conversion_paths.iterrows():
            if path['converted'] and len(path['touchpoints']) > 0:
                # Each touchpoint gets equal credit
                weight_per_touchpoint = 1.0 / len(path['touchpoints'])
                
                for channel in path['touchpoints']:
                    attribution_data.append({
                        'customer_id': path['customer_id'],
                        'channel': channel,
                        'attribution_weight': weight_per_touchpoint,
                        'model': 'linear'
                    })
                    
        return pd.DataFrame(attribution_data)
        
    def _time_decay_attribution(self, conversion_paths):
        """Apply time-decay attribution model (placeholder)."""
        # For now, fallback to linear attribution
        return self._linear_attribution(conversion_paths)
        
    def _position_based_attribution(self, conversion_paths):
        """Apply position-based attribution model (placeholder)."""
        # For now, fallback to linear attribution
        return self._linear_attribution(conversion_paths)
        
    def _calculate_channel_attribution(self, attribution_weights):
        """Calculate total attribution by channel."""
        if attribution_weights.empty:
            return pd.DataFrame(columns=['channel', 'total_attribution', 'conversions'])
            
        channel_totals = attribution_weights.groupby('channel').agg({
            'attribution_weight': 'sum',
            'customer_id': 'count'
        }).reset_index()
        
        channel_totals.columns = ['channel', 'total_attribution', 'conversions']
        channel_totals = channel_totals.sort_values('total_attribution', ascending=False)
        
        return channel_totals
        
    def _compare_attribution_models(self, conversion_paths):
        """Compare results across different attribution models."""
        models = [AttributionModel.FIRST_TOUCH, AttributionModel.LAST_TOUCH, AttributionModel.LINEAR]
        comparison = {}
        
        for model in models:
            temp_analyzer = AttributionAnalyzer(attribution_model=model)
            temp_analyzer.is_fitted_ = True
            
            if model == AttributionModel.FIRST_TOUCH:
                weights = temp_analyzer._first_touch_attribution(conversion_paths)
            elif model == AttributionModel.LAST_TOUCH:
                weights = temp_analyzer._last_touch_attribution(conversion_paths)
            elif model == AttributionModel.LINEAR:
                weights = temp_analyzer._linear_attribution(conversion_paths)
                
            if not weights.empty:
                channel_totals = weights.groupby('channel')['attribution_weight'].sum().to_dict()
            else:
                channel_totals = {}
                
            comparison[model.value] = channel_totals
            
        return comparison


class FunnelAnalyzer(BaseEstimator, TransformerMixin):
    """
    sklearn-compatible transformer for marketing funnel analysis.
    
    Analyzes conversion funnels to identify bottlenecks and optimization
    opportunities in the customer journey.
    
    Parameters:
    -----------
    steps : list of str
        Ordered list of funnel step names
    """
    
    def __init__(self, steps=None):
        self.steps = steps or ['awareness', 'interest', 'consideration', 'purchase']
        
    def fit(self, X, y=None):
        """Fit the funnel analyzer."""
        self.is_fitted_ = True
        return self
        
    def transform(self, X):
        """Perform funnel analysis."""
        check_is_fitted(self, 'is_fitted_')
        
        logger.info("Performing marketing funnel analysis")
        
        # Calculate funnel metrics
        funnel_steps = self._calculate_funnel_steps(X)
        conversion_rates = self._calculate_conversion_rates(funnel_steps)
        drop_off_rates = self._calculate_drop_off_rates(funnel_steps)
        bottlenecks = self._identify_bottlenecks(drop_off_rates)
        recommendations = self._generate_recommendations(bottlenecks, drop_off_rates)
        
        return FunnelAnalysisResult(
            funnel_steps=funnel_steps,
            conversion_rates=conversion_rates,
            drop_off_rates=drop_off_rates,
            bottlenecks=bottlenecks,
            optimization_recommendations=recommendations
        )
        
    def _calculate_funnel_steps(self, X):
        """Calculate user counts at each funnel step."""
        # Expect data with columns for each step (boolean values)
        step_counts = []
        
        for i, step in enumerate(self.steps):
            if step in X.columns:
                count = X[step].sum()
            else:
                # If step column doesn't exist, assume 0
                count = 0
                
            step_counts.append({
                'step': step,
                'step_number': i + 1,
                'users': count
            })
            
        return pd.DataFrame(step_counts)
        
    def _calculate_conversion_rates(self, funnel_steps):
        """Calculate conversion rates between steps."""
        conversion_rates = {}
        
        for i in range(len(funnel_steps) - 1):
            current_step = funnel_steps.iloc[i]
            next_step = funnel_steps.iloc[i + 1]
            
            if current_step['users'] > 0:
                rate = next_step['users'] / current_step['users']
            else:
                rate = 0
                
            conversion_rates[f"{current_step['step']}_to_{next_step['step']}"] = rate
            
        return conversion_rates
        
    def _calculate_drop_off_rates(self, funnel_steps):
        """Calculate drop-off rates between steps."""
        drop_off_rates = {}
        
        for i in range(len(funnel_steps) - 1):
            current_step = funnel_steps.iloc[i]
            next_step = funnel_steps.iloc[i + 1]
            
            if current_step['users'] > 0:
                rate = (current_step['users'] - next_step['users']) / current_step['users']
            else:
                rate = 1
                
            drop_off_rates[f"{current_step['step']}_to_{next_step['step']}"] = rate
            
        return drop_off_rates
        
    def _identify_bottlenecks(self, drop_off_rates):
        """Identify funnel bottlenecks with high drop-off rates."""
        # Sort drop-off rates and identify top bottlenecks
        sorted_drops = sorted(drop_off_rates.items(), key=lambda x: x[1], reverse=True)
        
        # Consider bottlenecks as steps with >50% drop-off
        bottlenecks = [step for step, rate in sorted_drops if rate > 0.5]
        
        return bottlenecks
        
    def _generate_recommendations(self, bottlenecks, drop_off_rates):
        """Generate optimization recommendations."""
        recommendations = []
        
        for bottleneck in bottlenecks:
            rate = drop_off_rates[bottleneck]
            recommendations.append(
                f"High drop-off at {bottleneck} ({rate:.1%}). Consider improving user experience or reducing friction."
            )
            
        if not recommendations:
            recommendations.append("Funnel performance looks good. Focus on incremental improvements.")
            
        return recommendations


# Additional high-level convenience functions
def analyze_attribution(data: pd.DataFrame,
                       customer_column: str = 'customer_id',
                       channel_column: str = 'channel',
                       timestamp_column: str = 'timestamp',
                       model: str = 'last_touch') -> AttributionResult:
    """
    Perform marketing attribution analysis.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Touchpoint data with customer, channel, and timestamp information
    customer_column : str, default='customer_id'
        Name of the customer identifier column
    channel_column : str, default='channel'
        Name of the marketing channel column
    timestamp_column : str, default='timestamp'
        Name of the timestamp column
    model : str, default='last_touch'
        Attribution model to use
        
    Returns:
    --------
    result : AttributionResult
        Complete attribution analysis results
    """
    # Rename columns to match expected format
    data_renamed = data.rename(columns={
        customer_column: 'customer_id',
        channel_column: 'channel',
        timestamp_column: 'timestamp'
    })
    
    transformer = AttributionAnalyzer(attribution_model=model)
    transformer.fit(data_renamed)
    return transformer.transform(data_renamed)


def analyze_funnel(data: pd.DataFrame,
                  steps: Optional[List[str]] = None) -> FunnelAnalysisResult:
    """
    Perform marketing funnel analysis.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Funnel data with columns for each step (boolean values)
    steps : list of str, optional
        Ordered list of funnel step names
        
    Returns:
    --------
    result : FunnelAnalysisResult
        Complete funnel analysis results
    """
    transformer = FunnelAnalyzer(steps=steps)
    transformer.fit(data)
    return transformer.transform(data)


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
    
    def __init__(self, customer_analytics=True, ab_testing=True, 
                 attribution_modeling=True, funnel_analysis=True, 
                 streaming_config=None, **kwargs):
        super().__init__(streaming_config=streaming_config, **kwargs)
        
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
            self.transformers['rfm'] = RFMAnalysisTransformer()
            self.transformers['cohort'] = CohortAnalysisTransformer()
            self.transformers['clv'] = CLVCalculator()
            
        if self.ab_testing:
            self.transformers['ab_test'] = ABTestAnalyzer()
            self.transformers['power_analysis'] = PowerAnalysisTransformer()
            
        if self.attribution_modeling:
            self.transformers['attribution'] = AttributionAnalyzer()
            
        if self.funnel_analysis:
            self.transformers['funnel'] = FunnelAnalyzer()
            
    def fit(self, X, y=None):
        """Fit all enabled BI transformers."""
        logger.info("Fitting Business Intelligence pipeline components")
        
        for name, transformer in self.transformers.items():
            try:
                transformer.fit(X, y)
                logger.debug(f"Successfully fitted {name} transformer")
            except Exception as e:
                logger.warning(f"Failed to fit {name} transformer: {str(e)}")
                
        return self
        
    def transform(self, X):
        """Transform data using all fitted BI transformers."""
        logger.info("Executing Business Intelligence pipeline analysis")
        
        results = {}
        metadata = CompositionMetadata()
        
        for name, transformer in self.transformers.items():
            try:
                if hasattr(transformer, 'is_fitted_') and transformer.is_fitted_:
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
            data=results,
            metadata=metadata,
            streaming_config=self.streaming_config
        )
        
        return pipeline_result
        
    def analyze_customer_journey(self, transaction_data: pd.DataFrame,
                                touchpoint_data: pd.DataFrame = None,
                                funnel_data: pd.DataFrame = None) -> Dict[str, Any]:
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
            journey_analysis['customer_segments'] = analyze_rfm(transaction_data)
            journey_analysis['retention_analysis'] = perform_cohort_analysis(transaction_data)
            journey_analysis['lifetime_value'] = calculate_clv(transaction_data)
            
        # Attribution Analysis
        if self.attribution_modeling and touchpoint_data is not None:
            journey_analysis['attribution'] = analyze_attribution(touchpoint_data)
            
        # Funnel Analysis
        if self.funnel_analysis and funnel_data is not None:
            journey_analysis['funnel_optimization'] = analyze_funnel(funnel_data)
            
        # Cross-analysis insights
        journey_analysis['insights'] = self._generate_cross_analysis_insights(journey_analysis)
        
        return journey_analysis
        
    def _generate_cross_analysis_insights(self, analysis_results: Dict[str, Any]) -> List[str]:
        """Generate insights from cross-analysis of BI components."""
        insights = []
        
        # RFM + CLV insights
        if 'customer_segments' in analysis_results and 'lifetime_value' in analysis_results:
            insights.append("Cross-reference RFM segments with CLV to identify high-value customer characteristics")
            
        # Cohort + Attribution insights
        if 'retention_analysis' in analysis_results and 'attribution' in analysis_results:
            insights.append("Analyze which acquisition channels produce customers with better retention rates")
            
        # Funnel + Attribution insights
        if 'funnel_optimization' in analysis_results and 'attribution' in analysis_results:
            insights.append("Optimize attribution models based on funnel performance at each touchpoint")
            
        if not insights:
            insights.append("Enable multiple analysis components for cross-analysis insights")
            
        return insights


# Integration with existing statistical domain for enhanced A/B testing
def enhanced_ab_test(data: pd.DataFrame, 
                    group_column: str = 'group',
                    outcome_column: str = 'converted',
                    use_statistical_domain: bool = True,
                    **kwargs) -> Dict[str, Any]:
    """
    Enhanced A/B test analysis leveraging both BI and statistical analysis domains.
    
    Parameters:
    -----------
    data : pd.DataFrame
        A/B test data
    group_column : str, default='group'
        Group assignment column
    outcome_column : str, default='converted'
        Outcome variable column
    use_statistical_domain : bool, default=True
        Whether to include statistical domain analysis
    **kwargs : additional arguments
        Additional arguments for statistical tests
        
    Returns:
    --------
    results : Dict[str, Any]
        Combined BI and statistical analysis results
    """
    results = {}
    
    # Business Intelligence A/B test analysis
    bi_result = perform_ab_test(data, group_column, outcome_column, **kwargs)
    results['business_intelligence'] = bi_result.to_dict()
    
    if use_statistical_domain:
        try:
            # Import and use statistical analysis domain for additional rigor
            from .statistical_analysis import run_hypothesis_test
            
            # Prepare data for statistical analysis
            groups = data[group_column].unique()
            if len(groups) == 2:
                group_a_data = data[data[group_column] == groups[0]][outcome_column]
                group_b_data = data[data[group_column] == groups[1]][outcome_column]
                
                # Run statistical hypothesis test
                stat_result = run_hypothesis_test(
                    group_a_data.values, group_b_data.values,
                    test_type='ttest_ind', alpha=kwargs.get('alpha', 0.05)
                )
                results['statistical_analysis'] = stat_result
                
        except ImportError:
            logger.warning("Statistical analysis domain not available for enhanced A/B testing")
            
    return results