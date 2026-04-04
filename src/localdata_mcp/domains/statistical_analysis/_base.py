"""
Statistical Analysis Domain - Base types and shared imports.

Contains the StatisticalTestResult dataclass and common imports used
across all statistical analysis sub-modules.
"""

import warnings
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import time
import json

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import chi2_contingency, pearsonr, spearmanr, kendalltau
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted
import statsmodels.api as sm
from statsmodels.stats.power import ttest_power, tt_solve_power
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.anova import anova_lm
from statsmodels.formula.api import ols

from ...logging_manager import get_logger

logger = get_logger(__name__)

# Suppress specific warnings that are not critical for our use case
warnings.filterwarnings("ignore", category=RuntimeWarning, module="scipy.stats")
warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")


@dataclass
class StatisticalTestResult:
    """Standardized result structure for statistical tests."""

    test_name: str
    statistic: float
    p_value: float
    degrees_of_freedom: Optional[Union[int, Tuple[int, int]]] = None
    effect_size: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    interpretation: str = ""
    assumptions_met: Dict[str, bool] = None
    additional_info: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary format."""
        result_dict = {
            "test_name": self.test_name,
            "statistic": self.statistic,
            "p_value": self.p_value,
            "interpretation": self.interpretation,
        }

        if self.degrees_of_freedom is not None:
            result_dict["degrees_of_freedom"] = self.degrees_of_freedom
        if self.effect_size is not None:
            result_dict["effect_size"] = self.effect_size
        if self.confidence_interval is not None:
            result_dict["confidence_interval"] = self.confidence_interval
        if self.assumptions_met is not None:
            result_dict["assumptions_met"] = self.assumptions_met
        if self.additional_info is not None:
            result_dict["additional_info"] = self.additional_info

        return result_dict
