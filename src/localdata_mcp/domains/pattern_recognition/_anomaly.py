"""
Pattern Recognition - Anomaly Detection Transformer.

Advanced anomaly detection with multiple algorithms (Isolation Forest, One-Class SVM,
Local Outlier Factor, statistical methods).
"""

import warnings
from typing import Any, Dict, Optional
import time

import numpy as np
from sklearn.utils.validation import check_array, check_is_fitted
from sklearn.preprocessing import StandardScaler

# Anomaly detection
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from scipy import stats

from ...logging_manager import get_logger
from ...pipeline.base import AnalysisPipelineBase
from ._results import AnomalyDetectionResult

logger = get_logger(__name__)

# Suppress specific warnings that are not critical for our use case
warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


class AnomalyDetectionTransformer(AnalysisPipelineBase):
    """Advanced anomaly detection transformer with multiple algorithms."""

    def __init__(
        self,
        algorithm: str = "isolation_forest",
        contamination: Optional[float] = None,
        standardize: bool = True,
        random_state: int = 42,
        **algorithm_params,
    ):
        """
        Initialize anomaly detection transformer.

        Parameters:
        -----------
        algorithm : str
            Detection algorithm ('isolation_forest', 'one_class_svm', 'lof', 'statistical')
        contamination : float, optional
            Expected proportion of outliers (auto-estimated if None)
        standardize : bool
            Whether to standardize features before detection
        random_state : int
            Random state for reproducibility
        **algorithm_params
            Additional parameters for specific algorithms
        """
        super().__init__(analytical_intention=f"{algorithm} anomaly detection")
        self.algorithm = algorithm
        self.contamination = contamination
        self.standardize = standardize
        self.random_state = random_state
        self.algorithm_params = algorithm_params

        # Initialize components
        self.scaler_ = None
        self.detector_ = None
        self.threshold_ = None

    def get_analysis_type(self) -> str:
        """Get the specific analysis type this pipeline performs."""
        return f"{self.algorithm}_anomaly_detection"

    def _configure_analysis_pipeline(self):
        """Configure analysis steps based on intention and complexity level."""
        return [self.fit]

    def _execute_analysis_step(self, step, data, context):
        """Execute individual analysis step with error handling and metadata."""
        result = step(data, context) if callable(step) else step
        return result, {}

    def _execute_streaming_analysis(self, data):
        """Execute analysis with streaming support for large datasets."""
        return self._execute_standard_analysis(data)

    def _execute_standard_analysis(self, data):
        """Execute analysis on full dataset in memory."""
        return self.detector_ if hasattr(self, "detector_") else None, {}

    def fit(self, X, y=None):
        """Fit the anomaly detection model to the data."""
        logger.info(
            f"Fitting anomaly detection transformer with algorithm: {self.algorithm}"
        )
        start_time = time.time()

        # Validate input
        X = check_array(X, accept_sparse=False)

        # Standardize if requested
        if self.standardize and self.algorithm != "statistical":
            self.scaler_ = StandardScaler()
            X_scaled = self.scaler_.fit_transform(X)
        else:
            X_scaled = X

        # Estimate contamination if not provided
        if self.contamination is None:
            if self.algorithm == "statistical":
                self.contamination_ = 0.05  # 5% for statistical methods
            else:
                self.contamination_ = min(0.1, max(0.01, 100 / X.shape[0]))  # Adaptive
        else:
            self.contamination_ = self.contamination

        # Create and fit detector
        self.detector_ = self._create_detector()

        try:
            if self.algorithm == "statistical":
                # Statistical methods don't need fitting
                pass
            else:
                self.detector_.fit(X_scaled)
        except Exception as e:
            logger.error(f"Anomaly detection fitting failed: {e}")
            raise

        self.fit_time_ = time.time() - start_time
        logger.info(f"Anomaly detection completed in {self.fit_time_:.3f} seconds")

        return self

    def transform(self, X):
        """Transform data by returning anomaly labels."""
        return self.predict(X)

    def predict(self, X):
        """Predict anomaly labels for data."""
        check_is_fitted(self, "detector_")

        start_time = time.time()

        # Scale if necessary
        if self.scaler_ is not None:
            X_scaled = self.scaler_.transform(X)
        else:
            X_scaled = X

        # Predict anomalies
        try:
            if self.algorithm == "statistical":
                labels = self._statistical_detection(X_scaled)
            else:
                labels = self.detector_.predict(X_scaled)
        except Exception as e:
            logger.error(f"Anomaly detection prediction failed: {e}")
            raise

        self.prediction_time_ = time.time() - start_time
        return labels

    def decision_function(self, X):
        """Compute anomaly scores for data."""
        check_is_fitted(self, "detector_")

        # Scale if necessary
        if self.scaler_ is not None:
            X_scaled = self.scaler_.transform(X)
        else:
            X_scaled = X

        # Get anomaly scores
        if self.algorithm == "statistical":
            return self._statistical_scores(X_scaled)
        elif hasattr(self.detector_, "decision_function"):
            return self.detector_.decision_function(X_scaled)
        elif hasattr(self.detector_, "score_samples"):
            return self.detector_.score_samples(X_scaled)
        else:
            return np.zeros(X.shape[0])  # Fallback

    def _create_detector(self):
        """Create detector based on algorithm choice."""
        base_params = (
            {"random_state": self.random_state}
            if "random_state" in self._get_valid_params()
            else {}
        )
        params = {**base_params, **self.algorithm_params}

        if self.algorithm == "isolation_forest":
            params["contamination"] = self.contamination_
            return IsolationForest(**params)
        elif self.algorithm == "one_class_svm":
            # One-Class SVM doesn't use contamination directly
            return OneClassSVM(**params)
        elif self.algorithm == "lof":
            params["contamination"] = self.contamination_
            params["novelty"] = True
            return LocalOutlierFactor(**params)
        elif self.algorithm == "statistical":
            # Statistical methods don't need a detector object
            return None
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")

    def _get_valid_params(self):
        """Get valid parameters for the chosen algorithm."""
        param_sets = {
            "isolation_forest": [
                "contamination",
                "n_estimators",
                "max_samples",
                "random_state",
            ],
            "one_class_svm": ["kernel", "gamma", "nu"],
            "lof": ["contamination", "n_neighbors", "algorithm", "metric"],
            "statistical": ["method", "threshold_std"],
        }
        return param_sets.get(self.algorithm, [])

    def _statistical_detection(self, X):
        """Perform statistical anomaly detection."""
        method = self.algorithm_params.get("method", "zscore")
        threshold_std = self.algorithm_params.get("threshold_std", 3.0)

        if method == "zscore":
            # Z-score method
            z_scores = np.abs(stats.zscore(X, axis=0))
            max_z_scores = np.max(z_scores, axis=1)
            labels = np.where(max_z_scores > threshold_std, -1, 1)
        elif method == "iqr":
            # Interquartile range method
            Q1 = np.percentile(X, 25, axis=0)
            Q3 = np.percentile(X, 75, axis=0)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outliers = np.any((X < lower_bound) | (X > upper_bound), axis=1)
            labels = np.where(outliers, -1, 1)
        else:
            raise ValueError(f"Unknown statistical method: {method}")

        return labels

    def _statistical_scores(self, X):
        """Compute statistical anomaly scores."""
        method = self.algorithm_params.get("method", "zscore")

        if method == "zscore":
            z_scores = np.abs(stats.zscore(X, axis=0))
            return np.max(z_scores, axis=1)
        elif method == "iqr":
            Q1 = np.percentile(X, 25, axis=0)
            Q3 = np.percentile(X, 75, axis=0)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            lower_violations = np.maximum(0, lower_bound - X)
            upper_violations = np.maximum(0, X - upper_bound)
            return np.max(lower_violations + upper_violations, axis=1)
        else:
            return np.zeros(X.shape[0])

    def get_anomaly_result(self, X, y_true=None) -> AnomalyDetectionResult:
        """Generate comprehensive anomaly detection results."""
        check_is_fitted(self, "detector_")

        # Get predictions and scores
        start_time = time.time()
        labels = self.predict(X)
        scores = self.decision_function(X)
        prediction_time = time.time() - start_time

        n_samples = len(labels)
        n_anomalies = np.sum(labels == -1)

        # Quality metrics (if ground truth available)
        precision = None
        recall = None
        f1_score = None
        auc_roc = None

        if y_true is not None:
            from sklearn.metrics import (
                precision_score,
                recall_score,
                f1_score as f1,
                roc_auc_score,
            )

            try:
                # Convert to binary format (1 for normal, 0 for anomaly)
                y_pred_binary = (labels == 1).astype(int)
                y_true_binary = (y_true == 1).astype(int)

                precision = precision_score(y_true_binary, y_pred_binary)
                recall = recall_score(y_true_binary, y_pred_binary)
                f1_score = f1(y_true_binary, y_pred_binary)

                if len(np.unique(y_true_binary)) > 1:
                    auc_roc = roc_auc_score(y_true_binary, scores)
            except Exception as e:
                logger.warning(f"Failed to calculate quality metrics: {e}")

        # Score statistics
        score_statistics = {
            "mean_score": float(np.mean(scores)),
            "std_score": float(np.std(scores)),
            "min_score": float(np.min(scores)),
            "max_score": float(np.max(scores)),
            "median_score": float(np.median(scores)),
        }

        # Anomaly statistics
        if n_anomalies > 0:
            anomaly_indices = np.where(labels == -1)[0]
            anomaly_scores_subset = scores[anomaly_indices]
            normal_scores_subset = scores[labels == 1]

            anomaly_statistics = {
                "anomaly_indices": anomaly_indices.tolist(),
                "mean_anomaly_score": float(np.mean(anomaly_scores_subset)),
                "mean_normal_score": float(np.mean(normal_scores_subset))
                if len(normal_scores_subset) > 0
                else None,
                "score_separation": float(
                    np.mean(anomaly_scores_subset) - np.mean(normal_scores_subset)
                )
                if len(normal_scores_subset) > 0
                else None,
            }
        else:
            anomaly_statistics = {
                "anomaly_indices": [],
                "mean_anomaly_score": None,
                "mean_normal_score": float(np.mean(scores)),
                "score_separation": None,
            }

        return AnomalyDetectionResult(
            algorithm=self.algorithm,
            n_samples=n_samples,
            n_anomalies=n_anomalies,
            anomaly_labels=labels,
            anomaly_scores=scores,
            threshold=self.threshold_,
            contamination=self.contamination_,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            auc_roc=auc_roc,
            score_statistics=score_statistics,
            anomaly_statistics=anomaly_statistics,
            algorithm_params=self.algorithm_params,
            fit_time=self.fit_time_,
            prediction_time=prediction_time,
        )
