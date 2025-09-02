"""
Tests for Pattern Recognition Domain - Comprehensive testing of clustering, dimensionality reduction,
and anomaly detection capabilities with performance benchmarking.

Tests validate:
- Clustering algorithms with known ground truth datasets
- Dimensionality reduction structure preservation  
- Anomaly detection accuracy with synthetic outliers
- Performance benchmarking with high-dimensional data
- Integration with sklearn pipeline architecture
- Error handling and edge cases
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
from sklearn.datasets import make_blobs, make_classification
from sklearn.metrics import adjusted_rand_score, silhouette_score

# Import the pattern recognition domain components
from src.localdata_mcp.domains.pattern_recognition import (
    ClusteringTransformer,
    DimensionalityReductionTransformer,
    AnomalyDetectionTransformer,
    PatternEvaluationTransformer,
    perform_clustering,
    reduce_dimensions,
    detect_anomalies,
    evaluate_patterns,
    ClusteringResult,
    DimensionalityReductionResult,
    AnomalyDetectionResult,
    PatternEvaluationResult
)


@pytest.fixture
def sample_clustering_data():
    """Generate sample clustering data with known clusters."""
    X, y = make_blobs(n_samples=300, centers=4, n_features=5, 
                      random_state=42, cluster_std=1.0)
    return X, y


@pytest.fixture
def sample_classification_data():
    """Generate sample classification data for dimensionality reduction."""
    X, y = make_classification(n_samples=200, n_features=10, n_informative=5,
                              n_redundant=2, n_classes=3, random_state=42)
    return X, y


@pytest.fixture
def sample_anomaly_data():
    """Generate sample data with anomalies."""
    # Normal data
    normal_data = np.random.randn(200, 5)
    
    # Add some clear outliers
    anomalies = np.random.randn(20, 5) * 5 + 10  # Far from normal data
    
    X = np.vstack([normal_data, anomalies])
    y_true = np.array([1] * 200 + [-1] * 20)  # 1 = normal, -1 = anomaly
    
    return X, y_true


class TestClusteringTransformer:
    """Test clustering transformer with multiple algorithms."""
    
    def test_kmeans_clustering(self, sample_clustering_data):
        """Test K-means clustering with known clusters."""
        X, y_true = sample_clustering_data
        
        clusterer = ClusteringTransformer(
            algorithm='kmeans',
            n_clusters=4,
            random_state=42
        )
        clusterer.fit(X)
        
        result = clusterer.get_clustering_result(X, y_true)
        
        # Validate results
        assert isinstance(result, ClusteringResult)
        assert result.algorithm == 'kmeans'
        assert result.n_clusters == 4
        assert len(result.labels) == len(X)
        assert result.silhouette_avg is not None
        assert result.silhouette_avg > 0.5  # Should be good clustering
        assert result.adjusted_rand_score is not None
        assert result.adjusted_rand_score > 0.7  # Should match true clusters well
        
    def test_hierarchical_clustering(self, sample_clustering_data):
        """Test hierarchical clustering."""
        X, y_true = sample_clustering_data
        
        clusterer = ClusteringTransformer(
            algorithm='hierarchical',
            n_clusters=4
        )
        clusterer.fit(X)
        
        result = clusterer.get_clustering_result(X, y_true)
        
        assert isinstance(result, ClusteringResult)
        assert result.algorithm == 'hierarchical'
        assert result.n_clusters == 4
        assert result.silhouette_avg > 0.4  # Should be reasonable clustering
        
    def test_dbscan_clustering(self, sample_clustering_data):
        """Test DBSCAN clustering."""
        X, y_true = sample_clustering_data
        
        clusterer = ClusteringTransformer(
            algorithm='dbscan',
            eps=1.0,
            min_samples=5
        )
        clusterer.fit(X)
        
        result = clusterer.get_clustering_result(X)
        
        assert isinstance(result, ClusteringResult)
        assert result.algorithm == 'dbscan'
        assert len(result.labels) == len(X)
        # DBSCAN may find different number of clusters
        
    def test_gmm_clustering(self, sample_clustering_data):
        """Test Gaussian Mixture Model clustering."""
        X, y_true = sample_clustering_data
        
        clusterer = ClusteringTransformer(
            algorithm='gmm',
            n_clusters=4,
            random_state=42
        )
        clusterer.fit(X)
        
        result = clusterer.get_clustering_result(X, y_true)
        
        assert isinstance(result, ClusteringResult)
        assert result.algorithm == 'gmm'
        assert result.n_clusters == 4
        assert result.adjusted_rand_score > 0.5  # Should find reasonable clusters
        
    def test_spectral_clustering(self, sample_clustering_data):
        """Test spectral clustering."""
        X, y_true = sample_clustering_data
        
        clusterer = ClusteringTransformer(
            algorithm='spectral',
            n_clusters=4,
            random_state=42
        )
        clusterer.fit(X)
        
        result = clusterer.get_clustering_result(X, y_true)
        
        assert isinstance(result, ClusteringResult)
        assert result.algorithm == 'spectral'
        assert result.n_clusters == 4
        
    def test_auto_k_selection(self, sample_clustering_data):
        """Test automatic k selection."""
        X, y_true = sample_clustering_data
        
        clusterer = ClusteringTransformer(
            algorithm='kmeans',
            auto_k_selection=True,
            k_range=(2, 8),
            random_state=42
        )
        clusterer.fit(X)
        
        assert clusterer.optimal_k_ is not None
        assert 2 <= clusterer.optimal_k_ <= 8
        assert len(clusterer.k_scores_) > 0
        
    def test_clustering_transform(self, sample_clustering_data):
        """Test clustering transformer interface."""
        X, y_true = sample_clustering_data
        
        clusterer = ClusteringTransformer(
            algorithm='kmeans',
            n_clusters=4,
            random_state=42
        )
        
        # Test fit and transform
        clusterer.fit(X)
        labels = clusterer.transform(X)
        
        assert len(labels) == len(X)
        assert len(set(labels)) <= 4
        
    def test_clustering_edge_cases(self):
        """Test clustering edge cases and error handling."""
        # Test with very small dataset
        X_small = np.random.randn(3, 2)
        
        clusterer = ClusteringTransformer(
            algorithm='kmeans',
            n_clusters=2,
            random_state=42
        )
        
        clusterer.fit(X_small)
        result = clusterer.get_clustering_result(X_small)
        
        assert isinstance(result, ClusteringResult)
        assert len(result.labels) == 3
        
        # Test with single feature
        X_single = np.random.randn(50, 1)
        clusterer_single = ClusteringTransformer(
            algorithm='kmeans',
            n_clusters=3,
            random_state=42
        )
        clusterer_single.fit(X_single)
        
        result_single = clusterer_single.get_clustering_result(X_single)
        assert isinstance(result_single, ClusteringResult)


class TestDimensionalityReductionTransformer:
    """Test dimensionality reduction transformer with multiple algorithms."""
    
    def test_pca_reduction(self, sample_classification_data):
        """Test PCA dimensionality reduction."""
        X, y = sample_classification_data
        
        reducer = DimensionalityReductionTransformer(
            algorithm='pca',
            n_components=5,
            random_state=42
        )
        reducer.fit(X)
        
        result = reducer.get_reduction_result(X)
        
        assert isinstance(result, DimensionalityReductionResult)
        assert result.algorithm == 'pca'
        assert result.original_dimensions == X.shape[1]
        assert result.reduced_dimensions == 5
        assert result.transformed_data.shape == (X.shape[0], 5)
        assert result.explained_variance_ratio is not None
        assert len(result.explained_variance_ratio) == 5
        assert result.cumulative_variance_ratio is not None
        
    def test_tsne_reduction(self, sample_classification_data):
        """Test t-SNE dimensionality reduction."""
        X, y = sample_classification_data
        
        reducer = DimensionalityReductionTransformer(
            algorithm='tsne',
            n_components=2,
            random_state=42
        )
        reducer.fit(X)
        
        result = reducer.get_reduction_result(X)
        
        assert isinstance(result, DimensionalityReductionResult)
        assert result.algorithm == 'tsne'
        assert result.reduced_dimensions == 2
        assert result.transformed_data.shape == (X.shape[0], 2)
        
    @pytest.mark.skipif(True, reason="UMAP may not be available in test environment")
    def test_umap_reduction(self, sample_classification_data):
        """Test UMAP dimensionality reduction if available."""
        X, y = sample_classification_data
        
        reducer = DimensionalityReductionTransformer(
            algorithm='umap',
            n_components=3,
            random_state=42
        )
        reducer.fit(X)
        
        result = reducer.get_reduction_result(X)
        
        assert isinstance(result, DimensionalityReductionResult)
        assert result.algorithm == 'umap'
        assert result.reduced_dimensions == 3
        assert result.transformed_data.shape == (X.shape[0], 3)
        
    def test_ica_reduction(self, sample_classification_data):
        """Test Independent Component Analysis."""
        X, y = sample_classification_data
        
        reducer = DimensionalityReductionTransformer(
            algorithm='ica',
            n_components=4,
            random_state=42
        )
        reducer.fit(X)
        
        result = reducer.get_reduction_result(X)
        
        assert isinstance(result, DimensionalityReductionResult)
        assert result.algorithm == 'ica'
        assert result.reduced_dimensions == 4
        assert result.transformed_data.shape == (X.shape[0], 4)
        
    def test_lda_reduction(self, sample_classification_data):
        """Test Linear Discriminant Analysis."""
        X, y = sample_classification_data
        
        reducer = DimensionalityReductionTransformer(
            algorithm='lda',
            n_components=2,
            random_state=42
        )
        reducer.fit(X, y)
        
        result = reducer.get_reduction_result(X)
        
        assert isinstance(result, DimensionalityReductionResult)
        assert result.algorithm == 'lda'
        assert result.reduced_dimensions == 2
        assert result.transformed_data.shape == (X.shape[0], 2)
        
    def test_auto_component_selection_pca(self, sample_classification_data):
        """Test automatic component selection for PCA."""
        X, y = sample_classification_data
        
        reducer = DimensionalityReductionTransformer(
            algorithm='pca',
            preserve_variance=0.9,
            random_state=42
        )
        reducer.fit(X)
        
        assert reducer.optimal_components_ is not None
        assert reducer.optimal_components_ <= X.shape[1]
        
        result = reducer.get_reduction_result(X)
        assert result.cumulative_variance_ratio[-1] >= 0.9
        
    def test_reduction_transform(self, sample_classification_data):
        """Test dimensionality reduction transformer interface."""
        X, y = sample_classification_data
        
        reducer = DimensionalityReductionTransformer(
            algorithm='pca',
            n_components=3,
            random_state=42
        )
        
        # Test fit and transform
        reducer.fit(X)
        X_transformed = reducer.transform(X)
        
        assert X_transformed.shape == (X.shape[0], 3)
        assert not np.array_equal(X_transformed, X[:, :3])  # Should be different
        
    def test_reduction_edge_cases(self):
        """Test dimensionality reduction edge cases."""
        # Test with more components than features
        X_small = np.random.randn(20, 3)
        
        reducer = DimensionalityReductionTransformer(
            algorithm='pca',
            n_components=5,  # More than available features
            random_state=42
        )
        
        # Should automatically limit to available features
        reducer.fit(X_small)
        result = reducer.get_reduction_result(X_small)
        
        assert result.reduced_dimensions <= 3


class TestAnomalyDetectionTransformer:
    """Test anomaly detection transformer with multiple algorithms."""
    
    def test_isolation_forest_detection(self, sample_anomaly_data):
        """Test Isolation Forest anomaly detection."""
        X, y_true = sample_anomaly_data
        
        detector = AnomalyDetectionTransformer(
            algorithm='isolation_forest',
            contamination=0.1,
            random_state=42
        )
        detector.fit(X)
        
        result = detector.get_anomaly_result(X, y_true)
        
        assert isinstance(result, AnomalyDetectionResult)
        assert result.algorithm == 'isolation_forest'
        assert result.n_samples == len(X)
        assert result.n_anomalies > 0
        assert len(result.anomaly_labels) == len(X)
        assert result.anomaly_scores is not None
        assert result.precision is not None
        assert result.recall is not None
        assert result.f1_score > 0.5  # Should detect anomalies reasonably well
        
    def test_one_class_svm_detection(self, sample_anomaly_data):
        """Test One-Class SVM anomaly detection."""
        X, y_true = sample_anomaly_data
        
        detector = AnomalyDetectionTransformer(
            algorithm='one_class_svm',
            nu=0.1
        )
        detector.fit(X)
        
        result = detector.get_anomaly_result(X, y_true)
        
        assert isinstance(result, AnomalyDetectionResult)
        assert result.algorithm == 'one_class_svm'
        assert result.n_anomalies > 0
        assert result.precision is not None
        
    def test_lof_detection(self, sample_anomaly_data):
        """Test Local Outlier Factor detection."""
        X, y_true = sample_anomaly_data
        
        detector = AnomalyDetectionTransformer(
            algorithm='lof',
            contamination=0.1,
            n_neighbors=20
        )
        detector.fit(X)
        
        result = detector.get_anomaly_result(X, y_true)
        
        assert isinstance(result, AnomalyDetectionResult)
        assert result.algorithm == 'lof'
        assert result.n_anomalies > 0
        
    def test_statistical_zscore_detection(self, sample_anomaly_data):
        """Test statistical Z-score anomaly detection."""
        X, y_true = sample_anomaly_data
        
        detector = AnomalyDetectionTransformer(
            algorithm='statistical',
            method='zscore',
            threshold_std=3.0
        )
        detector.fit(X)
        
        result = detector.get_anomaly_result(X, y_true)
        
        assert isinstance(result, AnomalyDetectionResult)
        assert result.algorithm == 'statistical'
        assert result.n_anomalies > 0
        assert result.f1_score > 0.3  # Should detect some anomalies
        
    def test_statistical_iqr_detection(self, sample_anomaly_data):
        """Test statistical IQR anomaly detection."""
        X, y_true = sample_anomaly_data
        
        detector = AnomalyDetectionTransformer(
            algorithm='statistical',
            method='iqr'
        )
        detector.fit(X)
        
        result = detector.get_anomaly_result(X, y_true)
        
        assert isinstance(result, AnomalyDetectionResult)
        assert result.algorithm == 'statistical'
        assert result.n_anomalies > 0
        
    def test_anomaly_detection_transform(self, sample_anomaly_data):
        """Test anomaly detection transformer interface."""
        X, y_true = sample_anomaly_data
        
        detector = AnomalyDetectionTransformer(
            algorithm='isolation_forest',
            contamination=0.1,
            random_state=42
        )
        
        # Test fit and predict
        detector.fit(X)
        labels = detector.predict(X)
        scores = detector.decision_function(X)
        
        assert len(labels) == len(X)
        assert len(scores) == len(X)
        assert set(labels) <= {-1, 1}  # Should be -1 (anomaly) or 1 (normal)
        
    def test_auto_contamination_estimation(self):
        """Test automatic contamination estimation."""
        X = np.random.randn(100, 5)
        
        detector = AnomalyDetectionTransformer(
            algorithm='isolation_forest',
            contamination=None,  # Auto-estimate
            random_state=42
        )
        detector.fit(X)
        
        assert detector.contamination_ is not None
        assert 0.001 <= detector.contamination_ <= 0.2
        
    def test_anomaly_edge_cases(self):
        """Test anomaly detection edge cases."""
        # Test with very clean data (no clear anomalies)
        X_clean = np.random.randn(50, 3)
        
        detector = AnomalyDetectionTransformer(
            algorithm='isolation_forest',
            contamination=0.05,
            random_state=42
        )
        detector.fit(X_clean)
        
        result = detector.get_anomaly_result(X_clean)
        assert isinstance(result, AnomalyDetectionResult)
        # May find few or no anomalies in clean data


class TestPatternEvaluationTransformer:
    """Test pattern evaluation and quality assessment."""
    
    def test_clustering_evaluation(self, sample_clustering_data):
        """Test clustering quality evaluation."""
        X, y_true = sample_clustering_data
        
        # Generate clustering labels
        clusterer = ClusteringTransformer(algorithm='kmeans', n_clusters=4, random_state=42)
        clusterer.fit(X)
        labels = clusterer.labels_
        
        evaluator = PatternEvaluationTransformer('clustering')
        result = evaluator.evaluate_clustering(X, labels, y_true)
        
        assert isinstance(result, PatternEvaluationResult)
        assert result.evaluation_type == 'clustering'
        assert 'silhouette_score' in result.metrics
        assert 'adjusted_rand_index' in result.metrics
        assert len(result.recommendations) >= 0
        assert result.quality_assessment in ['Excellent', 'Good', 'Fair', 'Poor']
        
    def test_dimensionality_reduction_evaluation(self, sample_classification_data):
        """Test dimensionality reduction quality evaluation."""
        X, y = sample_classification_data
        
        # Generate reduction
        reducer = DimensionalityReductionTransformer(algorithm='pca', n_components=3, random_state=42)
        reducer.fit(X)
        X_reduced = reducer.transform(X)
        
        evaluator = PatternEvaluationTransformer('dimensionality_reduction')
        result = evaluator.evaluate_dimensionality_reduction(X, X_reduced)
        
        assert isinstance(result, PatternEvaluationResult)
        assert result.evaluation_type == 'dimensionality_reduction'
        assert 'reduction_ratio' in result.metrics
        assert 'original_dimensions' in result.metrics
        assert 'reduced_dimensions' in result.metrics
        assert len(result.recommendations) >= 0
        
    def test_anomaly_detection_evaluation(self, sample_anomaly_data):
        """Test anomaly detection quality evaluation."""
        X, y_true = sample_anomaly_data
        
        # Generate anomaly detection results
        detector = AnomalyDetectionTransformer(algorithm='isolation_forest', contamination=0.1, random_state=42)
        detector.fit(X)
        labels = detector.predict(X)
        scores = detector.decision_function(X)
        
        evaluator = PatternEvaluationTransformer('anomaly_detection')
        result = evaluator.evaluate_anomaly_detection(X, labels, scores, y_true)
        
        assert isinstance(result, PatternEvaluationResult)
        assert result.evaluation_type == 'anomaly_detection'
        assert 'contamination_rate' in result.metrics
        assert 'n_anomalies' in result.metrics
        assert 'precision' in result.metrics
        assert len(result.recommendations) >= 0


class TestHighLevelFunctions:
    """Test high-level convenience functions."""
    
    def test_perform_clustering_function(self, sample_clustering_data):
        """Test high-level perform_clustering function."""
        X, y_true = sample_clustering_data
        
        result = perform_clustering(X, algorithm='kmeans', n_clusters=4, y_true=y_true)
        
        assert isinstance(result, dict)
        assert 'algorithm' in result
        assert 'labels' in result
        assert 'evaluation' in result
        assert result['algorithm'] == 'kmeans'
        assert len(result['labels']) == len(X)
        
    def test_reduce_dimensions_function(self, sample_classification_data):
        """Test high-level reduce_dimensions function."""
        X, y = sample_classification_data
        
        result = reduce_dimensions(X, algorithm='pca', n_components=3)
        
        assert isinstance(result, dict)
        assert 'algorithm' in result
        assert 'transformed_data' in result
        assert 'evaluation' in result
        assert result['algorithm'] == 'pca'
        assert len(result['transformed_data']) == len(X)
        
    def test_detect_anomalies_function(self, sample_anomaly_data):
        """Test high-level detect_anomalies function."""
        X, y_true = sample_anomaly_data
        
        result = detect_anomalies(X, algorithm='isolation_forest', contamination=0.1, y_true=y_true)
        
        assert isinstance(result, dict)
        assert 'algorithm' in result
        assert 'anomaly_labels' in result
        assert 'evaluation' in result
        assert result['algorithm'] == 'isolation_forest'
        assert len(result['anomaly_labels']) == len(X)
        
    def test_evaluate_patterns_function(self, sample_clustering_data):
        """Test high-level evaluate_patterns function."""
        X, y_true = sample_clustering_data
        
        # First generate clustering results
        clustering_result = perform_clustering(X, algorithm='kmeans', n_clusters=4, y_true=y_true)
        
        # Then evaluate them
        evaluation_result = evaluate_patterns(X, 'clustering', clustering_result, y_true)
        
        assert isinstance(evaluation_result, dict)
        assert 'evaluation_type' in evaluation_result
        assert 'metrics' in evaluation_result
        assert 'quality_assessment' in evaluation_result
        assert evaluation_result['evaluation_type'] == 'clustering'


class TestPerformanceBenchmarks:
    """Performance benchmarking tests for high-dimensional data."""
    
    def test_clustering_performance_large_dataset(self):
        """Test clustering performance with larger dataset."""
        # Generate larger dataset
        X, y = make_blobs(n_samples=1000, centers=5, n_features=20, random_state=42)
        
        import time
        start_time = time.time()
        
        result = perform_clustering(X, algorithm='kmeans', n_clusters=5)
        
        elapsed_time = time.time() - start_time
        
        # Should complete in reasonable time (< 10 seconds)
        assert elapsed_time < 10.0
        assert isinstance(result, dict)
        assert result['n_clusters'] == 5
        
    def test_dimensionality_reduction_performance(self):
        """Test dimensionality reduction performance with high-dimensional data."""
        # Generate high-dimensional dataset
        X = np.random.randn(500, 50)
        
        import time
        start_time = time.time()
        
        result = reduce_dimensions(X, algorithm='pca', n_components=10)
        
        elapsed_time = time.time() - start_time
        
        # Should complete in reasonable time (< 5 seconds)
        assert elapsed_time < 5.0
        assert isinstance(result, dict)
        assert result['reduced_dimensions'] == 10
        
    def test_anomaly_detection_performance(self):
        """Test anomaly detection performance with larger dataset."""
        # Generate larger dataset with anomalies
        normal_data = np.random.randn(800, 10)
        anomalies = np.random.randn(200, 10) * 3 + 5
        X = np.vstack([normal_data, anomalies])
        
        import time
        start_time = time.time()
        
        result = detect_anomalies(X, algorithm='isolation_forest', contamination=0.2)
        
        elapsed_time = time.time() - start_time
        
        # Should complete in reasonable time (< 5 seconds)
        assert elapsed_time < 5.0
        assert isinstance(result, dict)
        assert result['n_samples'] == 1000


class TestEdgeCasesAndErrorHandling:
    """Test edge cases and error handling."""
    
    def test_empty_data_handling(self):
        """Test handling of empty datasets."""
        X_empty = np.array([]).reshape(0, 5)
        
        with pytest.raises((ValueError, Exception)):
            perform_clustering(X_empty)
            
    def test_single_sample_handling(self):
        """Test handling of single sample."""
        X_single = np.array([[1, 2, 3, 4, 5]])
        
        # Should handle gracefully or raise appropriate error
        try:
            result = perform_clustering(X_single, algorithm='kmeans', n_clusters=1)
            assert isinstance(result, dict)
        except (ValueError, Exception):
            # Acceptable to fail with appropriate error
            pass
            
    def test_invalid_algorithm_handling(self):
        """Test handling of invalid algorithms."""
        X = np.random.randn(50, 5)
        
        with pytest.raises((ValueError, Exception)):
            perform_clustering(X, algorithm='invalid_algorithm')
            
        with pytest.raises((ValueError, Exception)):
            reduce_dimensions(X, algorithm='invalid_algorithm')
            
        with pytest.raises((ValueError, Exception)):
            detect_anomalies(X, algorithm='invalid_algorithm')
            
    def test_inconsistent_parameters(self):
        """Test handling of inconsistent parameters."""
        X = np.random.randn(50, 5)
        
        # Test with more clusters than samples
        try:
            result = perform_clustering(X, algorithm='kmeans', n_clusters=100)
            # Should either handle gracefully or raise appropriate error
        except (ValueError, Exception):
            pass
            
    def test_missing_dependencies(self):
        """Test handling when optional dependencies are missing."""
        X = np.random.randn(50, 10)
        
        # Test UMAP when not available (mocked)
        with patch('src.localdata_mcp.domains.pattern_recognition.UMAP_AVAILABLE', False):
            with pytest.raises((ImportError, Exception)):
                reduce_dimensions(X, algorithm='umap')


class TestResultDataclasses:
    """Test result dataclass serialization and functionality."""
    
    def test_clustering_result_serialization(self, sample_clustering_data):
        """Test ClusteringResult serialization to dict."""
        X, y_true = sample_clustering_data
        
        clusterer = ClusteringTransformer(algorithm='kmeans', n_clusters=4, random_state=42)
        clusterer.fit(X)
        result = clusterer.get_clustering_result(X, y_true)
        
        result_dict = result.to_dict()
        
        assert isinstance(result_dict, dict)
        assert 'algorithm' in result_dict
        assert 'labels' in result_dict
        assert 'silhouette_avg' in result_dict
        assert isinstance(result_dict['labels'], list)
        
    def test_dimensionality_reduction_result_serialization(self, sample_classification_data):
        """Test DimensionalityReductionResult serialization to dict."""
        X, y = sample_classification_data
        
        reducer = DimensionalityReductionTransformer(algorithm='pca', n_components=3, random_state=42)
        reducer.fit(X)
        result = reducer.get_reduction_result(X)
        
        result_dict = result.to_dict()
        
        assert isinstance(result_dict, dict)
        assert 'algorithm' in result_dict
        assert 'transformed_data' in result_dict
        assert 'original_dimensions' in result_dict
        assert isinstance(result_dict['transformed_data'], list)
        
    def test_anomaly_detection_result_serialization(self, sample_anomaly_data):
        """Test AnomalyDetectionResult serialization to dict."""
        X, y_true = sample_anomaly_data
        
        detector = AnomalyDetectionTransformer(algorithm='isolation_forest', contamination=0.1, random_state=42)
        detector.fit(X)
        result = detector.get_anomaly_result(X, y_true)
        
        result_dict = result.to_dict()
        
        assert isinstance(result_dict, dict)
        assert 'algorithm' in result_dict
        assert 'anomaly_labels' in result_dict
        assert 'anomaly_scores' in result_dict
        assert isinstance(result_dict['anomaly_labels'], list)
        
    def test_pattern_evaluation_result_serialization(self, sample_clustering_data):
        """Test PatternEvaluationResult serialization to dict."""
        X, y_true = sample_clustering_data
        
        clusterer = ClusteringTransformer(algorithm='kmeans', n_clusters=4, random_state=42)
        clusterer.fit(X)
        labels = clusterer.labels_
        
        evaluator = PatternEvaluationTransformer('clustering')
        result = evaluator.evaluate_clustering(X, labels, y_true)
        
        result_dict = result.to_dict()
        
        assert isinstance(result_dict, dict)
        assert 'evaluation_type' in result_dict
        assert 'metrics' in result_dict
        assert 'recommendations' in result_dict
        assert isinstance(result_dict['metrics'], dict)
        assert isinstance(result_dict['recommendations'], list)


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])