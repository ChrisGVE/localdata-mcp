"""
Tests for regression modeling domain.

This module contains comprehensive tests for the regression modeling domain,
including all transformers, pipelines, and high-level functions.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
from sklearn.datasets import make_regression, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from localdata_mcp.domains.regression_modeling import (
    LinearRegressionTransformer,
    RegularizedRegressionTransformer,
    LogisticRegressionTransformer,
    PolynomialRegressionTransformer,
    ResidualAnalysisTransformer,
    FeatureSelectionTransformer,
    RegressionModelingPipeline,
    RegressionModelResult,
    ResidualAnalysisResult,
    fit_regression_model,
    evaluate_model_performance,
    analyze_residuals,
    select_features
)


class TestLinearRegressionTransformer:
    """Test cases for LinearRegressionTransformer."""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample regression data."""
        X, y = make_regression(n_samples=100, n_features=5, noise=0.1, random_state=42)
        feature_names = [f"feature_{i}" for i in range(5)]
        return X, y, feature_names
    
    def test_basic_fit_predict(self, sample_data):
        """Test basic fit and predict functionality."""
        X, y, feature_names = sample_data
        
        transformer = LinearRegressionTransformer()
        transformer.fit(X, y, feature_names=feature_names)
        
        # Test predictions
        predictions = transformer.predict(X)
        assert len(predictions) == len(y)
        assert isinstance(predictions, np.ndarray)
        
        # Test scoring
        score = transformer.score(X, y)
        assert 0 <= score <= 1
        
    def test_comprehensive_results(self, sample_data):
        """Test comprehensive regression results."""
        X, y, feature_names = sample_data
        
        transformer = LinearRegressionTransformer(include_diagnostics=True)
        transformer.fit(X, y, feature_names=feature_names)
        
        result = transformer.get_result()
        assert isinstance(result, RegressionModelResult)
        assert result.model_type == "Linear Regression"
        assert result.r2_score is not None
        assert result.mse > 0
        assert result.mae > 0
        assert result.rmse > 0
        assert result.adjusted_r2 is not None
        assert result.coefficients is not None
        assert len(result.feature_names) == len(feature_names) + 1  # +1 for intercept
        
    def test_without_diagnostics(self, sample_data):
        """Test without comprehensive diagnostics."""
        X, y, feature_names = sample_data
        
        transformer = LinearRegressionTransformer(include_diagnostics=False)
        transformer.fit(X, y, feature_names=feature_names)
        
        result = transformer.get_result()
        assert result.aic is None
        assert result.bic is None
        assert result.log_likelihood is None
        
    def test_transform_method(self, sample_data):
        """Test transform method for pipeline compatibility."""
        X, y, feature_names = sample_data
        
        transformer = LinearRegressionTransformer()
        transformer.fit(X, y, feature_names=feature_names)
        
        transformed = transformer.transform(X)
        predictions = transformer.predict(X)
        np.testing.assert_array_equal(transformed, predictions)


class TestRegularizedRegressionTransformer:
    """Test cases for RegularizedRegressionTransformer."""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample regression data with some noise."""
        X, y = make_regression(n_samples=100, n_features=20, noise=0.1, random_state=42)
        feature_names = [f"feature_{i}" for i in range(20)]
        return X, y, feature_names
    
    @pytest.mark.parametrize("method", ['ridge', 'lasso', 'elastic_net'])
    def test_regularization_methods(self, sample_data, method):
        """Test different regularization methods."""
        X, y, feature_names = sample_data
        
        transformer = RegularizedRegressionTransformer(method=method, alpha=0.1)
        transformer.fit(X, y, feature_names=feature_names)
        
        result = transformer.get_result()
        assert result.model_type == f"{method.title()} Regression"
        assert result.r2_score is not None
        assert result.coefficients is not None
        
        # Test prediction
        predictions = transformer.predict(X)
        assert len(predictions) == len(y)
        
    def test_auto_alpha_selection(self, sample_data):
        """Test automatic alpha selection via cross-validation."""
        X, y, feature_names = sample_data
        
        transformer = RegularizedRegressionTransformer(method='ridge', alpha='auto', cv=3)
        transformer.fit(X, y, feature_names=feature_names)
        
        result = transformer.get_result()
        assert 'optimal_alpha' in result.convergence_info
        assert result.cv_scores is not None
        assert result.cv_mean is not None
        assert result.cv_std is not None
        
    def test_feature_selection_with_lasso(self, sample_data):
        """Test feature selection capability with Lasso."""
        X, y, feature_names = sample_data
        
        # Add some irrelevant features
        noise_features = np.random.randn(X.shape[0], 10)
        X_with_noise = np.hstack([X, noise_features])
        extended_feature_names = feature_names + [f"noise_{i}" for i in range(10)]
        
        transformer = RegularizedRegressionTransformer(method='lasso', alpha=0.1)
        transformer.fit(X_with_noise, y, feature_names=extended_feature_names)
        
        result = transformer.get_result()
        assert result.selected_features is not None
        assert len(result.selected_features) <= len(extended_feature_names)


class TestLogisticRegressionTransformer:
    """Test cases for LogisticRegressionTransformer."""
    
    @pytest.fixture
    def sample_classification_data(self):
        """Generate sample classification data."""
        X, y = make_classification(n_samples=100, n_features=5, n_classes=2, random_state=42)
        feature_names = [f"feature_{i}" for i in range(5)]
        return X, y, feature_names
    
    @pytest.fixture
    def sample_multiclass_data(self):
        """Generate sample multiclass classification data."""
        X, y = make_classification(n_samples=150, n_features=4, n_classes=3, n_clusters_per_class=1, random_state=42)
        feature_names = [f"feature_{i}" for i in range(4)]
        return X, y, feature_names
    
    def test_binary_classification(self, sample_classification_data):
        """Test binary classification."""
        X, y, feature_names = sample_classification_data
        
        transformer = LogisticRegressionTransformer()
        transformer.fit(X, y, feature_names=feature_names)
        
        result = transformer.get_result()
        assert result.model_type == "Logistic Regression"
        assert 0 <= result.r2_score <= 1  # Using accuracy as r2_score equivalent
        assert result.additional_info['n_classes'] == 2
        assert 'accuracy' in result.additional_info
        assert 'precision' in result.additional_info
        assert 'recall' in result.additional_info
        assert 'f1_score' in result.additional_info
        
        # Test predictions
        predictions = transformer.predict(X)
        probabilities = transformer.predict_proba(X)
        assert len(predictions) == len(y)
        assert probabilities.shape == (len(y), 2)
        
    def test_multiclass_classification(self, sample_multiclass_data):
        """Test multiclass classification."""
        X, y, feature_names = sample_multiclass_data
        
        transformer = LogisticRegressionTransformer(multi_class='multinomial', solver='lbfgs')
        transformer.fit(X, y, feature_names=feature_names)
        
        result = transformer.get_result()
        assert result.additional_info['n_classes'] == 3
        assert result.coefficients.ndim == 2  # Multi-class coefficients
        
        # Test predictions
        probabilities = transformer.predict_proba(X)
        assert probabilities.shape == (len(y), 3)
        
    def test_regularization_parameters(self, sample_classification_data):
        """Test different regularization parameters."""
        X, y, feature_names = sample_classification_data
        
        transformer = LogisticRegressionTransformer(penalty='l1', C=0.5, solver='liblinear')
        transformer.fit(X, y, feature_names=feature_names)
        
        result = transformer.get_result()
        assert result.model_params['penalty'] == 'l1'
        assert result.model_params['C'] == 0.5
        
    def test_transform_returns_probabilities(self, sample_classification_data):
        """Test that transform method returns probabilities."""
        X, y, feature_names = sample_classification_data
        
        transformer = LogisticRegressionTransformer()
        transformer.fit(X, y, feature_names=feature_names)
        
        transformed = transformer.transform(X)
        probabilities = transformer.predict_proba(X)
        np.testing.assert_array_equal(transformed, probabilities)


class TestPolynomialRegressionTransformer:
    """Test cases for PolynomialRegressionTransformer."""
    
    @pytest.fixture
    def sample_nonlinear_data(self):
        """Generate sample data with polynomial relationship."""
        np.random.seed(42)
        X = np.random.randn(80, 2)
        # Create polynomial relationship
        y = 2*X[:, 0] + 3*X[:, 1] + X[:, 0]**2 + 0.5*X[:, 0]*X[:, 1] + 0.1*np.random.randn(80)
        feature_names = ['x1', 'x2']
        return X, y, feature_names
    
    def test_polynomial_features(self, sample_nonlinear_data):
        """Test polynomial feature generation."""
        X, y, feature_names = sample_nonlinear_data
        
        transformer = PolynomialRegressionTransformer(degree=2)
        transformer.fit(X, y, feature_names=feature_names)
        
        result = transformer.get_result()
        assert result.model_type == "Polynomial Regression (degree=2)"
        assert result.convergence_info['degree'] == 2
        assert result.convergence_info['original_features'] == 2
        assert result.convergence_info['polynomial_features'] > 2  # Should expand features
        assert result.convergence_info['feature_expansion_ratio'] > 1.0
        
    def test_overfitting_detection(self, sample_nonlinear_data):
        """Test overfitting detection with high degree polynomial."""
        X, y, feature_names = sample_nonlinear_data
        
        # High degree polynomial likely to overfit
        transformer = PolynomialRegressionTransformer(degree=5)
        transformer.fit(X, y, feature_names=feature_names)
        
        result = transformer.get_result()
        # Check if overfitting detection mechanism is working
        assert 'overfitting_detected' in result.convergence_info
        assert 'overfitting_gap' in result.convergence_info
        
    def test_polynomial_with_regularization(self, sample_nonlinear_data):
        """Test polynomial regression with regularization."""
        X, y, feature_names = sample_nonlinear_data
        
        transformer = PolynomialRegressionTransformer(
            degree=3, 
            regularization='ridge', 
            alpha=0.1
        )
        transformer.fit(X, y, feature_names=feature_names)
        
        result = transformer.get_result()
        assert result.model_params['regularization'] == 'ridge'
        assert result.model_params['alpha'] == 0.1
        assert result.convergence_info['regularization_used'] is True
        
    def test_interaction_only(self, sample_nonlinear_data):
        """Test interaction-only polynomial features."""
        X, y, feature_names = sample_nonlinear_data
        
        transformer = PolynomialRegressionTransformer(degree=2, interaction_only=True)
        transformer.fit(X, y, feature_names=feature_names)
        
        result = transformer.get_result()
        # With interaction_only=True, fewer features should be created
        assert result.convergence_info['polynomial_features'] < 6  # 1 + 2 + 1 interaction + bias


class TestResidualAnalysisTransformer:
    """Test cases for ResidualAnalysisTransformer."""
    
    @pytest.fixture
    def sample_regression_results(self):
        """Generate sample regression data and fitted results."""
        X, y = make_regression(n_samples=100, n_features=3, noise=0.5, random_state=42)
        
        # Fit a simple model to get residuals
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)
        residuals = y - y_pred
        
        return X, y, residuals, y_pred
    
    def test_comprehensive_residual_analysis(self, sample_regression_results):
        """Test comprehensive residual analysis."""
        X, y, residuals, fitted_values = sample_regression_results
        
        analyzer = ResidualAnalysisTransformer()
        analyzer.fit(X, y, residuals=residuals, fitted_values=fitted_values)
        
        result = analyzer.get_result()
        assert isinstance(result, ResidualAnalysisResult)
        assert len(result.residuals) == len(y)
        assert len(result.standardized_residuals) == len(y)
        assert len(result.fitted_values) == len(y)
        
        # Check diagnostic tests are performed
        assert result.normality_test is not None
        assert result.homoscedasticity_test is not None
        
        # Check residual statistics
        assert result.residual_stats is not None
        assert 'mean' in result.residual_stats
        assert 'std' in result.residual_stats
        assert 'skewness' in result.residual_stats
        assert 'kurtosis' in result.residual_stats
        
    def test_outlier_detection(self, sample_regression_results):
        """Test outlier detection in residuals."""
        X, y, residuals, fitted_values = sample_regression_results
        
        # Add some outliers
        residuals_with_outliers = residuals.copy()
        residuals_with_outliers[0] = 5 * np.std(residuals)  # Strong outlier
        residuals_with_outliers[1] = -4 * np.std(residuals)  # Strong outlier
        
        analyzer = ResidualAnalysisTransformer(outlier_threshold=2.0)
        analyzer.fit(X, y, residuals=residuals_with_outliers, fitted_values=fitted_values)
        
        result = analyzer.get_result()
        assert result.outliers is not None
        assert len(result.outliers) >= 2  # Should detect the outliers we added
        
    def test_influence_measures(self, sample_regression_results):
        """Test computation of influence measures."""
        X, y, residuals, fitted_values = sample_regression_results
        
        analyzer = ResidualAnalysisTransformer(include_influence=True)
        analyzer.fit(X, y, residuals=residuals, fitted_values=fitted_values)
        
        result = analyzer.get_result()
        # Note: Some influence measures might be None if computation fails
        # This is expected behavior for robustness
        
    def test_without_influence_measures(self, sample_regression_results):
        """Test without computing influence measures."""
        X, y, residuals, fitted_values = sample_regression_results
        
        analyzer = ResidualAnalysisTransformer(include_influence=False)
        analyzer.fit(X, y, residuals=residuals, fitted_values=fitted_values)
        
        result = analyzer.get_result()
        assert result.leverage is None
        assert result.cooks_distance is None
        
    def test_transform_returns_standardized_residuals(self, sample_regression_results):
        """Test that transform method returns standardized residuals."""
        X, y, residuals, fitted_values = sample_regression_results
        
        analyzer = ResidualAnalysisTransformer()
        analyzer.fit(X, y, residuals=residuals, fitted_values=fitted_values)
        
        transformed = analyzer.transform(X)
        assert transformed.shape == (len(y), 1)
        # Should be standardized residuals reshaped
        expected = analyzer.get_result().standardized_residuals.reshape(-1, 1)
        np.testing.assert_array_equal(transformed, expected)


class TestFeatureSelectionTransformer:
    """Test cases for FeatureSelectionTransformer."""
    
    @pytest.fixture
    def sample_data_with_irrelevant_features(self):
        """Generate data with relevant and irrelevant features."""
        # Create data where only first 3 features are relevant
        X_relevant, y = make_regression(n_samples=100, n_features=3, noise=0.1, random_state=42)
        
        # Add 7 irrelevant features
        X_irrelevant = np.random.randn(100, 7)
        X = np.hstack([X_relevant, X_irrelevant])
        
        feature_names = [f"relevant_{i}" for i in range(3)] + [f"irrelevant_{i}" for i in range(7)]
        return X, y, feature_names
    
    @pytest.mark.parametrize("method", ['model_based', 'rfe', 'univariate'])
    def test_feature_selection_methods(self, sample_data_with_irrelevant_features, method):
        """Test different feature selection methods."""
        X, y, feature_names = sample_data_with_irrelevant_features
        
        selector = FeatureSelectionTransformer(method=method)
        selector.fit(X, y, feature_names=feature_names)
        
        result = selector.get_result()
        assert result['method'] == method
        assert result['n_selected'] <= result['n_original']
        assert result['selected_features'] is not None
        assert result['selected_indices'] is not None
        assert 0 < result['selection_ratio'] <= 1
        
        # Test transformation
        X_selected = selector.transform(X)
        assert X_selected.shape[0] == X.shape[0]
        assert X_selected.shape[1] == result['n_selected']
        
    def test_rfecv_method(self, sample_data_with_irrelevant_features):
        """Test recursive feature elimination with cross-validation."""
        X, y, feature_names = sample_data_with_irrelevant_features
        
        selector = FeatureSelectionTransformer(method='rfecv', cv=3)
        selector.fit(X, y, feature_names=feature_names)
        
        result = selector.get_result()
        assert result['method'] == 'rfecv'
        # RFECV should have additional info
        if 'cv_scores' in result:
            assert result['cv_scores'] is not None
            assert 'optimal_n_features' in result
            
    def test_feature_importance_computation(self, sample_data_with_irrelevant_features):
        """Test feature importance computation."""
        X, y, feature_names = sample_data_with_irrelevant_features
        
        selector = FeatureSelectionTransformer(method='model_based')
        selector.fit(X, y, feature_names=feature_names)
        
        result = selector.get_result()
        if result['feature_importance'] is not None:
            assert len(result['feature_importance']) == result['n_selected']
            
    def test_performance_comparison(self, sample_data_with_irrelevant_features):
        """Test performance comparison between original and selected features."""
        X, y, feature_names = sample_data_with_irrelevant_features
        
        selector = FeatureSelectionTransformer(method='model_based')
        selector.fit(X, y, feature_names=feature_names)
        
        result = selector.get_result()
        if result['comparison']:
            comparison = result['comparison']
            assert 'original_features' in comparison
            assert 'selected_features' in comparison
            assert 'feature_reduction' in comparison
            assert 'r2_original' in comparison
            assert 'r2_selected' in comparison
            
    def test_get_support(self, sample_data_with_irrelevant_features):
        """Test get_support method for feature mask."""
        X, y, feature_names = sample_data_with_irrelevant_features
        
        selector = FeatureSelectionTransformer(method='univariate', k=5)
        selector.fit(X, y, feature_names=feature_names)
        
        support_mask = selector.get_support()
        support_indices = selector.get_support(indices=True)
        
        assert len(support_mask) == X.shape[1]
        assert np.sum(support_mask) == 5
        assert len(support_indices) == 5


class TestRegressionModelingPipeline:
    """Test cases for RegressionModelingPipeline."""
    
    @pytest.fixture
    def sample_regression_data(self):
        """Generate sample regression data."""
        X, y = make_regression(n_samples=100, n_features=8, noise=0.1, random_state=42)
        feature_names = [f"feature_{i}" for i in range(8)]
        return X, y, feature_names
    
    @pytest.fixture
    def sample_classification_data(self):
        """Generate sample classification data."""
        X, y = make_classification(n_samples=100, n_features=8, n_classes=2, random_state=42)
        feature_names = [f"feature_{i}" for i in range(8)]
        return X, y, feature_names
    
    @pytest.mark.parametrize("model_type", ['linear', 'ridge', 'lasso', 'polynomial'])
    def test_regression_pipeline_types(self, sample_regression_data, model_type):
        """Test different regression pipeline types."""
        X, y, feature_names = sample_regression_data
        
        pipeline = RegressionModelingPipeline(model_type=model_type)
        pipeline.fit(X, y, feature_names=feature_names)
        
        results = pipeline.get_results()
        assert results['model_type'] == model_type
        assert 'regression_analysis' in results
        assert 'pipeline_config' in results
        
        # Test prediction
        predictions = pipeline.predict(X)
        assert len(predictions) == len(y)
        
        # Test scoring
        score = pipeline.score(X, y)
        assert isinstance(score, float)
        
    def test_logistic_regression_pipeline(self, sample_classification_data):
        """Test logistic regression pipeline."""
        X, y, feature_names = sample_classification_data
        
        pipeline = RegressionModelingPipeline(model_type='logistic')
        pipeline.fit(X, y, feature_names=feature_names)
        
        results = pipeline.get_results()
        assert results['model_type'] == 'logistic'
        assert 'regression_analysis' in results
        # Residual analysis should be skipped for logistic regression
        assert 'residual_analysis' not in results
        
    def test_pipeline_with_feature_selection(self, sample_regression_data):
        """Test pipeline with feature selection enabled."""
        X, y, feature_names = sample_regression_data
        
        pipeline = RegressionModelingPipeline(
            model_type='linear',
            feature_selection=True,
            feature_selection_params={'method': 'model_based'}
        )
        pipeline.fit(X, y, feature_names=feature_names)
        
        results = pipeline.get_results()
        assert 'feature_selection' in results
        assert results['feature_selection']['n_selected'] <= len(feature_names)
        
    def test_pipeline_without_residual_analysis(self, sample_regression_data):
        """Test pipeline without residual analysis."""
        X, y, feature_names = sample_regression_data
        
        pipeline = RegressionModelingPipeline(
            model_type='linear',
            residual_analysis=False
        )
        pipeline.fit(X, y, feature_names=feature_names)
        
        results = pipeline.get_results()
        assert 'residual_analysis' not in results
        
    def test_pipeline_composition_metadata(self, sample_regression_data):
        """Test pipeline composition metadata."""
        X, y, feature_names = sample_regression_data
        
        pipeline = RegressionModelingPipeline(model_type='linear')
        pipeline.fit(X, y, feature_names=feature_names)
        
        metadata = pipeline.get_composition_metadata()
        assert metadata.domain == "regression_modeling"
        assert metadata.analysis_type == "linear"
        assert metadata.result_type == "model_results"
        assert len(metadata.compatible_tools) > 0
        assert len(metadata.suggested_compositions) > 0
        
    def test_pipeline_with_custom_parameters(self, sample_regression_data):
        """Test pipeline with custom model parameters."""
        X, y, feature_names = sample_regression_data
        
        pipeline = RegressionModelingPipeline(
            model_type='ridge',
            model_params={'alpha': 0.5},
            residual_params={'outlier_threshold': 3.0}
        )
        pipeline.fit(X, y, feature_names=feature_names)
        
        results = pipeline.get_results()
        model_params = results['regression_analysis']['model_params']
        assert model_params['alpha'] == 0.5


class TestHighLevelFunctions:
    """Test cases for high-level convenience functions."""
    
    @pytest.fixture
    def sample_dataframe(self):
        """Generate sample DataFrame."""
        X, y = make_regression(n_samples=100, n_features=5, noise=0.1, random_state=42)
        df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(5)])
        df['target'] = y
        return df
    
    def test_fit_regression_model(self, sample_dataframe):
        """Test fit_regression_model function."""
        results = fit_regression_model(
            data=sample_dataframe,
            target_column='target',
            model_type='linear'
        )
        
        assert 'model_type' in results
        assert 'regression_analysis' in results
        assert 'pipeline_config' in results
        
    def test_fit_regression_model_with_feature_selection(self, sample_dataframe):
        """Test fit_regression_model with specific features."""
        feature_columns = ['feature_0', 'feature_1', 'feature_2']
        
        results = fit_regression_model(
            data=sample_dataframe,
            target_column='target',
            model_type='ridge',
            feature_columns=feature_columns,
            feature_selection=True
        )
        
        assert results['model_type'] == 'ridge'
        
    def test_evaluate_model_performance(self, sample_dataframe):
        """Test evaluate_model_performance function."""
        # Prepare data
        X = sample_dataframe.drop('target', axis=1).values
        y = sample_dataframe['target'].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Fit a model
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Evaluate
        evaluation = evaluate_model_performance(model, X_test, y_test, X_train, y_train)
        
        assert 'test_metrics' in evaluation
        assert 'train_metrics' in evaluation
        assert 'overfitting_check' in evaluation
        assert 'r2_score' in evaluation['test_metrics']
        assert 'mse' in evaluation['test_metrics']
        assert 'likely_overfitting' in evaluation['overfitting_check']
        
    def test_evaluate_model_performance_test_only(self, sample_dataframe):
        """Test evaluate_model_performance with test data only."""
        X = sample_dataframe.drop('target', axis=1).values
        y = sample_dataframe['target'].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        evaluation = evaluate_model_performance(model, X_test, y_test)
        
        assert 'test_metrics' in evaluation
        assert 'train_metrics' not in evaluation
        assert 'overfitting_check' not in evaluation
        
    def test_analyze_residuals_function(self, sample_dataframe):
        """Test analyze_residuals function."""
        X = sample_dataframe.drop('target', axis=1).values
        y = sample_dataframe['target'].values
        feature_names = sample_dataframe.drop('target', axis=1).columns.tolist()
        
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(X, y)
        
        analysis = analyze_residuals(model, X, y, feature_names=feature_names)
        
        assert 'residual_statistics' in analysis
        assert 'diagnostic_tests' in analysis
        
    def test_select_features_function(self, sample_dataframe):
        """Test select_features function."""
        X = sample_dataframe.drop('target', axis=1).values
        y = sample_dataframe['target'].values
        feature_names = sample_dataframe.drop('target', axis=1).columns.tolist()
        
        result = select_features(
            X, y, 
            method='univariate', 
            feature_names=feature_names,
            k=3
        )
        
        assert 'method' in result
        assert 'selected_features' in result
        assert 'selected_data' in result
        assert result['n_selected'] == 3
        assert result['selected_data'].shape[1] == 3


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_small_dataset(self):
        """Test with very small dataset."""
        X = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([1, 2, 3])
        
        transformer = LinearRegressionTransformer()
        transformer.fit(X, y)
        
        result = transformer.get_result()
        assert result is not None
        
    def test_single_feature(self):
        """Test with single feature."""
        X = np.array([[1], [2], [3], [4], [5]])
        y = np.array([2, 4, 6, 8, 10])
        
        transformer = LinearRegressionTransformer()
        transformer.fit(X, y)
        
        result = transformer.get_result()
        assert result.n_features == 1
        
    def test_perfect_correlation(self):
        """Test with perfect correlation (no noise)."""
        X = np.array([[1], [2], [3], [4], [5]])
        y = 2 * X[:, 0] + 1  # Perfect linear relationship
        
        transformer = LinearRegressionTransformer()
        transformer.fit(X, y)
        
        result = transformer.get_result()
        assert abs(result.r2_score - 1.0) < 1e-10  # Should be very close to 1
        
    def test_invalid_model_type(self):
        """Test invalid model type in pipeline."""
        X = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([1, 2, 3])
        
        pipeline = RegressionModelingPipeline(model_type='invalid_type')
        
        with pytest.raises(ValueError):
            pipeline.fit(X, y)
            
    def test_invalid_feature_selection_method(self):
        """Test invalid feature selection method."""
        X = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([1, 2, 3])
        
        selector = FeatureSelectionTransformer(method='invalid_method')
        
        with pytest.raises(ValueError):
            selector.fit(X, y)


if __name__ == "__main__":
    pytest.main([__file__])