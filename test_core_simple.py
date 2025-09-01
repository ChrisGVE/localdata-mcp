"""
Simple test script for DataSciencePipeline basic functionality.
Tests core sklearn compatibility and enhanced features.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_classification, make_regression

# Direct import to avoid __init__.py dependency issues
from src.localdata_mcp.pipeline.core import DataSciencePipeline
from src.localdata_mcp.pipeline.base import PipelineState, StreamingConfig


def test_basic_initialization():
    """Test basic initialization."""
    print("Testing basic initialization...")
    
    pipeline = DataSciencePipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(n_estimators=5, random_state=42))
    ])
    
    assert pipeline.state == PipelineState.INITIALIZED
    assert len(pipeline.steps) == 2
    assert pipeline.analytical_intention == "General data science pipeline"
    assert pipeline.composition_aware is True
    assert pipeline.pipeline_id is not None
    
    print("✓ Basic initialization test passed")


def test_sklearn_compatibility():
    """Test sklearn Pipeline API compatibility."""
    print("Testing sklearn compatibility...")
    
    # Create sample data
    X, y = make_classification(n_samples=50, n_features=4, random_state=42)
    X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(4)])
    
    pipeline = DataSciencePipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(n_estimators=5, random_state=42))
    ])
    
    # Test sklearn methods work
    pipeline.fit(X_df, y)
    predictions = pipeline.predict(X_df)
    fit_transformed = pipeline.fit_transform(X_df, y)
    
    assert pipeline.state == PipelineState.COMPLETED
    assert len(predictions) == len(X_df)
    assert fit_transformed.shape[0] == X_df.shape[0]
    
    # Test sklearn properties are accessible
    assert hasattr(pipeline, 'named_steps')
    assert 'scaler' in pipeline.named_steps
    assert 'classifier' in pipeline.named_steps
    
    print("✓ sklearn compatibility test passed")


def test_enhanced_features():
    """Test enhanced LocalData MCP features."""
    print("Testing enhanced features...")
    
    X, y = make_regression(n_samples=30, n_features=3, random_state=42)
    X_df = pd.DataFrame(X, columns=['price', 'size', 'location'])
    
    pipeline = DataSciencePipeline(
        steps=[('scaler', StandardScaler()), ('regressor', LinearRegression())],
        analytical_intention="Predict housing prices",
        composition_aware=True
    )
    
    pipeline.fit(X_df, y)
    result = pipeline.transform(X_df)
    
    # Check enhanced features
    assert pipeline._fit_time is not None
    assert pipeline._fit_time > 0
    assert pipeline._transform_time is not None
    assert pipeline._transform_time > 0
    
    # Check metadata generation
    step_metadata = pipeline.get_step_metadata()
    assert len(step_metadata) == 2  # scaler + regressor
    
    # Check pipeline result
    pipeline_result = pipeline.get_pipeline_result()
    assert pipeline_result is not None
    assert pipeline_result.success is True
    assert pipeline_result.execution_time_seconds > 0
    assert pipeline_result.metadata['pipeline_id'] == pipeline.pipeline_id
    
    print("✓ Enhanced features test passed")


def test_streaming_configuration():
    """Test streaming configuration."""
    print("Testing streaming configuration...")
    
    streaming_config = StreamingConfig(
        enabled=True,
        threshold_mb=50,
        chunk_size_adaptive=True
    )
    
    pipeline = DataSciencePipeline(
        steps=[('scaler', StandardScaler()), ('regressor', LinearRegression())],
        analytical_intention="Test streaming configuration",
        streaming_config=streaming_config
    )
    
    assert pipeline.streaming_config.enabled is True
    assert pipeline.streaming_config.threshold_mb == 50
    assert pipeline.streaming_config.chunk_size_adaptive is True
    
    print("✓ Streaming configuration test passed")


def test_metadata_tracking():
    """Test metadata tracking capabilities."""
    print("Testing metadata tracking...")
    
    X, y = make_classification(n_samples=30, n_features=4, random_state=42)
    X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(4)])
    
    pipeline = DataSciencePipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(n_estimators=3, random_state=42))
    ], composition_aware=True)
    
    pipeline.fit(X_df, y)
    pipeline.transform(X_df)
    
    # Check execution context is populated
    assert 'data_profile' in pipeline._execution_context
    data_profile = pipeline._execution_context['data_profile']
    assert data_profile['shape'] == X_df.shape
    assert 'memory_usage_mb' in data_profile
    assert 'column_types' in data_profile
    
    # Check composition metadata
    metadata = pipeline.composition_metadata
    assert metadata is not None
    assert metadata.domain == "classification"
    assert metadata.analysis_type == "classification"
    assert metadata.input_schema['columns'] == [f'feature_{i}' for i in range(4)]
    
    print("✓ Metadata tracking test passed")


def test_error_handling():
    """Test basic error handling."""
    print("Testing error handling...")
    
    # Test unfitted pipeline error
    X = pd.DataFrame({'feature_1': [1, 2, 3], 'feature_2': [4, 5, 6]})
    
    pipeline = DataSciencePipeline([
        ('scaler', StandardScaler()),
        ('regressor', LinearRegression())
    ])
    
    try:
        pipeline.transform(X)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Pipeline not fitted" in str(e)
    
    print("✓ Error handling test passed")


def main():
    """Run all tests."""
    print("Running DataSciencePipeline core functionality tests...\n")
    
    try:
        test_basic_initialization()
        test_sklearn_compatibility()
        test_enhanced_features()
        test_streaming_configuration() 
        test_metadata_tracking()
        test_error_handling()
        
        print(f"\n🎉 All tests passed! DataSciencePipeline is working correctly.")
        print(f"✓ sklearn Pipeline API compatibility maintained")
        print(f"✓ Enhanced LocalData MCP features functional")
        print(f"✓ Metadata tracking and composition awareness working")
        print(f"✓ Streaming configuration support ready")
        print(f"✓ Error handling implemented")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)