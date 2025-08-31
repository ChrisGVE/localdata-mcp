#!/usr/bin/env python3
"""Demo script showcasing the Intelligent Token Management System.

This script demonstrates the key features of the TokenManager implementation
for LocalData MCP v1.3.1, showing how it provides rich metadata for LLM
decision-making about data handling.
"""

import pandas as pd
import numpy as np
from src.localdata_mcp.token_manager import get_token_manager

def demo_basic_estimation():
    """Demonstrate basic token estimation with different data types."""
    print("=== Basic Token Estimation Demo ===")
    
    # Create sample DataFrame with different data types
    df = pd.DataFrame({
        'id': range(1000),  # Numeric: ~1 token each
        'name': [f'User_{i}' for i in range(1000)],  # Short text
        'description': [f'This is a longer description for user {i} with more content' for i in range(1000)],  # Long text
        'score': np.random.uniform(0, 100, 1000),  # Float values
        'is_active': np.random.choice([True, False], 1000),  # Boolean
        'created_at': pd.date_range('2023-01-01', periods=1000, freq='H')  # Datetime
    })
    
    token_manager = get_token_manager()
    estimation = token_manager.estimate_tokens_from_dataframe(df)
    
    print(f"Dataset: {estimation.total_rows} rows × {len(df.columns)} columns")
    print(f"Total estimated tokens: {estimation.total_tokens:,}")
    print(f"Tokens per row: {estimation.tokens_per_row:.1f}")
    print(f"Confidence: {estimation.confidence:.1%}")
    print(f"Risk level: {estimation.risk_level}")
    print(f"Memory risk: {estimation.memory_risk}")
    
    print("\nColumn breakdown:")
    print(f"  Numeric columns: {len(estimation.numeric_columns)} - {estimation.numeric_columns}")
    print(f"  Text columns: {len(estimation.text_columns)} - {estimation.text_columns}")
    print(f"  Other columns: {len(estimation.other_columns)} - {estimation.other_columns}")
    
    print("\nToken breakdown by column:")
    for col, tokens in estimation.column_token_breakdown.items():
        print(f"  {col}: {tokens} tokens per value")
    
    print(f"\nJSON overhead: {estimation.json_overhead_per_row} tokens per row")
    print()

def demo_response_metadata():
    """Demonstrate rich response metadata generation."""
    print("=== Response Metadata Demo ===")
    
    # Create a larger text-heavy dataset
    large_df = pd.DataFrame({
        'article_id': range(5000),
        'title': [f'Article Title Number {i} - A Comprehensive Analysis' for i in range(5000)],
        'content': [f'This is the full article content for article {i}. ' + 
                   'It contains multiple paragraphs of detailed information, analysis, and conclusions. ' * 3
                   for i in range(5000)],
        'category': np.random.choice(['Tech', 'Science', 'Business', 'Health'], 5000),
        'word_count': np.random.randint(100, 2000, 5000)
    })
    
    token_manager = get_token_manager()
    estimation = token_manager.estimate_tokens_from_dataframe(large_df)
    metadata = token_manager.get_response_metadata(estimation)
    
    print(f"Dataset: {metadata.row_count} rows")
    print(f"Response size category: {metadata.response_size_category}")
    print(f"Estimated tokens: {metadata.estimated_tokens:,}")
    print(f"Estimated memory: {metadata.estimated_memory_mb:.1f} MB")
    print(f"Data density: {metadata.data_density}")
    print(f"Text heavy: {metadata.text_heavy}")
    print(f"Streaming recommended: {metadata.streaming_recommended}")
    print(f"Processing complexity: {metadata.processing_complexity}")
    print(f"Estimated response time: {metadata.estimated_response_time:.2f} seconds")
    
    print("\nModel compatibility:")
    for model, compatibility in metadata.model_compatibility.items():
        fits = "✓" if compatibility['fits_in_context'] else "✗"
        utilization = compatibility['utilization_percent']
        print(f"  {model}: {fits} ({utilization:.1f}% of context)")
    
    if metadata.chunking_recommendation and metadata.chunking_recommendation.should_chunk:
        chunk_rec = metadata.chunking_recommendation
        print(f"\nChunking recommended:")
        print(f"  Strategy: {chunk_rec.strategy}")
        print(f"  Chunk size: {chunk_rec.recommended_chunk_size} rows")
        print(f"  Estimated chunks: {chunk_rec.estimated_chunks}")
        print(f"  Rationale: {chunk_rec.chunk_size_rationale}")
    
    print("\nSampling options:")
    if metadata.sampling_options['recommended']:
        print(f"  Sampling recommended: {metadata.sampling_options['reason']}")
        for option_name, option_info in metadata.sampling_options['options'].items():
            print(f"    {option_name}: {option_info['rows']} rows ({option_info['percentage']:.1f}%) = {option_info['estimated_tokens']:,} tokens")
    else:
        print(f"  Sampling not recommended: {metadata.sampling_options['reason']}")
    
    print()

def demo_query_result_estimation():
    """Demonstrate query result estimation (pre-execution)."""
    print("=== Query Result Estimation Demo ===")
    
    # Simulate a sample row from LIMIT 1 query
    sample_row = pd.Series({
        'user_id': 12345,
        'username': 'john_doe_example',
        'email': 'john.doe@company.com',
        'full_name': 'John Michael Doe',
        'bio': 'Senior Software Engineer with 8+ years of experience in Python, JavaScript, and cloud technologies.',
        'created_at': '2023-01-15 14:30:00',
        'last_login': '2024-12-30 09:45:23',
        'is_verified': True,
        'subscription_level': 'premium'
    })
    
    # Simulate different query result sizes
    result_sizes = [100, 1000, 10000, 100000]
    
    token_manager = get_token_manager()
    
    for size in result_sizes:
        sample_df = pd.DataFrame([sample_row])
        estimation = token_manager.estimate_tokens_for_query_result(size, sample_df)
        
        print(f"Query result: {size:,} rows")
        print(f"  Estimated tokens: {estimation.total_tokens:,}")
        print(f"  Tokens per row: {estimation.tokens_per_row:.1f}")
        print(f"  Risk level: {estimation.risk_level}")
        print(f"  Confidence: {estimation.confidence:.1%} (extrapolated from {estimation.sample_size} sample)")
        
        # Check context window compatibility
        gpt4_fits = estimation.fits_in_context.get('gpt-4', False)
        claude_fits = estimation.fits_in_context.get('claude-3-sonnet', False)
        print(f"  Fits in GPT-4: {'✓' if gpt4_fits else '✗'}")
        print(f"  Fits in Claude: {'✓' if claude_fits else '✗'}")
        
        if estimation.recommended_chunk_size:
            print(f"  Recommended chunk size: {estimation.recommended_chunk_size} rows")
        print()

def demo_performance_characteristics():
    """Demonstrate performance optimization features."""
    print("=== Performance Characteristics Demo ===")
    
    datasets = [
        ("Small numeric", pd.DataFrame({'nums': range(100)})),
        ("Medium mixed", pd.DataFrame({
            'id': range(1000),
            'text': ['Short text'] * 1000,
            'value': np.random.uniform(0, 100, 1000)
        })),
        ("Large text-heavy", pd.DataFrame({
            'id': range(5000),
            'content': ['This is a much longer piece of text content that will require more tokens to represent accurately in the language model context'] * 5000
        }))
    ]
    
    token_manager = get_token_manager()
    
    for name, df in datasets:
        estimation = token_manager.estimate_tokens_from_dataframe(df)
        metadata = token_manager.get_response_metadata(estimation)
        
        print(f"{name}:")
        print(f"  Rows: {len(df):,}, Columns: {len(df.columns)}")
        print(f"  Estimation method: {estimation.estimation_method}")
        print(f"  Sample size: {estimation.sample_size}")
        print(f"  Confidence: {estimation.confidence:.1%}")
        print(f"  Processing complexity: {metadata.processing_complexity}")
        print(f"  Response time estimate: {metadata.estimated_response_time:.3f}s")
        print(f"  Memory estimate: {metadata.estimated_memory_mb:.1f} MB")
        print(f"  Streaming recommended: {metadata.streaming_recommended}")
        print()

if __name__ == '__main__':
    print("🚀 LocalData MCP v1.3.1 - Intelligent Token Management System Demo")
    print("=" * 70)
    print()
    
    demo_basic_estimation()
    demo_response_metadata()
    demo_query_result_estimation()
    demo_performance_characteristics()
    
    print("=" * 70)
    print("✅ Demo complete! The TokenManager provides rich metadata to help")
    print("   LLMs make intelligent decisions about data handling, chunking,")
    print("   and response optimization.")