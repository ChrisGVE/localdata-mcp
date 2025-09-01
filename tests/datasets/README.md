# Dataset Generators for LocalData MCP Stress Testing

This directory contains generators for creating large, realistic datasets to stress test LocalData MCP's performance, memory management, and query optimization capabilities.

## Available Generators

### Social Media Dataset Generator (`social_media_generator.py`)

Generates a comprehensive social media dataset for testing complex graph relationships and text-heavy data processing.

**Specifications:**
- **Size**: 6GB SQLite database
- **Scale**: 25M+ posts, 5M+ users, 500M+ interactions
- **Features**: 
  - Realistic user behavioral patterns (influencers, active users, casual users, lurkers, bots)
  - Complex social graph with power-law follower distributions
  - Temporal posting patterns and viral content cascades
  - Rich text content with hashtags, mentions, and realistic engagement
  - Network effects modeling for viral prediction testing

**Usage:**
```bash
cd tests/datasets
python social_media_generator.py
```

**Test Scenarios:**
- Content analysis and text processing performance
- User engagement pattern analysis
- Viral content prediction algorithms
- Complex JOIN operations across large tables
- String processing and search optimization
- Graph relationship traversals

## Requirements

Install required dependencies:
```bash
pip install -r requirements.txt
```

## Output Location

Generated datasets are saved to `tests/data/stress_test/` directory.

## Performance Expectations

The social media generator is designed to:
- Generate realistic data distributions (not random)
- Maintain referential integrity across all relationships
- Create complex query scenarios for testing
- Support performance benchmarking integration
- Include comprehensive error handling

**Estimated Generation Time:**
- 6GB Social Media Dataset: 30-60 minutes (depending on hardware)
- Memory usage: ~2-4GB during generation
- Requires ~15GB free disk space (for temporary files and final database)

## Integration with LocalData MCP Testing

These generators are specifically designed to work with the LocalData MCP stress testing framework, providing realistic datasets that challenge:

1. **Memory Management**: Large result sets that test streaming architecture
2. **Query Optimization**: Complex queries requiring intelligent execution planning  
3. **String Processing**: Text-heavy data for testing search and filter performance
4. **Concurrent Access**: Multiple simultaneous connections to large databases
5. **Resource Limits**: Operations that approach system boundaries gracefully

## Adding New Generators

To add a new dataset generator:

1. Create a new Python file following the pattern: `{domain}_generator.py`
2. Implement a generator class with realistic data patterns
3. Add comprehensive documentation and usage examples
4. Update `__init__.py` to export your generator
5. Add test scenarios to this README

## Troubleshooting

**Common Issues:**
- **Insufficient Disk Space**: Ensure at least 2x the target database size in free space
- **Memory Errors**: Reduce batch sizes in generator configuration
- **Long Generation Times**: Normal for large datasets; use verbose output to monitor progress
- **Database Locks**: Ensure no other processes are accessing the output database file

**Performance Optimization:**
- Run generators on fast storage (SSD preferred)
- Ensure adequate RAM (8GB+ recommended for large datasets)
- Close other applications to maximize available resources
- Use batch processing for extremely large datasets