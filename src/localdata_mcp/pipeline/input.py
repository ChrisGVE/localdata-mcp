"""
Input Stage Pipeline - Data Source to DataFrame Conversion

This module implements the input stage of the Core Pipeline Framework,
handling SQL/CSV/JSON → DataFrame conversion with streaming compatibility
and intention-driven interface design.

Key Features:
- Automatic data source detection and format conversion
- Streaming-compatible processing for large datasets
- LLM-friendly intention-driven configuration
- Rich metadata generation for downstream composition
"""

import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import pandas as pd
from sqlalchemy.engine import Engine

from .base import (
    AnalysisPipelineBase, 
    DataSourceType,
    StreamingConfig, 
    CompositionMetadata,
    PipelineError,
    ErrorClassification
)
from ..streaming_executor import (
    StreamingQueryExecutor,
    create_streaming_source,
    StreamingDataSource
)
from ..enhanced_database_manager import get_enhanced_database_manager
from ..logging_manager import get_logger

logger = get_logger(__name__)


class DataInputPipeline(AnalysisPipelineBase):
    """
    Input stage pipeline with streaming compatibility and LLM-friendly interface.
    
    First Principle: Intention-Driven Interface
    - LLM agents express what data they want, not how to get it
    - Automatic format detection and conversion
    - Intelligent streaming decisions based on data characteristics
    """
    
    def __init__(self, 
                 analytical_intention: str,
                 data_source: Union[str, Dict[str, Any], pd.DataFrame],
                 streaming_config: Optional[StreamingConfig] = None,
                 source_parameters: Optional[Dict[str, Any]] = None):
        """
        Initialize input pipeline with intention-driven parameters.
        
        Args:
            analytical_intention: Natural language description of analytical intent
            data_source: SQL query, file path, database connection, or DataFrame
            streaming_config: Configuration for streaming execution
            source_parameters: Additional parameters for data source (sheet_name, etc.)
        """
        super().__init__(
            analytical_intention=analytical_intention,
            streaming_config=streaming_config or StreamingConfig(),
            composition_aware=True
        )
        
        self.data_source = data_source
        self.source_parameters = source_parameters or {}
        
        # Internal state for streaming
        self._streaming_source: Optional[StreamingDataSource] = None
        self._data_source_type: Optional[DataSourceType] = None
        self._source_metadata: Dict[str, Any] = {}
        
        # Enhanced database manager for SQL sources
        self._db_manager = get_enhanced_database_manager()
        
        logger.info("DataInputPipeline initialized",
                   intention=analytical_intention,
                   source_type=type(data_source).__name__)
    
    def get_analysis_type(self) -> str:
        """Get the analysis type - input stage."""
        return "data_input"
    
    def _configure_analysis_pipeline(self) -> list:
        """Configure input processing steps."""
        return [
            self._detect_data_source_type,
            self._validate_data_source,
            self._create_streaming_source,
            self._extract_source_metadata
        ]
    
    def _execute_analysis_step(self, step, data, context) -> Tuple[Any, Dict[str, Any]]:
        """Execute individual input processing step."""
        step_name = step.__name__
        start_time = time.time()
        
        try:
            result = step(data, context)
            execution_time = time.time() - start_time
            
            metadata = {
                "step": step_name,
                "execution_time": execution_time,
                "success": True
            }
            
            return result, metadata
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Input step {step_name} failed: {e}")
            
            metadata = {
                "step": step_name,
                "execution_time": execution_time,
                "success": False,
                "error": str(e)
            }
            
            raise PipelineError(
                f"Input processing step '{step_name}' failed: {e}",
                ErrorClassification.CONFIGURATION_ERROR,
                "input_processing",
                context={"step": step_name, "data_source": str(self.data_source)}
            )
    
    def _execute_streaming_analysis(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Execute input processing with streaming support."""
        if self._streaming_source is None:
            raise PipelineError(
                "Streaming source not configured",
                ErrorClassification.CONFIGURATION_ERROR,
                "streaming_setup"
            )
        
        # Execute streaming data loading
        executor = StreamingQueryExecutor()
        query_id = f"input_pipeline_{int(time.time() * 1000)}"
        
        first_chunk, streaming_metadata = executor.execute_streaming(
            self._streaming_source, 
            query_id,
            initial_chunk_size=self.streaming_config.initial_chunk_size
        )
        
        # Build enriched metadata for composition
        enriched_metadata = self._build_input_metadata(first_chunk, streaming_metadata)
        
        return first_chunk, enriched_metadata
    
    def _execute_standard_analysis(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Execute input processing on full dataset in memory."""
        start_time = time.time()
        
        # If data_source is already a DataFrame, return it with metadata
        if isinstance(self.data_source, pd.DataFrame):
            result_data = self.data_source.copy()
        else:
            # Load data from source
            result_data = self._load_data_from_source()
        
        execution_time = time.time() - start_time
        
        # Build metadata
        metadata = {
            "data_source_type": self._data_source_type.value if self._data_source_type else "unknown",
            "execution_time": execution_time,
            "streaming_enabled": False,
            "rows_loaded": len(result_data),
            "columns_loaded": len(result_data.columns),
            "memory_usage_mb": result_data.memory_usage(deep=True).sum() / (1024 * 1024),
            "source_metadata": self._source_metadata
        }
        
        return result_data, metadata
    
    def _detect_data_source_type(self, data: Any, context: Dict[str, Any]) -> DataSourceType:
        """Detect the type of data source being used."""
        if isinstance(self.data_source, pd.DataFrame):
            self._data_source_type = DataSourceType.DATAFRAME
        elif isinstance(self.data_source, str):
            # Check if it's a file path or SQL query
            if self._is_file_path(self.data_source):
                file_ext = Path(self.data_source).suffix.lower()
                if file_ext == '.csv':
                    self._data_source_type = DataSourceType.CSV_FILE
                elif file_ext in ['.xlsx', '.xls']:
                    self._data_source_type = DataSourceType.EXCEL_FILE
                elif file_ext == '.json':
                    self._data_source_type = DataSourceType.JSON_FILE
                elif file_ext == '.parquet':
                    self._data_source_type = DataSourceType.PARQUET_FILE
                else:
                    raise ValueError(f"Unsupported file format: {file_ext}")
            else:
                # Assume it's a SQL query
                self._data_source_type = DataSourceType.SQL_QUERY
        elif isinstance(self.data_source, dict):
            # Database connection info
            self._data_source_type = DataSourceType.DATABASE_CONNECTION
        else:
            raise ValueError(f"Unsupported data source type: {type(self.data_source)}")
        
        logger.info(f"Detected data source type: {self._data_source_type.value}")
        return self._data_source_type
    
    def _validate_data_source(self, data: Any, context: Dict[str, Any]) -> bool:
        """Validate that the data source is accessible and valid."""
        try:
            if self._data_source_type == DataSourceType.DATAFRAME:
                if self.data_source.empty:
                    logger.warning("Input DataFrame is empty")
                return True
                
            elif self._data_source_type in [DataSourceType.CSV_FILE, DataSourceType.EXCEL_FILE, 
                                           DataSourceType.JSON_FILE, DataSourceType.PARQUET_FILE]:
                # Check file exists and is readable
                file_path = Path(self.data_source)
                if not file_path.exists():
                    raise FileNotFoundError(f"Data file not found: {self.data_source}")
                if not file_path.is_file():
                    raise ValueError(f"Path is not a file: {self.data_source}")
                return True
                
            elif self._data_source_type == DataSourceType.SQL_QUERY:
                # Validate SQL query format (basic check)
                query = str(self.data_source).strip().upper()
                if not any(query.startswith(cmd) for cmd in ['SELECT', 'WITH', 'SHOW', 'EXPLAIN']):
                    logger.warning("Data source may not be a valid SQL query")
                return True
                
            elif self._data_source_type == DataSourceType.DATABASE_CONNECTION:
                # Validate connection info has required fields
                required_fields = ['name', 'connection_string']
                missing = [field for field in required_fields if field not in self.data_source]
                if missing:
                    raise ValueError(f"Database connection missing required fields: {missing}")
                return True
                
            return True
            
        except Exception as e:
            raise PipelineError(
                f"Data source validation failed: {e}",
                ErrorClassification.CONFIGURATION_ERROR,
                "data_source_validation",
                context={"data_source_type": self._data_source_type.value}
            )
    
    def _create_streaming_source(self, data: Any, context: Dict[str, Any]) -> StreamingDataSource:
        """Create appropriate streaming data source."""
        try:
            if self._data_source_type == DataSourceType.DATAFRAME:
                # For DataFrames, we don't create a streaming source in the traditional sense
                # Instead, we'll chunk the DataFrame if it's large enough
                self._streaming_source = None  # Will be handled differently
                return None
                
            elif self._data_source_type in [DataSourceType.CSV_FILE, DataSourceType.EXCEL_FILE, 
                                           DataSourceType.JSON_FILE, DataSourceType.PARQUET_FILE]:
                # Create file streaming source
                file_type = self._data_source_type.value.replace('_file', '')
                self._streaming_source = create_streaming_source(
                    file_path=self.data_source,
                    file_type=file_type,
                    **self.source_parameters
                )
                
            elif self._data_source_type == DataSourceType.SQL_QUERY:
                # Need database connection for SQL queries
                # This would typically come from the connection manager
                engine = self._get_sql_engine()
                self._streaming_source = create_streaming_source(
                    engine=engine,
                    query=self.data_source
                )
                
            elif self._data_source_type == DataSourceType.DATABASE_CONNECTION:
                # Create database connection and then streaming source
                conn_info = self.data_source
                engine = self._create_database_connection(conn_info)
                query = conn_info.get('query', 'SELECT * FROM main_table LIMIT 1000')
                self._streaming_source = create_streaming_source(
                    engine=engine,
                    query=query
                )
            
            return self._streaming_source
            
        except Exception as e:
            raise PipelineError(
                f"Failed to create streaming source: {e}",
                ErrorClassification.EXTERNAL_DEPENDENCY_ERROR,
                "streaming_source_creation",
                context={"data_source_type": self._data_source_type.value}
            )
    
    def _extract_source_metadata(self, data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract metadata from the data source."""
        metadata = {
            "source_type": self._data_source_type.value,
            "analytical_intention": self.analytical_intention,
            "timestamp": time.time()
        }
        
        try:
            if self._data_source_type == DataSourceType.DATAFRAME:
                df = self.data_source
                metadata.update({
                    "shape": df.shape,
                    "columns": df.columns.tolist(),
                    "dtypes": dict(df.dtypes),
                    "memory_usage_mb": df.memory_usage(deep=True).sum() / (1024 * 1024)
                })
                
            elif self._data_source_type in [DataSourceType.CSV_FILE, DataSourceType.EXCEL_FILE, 
                                           DataSourceType.JSON_FILE, DataSourceType.PARQUET_FILE]:
                file_path = Path(self.data_source)
                metadata.update({
                    "file_path": str(file_path),
                    "file_size_mb": file_path.stat().st_size / (1024 * 1024),
                    "file_extension": file_path.suffix,
                    "source_parameters": self.source_parameters
                })
                
            elif self._data_source_type == DataSourceType.SQL_QUERY:
                metadata.update({
                    "query": self.data_source,
                    "query_length": len(str(self.data_source)),
                    "estimated_complexity": self._estimate_query_complexity(self.data_source)
                })
            
            self._source_metadata = metadata
            return metadata
            
        except Exception as e:
            logger.warning(f"Failed to extract source metadata: {e}")
            return metadata
    
    def _load_data_from_source(self) -> pd.DataFrame:
        """Load data from source when not using streaming."""
        if self._data_source_type == DataSourceType.DATAFRAME:
            return self.data_source.copy()
            
        elif self._data_source_type == DataSourceType.CSV_FILE:
            return pd.read_csv(self.data_source, **self.source_parameters)
            
        elif self._data_source_type == DataSourceType.EXCEL_FILE:
            return pd.read_excel(self.data_source, **self.source_parameters)
            
        elif self._data_source_type == DataSourceType.JSON_FILE:
            return pd.read_json(self.data_source, **self.source_parameters)
            
        elif self._data_source_type == DataSourceType.PARQUET_FILE:
            return pd.read_parquet(self.data_source, **self.source_parameters)
            
        elif self._data_source_type == DataSourceType.SQL_QUERY:
            engine = self._get_sql_engine()
            return pd.read_sql(self.data_source, engine)
            
        else:
            raise PipelineError(
                f"Cannot load data from source type: {self._data_source_type.value}",
                ErrorClassification.CONFIGURATION_ERROR,
                "data_loading"
            )
    
    def _build_input_metadata(self, sample_data: pd.DataFrame, streaming_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Build enriched metadata for composition and downstream tools."""
        base_metadata = {
            "input_pipeline": {
                "analytical_intention": self.analytical_intention,
                "data_source_type": self._data_source_type.value,
                "streaming_enabled": True,
                "sample_rows": len(sample_data),
                "total_columns": len(sample_data.columns)
            },
            "data_characteristics": {
                "columns": sample_data.columns.tolist(),
                "dtypes": dict(sample_data.dtypes),
                "null_counts": sample_data.isnull().sum().to_dict(),
                "numeric_columns": sample_data.select_dtypes(include=['number']).columns.tolist(),
                "categorical_columns": sample_data.select_dtypes(include=['object', 'category']).columns.tolist(),
                "datetime_columns": sample_data.select_dtypes(include=['datetime64']).columns.tolist()
            },
            "streaming_info": streaming_metadata,
            "source_metadata": self._source_metadata,
            "composition_context": {
                "ready_for_preprocessing": True,
                "ready_for_analysis": len(sample_data) > 0,
                "suggested_next_steps": self._suggest_next_pipeline_steps(sample_data),
                "data_quality_indicators": self._assess_data_quality(sample_data)
            }
        }
        
        return base_metadata
    
    def _suggest_next_pipeline_steps(self, data: pd.DataFrame) -> list:
        """Suggest appropriate next pipeline steps based on data characteristics."""
        suggestions = []
        
        # Check if preprocessing is needed
        null_percentage = (data.isnull().sum().sum() / (len(data) * len(data.columns))) * 100
        if null_percentage > 5:
            suggestions.append({
                "step": "preprocessing",
                "reason": f"Data has {null_percentage:.1f}% missing values",
                "recommendation": "DataPreprocessingPipeline with AUTO complexity"
            })
        
        # Suggest analysis type based on intention and data
        intention_lower = self.analytical_intention.lower()
        numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
        datetime_cols = data.select_dtypes(include=['datetime64']).columns.tolist()
        
        if datetime_cols and 'forecast' in intention_lower:
            suggestions.append({
                "step": "time_series_analysis",
                "reason": "DateTime columns detected with forecasting intention",
                "recommendation": "TimeSeriesAnalysisPipeline"
            })
        elif len(numeric_cols) >= 2 and 'correlation' in intention_lower:
            suggestions.append({
                "step": "statistical_analysis",
                "reason": "Multiple numeric columns with correlation intention",
                "recommendation": "StatisticalAnalysisPipeline"
            })
        elif 'predict' in intention_lower or 'classify' in intention_lower:
            suggestions.append({
                "step": "machine_learning",
                "reason": "Prediction/classification intention detected",
                "recommendation": "MachineLearningPipeline"
            })
        
        return suggestions
    
    def _assess_data_quality(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Assess data quality indicators."""
        return {
            "completeness_score": (1 - data.isnull().sum().sum() / (len(data) * len(data.columns))) * 100,
            "row_count": len(data),
            "column_count": len(data.columns),
            "duplicate_rows": data.duplicated().sum(),
            "memory_usage_mb": data.memory_usage(deep=True).sum() / (1024 * 1024)
        }
    
    # Utility methods
    def _is_file_path(self, source: str) -> bool:
        """Check if source string is a file path."""
        return (Path(source).exists() or 
                any(source.lower().endswith(ext) for ext in ['.csv', '.xlsx', '.xls', '.json', '.parquet']))
    
    def _get_sql_engine(self) -> Engine:
        """Get SQL engine from database manager."""
        # This would integrate with the connection manager
        # For now, return a placeholder - actual implementation would
        # use the enhanced database manager to get the appropriate engine
        raise NotImplementedError("SQL engine integration with connection manager needed")
    
    def _create_database_connection(self, conn_info: Dict[str, Any]) -> Engine:
        """Create database connection from connection info."""
        # Use enhanced database manager to create connection
        result = self._db_manager.connect_database_with_error_handling(
            name=conn_info['name'],
            db_type=conn_info.get('db_type', 'sqlite'),
            conn_string=conn_info['connection_string']
        )
        
        if not result['success']:
            raise PipelineError(
                f"Failed to create database connection: {result.get('error', {}).get('message', 'Unknown error')}",
                ErrorClassification.EXTERNAL_DEPENDENCY_ERROR,
                "database_connection"
            )
        
        # Return engine from connection manager
        # This is a placeholder - actual implementation would extract engine from result
        raise NotImplementedError("Database connection integration needed")
    
    def _estimate_query_complexity(self, query: str) -> str:
        """Estimate SQL query complexity."""
        query_upper = query.upper()
        
        # Simple heuristic based on keywords
        complex_keywords = ['JOIN', 'SUBQUERY', 'WINDOW', 'CTE', 'UNION', 'INTERSECT', 'EXCEPT']
        medium_keywords = ['GROUP BY', 'ORDER BY', 'HAVING', 'DISTINCT']
        
        if any(keyword in query_upper for keyword in complex_keywords):
            return "complex"
        elif any(keyword in query_upper for keyword in medium_keywords):
            return "medium"
        else:
            return "simple"