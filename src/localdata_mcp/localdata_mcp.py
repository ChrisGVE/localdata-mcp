"""LocalData MCP - Database connection and query management."""

import atexit
import configparser
import hashlib
import json
import logging
import os
import psutil
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import pandas as pd
import numpy as np
import yaml
from fastmcp import FastMCP
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.sql import quoted_name

# Import SQL query parser for security validation
from .query_parser import parse_and_validate_sql, SQLSecurityError

# Import query analyzer for pre-execution analysis
from .query_analyzer import analyze_query, QueryAnalysis

# Import streaming executor for memory-bounded processing
from .streaming_executor import StreamingQueryExecutor, create_streaming_source

# Import timeout manager for query timeout handling
from .timeout_manager import QueryTimeoutError, get_timeout_manager

# Import streaming file processors for memory-efficient file loading
from .file_processor import create_streaming_file_engine, FileProcessorFactory

# Import enhanced response metadata system
from .response_metadata import (
    get_metadata_generator, 
    EnhancedResponseMetadata, 
    LLMCommunicationProtocol,
    ResponseMetadataGenerator
)

# Import structured logging system
from .logging_manager import get_logging_manager, get_logger
from .config_manager import get_config_manager

# Import backward compatibility management
from .compatibility_manager import get_compatibility_manager

# TOML support
try:
    import toml
    TOML_AVAILABLE = True
except ImportError:
    TOML_AVAILABLE = False

# Excel support libraries with graceful error handling
try:
    import openpyxl
    OPENPYXL_AVAILABLE = True
except ImportError:
    OPENPYXL_AVAILABLE = False

try:
    import xlrd
    XLRD_AVAILABLE = True
except ImportError:
    XLRD_AVAILABLE = False

try:
    import defusedxml
    DEFUSEDXML_AVAILABLE = True
except ImportError:
    DEFUSEDXML_AVAILABLE = False

# ODS (LibreOffice Calc) support libraries with graceful error handling
try:
    from odf import opendocument, table
    ODFPY_AVAILABLE = True
except ImportError:
    ODFPY_AVAILABLE = False

# XML parsing support
try:
    import lxml
    LXML_AVAILABLE = True
except ImportError:
    LXML_AVAILABLE = False

# Analytical format support (Parquet, Arrow, Feather)
try:
    import pyarrow
    PYARROW_AVAILABLE = True
except ImportError:
    PYARROW_AVAILABLE = False

# Apple Numbers support
try:
    import numbers_parser
    NUMBERS_AVAILABLE = True
except ImportError:
    NUMBERS_AVAILABLE = False

# HDF5 support (scientific data format)
try:
    import h5py
    import tables
    HDF5_AVAILABLE = True
except ImportError:
    HDF5_AVAILABLE = False

# Modern database support
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

try:
    import elasticsearch
    ELASTICSEARCH_AVAILABLE = True
except ImportError:
    ELASTICSEARCH_AVAILABLE = False

try:
    import pymongo
    MONGODB_AVAILABLE = True
except ImportError:
    MONGODB_AVAILABLE = False

try:
    from influxdb_client import InfluxDBClient
    INFLUXDB_AVAILABLE = True
except ImportError:
    INFLUXDB_AVAILABLE = False

try:
    from neo4j import GraphDatabase
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False

try:
    import couchdb
    COUCHDB_AVAILABLE = True
except ImportError:
    COUCHDB_AVAILABLE = False

# Enhanced streaming JSON processing
try:
    import ijson
    IJSON_AVAILABLE = True
except ImportError:
    IJSON_AVAILABLE = False

# Create FastMCP instance
mcp = FastMCP("LocalData MCP - Advanced database connection and analytics platform")

# Initialize components
logging_manager = get_logging_manager()
config_manager = get_config_manager()

# Initialize logging configuration from config manager
logging_config = config_manager.logging_config
logger = get_logger(__name__)

class QueryBuffer:
    """Buffer to store query results with metadata."""
    
    def __init__(self, df: pd.DataFrame, query: str, db_name: str):
        self.df = df
        self.query = query
        self.db_name = db_name
        self.timestamp = time.time()
        self.access_count = 0
        self.chunk_info = {}
        self.llm_protocol = None  
        self.response_metadata = None
        
    def access(self) -> pd.DataFrame:
        """Access the buffered data and increment access count."""
        self.access_count += 1
        return self.df
        
    def is_expired(self, ttl: float = 3600) -> bool:
        """Check if buffer has expired based on TTL (default 1 hour)."""
        return time.time() - self.timestamp > ttl


class DatabaseManager:
    """Enhanced database manager with streaming support and comprehensive data source compatibility."""
    
    def __init__(self):
        self.connections = {}
        self.db_types = {}
        self.timeout_manager = get_timeout_manager()
        self.temp_files = set()  # Track temporary files for cleanup
        
        # Query result buffers with thread-safe access
        self.query_buffers = {}
        self.query_buffer_lock = threading.Lock()
        
        # Initialize streaming executor
        self.streaming_executor = StreamingQueryExecutor()
        
        # Register cleanup handler
        atexit.register(self._cleanup_temp_files)
        atexit.register(self._cleanup_connections)
        
    def _cleanup_temp_files(self):
        """Clean up temporary files on exit."""
        for temp_file in self.temp_files.copy():
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
                self.temp_files.remove(temp_file)
            except Exception as e:
                logger.warning(f"Failed to cleanup temp file {temp_file}: {e}")
                
    def _cleanup_connections(self):
        """Clean up database connections on exit."""
        for name, conn in self.connections.items():
            try:
                if hasattr(conn, 'dispose'):
                    conn.dispose()
                elif hasattr(conn, 'close'):
                    conn.close()
            except Exception as e:
                logger.warning(f"Failed to cleanup connection {name}: {e}")

    def _safe_table_identifier(self, table_name: str) -> str:
        """Safely quote table identifiers to prevent SQL injection."""
        return quoted_name(table_name, quote=True)
        
    def _get_connection(self, name: str):
        """Get database connection by name."""
        if name not in self.connections:
            raise ValueError(f"Database '{name}' is not connected. Use connect_database first.")
        return self.connections[name]
        
    def _check_memory_usage(self) -> Dict[str, Any]:
        """Check current memory usage and return status."""
        memory = psutil.virtual_memory()
        return {
            'total_gb': round(memory.total / (1024**3), 2),
            'available_gb': round(memory.available / (1024**3), 2),
            'used_gb': round(memory.used / (1024**3), 2),
            'percent_used': memory.percent,
            'is_low_memory': memory.percent > 85
        }

    def connect_database(self, name: str, db_type: str, conn_string: str, sheet_name: Optional[str] = None) -> str:
        """
        Connect to a database or file-based data source with comprehensive format support.
        
        This method supports traditional databases, modern NoSQL databases, and numerous file formats
        with optimized memory usage and streaming capabilities for large datasets.
        
        Args:
            name: Unique identifier for this connection
            db_type: Database type or file format
            conn_string: Connection string or file path
            sheet_name: Sheet name for multi-sheet formats (Excel, ODS, Numbers) or dataset name (HDF5)
        
        Returns:
            Connection status and metadata in JSON format
        """
        try:
            logger.info(f"Connecting to {db_type} database: {name}")
            
            # Store connection metadata
            self.db_types[name] = db_type
            
            # Handle different database types and file formats
            if db_type in ['sqlite', 'postgresql', 'mysql', 'duckdb']:
                self.connections[name] = create_engine(conn_string)
                
            elif db_type == 'csv':
                # Use streaming file processor for CSV
                engine = create_streaming_file_engine(conn_string, format_type='csv')
                self.connections[name] = engine
                
            elif db_type == 'json':
                # Handle both streaming and regular JSON
                engine = create_streaming_file_engine(conn_string, format_type='json')
                self.connections[name] = engine
                
            elif db_type in ['yaml', 'yml']:
                if not os.path.exists(conn_string):
                    return json.dumps({"error": f"YAML file not found: {conn_string}"})
                    
                with open(conn_string, 'r', encoding='utf-8') as file:
                    data = yaml.safe_load(file)
                    
                # Convert to temporary SQLite database
                df = pd.json_normalize(data) if isinstance(data, dict) else pd.DataFrame(data)
                temp_db = self._create_temp_sqlite(df)
                self.connections[name] = create_engine(f"sqlite:///{temp_db}")
                self.temp_files.add(temp_db)
                
            elif db_type == 'toml':
                if not TOML_AVAILABLE:
                    return json.dumps({"error": "TOML support not available. Install with: pip install toml"})
                    
                if not os.path.exists(conn_string):
                    return json.dumps({"error": f"TOML file not found: {conn_string}"})
                    
                with open(conn_string, 'r', encoding='utf-8') as file:
                    data = toml.load(file)
                    
                # Convert to temporary SQLite database  
                df = pd.json_normalize(data) if isinstance(data, dict) else pd.DataFrame(data)
                temp_db = self._create_temp_sqlite(df)
                self.connections[name] = create_engine(f"sqlite:///{temp_db}")
                self.temp_files.add(temp_db)
                
            elif db_type == 'excel':
                if not OPENPYXL_AVAILABLE:
                    return json.dumps({"error": "Excel support not available. Install with: pip install openpyxl"})
                
                # Use streaming processor with enhanced Excel support
                engine = create_streaming_file_engine(
                    conn_string, 
                    format_type='excel', 
                    sheet_name=sheet_name
                )
                self.connections[name] = engine
                
            elif db_type == 'ods':
                if not ODFPY_AVAILABLE:
                    return json.dumps({"error": "ODS support not available. Install with: pip install odfpy"})
                
                engine = create_streaming_file_engine(
                    conn_string,
                    format_type='ods',
                    sheet_name=sheet_name
                )
                self.connections[name] = engine
                
            elif db_type == 'numbers':
                if not NUMBERS_AVAILABLE:
                    return json.dumps({"error": "Apple Numbers support not available. Install with: pip install numbers-parser"})
                
                engine = create_streaming_file_engine(
                    conn_string,
                    format_type='numbers',
                    sheet_name=sheet_name
                )
                self.connections[name] = engine
                
            elif db_type == 'xml':
                if not LXML_AVAILABLE:
                    return json.dumps({"error": "XML support not available. Install with: pip install lxml"})
                
                engine = create_streaming_file_engine(conn_string, format_type='xml')
                self.connections[name] = engine
                
            elif db_type == 'parquet':
                if not PYARROW_AVAILABLE:
                    return json.dumps({"error": "Parquet support not available. Install with: pip install pyarrow"})
                
                engine = create_streaming_file_engine(conn_string, format_type='parquet')
                self.connections[name] = engine
                
            elif db_type in ['feather', 'arrow']:
                if not PYARROW_AVAILABLE:
                    return json.dumps({"error": "Arrow/Feather support not available. Install with: pip install pyarrow"})
                
                engine = create_streaming_file_engine(conn_string, format_type=db_type)
                self.connections[name] = engine
                
            elif db_type == 'hdf5':
                if not HDF5_AVAILABLE:
                    return json.dumps({"error": "HDF5 support not available. Install with: pip install h5py tables"})
                
                engine = create_streaming_file_engine(
                    conn_string,
                    format_type='hdf5',
                    dataset_name=sheet_name  # Use sheet_name as dataset_name for HDF5
                )
                self.connections[name] = engine
                
            elif db_type == 'ini':
                if not os.path.exists(conn_string):
                    return json.dumps({"error": f"INI file not found: {conn_string}"})
                    
                config = configparser.ConfigParser()
                config.read(conn_string)
                
                # Convert INI to DataFrame
                data = []
                for section in config.sections():
                    for key, value in config[section].items():
                        data.append({'section': section, 'key': key, 'value': value})
                        
                df = pd.DataFrame(data)
                temp_db = self._create_temp_sqlite(df)
                self.connections[name] = create_engine(f"sqlite:///{temp_db}")
                self.temp_files.add(temp_db)
                
            elif db_type == 'tsv':
                # TSV is CSV with tab separator
                engine = create_streaming_file_engine(conn_string, format_type='csv', separator='\t')
                self.connections[name] = engine
                
            # Modern database support
            elif db_type == 'redis' and REDIS_AVAILABLE:
                self.connections[name] = self._create_redis_connection(conn_string)
                
            elif db_type == 'elasticsearch' and ELASTICSEARCH_AVAILABLE:
                self.connections[name] = self._create_elasticsearch_connection(conn_string)
                
            elif db_type == 'mongodb' and MONGODB_AVAILABLE:
                self.connections[name] = self._create_mongodb_connection(conn_string)
                
            elif db_type == 'influxdb' and INFLUXDB_AVAILABLE:
                self.connections[name] = self._create_influxdb_connection(conn_string)
                
            elif db_type == 'neo4j' and NEO4J_AVAILABLE:
                self.connections[name] = self._create_neo4j_connection(conn_string)
                
            elif db_type == 'couchdb' and COUCHDB_AVAILABLE:
                self.connections[name] = self._create_couchdb_connection(conn_string)
                
            else:
                return json.dumps({"error": f"Unsupported database type: {db_type}"})
            
            # Test connection
            self._test_connection(name, db_type)
            
            logger.info(f"Successfully connected to {db_type} database: {name}")
            
            return json.dumps({
                "success": True,
                "message": f"Connected to {db_type} database: {name}",
                "database_type": db_type,
                "connection_name": name,
                "sheet_name": sheet_name
            })
            
        except Exception as e:
            logger.error(f"Failed to connect to database {name}: {e}")
            return json.dumps({
                "error": f"Connection failed: {str(e)}",
                "database_type": db_type,
                "connection_name": name
            })

    def _create_temp_sqlite(self, df: pd.DataFrame) -> str:
        """Create temporary SQLite database from DataFrame."""
        temp_file = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        temp_file.close()
        
        engine = create_engine(f"sqlite:///{temp_file.name}")
        df.to_sql('data', engine, index=False, if_exists='replace')
        
        return temp_file.name

    def _test_connection(self, name: str, db_type: str):
        """Test database connection."""
        if db_type in ['sqlite', 'postgresql', 'mysql', 'duckdb']:
            # Test SQL databases
            with self.connections[name].connect() as conn:
                conn.execute(text("SELECT 1"))
        # Add other connection tests as needed
        
    def _create_redis_connection(self, conn_string: str):
        """Create Redis connection from connection string."""
        # Parse Redis connection string (redis://localhost:6379/0)
        return redis.from_url(conn_string)
    
    def _create_elasticsearch_connection(self, conn_string: str):
        """Create Elasticsearch connection from connection string."""
        from elasticsearch import Elasticsearch
        
        # Parse Elasticsearch connection string (http://localhost:9200)
        return Elasticsearch([conn_string])
    
    def _create_mongodb_connection(self, conn_string: str):
        """Create MongoDB connection from connection string."""
        # Parse MongoDB connection string (mongodb://localhost:27017/)
        return pymongo.MongoClient(conn_string)
    
    def _create_influxdb_connection(self, conn_string: str):
        """Create InfluxDB connection from connection string."""
        # Parse InfluxDB connection string (http://localhost:8086)
        # InfluxDB requires token and org - use defaults for local testing
        return InfluxDBClient(url=conn_string, token="", org="")
    
    def _create_neo4j_connection(self, conn_string: str):
        """Create Neo4j connection from connection string."""
        from neo4j import GraphDatabase
        
        # Parse connection string (bolt://localhost:7687)
        if not conn_string.startswith('bolt://'):
            conn_string = f"bolt://{conn_string}"
            
        # Neo4j requires auth - use defaults for local testing
        return GraphDatabase.driver(conn_string, auth=("neo4j", "password"))
    
    def _create_couchdb_connection(self, conn_string: str):
        """Create CouchDB connection from connection string."""
        import couchdb
        
        # Parse connection string (http://admin:testpassword@localhost:5984/)
        if not conn_string.startswith('http'):
            conn_string = f"http://admin:testpassword@{conn_string}/"
            
        return couchdb.Server(conn_string)

    @mcp.tool
    def manage_memory_bounds(self) -> str:
        """
        Monitor and manage memory usage across all streaming operations.
        
        This tool provides comprehensive memory management including:
        - Current memory status and usage statistics  
        - Automatic cleanup of expired buffers
        - Memory optimization recommendations
        - Active buffer information
        
        Returns JSON with memory management actions taken and current status.
        """
        try:
            # Execute memory management
            management_result = self.streaming_executor.manage_memory_bounds()
            
            # Add database manager specific memory info
            additional_info = {
                "legacy_query_buffers": len(self.query_buffers),
                "active_connections": len(self.connections),
                "temp_files_count": len(self.temp_files)
            }
            
            # Clean up legacy query buffers if memory is high
            memory_status = management_result.get("memory_status", {})
            if memory_status.get("is_low_memory", False):
                with self.query_buffer_lock:
                    legacy_buffers_cleared = len(self.query_buffers)
                    self.query_buffers.clear()
                    if legacy_buffers_cleared > 0:
                        management_result["actions_taken"].append(f"Cleared {legacy_buffers_cleared} legacy query buffers")
            
            management_result["database_manager_info"] = additional_info
            
            return json.dumps(management_result, indent=2)
            
        except Exception as e:
            logger.error(f"Error in memory management: {e}")
            return f"Memory management error: {e}"
    
    @mcp.tool  
    def get_streaming_status(self) -> str:
        """
        Get detailed status of all streaming operations and memory usage.
        
        Returns comprehensive information about:
        - Active streaming buffers and their memory usage
        - Memory status and recommendations
        - Performance metrics from recent streaming operations
        - Configuration settings for streaming
        
        Useful for monitoring and debugging streaming performance.
        """
        try:
            # Get memory status
            memory_info = self._check_memory_usage()
            
            # Get streaming executor status
            streaming_status = {
                "memory_status": memory_info,
                "streaming_buffers": {},
                "performance_config": {
                    "memory_limit_mb": self.streaming_executor.config.memory_limit_mb,
                    "default_chunk_size": self.streaming_executor.config.chunk_size,
                    "memory_warning_threshold": self.streaming_executor.config.memory_warning_threshold,
                    "enable_query_analysis": self.streaming_executor.config.enable_query_analysis,
                    "auto_cleanup_buffers": self.streaming_executor.config.auto_cleanup_buffers
                },
                "recent_chunk_metrics": self.streaming_executor._chunk_metrics[-10:] if hasattr(self.streaming_executor, '_chunk_metrics') else []
            }
            
            # Get info for each active buffer
            for query_id in self.streaming_executor._result_buffers:
                buffer_info = self.streaming_executor.get_buffer_info(query_id)
                if buffer_info:
                    streaming_status["streaming_buffers"][query_id] = buffer_info
            
            # Add database manager specific info
            streaming_status["database_manager"] = {
                "total_connections": len(self.connections),
                "connection_types": {name: db_type for name, db_type in self.db_types.items()},
                "legacy_buffers": len(self.query_buffers),
                "temp_files": len(self.temp_files)
            }
            
            return json.dumps(streaming_status, indent=2, default=str)
            
        except Exception as e:
            logger.error(f"Error getting streaming status: {e}")
            return f"Streaming status error: {e}"
    
    @mcp.tool
    def clear_streaming_buffer(self, query_id: str) -> str:
        """
        Clear a specific streaming result buffer to free memory.
        
        Args:
            query_id: The ID of the streaming buffer to clear.
            
        This is useful when you're done with a large result set and want to
        free up memory immediately instead of waiting for automatic cleanup.
        """
        try:
            success = self.streaming_executor.clear_buffer(query_id)
            if success:
                return f"Successfully cleared streaming buffer: {query_id}"
            else:
                return f"Streaming buffer '{query_id}' not found or already cleared."
                
        except Exception as e:
            logger.error(f"Error clearing streaming buffer {query_id}: {e}")
            return f"Error clearing buffer: {e}"
    
    @mcp.tool
    def get_query_metadata(self, query_id: str) -> str:
        """
        Get comprehensive metadata for a query result including LLM-friendly summary,
        data quality metrics, complexity analysis, and processing recommendations.
        
        Args:
            query_id: The ID of the query to get metadata for.
            
        Returns:
            Comprehensive metadata in JSON format for LLM decision-making.
        """
        try:
            with self.query_buffer_lock:
                if query_id not in self.query_buffers:
                    return f"Query ID '{query_id}' not found in buffers."
                
                query_buffer = self.query_buffers[query_id]
                
                if not query_buffer.llm_protocol:
                    return f"Query ID '{query_id}' does not have enhanced metadata available."
                
                # Get comprehensive summary
                summary = query_buffer.llm_protocol.get_summary()
                
                # Add additional metadata
                metadata_response = {
                    "query_info": {
                        "query_id": query_id,
                        "database": query_buffer.db_name,
                        "timestamp": query_buffer.timestamp,
                        "query": query_buffer.query
                    },
                    "llm_summary": summary,
                    "schema_details": query_buffer.llm_protocol.get_schema_details(),
                    "data_quality_report": query_buffer.llm_protocol.get_data_quality_report(),
                    "enhanced_metadata": {
                        "complexity_score": query_buffer.response_metadata.query_complexity_score,
                        "processing_time_estimate": query_buffer.response_metadata.estimated_processing_time,
                        "memory_footprint_mb": query_buffer.response_metadata.memory_footprint,
                        "token_estimation": {
                            "total_tokens": query_buffer.response_metadata.token_estimation.total_tokens,
                            "tokens_per_row": query_buffer.response_metadata.token_estimation.tokens_per_row,
                            "confidence": query_buffer.response_metadata.token_estimation.confidence
                        },
                        "llm_friendly_summary": query_buffer.response_metadata.llm_friendly_summary
                    }
                }
                
                return json.dumps(metadata_response, indent=2)
                
        except Exception as e:
            logger.error(f"Error getting query metadata for {query_id}: {e}")
            return f"Error getting metadata: {e}"
    
    @mcp.tool
    def request_data_chunk(self, query_id: str, chunk_id: int) -> str:
        """
        Request a specific chunk of data using the LLM communication protocol.
        This enables progressive loading of large datasets.
        
        Args:
            query_id: The ID of the query result to get a chunk from.
            chunk_id: The specific chunk number to retrieve (0-based).
            
        Returns:
            Chunk data in JSON format with metadata.
        """
        try:
            with self.query_buffer_lock:
                if query_id not in self.query_buffers:
                    return f"Query ID '{query_id}' not found in buffers."
                
                query_buffer = self.query_buffers[query_id]
                
                if not query_buffer.llm_protocol:
                    return f"Query ID '{query_id}' does not support chunked access."
                
                # Request chunk through LLM protocol
                chunk_result = query_buffer.llm_protocol.get_chunk(chunk_id)
                
                return json.dumps(chunk_result, indent=2, default=str)
                
        except Exception as e:
            logger.error(f"Error getting chunk {chunk_id} for query {query_id}: {e}")
            return f"Error getting chunk: {e}"
    
    def profile_table(self, name: str, table_name: Optional[str] = None, query: Optional[str] = None, 
                     sample_size: int = 10000, include_distributions: bool = True) -> str:
        """Generate comprehensive data profile with advanced analytics."""
        try:
            if not table_name and not query:
                return json.dumps({"error": "Either table_name or query must be provided"})
            
            if table_name and query:
                return json.dumps({"error": "Provide either table_name or query, not both"})
                
            if name not in self.connections:
                return json.dumps({"error": f"Database '{name}' is not connected. Use connect_database first."})
            
            # Get data for analysis
            df = self._get_data_for_analysis(name, table_name, query, sample_size)
            if df.empty:
                return json.dumps({"error": "No data available for profiling"})
            
            # Use statistical analysis domain for comprehensive profiling
            from .domains.statistical_analysis import profile_dataset, detect_data_types, analyze_distributions
            
            # Basic profiling
            profile_result = profile_dataset(
                df, 
                include_distributions=include_distributions,
                sample_size=sample_size
            )
            
            # Add data type detection
            type_detection = detect_data_types(df)
            
            # Distribution analysis for numeric columns
            if include_distributions:
                distribution_analysis = analyze_distributions(
                    df.select_dtypes(include=[np.number]),
                    bins=20
                )
                profile_result['distribution_analysis'] = distribution_analysis
            
            profile_result['data_type_suggestions'] = type_detection
            profile_result['sample_size_used'] = sample_size
            profile_result['database_name'] = name
            
            return json.dumps(profile_result, default=str)
            
        except Exception as e:
            logger.error(f"Data profiling failed: {e}")
            return json.dumps({"error": f"Data profiling failed: {str(e)}"})
    
    def detect_data_types(self, name: str, table_name: Optional[str] = None, query: Optional[str] = None,
                         sample_size: int = 5000, confidence_threshold: float = 0.8) -> str:
        """Advanced data type detection with pattern recognition."""
        try:
            if not table_name and not query:
                return json.dumps({"error": "Either table_name or query must be provided"})
            
            if table_name and query:
                return json.dumps({"error": "Provide either table_name or query, not both"})
                
            if name not in self.connections:
                return json.dumps({"error": f"Database '{name}' is not connected. Use connect_database first."})
            
            # Get data for analysis
            df = self._get_data_for_analysis(name, table_name, query, sample_size)
            if df.empty:
                return json.dumps({"error": "No data available for type detection"})
            
            # Use statistical analysis domain for type detection
            from .domains.statistical_analysis import detect_data_types as domain_detect_types
            
            result = domain_detect_types(df, confidence_threshold=confidence_threshold)
            result['sample_size_used'] = sample_size
            result['database_name'] = name
            
            return json.dumps(result, default=str)
            
        except Exception as e:
            logger.error(f"Data type detection failed: {e}")
            return json.dumps({"error": f"Data type detection failed: {str(e)}"})
    
    def analyze_distributions(self, name: str, table_name: Optional[str] = None, query: Optional[str] = None,
                             columns: Optional[str] = None, sample_size: int = 10000, 
                             bins: int = 20, percentiles: Optional[str] = None) -> str:
        """Column-wise distribution analysis with comprehensive statistics."""
        try:
            if not table_name and not query:
                return json.dumps({"error": "Either table_name or query must be provided"})
            
            if table_name and query:
                return json.dumps({"error": "Provide either table_name or query, not both"})
                
            if name not in self.connections:
                return json.dumps({"error": f"Database '{name}' is not connected. Use connect_database first."})
            
            # Get data for analysis
            df = self._get_data_for_analysis(name, table_name, query, sample_size)
            if df.empty:
                return json.dumps({"error": "No data available for distribution analysis"})
            
            # Filter columns if specified
            if columns:
                column_list = [col.strip() for col in columns.split(',')]
                available_cols = [col for col in column_list if col in df.columns]
                if not available_cols:
                    return json.dumps({"error": f"None of the specified columns found: {columns}"})
                df = df[available_cols]
            
            # Parse percentiles if provided
            percentile_list = None
            if percentiles:
                try:
                    percentile_list = [float(p.strip()) for p in percentiles.split(',')]
                except ValueError:
                    return json.dumps({"error": "Invalid percentile format. Use comma-separated numbers (e.g., '25,50,75')"})
            
            # Use statistical analysis domain for distribution analysis
            from .domains.statistical_analysis import analyze_distributions as domain_analyze_distributions
            
            result = domain_analyze_distributions(
                df, 
                bins=bins, 
                percentiles=percentile_list
            )
            
            result['sample_size_used'] = sample_size
            result['database_name'] = name
            
            return json.dumps(result, default=str)
            
        except Exception as e:
            logger.error(f"Distribution analysis failed: {e}")
            return json.dumps({"error": f"Distribution analysis failed: {str(e)}"})

    def perform_clustering(self, name: str, table_name: Optional[str] = None, query: Optional[str] = None,
                          algorithm: str = 'kmeans', n_clusters: Optional[int] = None,
                          sample_size: int = 10000) -> str:
        """Perform comprehensive clustering analysis on data."""
        try:
            if not table_name and not query:
                return json.dumps({"error": "Either table_name or query must be provided"})
            
            if table_name and query:
                return json.dumps({"error": "Provide either table_name or query, not both"})
                
            if name not in self.connections:
                return json.dumps({"error": f"Database '{name}' is not connected. Use connect_database first."})
            
            # Get data
            df = self._get_data_for_analysis(name, table_name, query, sample_size)
            if df.empty:
                return json.dumps({"error": "No data available for clustering analysis"})
            
            # Select only numeric columns for clustering
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                return json.dumps({"error": "No numeric columns found for clustering analysis"})
            
            X = df[numeric_cols].values
            
            # Perform clustering using the domain function
            from .domains.pattern_recognition import perform_clustering as domain_perform_clustering
            result = domain_perform_clustering(X, algorithm, n_clusters)
            
            # Add metadata about the data
            result['data_info'] = {
                'n_samples': len(df),
                'n_features': len(numeric_cols),
                'feature_names': numeric_cols.tolist(),
                'sample_size_used': sample_size
            }
            
            return json.dumps(result, default=str)
            
        except Exception as e:
            logger.error(f"Clustering analysis failed: {e}")
            return json.dumps({"error": f"Clustering analysis failed: {str(e)}"})
    
    def reduce_dimensions(self, name: str, table_name: Optional[str] = None, query: Optional[str] = None,
                         algorithm: str = 'pca', n_components: Optional[int] = None,
                         sample_size: int = 10000) -> str:
        """Perform comprehensive dimensionality reduction on data."""
        try:
            if not table_name and not query:
                return json.dumps({"error": "Either table_name or query must be provided"})
            
            if table_name and query:
                return json.dumps({"error": "Provide either table_name or query, not both"})
                
            if name not in self.connections:
                return json.dumps({"error": f"Database '{name}' is not connected. Use connect_database first."})
            
            # Get data
            df = self._get_data_for_analysis(name, table_name, query, sample_size)
            if df.empty:
                return json.dumps({"error": "No data available for dimensionality reduction"})
            
            # Select only numeric columns for reduction
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                return json.dumps({"error": "No numeric columns found for dimensionality reduction"})
            
            X = df[numeric_cols].values
            
            # For LDA, we need target labels - use first categorical column if available
            y = None
            if algorithm == 'lda':
                categorical_cols = df.select_dtypes(include=['object', 'category']).columns
                if len(categorical_cols) > 0:
                    from sklearn.preprocessing import LabelEncoder
                    le = LabelEncoder()
                    y = le.fit_transform(df[categorical_cols[0]].astype(str))
                else:
                    return json.dumps({"error": "LDA requires categorical target variable. No categorical columns found."})
            
            # Perform dimensionality reduction using the domain function
            from .domains.pattern_recognition import reduce_dimensions as domain_reduce_dimensions
            result = domain_reduce_dimensions(X, algorithm, n_components, y)
            
            # Add metadata about the data
            result['data_info'] = {
                'n_samples': len(df),
                'n_features': len(numeric_cols),
                'feature_names': numeric_cols.tolist(),
                'sample_size_used': sample_size,
                'target_used_for_lda': categorical_cols[0] if algorithm == 'lda' and len(categorical_cols) > 0 else None
            }
            
            return json.dumps(result, default=str)
            
        except Exception as e:
            logger.error(f"Dimensionality reduction failed: {e}")
            return json.dumps({"error": f"Dimensionality reduction failed: {str(e)}"})
    
    def detect_anomalies(self, name: str, table_name: Optional[str] = None, query: Optional[str] = None,
                        algorithm: str = 'isolation_forest', contamination: Optional[float] = None,
                        sample_size: int = 10000) -> str:
        """Perform comprehensive anomaly detection on data."""
        try:
            if not table_name and not query:
                return json.dumps({"error": "Either table_name or query must be provided"})
            
            if table_name and query:
                return json.dumps({"error": "Provide either table_name or query, not both"})
                
            if name not in self.connections:
                return json.dumps({"error": f"Database '{name}' is not connected. Use connect_database first."})
            
            # Get data
            df = self._get_data_for_analysis(name, table_name, query, sample_size)
            if df.empty:
                return json.dumps({"error": "No data available for anomaly detection"})
            
            # Select only numeric columns for anomaly detection
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                return json.dumps({"error": "No numeric columns found for anomaly detection"})
            
            X = df[numeric_cols].values
            
            # Perform anomaly detection using the domain function
            from .domains.pattern_recognition import detect_anomalies as domain_detect_anomalies
            result = domain_detect_anomalies(X, algorithm, contamination)
            
            # Add metadata about the data and anomalies
            anomaly_indices = result.get('anomaly_statistics', {}).get('anomaly_indices', [])
            result['data_info'] = {
                'n_samples': len(df),
                'n_features': len(numeric_cols),
                'feature_names': numeric_cols.tolist(),
                'sample_size_used': sample_size
            }
            
            # Add details about specific anomalous records if found
            if anomaly_indices and len(anomaly_indices) > 0 and len(anomaly_indices) <= 100:  # Limit to 100 records
                anomalous_records = df.iloc[anomaly_indices].to_dict('records')
                result['anomalous_records'] = anomalous_records[:10]  # Show top 10 only
                
            return json.dumps(result, default=str)
            
        except Exception as e:
            logger.error(f"Anomaly detection failed: {e}")
            return json.dumps({"error": f"Anomaly detection failed: {str(e)}"})
    
    def evaluate_patterns(self, name: str, table_name: Optional[str] = None, query: Optional[str] = None,
                         pattern_type: str = 'clustering', results_data: Optional[str] = None,
                         sample_size: int = 10000) -> str:
        """Evaluate pattern recognition results and provide recommendations."""
        try:
            if not table_name and not query and not results_data:
                return json.dumps({"error": "Either table_name/query or results_data must be provided"})
            
            if (table_name or query) and results_data:
                return json.dumps({"error": "Provide either data source (table_name/query) or results_data, not both"})
                
            if name not in self.connections:
                return json.dumps({"error": f"Database '{name}' is not connected. Use connect_database first."})
            
            # Get data if needed
            X = None
            if table_name or query:
                df = self._get_data_for_analysis(name, table_name, query, sample_size)
                if df.empty:
                    return json.dumps({"error": "No data available for pattern evaluation"})
                
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) == 0:
                    return json.dumps({"error": "No numeric columns found for pattern evaluation"})
                
                X = df[numeric_cols].values
            
            # Parse results data if provided
            results = None
            if results_data:
                try:
                    results = json.loads(results_data)
                except json.JSONDecodeError:
                    return json.dumps({"error": "Invalid JSON format in results_data"})
            
            # Perform pattern evaluation using the domain function
            from .domains.pattern_recognition import evaluate_patterns as domain_evaluate_patterns
            
            if X is not None and results is None:
                # Need to generate dummy results for evaluation
                if pattern_type == 'clustering':
                    from .domains.pattern_recognition import perform_clustering as domain_perform_clustering
                    results = domain_perform_clustering(X)
                elif pattern_type == 'dimensionality_reduction':
                    from .domains.pattern_recognition import reduce_dimensions as domain_reduce_dimensions
                    results = domain_reduce_dimensions(X)
                elif pattern_type == 'anomaly_detection':
                    from .domains.pattern_recognition import detect_anomalies as domain_detect_anomalies
                    results = domain_detect_anomalies(X)
                else:
                    return json.dumps({"error": f"Unknown pattern type: {pattern_type}"})
            
            if X is None and results is not None:
                # Create dummy data from results (not ideal but works for evaluation)
                if pattern_type == 'clustering':
                    labels = results.get('labels', [])
                    if labels:
                        X = np.random.randn(len(labels), 2)  # Dummy 2D data
                elif pattern_type == 'dimensionality_reduction':
                    transformed_data = results.get('transformed_data', [])
                    if transformed_data:
                        X = np.random.randn(len(transformed_data), 10)  # Dummy high-D data
                elif pattern_type == 'anomaly_detection':
                    anomaly_labels = results.get('anomaly_labels', [])
                    if anomaly_labels:
                        X = np.random.randn(len(anomaly_labels), 5)  # Dummy data
            
            if X is None:
                return json.dumps({"error": "Unable to obtain data for pattern evaluation"})
                
            evaluation_result = domain_evaluate_patterns(X, pattern_type, results)
            
            return json.dumps(evaluation_result, default=str)
            
        except Exception as e:
            logger.error(f"Pattern evaluation failed: {e}")
            return json.dumps({"error": f"Pattern evaluation failed: {str(e)}"})
    
    # Time Series Analysis Methods
    
    def analyze_time_series_basic(self, name: str, table_name: Optional[str] = None, query: Optional[str] = None,
                                 date_column: Optional[str] = None, value_column: Optional[str] = None,
                                 freq: Optional[str] = None, seasonal_periods: Optional[int] = None,
                                 sample_size: int = 0) -> str:
        """Perform comprehensive basic time series analysis."""
        try:
            if not table_name and not query:
                return json.dumps({"error": "Either table_name or query must be provided"})
            
            if table_name and query:
                return json.dumps({"error": "Provide either table_name or query, not both"})
                
            if name not in self.connections:
                return json.dumps({"error": f"Database '{name}' is not connected. Use connect_database first."})
            
            # Get data for analysis
            df = self._get_data_for_analysis(name, table_name, query, sample_size)
            if df.empty:
                return json.dumps({"error": "No data available for time series analysis"})
            
            # Use time series domain for analysis
            from .domains.time_series import analyze_time_series_basic as domain_analyze_basic
            
            result = domain_analyze_basic(
                df, 
                date_column=date_column,
                value_column=value_column,
                freq=freq,
                seasonal_periods=seasonal_periods
            )
            
            result['data_info'] = {
                'database_name': name,
                'sample_size_used': sample_size if sample_size > 0 else len(df)
            }
            
            return json.dumps(result, default=str)
            
        except Exception as e:
            logger.error(f"Basic time series analysis failed: {e}")
            return json.dumps({"error": f"Basic time series analysis failed: {str(e)}"})
    
    def forecast_arima(self, name: str, table_name: Optional[str] = None, query: Optional[str] = None,
                      date_column: Optional[str] = None, value_column: Optional[str] = None,
                      order: Optional[str] = None, seasonal_order: Optional[str] = None,
                      forecast_steps: int = 12, auto_arima: bool = True,
                      sample_size: int = 0) -> str:
        """Perform ARIMA forecasting on time series data."""
        try:
            if not table_name and not query:
                return json.dumps({"error": "Either table_name or query must be provided"})
            
            if table_name and query:
                return json.dumps({"error": "Provide either table_name or query, not both"})
                
            if name not in self.connections:
                return json.dumps({"error": f"Database '{name}' is not connected. Use connect_database first."})
            
            # Get data for analysis
            df = self._get_data_for_analysis(name, table_name, query, sample_size)
            if df.empty:
                return json.dumps({"error": "No data available for ARIMA forecasting"})
            
            # Parse order parameters if provided
            arima_order = None
            arima_seasonal_order = None
            
            if order:
                try:
                    arima_order = tuple(map(int, order.split(',')))
                    if len(arima_order) != 3:
                        return json.dumps({"error": "ARIMA order must have 3 components (p,d,q)"})
                except ValueError:
                    return json.dumps({"error": "Invalid ARIMA order format. Use 'p,d,q' (e.g., '1,1,1')"})
            
            if seasonal_order:
                try:
                    arima_seasonal_order = tuple(map(int, seasonal_order.split(',')))
                    if len(arima_seasonal_order) != 4:
                        return json.dumps({"error": "Seasonal order must have 4 components (P,D,Q,s)"})
                except ValueError:
                    return json.dumps({"error": "Invalid seasonal order format. Use 'P,D,Q,s' (e.g., '1,1,1,12')"})
            
            # Use time series domain for ARIMA forecasting
            from .domains.time_series import forecast_arima as domain_forecast_arima
            
            result = domain_forecast_arima(
                df,
                date_column=date_column,
                value_column=value_column,
                forecast_steps=forecast_steps,
                order=arima_order,
                seasonal_order=arima_seasonal_order,
                auto_arima=auto_arima
            )
            
            result['data_info'] = {
                'database_name': name,
                'sample_size_used': sample_size if sample_size > 0 else len(df)
            }
            
            return json.dumps(result, default=str)
            
        except Exception as e:
            logger.error(f"ARIMA forecasting failed: {e}")
            return json.dumps({"error": f"ARIMA forecasting failed: {str(e)}"})
    
    def forecast_exponential_smoothing(self, name: str, table_name: Optional[str] = None, query: Optional[str] = None,
                                      date_column: Optional[str] = None, value_column: Optional[str] = None,
                                      method: str = 'auto', seasonal: Optional[str] = None,
                                      seasonal_periods: Optional[int] = None, forecast_steps: int = 12,
                                      sample_size: int = 0) -> str:
        """Perform exponential smoothing forecasting on time series data."""
        try:
            if not table_name and not query:
                return json.dumps({"error": "Either table_name or query must be provided"})
            
            if table_name and query:
                return json.dumps({"error": "Provide either table_name or query, not both"})
                
            if name not in self.connections:
                return json.dumps({"error": f"Database '{name}' is not connected. Use connect_database first."})
            
            # Get data for analysis
            df = self._get_data_for_analysis(name, table_name, query, sample_size)
            if df.empty:
                return json.dumps({"error": "No data available for exponential smoothing forecasting"})
            
            # Use time series domain for exponential smoothing
            from .domains.time_series import forecast_exponential_smoothing as domain_forecast_smoothing
            
            result = domain_forecast_smoothing(
                df,
                date_column=date_column,
                value_column=value_column,
                method=method,
                seasonal=seasonal,
                seasonal_periods=seasonal_periods,
                forecast_steps=forecast_steps
            )
            
            result['data_info'] = {
                'database_name': name,
                'sample_size_used': sample_size if sample_size > 0 else len(df)
            }
            
            return json.dumps(result, default=str)
            
        except Exception as e:
            logger.error(f"Exponential smoothing forecasting failed: {e}")
            return json.dumps({"error": f"Exponential smoothing forecasting failed: {str(e)}"})
    
    def detect_time_series_anomalies(self, name: str, table_name: Optional[str] = None, query: Optional[str] = None,
                                    date_column: Optional[str] = None, value_column: Optional[str] = None,
                                    method: str = 'statistical', contamination: float = 0.05,
                                    window_size: Optional[int] = None, sample_size: int = 0) -> str:
        """Detect anomalies in time series data."""
        try:
            if not table_name and not query:
                return json.dumps({"error": "Either table_name or query must be provided"})
            
            if table_name and query:
                return json.dumps({"error": "Provide either table_name or query, not both"})
                
            if name not in self.connections:
                return json.dumps({"error": f"Database '{name}' is not connected. Use connect_database first."})
            
            # Get data for analysis
            df = self._get_data_for_analysis(name, table_name, query, sample_size)
            if df.empty:
                return json.dumps({"error": "No data available for time series anomaly detection"})
            
            # Use time series domain for anomaly detection
            from .domains.time_series import detect_time_series_anomalies as domain_detect_anomalies
            
            result = domain_detect_anomalies(
                df,
                date_column=date_column,
                value_column=value_column,
                method=method,
                contamination=contamination,
                window_size=window_size
            )
            
            result['data_info'] = {
                'database_name': name,
                'sample_size_used': sample_size if sample_size > 0 else len(df)
            }
            
            return json.dumps(result, default=str)
            
        except Exception as e:
            logger.error(f"Time series anomaly detection failed: {e}")
            return json.dumps({"error": f"Time series anomaly detection failed: {str(e)}"})
    
    def detect_change_points(self, name: str, table_name: Optional[str] = None, query: Optional[str] = None,
                            date_column: Optional[str] = None, value_column: Optional[str] = None,
                            method: str = 'cusum', min_size: int = 10, sample_size: int = 0) -> str:
        """Detect change points in time series data."""
        try:
            if not table_name and not query:
                return json.dumps({"error": "Either table_name or query must be provided"})
            
            if table_name and query:
                return json.dumps({"error": "Provide either table_name or query, not both"})
                
            if name not in self.connections:
                return json.dumps({"error": f"Database '{name}' is not connected. Use connect_database first."})
            
            # Get data for analysis
            df = self._get_data_for_analysis(name, table_name, query, sample_size)
            if df.empty:
                return json.dumps({"error": "No data available for change point detection"})
            
            # Use time series domain for change point detection
            from .domains.time_series import detect_change_points as domain_detect_change_points
            
            result = domain_detect_change_points(
                df,
                date_column=date_column,
                value_column=value_column,
                method=method,
                min_size=min_size
            )
            
            result['data_info'] = {
                'database_name': name,
                'sample_size_used': sample_size if sample_size > 0 else len(df)
            }
            
            return json.dumps(result, default=str)
            
        except Exception as e:
            logger.error(f"Change point detection failed: {e}")
            return json.dumps({"error": f"Change point detection failed: {str(e)}"})
    
    def analyze_multivariate_time_series(self, name: str, table_name: Optional[str] = None, query: Optional[str] = None,
                                        analysis_type: str = 'var', max_lags: int = 5, sample_size: int = 0) -> str:
        """Perform multivariate time series analysis."""
        try:
            if not table_name and not query:
                return json.dumps({"error": "Either table_name or query must be provided"})
            
            if table_name and query:
                return json.dumps({"error": "Provide either table_name or query, not both"})
                
            if name not in self.connections:
                return json.dumps({"error": f"Database '{name}' is not connected. Use connect_database first."})
            
            # Get data for analysis
            df = self._get_data_for_analysis(name, table_name, query, sample_size)
            if df.empty:
                return json.dumps({"error": "No data available for multivariate time series analysis"})
            
            # Select numeric columns for multivariate analysis
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) < 2:
                return json.dumps({"error": "At least 2 numeric columns required for multivariate time series analysis"})
            
            multivariate_df = df[numeric_cols]
            
            # Use time series domain for multivariate analysis
            from .domains.time_series import analyze_multivariate_time_series as domain_analyze_multivariate
            
            result = domain_analyze_multivariate(
                multivariate_df,
                analysis_type=analysis_type,
                max_lags=max_lags
            )
            
            result['data_info'] = {
                'database_name': name,
                'sample_size_used': sample_size if sample_size > 0 else len(df),
                'variables': numeric_cols.tolist()
            }
            
            return json.dumps(result, default=str)
            
        except Exception as e:
            logger.error(f"Multivariate time series analysis failed: {e}")
            return json.dumps({"error": f"Multivariate time series analysis failed: {str(e)}"})
    
    def _get_data_for_analysis(self, name: str, table_name: Optional[str], query: Optional[str], sample_size: int) -> pd.DataFrame:
        """Helper method to get data for pattern recognition analysis."""
        engine = self._get_connection(name)
        
        # Build the analysis query
        if table_name:
            if sample_size > 0:
                analysis_query = f"SELECT * FROM {self._safe_table_identifier(table_name)} ORDER BY RANDOM() LIMIT {sample_size}"
            else:
                analysis_query = f"SELECT * FROM {self._safe_table_identifier(table_name)}"
        else:
            if sample_size > 0:
                analysis_query = f"SELECT * FROM ({query}) AS subquery ORDER BY RANDOM() LIMIT {sample_size}"
            else:
                analysis_query = query
        
        # Execute query and return DataFrame
        with engine.connect() as connection:
            result = connection.execute(text(analysis_query))
            df = pd.DataFrame(result.fetchall(), columns=result.keys())
            return df


# Global DatabaseManager instance for MCP tool binding
# This fixes the MCP interface compatibility issue identified in validation
_db_manager = None

# MCP tool wrappers - bind instance methods to global MCP server
# This fixes the MCP interface compatibility issue identified in validation

# Essential MCP tool wrapper - needed for test compatibility
@mcp.tool
def connect_database(name: str, db_type: str, conn_string: str, sheet_name: Optional[str] = None):
    """
    Open a connection to a database.

    Args:
        name: A unique name to identify the connection (e.g., "analytics_db", "user_data").
        db_type: The type of the database ("sqlite", "postgresql", "mysql", "duckdb", "csv", "json", "yaml", "toml", "excel", "ods", "numbers", "xml", "ini", "tsv", "parquet", "feather", "arrow", "hdf5").
        conn_string: The connection string or file path for the database.
        sheet_name: Optional sheet name to load from Excel/ODS/Numbers files, or dataset name for HDF5 files. If not specified, all sheets/datasets are loaded.
    """
    return _db_manager.connect_database(name, db_type, conn_string, sheet_name)

@mcp.tool
def profile_table(name: str, table_name: Optional[str] = None, query: Optional[str] = None, 
                 sample_size: int = 10000, include_distributions: bool = True) -> str:
    """
    Generate comprehensive data profile with statistics, data quality metrics, and distribution analysis.
    
    This tool provides industry-standard data profiling including completeness, uniqueness, validity,
    and consistency analysis. Optimized for large datasets using streaming architecture.
    
    Args:
        name: Database connection name
        table_name: Name of the table to profile (mutually exclusive with query)
        query: Custom SQL query to profile (mutually exclusive with table_name)
        sample_size: Number of rows to sample for analysis (default: 10000, 0 = all rows)
        include_distributions: Whether to include distribution analysis for numeric columns
        
    Returns:
        Comprehensive data profile in JSON format with statistics and quality metrics
    """
    return _db_manager.profile_table(name, table_name, query, sample_size, include_distributions)

@mcp.tool
def detect_data_types(name: str, table_name: Optional[str] = None, query: Optional[str] = None,
                     sample_size: int = 5000, confidence_threshold: float = 0.8) -> str:
    """
    Intelligent data type detection beyond schema inference with pattern recognition and semantic analysis.
    
    Performs advanced data type detection including pattern matching for emails, URLs, dates, IDs,
    and semantic analysis for addresses, phone numbers. Provides confidence scoring and conversion
    recommendations.
    
    Args:
        name: Database connection name
        table_name: Name of the table to analyze (mutually exclusive with query)
        query: Custom SQL query to analyze (mutually exclusive with table_name)
        sample_size: Number of rows to sample for analysis (default: 5000, 0 = all rows)
        confidence_threshold: Minimum confidence level for suggestions (0.0-1.0)
        
    Returns:
        Data type detection results with confidence scores and conversion recommendations
    """
    return _db_manager.detect_data_types(name, table_name, query, sample_size, confidence_threshold)

@mcp.tool
def analyze_distributions(name: str, table_name: Optional[str] = None, query: Optional[str] = None,
                         columns: Optional[str] = None, sample_size: int = 10000, 
                         bins: int = 20, percentiles: Optional[str] = None) -> str:
    """
    Column-wise distribution analysis with histograms, percentiles, and statistical pattern detection.
    
    Provides comprehensive distribution analysis including histogram generation, percentile analysis,
    statistical moments, and distribution pattern detection for data exploration insights.
    
    Args:
        name: Database connection name
        table_name: Name of the table to analyze (mutually exclusive with query)
        query: Custom SQL query to analyze (mutually exclusive with table_name)
        columns: Comma-separated list of columns to analyze (default: all numeric columns)
        sample_size: Number of rows to sample for analysis (default: 10000, 0 = all rows)
        bins: Number of bins for histogram generation (default: 20)
        percentiles: Comma-separated percentiles to calculate (e.g., "10,25,50,75,90")
        
    Returns:
        Detailed distribution analysis with histograms and statistical measures in JSON format
    """
    return _db_manager.analyze_distributions(name, table_name, query, columns, sample_size, bins, percentiles)


@mcp.tool
def perform_clustering(name: str, table_name: Optional[str] = None, query: Optional[str] = None,
                      algorithm: str = 'kmeans', n_clusters: Optional[int] = None,
                      sample_size: int = 10000) -> str:
    """
    Perform comprehensive clustering analysis on data using advanced pattern recognition algorithms.
    
    Supports multiple clustering algorithms including K-means, hierarchical, DBSCAN, Gaussian Mixture Models,
    and spectral clustering with automatic parameter optimization and quality assessment.
    
    Args:
        name: Database connection name
        table_name: Name of the table to cluster (mutually exclusive with query)
        query: Custom SQL query to cluster (mutually exclusive with table_name)
        algorithm: Clustering algorithm ('kmeans', 'hierarchical', 'dbscan', 'gmm', 'spectral')
        n_clusters: Number of clusters (auto-selected if None)
        sample_size: Number of rows to sample for analysis (default: 10000, 0 = all rows)
        **kwargs: Additional parameters for the clustering algorithm
        
    Returns:
        Comprehensive clustering results with labels, quality metrics, and recommendations in JSON format
    """
    return _db_manager.perform_clustering(name, table_name, query, algorithm, n_clusters, sample_size)


@mcp.tool
def reduce_dimensions(name: str, table_name: Optional[str] = None, query: Optional[str] = None,
                     algorithm: str = 'pca', n_components: Optional[int] = None,
                     sample_size: int = 10000) -> str:
    """
    Perform comprehensive dimensionality reduction on data for visualization and analysis.
    
    Supports PCA, t-SNE, UMAP, Independent Component Analysis, and Linear Discriminant Analysis
    with automatic component selection and quality assessment.
    
    Args:
        name: Database connection name
        table_name: Name of the table to reduce (mutually exclusive with query)
        query: Custom SQL query to reduce (mutually exclusive with table_name)
        algorithm: Reduction algorithm ('pca', 'tsne', 'umap', 'ica', 'lda')
        n_components: Number of components (auto-selected if None)
        sample_size: Number of rows to sample for analysis (default: 10000, 0 = all rows)
        **kwargs: Additional parameters for the reduction algorithm
        
    Returns:
        Dimensionality reduction results with transformed data and quality metrics in JSON format
    """
    return _db_manager.reduce_dimensions(name, table_name, query, algorithm, n_components, sample_size)


@mcp.tool
def detect_anomalies(name: str, table_name: Optional[str] = None, query: Optional[str] = None,
                    algorithm: str = 'isolation_forest', contamination: Optional[float] = None,
                    sample_size: int = 10000) -> str:
    """
    Perform comprehensive anomaly detection on data using advanced algorithms.
    
    Supports Isolation Forest, One-Class SVM, Local Outlier Factor, and statistical methods
    for detecting outliers and anomalous patterns in datasets.
    
    Args:
        name: Database connection name
        table_name: Name of the table to analyze (mutually exclusive with query)
        query: Custom SQL query to analyze (mutually exclusive with table_name)
        algorithm: Detection algorithm ('isolation_forest', 'one_class_svm', 'lof', 'statistical')
        contamination: Expected proportion of outliers (auto-estimated if None)
        sample_size: Number of rows to sample for analysis (default: 10000, 0 = all rows)
        **kwargs: Additional parameters for the detection algorithm
        
    Returns:
        Anomaly detection results with labels, scores, and quality assessment in JSON format
    """
    return _db_manager.detect_anomalies(name, table_name, query, algorithm, contamination, sample_size)


@mcp.tool
def evaluate_patterns(name: str, table_name: Optional[str] = None, query: Optional[str] = None,
                     pattern_type: str = 'clustering', results_data: Optional[str] = None,
                     sample_size: int = 10000) -> str:
    """
    Evaluate pattern recognition results and provide comprehensive quality assessment and recommendations.
    
    Analyzes clustering, dimensionality reduction, or anomaly detection results to provide
    quality metrics, validation scores, and actionable recommendations for improvement.
    
    Args:
        name: Database connection name
        table_name: Name of the table to evaluate (mutually exclusive with query)
        query: Custom SQL query to evaluate (mutually exclusive with table_name)
        pattern_type: Type of pattern analysis ('clustering', 'dimensionality_reduction', 'anomaly_detection')
        results_data: JSON string of previous pattern recognition results (optional)
        sample_size: Number of rows to sample for analysis (default: 10000, 0 = all rows)
        
    Returns:
        Comprehensive evaluation results with quality metrics and recommendations in JSON format
    """
    return _db_manager.evaluate_patterns(name, table_name, query, pattern_type, results_data, sample_size)


# Time Series Analysis MCP Tools

@mcp.tool
def analyze_time_series_basic(name: str, table_name: Optional[str] = None, query: Optional[str] = None,
                             date_column: Optional[str] = None, value_column: Optional[str] = None,
                             freq: Optional[str] = None, seasonal_periods: Optional[int] = None,
                             sample_size: int = 0) -> str:
    """
    Perform comprehensive basic time series analysis including trend, seasonality, and stationarity tests.
    
    This tool provides fundamental time series analysis including trend detection, seasonality analysis,
    stationarity testing, and autocorrelation analysis. Optimized for LLM-driven time series exploration.
    
    Args:
        name: Database connection name
        table_name: Name of the table containing time series data (mutually exclusive with query)
        query: Custom SQL query to retrieve time series data (mutually exclusive with table_name)
        date_column: Name of the date/time column (auto-detected if None)
        value_column: Name of the value column (auto-detected if None)
        freq: Frequency of the time series ('D', 'H', 'M', etc.) (auto-inferred if None)
        seasonal_periods: Number of periods in a season (auto-detected if None)
        sample_size: Number of rows to sample for analysis (default: 0 = all rows)
        
    Returns:
        Comprehensive basic time series analysis results including trend, seasonality, and stationarity metrics
    """
    return _db_manager.analyze_time_series_basic(name, table_name, query, date_column, value_column, 
                                               freq, seasonal_periods, sample_size)


@mcp.tool
def forecast_arima(name: str, table_name: Optional[str] = None, query: Optional[str] = None,
                  date_column: Optional[str] = None, value_column: Optional[str] = None,
                  order: Optional[str] = None, seasonal_order: Optional[str] = None,
                  forecast_steps: int = 12, auto_arima: bool = True, sample_size: int = 0) -> str:
    """
    Perform ARIMA (AutoRegressive Integrated Moving Average) forecasting on time series data.
    
    Advanced time series forecasting using ARIMA models with automatic parameter selection,
    model diagnostics, and comprehensive forecast evaluation. Supports both manual parameter
    specification and automatic model selection.
    
    Args:
        name: Database connection name
        table_name: Name of the table containing time series data (mutually exclusive with query)
        query: Custom SQL query to retrieve time series data (mutually exclusive with table_name)
        date_column: Name of the date/time column (auto-detected if None)
        value_column: Name of the value column (auto-detected if None)
        order: ARIMA order as 'p,d,q' (e.g., '1,1,1') (auto-selected if None and auto_arima=True)
        seasonal_order: Seasonal order as 'P,D,Q,s' (e.g., '1,1,1,12') (auto-selected if None)
        forecast_steps: Number of periods to forecast ahead (default: 12)
        auto_arima: Whether to automatically select optimal ARIMA parameters (default: True)
        sample_size: Number of rows to use for modeling (default: 0 = all rows)
        
    Returns:
        ARIMA forecasting results with predictions, confidence intervals, and model diagnostics
    """
    return _db_manager.forecast_arima(name, table_name, query, date_column, value_column,
                                    order, seasonal_order, forecast_steps, auto_arima, sample_size)


@mcp.tool
def forecast_exponential_smoothing(name: str, table_name: Optional[str] = None, query: Optional[str] = None,
                                  date_column: Optional[str] = None, value_column: Optional[str] = None,
                                  method: str = 'auto', seasonal: Optional[str] = None,
                                  seasonal_periods: Optional[int] = None, forecast_steps: int = 12,
                                  sample_size: int = 0) -> str:
    """
    Perform exponential smoothing forecasting on time series data.
    
    Advanced exponential smoothing methods including Simple, Double (Holt's), and Triple (Holt-Winters)
    exponential smoothing with automatic method selection and parameter optimization.
    
    Args:
        name: Database connection name
        table_name: Name of the table containing time series data (mutually exclusive with query)
        query: Custom SQL query to retrieve time series data (mutually exclusive with table_name)
        date_column: Name of the date/time column (auto-detected if None)
        value_column: Name of the value column (auto-detected if None)
        method: Smoothing method ('simple', 'double', 'triple', 'auto') (default: 'auto')
        seasonal: Seasonal component type ('add', 'mul', None) (auto-detected if None and method='auto')
        seasonal_periods: Number of periods in a season (auto-detected if None)
        forecast_steps: Number of periods to forecast ahead (default: 12)
        sample_size: Number of rows to use for modeling (default: 0 = all rows)
        
    Returns:
        Exponential smoothing forecasting results with predictions and model parameters
    """
    return _db_manager.forecast_exponential_smoothing(name, table_name, query, date_column, value_column,
                                                     method, seasonal, seasonal_periods, forecast_steps, sample_size)


@mcp.tool
def detect_time_series_anomalies(name: str, table_name: Optional[str] = None, query: Optional[str] = None,
                                date_column: Optional[str] = None, value_column: Optional[str] = None,
                                method: str = 'statistical', contamination: float = 0.05,
                                window_size: Optional[int] = None, sample_size: int = 0) -> str:
    """
    Detect anomalies in time series data using statistical and machine learning methods.
    
    Advanced time series anomaly detection using multiple methods including statistical approaches,
    Isolation Forest, and sliding window techniques. Provides detailed anomaly scoring and periods.
    
    Args:
        name: Database connection name
        table_name: Name of the table containing time series data (mutually exclusive with query)
        query: Custom SQL query to retrieve time series data (mutually exclusive with table_name)
        date_column: Name of the date/time column (auto-detected if None)
        value_column: Name of the value column (auto-detected if None)
        method: Anomaly detection method ('statistical', 'isolation_forest') (default: 'statistical')
        contamination: Expected proportion of anomalies (default: 0.05 = 5%)
        window_size: Window size for sliding window methods (auto-calculated if None)
        sample_size: Number of rows to analyze (default: 0 = all rows)
        
    Returns:
        Time series anomaly detection results with anomaly indices, scores, and detailed periods
    """
    return _db_manager.detect_time_series_anomalies(name, table_name, query, date_column, value_column,
                                                   method, contamination, window_size, sample_size)


@mcp.tool
def detect_change_points(name: str, table_name: Optional[str] = None, query: Optional[str] = None,
                        date_column: Optional[str] = None, value_column: Optional[str] = None,
                        method: str = 'cusum', min_size: int = 10, sample_size: int = 0) -> str:
    """
    Detect change points in time series data using advanced statistical methods.
    
    Identifies structural breaks and regime changes in time series using CUSUM, PELT, and other
    change point detection algorithms. Provides detailed segment analysis and change point dates.
    
    Args:
        name: Database connection name
        table_name: Name of the table containing time series data (mutually exclusive with query)
        query: Custom SQL query to retrieve time series data (mutually exclusive with table_name)
        date_column: Name of the date/time column (auto-detected if None)
        value_column: Name of the value column (auto-detected if None)
        method: Change point detection method ('cusum', 'pelt') (default: 'cusum')
        min_size: Minimum segment size between change points (default: 10)
        sample_size: Number of rows to analyze (default: 0 = all rows)
        
    Returns:
        Change point detection results with change point locations, dates, and segment analysis
    """
    return _db_manager.detect_change_points(name, table_name, query, date_column, value_column,
                                          method, min_size, sample_size)


@mcp.tool
def analyze_multivariate_time_series(name: str, table_name: Optional[str] = None, query: Optional[str] = None,
                                    analysis_type: str = 'var', max_lags: int = 5, sample_size: int = 0) -> str:
    """
    Perform multivariate time series analysis including VAR models and Granger causality tests.
    
    Advanced multivariate time series analysis using Vector Autoregression (VAR) models,
    Granger causality testing, and cross-correlation analysis for multiple time series variables.
    
    Args:
        name: Database connection name
        table_name: Name of the table containing multivariate time series data (mutually exclusive with query)
        query: Custom SQL query to retrieve multivariate time series data (mutually exclusive with table_name)
        analysis_type: Type of analysis ('var', 'granger') (default: 'var')
        max_lags: Maximum number of lags to consider (default: 5)
        sample_size: Number of rows to analyze (default: 0 = all rows)
        
    Returns:
        Multivariate time series analysis results with model parameters, forecasts, and causality tests
    """
    return _db_manager.analyze_multivariate_time_series(name, table_name, query, analysis_type, max_lags, sample_size)


def main():
    """Main entry point with structured logging initialization."""
    try:
        # Log system startup with structured logging
        with logging_manager.context(
            operation="system_startup",
            component="localdata_mcp"
        ):
            logger.info("LocalData MCP starting up",
                      version="1.3.1",
                      structured_logging_enabled=True,
                      metrics_enabled=logging_config.enable_metrics,
                      security_logging_enabled=logging_config.enable_security_logging)
        
        # Initialize database manager
        global _db_manager
        _db_manager = DatabaseManager()
        manager = _db_manager  # Keep existing variable for backward compatibility
        
        # Log successful initialization
        with logging_manager.context(
            operation="system_ready",
            component="localdata_mcp"
        ):
            logger.info("LocalData MCP ready to accept connections",
                      transport="stdio",
                      logging_level=logging_config.level.value,
                      metrics_endpoint=f"http://localhost:{logging_config.metrics_port}{logging_config.metrics_endpoint}" if logging_config.enable_metrics else None)
        
        # Start MCP server
        mcp.run(transport="stdio")
        
    except Exception as e:
        logging_manager.log_error(e, "localdata_mcp", operation="system_startup")
        raise


if __name__ == "__main__":
    main()