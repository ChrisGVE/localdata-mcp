"""Memory-Bounded Streaming Pipeline for LocalData MCP.

This module implements intelligent streaming architecture that processes data in 
memory-safe chunks, preventing memory overflows while maintaining high performance.
Replaces batch processing with adaptive streaming based on available system resources.
"""

import gc
import logging
import time
import weakref
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Generator, Iterator, List, Optional, Tuple, Union
from abc import ABC, abstractmethod

import pandas as pd
import psutil
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine, Connection

# Import configuration system
from .config_manager import get_config_manager, PerformanceConfig

# Import query analyzer for intelligent chunking decisions
from .query_analyzer import analyze_query, QueryAnalysis


logger = logging.getLogger(__name__)


@dataclass
class MemoryStatus:
    """Current system memory status for adaptive decision making."""
    total_gb: float
    available_gb: float
    used_percent: float
    is_low_memory: bool
    recommended_chunk_size: int
    max_safe_chunk_size: int


@dataclass
class ChunkMetrics:
    """Metrics for a processed chunk."""
    chunk_number: int
    rows_processed: int
    memory_used_mb: float
    processing_time_seconds: float
    timestamp: float = field(default_factory=time.time)


class ResultBuffer:
    """Memory-bounded result buffer with automatic cleanup using weakref."""
    
    def __init__(self, query_id: str, db_name: str, query: str, 
                 max_memory_mb: int = 500):
        """Initialize result buffer with memory bounds.
        
        Args:
            query_id: Unique identifier for the query result
            db_name: Database name for the query
            query: The original SQL query
            max_memory_mb: Maximum memory to use for buffering
        """
        self.query_id = query_id
        self.db_name = db_name
        self.query = query
        self.max_memory_mb = max_memory_mb
        self.chunks: List[pd.DataFrame] = []
        self.total_rows = 0
        self.timestamp = time.time()
        self.is_complete = False
        self._current_memory_mb = 0.0
        
        # Use weakref for automatic cleanup
        self._cleanup_ref = weakref.finalize(self, self._cleanup_buffer)
    
    def add_chunk(self, chunk: pd.DataFrame) -> bool:
        """Add chunk to buffer if within memory limits.
        
        Returns:
            bool: True if chunk was added, False if memory limit exceeded
        """
        chunk_memory_mb = self._estimate_chunk_memory(chunk)
        
        if self._current_memory_mb + chunk_memory_mb > self.max_memory_mb:
            # Memory limit exceeded - don't buffer this chunk
            logger.warning(f"Buffer {self.query_id} memory limit ({self.max_memory_mb}MB) "
                         f"exceeded, not buffering chunk of {chunk_memory_mb:.1f}MB")
            return False
        
        self.chunks.append(chunk)
        self.total_rows += len(chunk)
        self._current_memory_mb += chunk_memory_mb
        
        logger.debug(f"Added chunk to buffer {self.query_id}: {len(chunk)} rows, "
                    f"{chunk_memory_mb:.1f}MB, total memory: {self._current_memory_mb:.1f}MB")
        return True
    
    def get_chunk_range(self, start_row: int, chunk_size: int) -> Optional[pd.DataFrame]:
        """Get specific range of rows from buffered chunks."""
        if not self.chunks:
            return None
        
        # Concatenate all chunks for range selection
        full_df = pd.concat(self.chunks, ignore_index=True)
        end_row = min(start_row + chunk_size, len(full_df))
        
        if start_row >= len(full_df):
            return None
        
        return full_df.iloc[start_row:end_row].copy()
    
    def clear(self):
        """Manually clear the buffer."""
        self.chunks.clear()
        self.total_rows = 0
        self._current_memory_mb = 0.0
        gc.collect()  # Force garbage collection
        
    def _estimate_chunk_memory(self, chunk: pd.DataFrame) -> float:
        """Estimate memory usage of a DataFrame chunk in MB."""
        return chunk.memory_usage(deep=True).sum() / (1024 * 1024)
    
    @staticmethod
    def _cleanup_buffer():
        """Cleanup method called by weakref when buffer is garbage collected."""
        gc.collect()


class StreamingDataSource(ABC):
    """Abstract base class for streaming data sources."""
    
    @abstractmethod
    def get_chunk_iterator(self, chunk_size: int) -> Generator[pd.DataFrame, None, None]:
        """Get iterator that yields data chunks."""
        pass
    
    @abstractmethod
    def estimate_total_rows(self) -> Optional[int]:
        """Estimate total number of rows if possible."""
        pass
    
    @abstractmethod
    def estimate_memory_per_row(self) -> float:
        """Estimate memory usage per row in bytes."""
        pass


class StreamingSQLSource(StreamingDataSource):
    """Streaming data source for SQL queries."""
    
    def __init__(self, engine: Engine, query: str, query_analysis: Optional[QueryAnalysis] = None):
        self.engine = engine
        self.query = query
        self.query_analysis = query_analysis
        self._connection: Optional[Connection] = None
        self._result_proxy = None
    
    def get_chunk_iterator(self, chunk_size: int) -> Generator[pd.DataFrame, None, None]:
        """Stream SQL results in chunks using server-side cursors."""
        try:
            self._connection = self.engine.connect()
            
            # Use server-side cursor for streaming (PostgreSQL)
            if self.engine.dialect.name == 'postgresql':
                # Enable server-side cursor
                self._connection = self._connection.execution_options(stream_results=True)
            
            result = self._connection.execute(text(self.query))
            
            # Get column names
            columns = list(result.keys())
            
            while True:
                # Fetch chunk of rows
                rows = result.fetchmany(chunk_size)
                if not rows:
                    break
                
                # Convert to DataFrame
                chunk = pd.DataFrame(rows, columns=columns)
                
                if not chunk.empty:
                    yield chunk
                else:
                    break
                    
        except Exception as e:
            logger.error(f"Error in SQL streaming: {e}")
            raise
        finally:
            self._cleanup()
    
    def estimate_total_rows(self) -> Optional[int]:
        """Use query analysis estimate if available."""
        if self.query_analysis:
            return self.query_analysis.estimated_rows
        return None
    
    def estimate_memory_per_row(self) -> float:
        """Use query analysis estimate if available."""
        if self.query_analysis:
            return self.query_analysis.estimated_row_size_bytes
        return 1024.0  # Default 1KB per row
    
    def _cleanup(self):
        """Clean up database resources."""
        if self._connection:
            try:
                self._connection.close()
            except Exception as e:
                logger.warning(f"Error closing SQL connection: {e}")
            finally:
                self._connection = None


class StreamingFileSource(StreamingDataSource):
    """Streaming data source for file formats."""
    
    def __init__(self, file_path: str, file_type: str, **kwargs):
        self.file_path = file_path
        self.file_type = file_type
        self.kwargs = kwargs
        self._estimated_rows: Optional[int] = None
        self._estimated_row_size: Optional[float] = None
    
    def get_chunk_iterator(self, chunk_size: int) -> Generator[pd.DataFrame, None, None]:
        """Stream file data in chunks based on file type."""
        try:
            if self.file_type == 'csv':
                yield from self._stream_csv(chunk_size)
            elif self.file_type == 'excel':
                yield from self._stream_excel(chunk_size)
            elif self.file_type == 'parquet':
                yield from self._stream_parquet(chunk_size)
            elif self.file_type == 'json':
                yield from self._stream_json(chunk_size)
            else:
                # Fallback to loading smaller chunks of unsupported formats
                yield from self._stream_generic(chunk_size)
                
        except Exception as e:
            logger.error(f"Error streaming {self.file_type} file {self.file_path}: {e}")
            raise
    
    def _stream_csv(self, chunk_size: int) -> Generator[pd.DataFrame, None, None]:
        """Stream CSV file using pandas chunksize parameter."""
        try:
            chunk_reader = pd.read_csv(self.file_path, chunksize=chunk_size, **self.kwargs)
            for chunk in chunk_reader:
                if not chunk.empty:
                    yield chunk
        except pd.errors.ParserError:
            # Fallback for CSV with no header
            chunk_reader = pd.read_csv(self.file_path, chunksize=chunk_size, header=None, **self.kwargs)
            for chunk in chunk_reader:
                if not chunk.empty:
                    yield chunk
    
    def _stream_excel(self, chunk_size: int) -> Generator[pd.DataFrame, None, None]:
        """Stream Excel file by processing sheets in chunks."""
        sheet_name = self.kwargs.get('sheet_name')
        
        with pd.ExcelFile(self.file_path) as excel_file:
            sheets_to_process = [sheet_name] if sheet_name else excel_file.sheet_names
            
            for sheet in sheets_to_process:
                try:
                    # For Excel, we can't stream directly, so we process smaller chunks
                    df = pd.read_excel(excel_file, sheet_name=sheet, **self.kwargs)
                    
                    if df.empty:
                        continue
                    
                    # Yield in chunks
                    for start in range(0, len(df), chunk_size):
                        chunk = df.iloc[start:start + chunk_size].copy()
                        if not chunk.empty:
                            yield chunk
                            
                except Exception as e:
                    logger.warning(f"Failed to process Excel sheet '{sheet}': {e}")
                    continue
    
    def _stream_parquet(self, chunk_size: int) -> Generator[pd.DataFrame, None, None]:
        """Stream Parquet file using pyarrow batch reading."""
        try:
            import pyarrow.parquet as pq
            
            parquet_file = pq.ParquetFile(self.file_path)
            
            # Read in batches
            for batch in parquet_file.iter_batches(batch_size=chunk_size):
                chunk = batch.to_pandas()
                if not chunk.empty:
                    yield chunk
                    
        except ImportError:
            # Fallback to pandas chunking
            df = pd.read_parquet(self.file_path, **self.kwargs)
            for start in range(0, len(df), chunk_size):
                chunk = df.iloc[start:start + chunk_size].copy()
                if not chunk.empty:
                    yield chunk
    
    def _stream_json(self, chunk_size: int) -> Generator[pd.DataFrame, None, None]:
        """Stream JSON file by processing in chunks."""
        # For JSON, we need to load and then chunk
        df = pd.read_json(self.file_path, **self.kwargs)
        
        for start in range(0, len(df), chunk_size):
            chunk = df.iloc[start:start + chunk_size].copy()
            if not chunk.empty:
                yield chunk
    
    def _stream_generic(self, chunk_size: int) -> Generator[pd.DataFrame, None, None]:
        """Generic streaming for unsupported file types."""
        # Load entire file and chunk it
        # This is not ideal but better than loading everything into memory at once
        logger.warning(f"Using generic streaming for {self.file_type} - may not be memory optimal")
        
        # Estimate file size and adjust chunk size if needed
        file_size = Path(self.file_path).stat().st_size
        if file_size > 100 * 1024 * 1024:  # > 100MB
            chunk_size = max(chunk_size // 2, 10)  # Reduce chunk size for large files
        
        # Load with appropriate reader (this part should be replaced with proper streaming)
        if self.file_type == 'yaml':
            import yaml
            with open(self.file_path, 'r') as f:
                data = yaml.safe_load(f)
            df = pd.json_normalize(data) if isinstance(data, (list, dict)) else pd.DataFrame(data)
        else:
            # Default to empty DataFrame
            df = pd.DataFrame()
        
        if not df.empty:
            for start in range(0, len(df), chunk_size):
                chunk = df.iloc[start:start + chunk_size].copy()
                if not chunk.empty:
                    yield chunk
    
    def estimate_total_rows(self) -> Optional[int]:
        """Estimate total rows by quick sampling."""
        if self._estimated_rows is not None:
            return self._estimated_rows
        
        try:
            # Quick estimation based on file type
            if self.file_type == 'csv':
                # Count lines in CSV (approximate)
                with open(self.file_path, 'r') as f:
                    self._estimated_rows = sum(1 for _ in f) - 1  # Subtract header
            elif self.file_type == 'parquet':
                try:
                    import pyarrow.parquet as pq
                    parquet_file = pq.ParquetFile(self.file_path)
                    self._estimated_rows = parquet_file.metadata.num_rows
                except ImportError:
                    pass
            
            return self._estimated_rows
        except Exception:
            return None
    
    def estimate_memory_per_row(self) -> float:
        """Estimate memory per row by sampling."""
        if self._estimated_row_size is not None:
            return self._estimated_row_size
        
        try:
            # Sample first chunk to estimate row size
            for chunk in self.get_chunk_iterator(100):  # Sample 100 rows
                if not chunk.empty:
                    memory_usage = chunk.memory_usage(deep=True).sum()
                    self._estimated_row_size = memory_usage / len(chunk)
                    return self._estimated_row_size
                break
        except Exception:
            pass
        
        return 1024.0  # Default 1KB per row


class StreamingQueryExecutor:
    """Memory-bounded streaming query executor."""
    
    def __init__(self, config: Optional[PerformanceConfig] = None):
        """Initialize streaming executor with configuration.
        
        Args:
            config: Performance configuration. If None, loads from config manager.
        """
        self.config = config or get_config_manager().get_performance_config()
        self._result_buffers: Dict[str, ResultBuffer] = {}
        self._chunk_metrics: List[ChunkMetrics] = []
        
        logger.info(f"StreamingQueryExecutor initialized with memory limit: {self.config.memory_limit_mb}MB, "
                   f"default chunk size: {self.config.chunk_size}")
    
    def execute_streaming(self, source: StreamingDataSource, query_id: str, 
                         initial_chunk_size: Optional[int] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Execute streaming query with adaptive memory management.
        
        Args:
            source: Streaming data source
            query_id: Unique identifier for buffering
            initial_chunk_size: Initial chunk size (adaptive if None)
        
        Returns:
            Tuple of (first_chunk_df, metadata_dict)
        """
        start_time = time.time()
        
        # Get initial memory status and chunk size
        memory_status = self._get_memory_status()
        chunk_size = initial_chunk_size or memory_status.recommended_chunk_size
        
        logger.info(f"Starting streaming execution for {query_id} with chunk_size={chunk_size}, "
                   f"available_memory={memory_status.available_gb:.1f}GB")
        
        # Initialize result buffer
        buffer_memory_limit = min(self.config.memory_limit_mb // 4, 500)  # Use 1/4 of limit, max 500MB
        result_buffer = ResultBuffer(query_id, "streaming", "streaming_query", buffer_memory_limit)
        self._result_buffers[query_id] = result_buffer
        
        first_chunk = None
        total_rows_processed = 0
        chunk_number = 0
        
        try:
            chunk_iterator = source.get_chunk_iterator(chunk_size)
            
            for chunk in chunk_iterator:
                chunk_start_time = time.time()
                chunk_number += 1
                
                # Check memory before processing chunk
                current_memory = self._get_memory_status()
                if current_memory.is_low_memory:
                    logger.warning(f"Low memory detected ({current_memory.used_percent:.1f}%), "
                                 f"reducing chunk size from {chunk_size} to {current_memory.recommended_chunk_size}")
                    chunk_size = current_memory.recommended_chunk_size
                
                # Process chunk
                processed_chunk = self._process_chunk(chunk, chunk_number)
                rows_in_chunk = len(processed_chunk)
                total_rows_processed += rows_in_chunk
                
                # Store first chunk for immediate return
                if first_chunk is None:
                    first_chunk = processed_chunk.copy()
                
                # Try to buffer the chunk
                buffered = result_buffer.add_chunk(processed_chunk)
                
                # Record chunk metrics
                chunk_time = time.time() - chunk_start_time
                chunk_memory = processed_chunk.memory_usage(deep=True).sum() / (1024 * 1024)
                
                metrics = ChunkMetrics(
                    chunk_number=chunk_number,
                    rows_processed=rows_in_chunk,
                    memory_used_mb=chunk_memory,
                    processing_time_seconds=chunk_time
                )
                self._chunk_metrics.append(metrics)
                
                logger.debug(f"Processed chunk {chunk_number}: {rows_in_chunk} rows, "
                           f"{chunk_memory:.1f}MB, {chunk_time:.3f}s, buffered={buffered}")
                
                # Adaptive chunk size adjustment based on performance
                if chunk_number > 1:
                    chunk_size = self._adapt_chunk_size(chunk_size, metrics, current_memory)
                
                # Break if we have enough data for initial response and memory is getting tight
                if chunk_number >= 3 and current_memory.is_low_memory and total_rows_processed >= chunk_size:
                    logger.info(f"Stopping initial streaming due to memory constraints after {chunk_number} chunks")
                    break
        
        except Exception as e:
            logger.error(f"Error during streaming execution: {e}")
            raise
        
        # Mark buffer as complete if we processed all data
        result_buffer.is_complete = True
        
        # Build metadata
        execution_time = time.time() - start_time
        final_memory = self._get_memory_status()
        
        metadata = {
            "query_id": query_id,
            "total_rows_processed": total_rows_processed,
            "chunks_processed": chunk_number,
            "execution_time_seconds": execution_time,
            "final_chunk_size": chunk_size,
            "memory_status": {
                "initial_available_gb": memory_status.available_gb,
                "final_available_gb": final_memory.available_gb,
                "final_used_percent": final_memory.used_percent
            },
            "estimated_total_rows": source.estimate_total_rows(),
            "streaming": True,
            "buffer_complete": result_buffer.is_complete
        }
        
        logger.info(f"Streaming execution completed: {total_rows_processed} rows in {execution_time:.3f}s, "
                   f"{chunk_number} chunks")
        
        return first_chunk or pd.DataFrame(), metadata
    
    def get_chunk_iterator(self, query_id: str, start_row: int = 0, 
                          chunk_size: Optional[int] = None) -> Iterator[pd.DataFrame]:
        """Get iterator for buffered query results.
        
        Args:
            query_id: Query result identifier
            start_row: Starting row (0-based)
            chunk_size: Chunk size for iteration
        
        Yields:
            DataFrame chunks
        """
        if query_id not in self._result_buffers:
            logger.error(f"Query result buffer '{query_id}' not found")
            return
        
        buffer = self._result_buffers[query_id]
        chunk_size = chunk_size or self.config.chunk_size
        
        current_row = start_row
        while True:
            chunk = buffer.get_chunk_range(current_row, chunk_size)
            if chunk is None or chunk.empty:
                break
            
            yield chunk
            current_row += len(chunk)
    
    def manage_memory_bounds(self) -> Dict[str, Any]:
        """Monitor and manage memory usage, cleaning up as needed.
        
        Returns:
            Dict with memory management actions taken
        """
        memory_status = self._get_memory_status()
        actions_taken = []
        
        if memory_status.is_low_memory:
            logger.warning(f"Memory usage high ({memory_status.used_percent:.1f}%), starting cleanup")
            
            # Clear oldest result buffers first
            buffers_to_clear = []
            buffer_ages = [(query_id, buffer.timestamp) for query_id, buffer in self._result_buffers.items()]
            buffer_ages.sort(key=lambda x: x[1])  # Sort by timestamp (oldest first)
            
            # Clear up to half of the buffers if memory is critical
            max_to_clear = len(buffer_ages) // 2 if memory_status.used_percent > 90 else len(buffer_ages) // 4
            
            for query_id, _ in buffer_ages[:max_to_clear]:
                self._result_buffers[query_id].clear()
                buffers_to_clear.append(query_id)
            
            # Remove cleared buffers from tracking
            for query_id in buffers_to_clear:
                del self._result_buffers[query_id]
            
            if buffers_to_clear:
                actions_taken.append(f"Cleared {len(buffers_to_clear)} result buffers")
            
            # Force garbage collection
            gc.collect()
            actions_taken.append("Forced garbage collection")
            
            # Clear old chunk metrics
            if len(self._chunk_metrics) > 100:
                self._chunk_metrics = self._chunk_metrics[-50:]  # Keep last 50
                actions_taken.append("Trimmed chunk metrics history")
        
        return {
            "memory_status": memory_status.__dict__,
            "actions_taken": actions_taken,
            "active_buffers": len(self._result_buffers),
            "chunk_metrics_count": len(self._chunk_metrics)
        }
    
    def _get_memory_status(self) -> MemoryStatus:
        """Get current memory status with recommendations."""
        try:
            memory = psutil.virtual_memory()
            
            total_gb = memory.total / (1024**3)
            available_gb = memory.available / (1024**3)
            used_percent = memory.percent
            
            # Determine if memory is low based on threshold
            is_low_memory = used_percent > (self.config.memory_warning_threshold * 100)
            
            # Calculate recommended chunk size based on available memory
            base_chunk_size = self.config.chunk_size
            
            if used_percent > 90:
                # Critical memory - very small chunks
                recommended_chunk_size = max(base_chunk_size // 8, 10)
                max_safe_chunk_size = max(base_chunk_size // 4, 25)
            elif used_percent > 80:
                # High memory - reduced chunks
                recommended_chunk_size = max(base_chunk_size // 4, 25)
                max_safe_chunk_size = max(base_chunk_size // 2, 50)
            elif used_percent > 60:
                # Medium memory - slightly reduced chunks
                recommended_chunk_size = max(base_chunk_size // 2, 50)
                max_safe_chunk_size = base_chunk_size
            else:
                # Plenty of memory - can use larger chunks
                recommended_chunk_size = base_chunk_size
                max_safe_chunk_size = min(base_chunk_size * 2, 1000)
            
            return MemoryStatus(
                total_gb=round(total_gb, 2),
                available_gb=round(available_gb, 2),
                used_percent=used_percent,
                is_low_memory=is_low_memory,
                recommended_chunk_size=recommended_chunk_size,
                max_safe_chunk_size=max_safe_chunk_size
            )
            
        except Exception as e:
            logger.warning(f"Could not get memory status: {e}")
            # Return safe defaults
            return MemoryStatus(
                total_gb=8.0,
                available_gb=4.0,
                used_percent=50.0,
                is_low_memory=False,
                recommended_chunk_size=self.config.chunk_size,
                max_safe_chunk_size=self.config.chunk_size
            )
    
    def _process_chunk(self, chunk: pd.DataFrame, chunk_number: int) -> pd.DataFrame:
        """Process a single chunk (placeholder for chunk-specific processing)."""
        # For now, just return the chunk as-is
        # This could be extended to include chunk-specific transformations,
        # data cleaning, or other processing steps
        return chunk
    
    def _adapt_chunk_size(self, current_chunk_size: int, metrics: ChunkMetrics, 
                         memory_status: MemoryStatus) -> int:
        """Adaptively adjust chunk size based on performance metrics and memory."""
        # Don't adjust too frequently
        if len(self._chunk_metrics) < 2:
            return current_chunk_size
        
        # Get recent performance trend
        recent_metrics = self._chunk_metrics[-3:] if len(self._chunk_metrics) >= 3 else self._chunk_metrics
        avg_processing_time = sum(m.processing_time_seconds for m in recent_metrics) / len(recent_metrics)
        avg_memory_per_row = sum(m.memory_used_mb / m.rows_processed for m in recent_metrics) / len(recent_metrics)
        
        # Adjust based on performance and memory constraints
        new_chunk_size = current_chunk_size
        
        # If processing is too slow, reduce chunk size
        if avg_processing_time > 2.0:  # More than 2 seconds per chunk
            new_chunk_size = max(current_chunk_size // 2, 10)
            logger.debug(f"Reducing chunk size due to slow processing: {current_chunk_size} -> {new_chunk_size}")
        
        # If memory per row is high, reduce chunk size
        elif avg_memory_per_row > 0.1:  # More than 0.1MB per row
            new_chunk_size = max(current_chunk_size // 2, 25)
            logger.debug(f"Reducing chunk size due to high memory usage: {current_chunk_size} -> {new_chunk_size}")
        
        # If processing is fast and memory is available, can increase chunk size
        elif avg_processing_time < 0.5 and not memory_status.is_low_memory:
            new_chunk_size = min(current_chunk_size * 2, memory_status.max_safe_chunk_size)
            logger.debug(f"Increasing chunk size due to good performance: {current_chunk_size} -> {new_chunk_size}")
        
        # Ensure we stay within memory bounds
        new_chunk_size = min(new_chunk_size, memory_status.max_safe_chunk_size)
        new_chunk_size = max(new_chunk_size, 10)  # Minimum 10 rows
        
        return new_chunk_size
    
    def clear_buffer(self, query_id: str) -> bool:
        """Clear a specific result buffer.
        
        Args:
            query_id: Buffer to clear
            
        Returns:
            bool: True if buffer was cleared
        """
        if query_id in self._result_buffers:
            self._result_buffers[query_id].clear()
            del self._result_buffers[query_id]
            logger.info(f"Cleared result buffer: {query_id}")
            return True
        return False
    
    def get_buffer_info(self, query_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a result buffer."""
        if query_id not in self._result_buffers:
            return None
        
        buffer = self._result_buffers[query_id]
        return {
            "query_id": buffer.query_id,
            "db_name": buffer.db_name,
            "total_rows": buffer.total_rows,
            "chunks_count": len(buffer.chunks),
            "memory_usage_mb": buffer._current_memory_mb,
            "memory_limit_mb": buffer.max_memory_mb,
            "timestamp": buffer.timestamp,
            "is_complete": buffer.is_complete
        }
    
    def cleanup_expired_buffers(self, max_age_seconds: int = 3600) -> int:
        """Clean up buffers older than specified age.
        
        Args:
            max_age_seconds: Maximum age of buffers to keep (default 1 hour)
            
        Returns:
            int: Number of buffers cleaned up
        """
        current_time = time.time()
        expired_buffers = []
        
        for query_id, buffer in self._result_buffers.items():
            if current_time - buffer.timestamp > max_age_seconds:
                expired_buffers.append(query_id)
        
        for query_id in expired_buffers:
            self.clear_buffer(query_id)
        
        if expired_buffers:
            logger.info(f"Cleaned up {len(expired_buffers)} expired buffers")
        
        return len(expired_buffers)


def create_streaming_source(engine: Engine = None, query: str = None, 
                          file_path: str = None, file_type: str = None,
                          query_analysis: QueryAnalysis = None, 
                          **kwargs) -> StreamingDataSource:
    """Factory function to create appropriate streaming data source.
    
    Args:
        engine: SQLAlchemy engine for SQL sources
        query: SQL query for SQL sources  
        file_path: File path for file sources
        file_type: File type for file sources
        query_analysis: Query analysis results for SQL sources
        **kwargs: Additional arguments for file readers
    
    Returns:
        StreamingDataSource: Appropriate streaming source
    """
    if engine and query:
        return StreamingSQLSource(engine, query, query_analysis)
    elif file_path and file_type:
        return StreamingFileSource(file_path, file_type, **kwargs)
    else:
        raise ValueError("Must provide either (engine, query) for SQL or (file_path, file_type) for files")