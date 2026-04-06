"""Streaming data sources for the streaming pipeline.

Contains abstract base class and concrete implementations for SQL and file-based
streaming data sources, plus a factory function for source creation.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generator, Optional

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Connection, Engine

from ..logging_manager import get_logger
from ..query_analyzer import QueryAnalysis

logger = get_logger(__name__)


class StreamingDataSource(ABC):
    """Abstract base class for streaming data sources."""

    @abstractmethod
    def get_chunk_iterator(
        self, chunk_size: int
    ) -> Generator[pd.DataFrame, None, None]:
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

    def __init__(
        self, engine: Engine, query: str, query_analysis: Optional[QueryAnalysis] = None
    ):
        self.engine = engine
        self.query = query
        self.query_analysis = query_analysis
        self._connection: Optional[Connection] = None
        self._result_proxy = None

    def get_chunk_iterator(
        self, chunk_size: int
    ) -> Generator[pd.DataFrame, None, None]:
        """Stream SQL results in chunks using server-side cursors."""
        try:
            self._connection = self.engine.connect()

            # Use server-side cursor for streaming (PostgreSQL)
            if self.engine.dialect.name == "postgresql":
                # Enable server-side cursor
                self._connection = self._connection.execution_options(
                    stream_results=True
                )

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

    def get_chunk_iterator(
        self, chunk_size: int
    ) -> Generator[pd.DataFrame, None, None]:
        """Stream file data in chunks based on file type."""
        try:
            if self.file_type == "csv":
                yield from self._stream_csv(chunk_size)
            elif self.file_type == "excel":
                yield from self._stream_excel(chunk_size)
            elif self.file_type == "parquet":
                yield from self._stream_parquet(chunk_size)
            elif self.file_type == "json":
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
            chunk_reader = pd.read_csv(
                self.file_path, chunksize=chunk_size, **self.kwargs
            )
            for chunk in chunk_reader:
                if not chunk.empty:
                    yield chunk
        except pd.errors.ParserError:
            # Fallback for CSV with no header
            chunk_reader = pd.read_csv(
                self.file_path, chunksize=chunk_size, header=None, **self.kwargs
            )
            for chunk in chunk_reader:
                if not chunk.empty:
                    yield chunk

    def _stream_excel(self, chunk_size: int) -> Generator[pd.DataFrame, None, None]:
        """Stream Excel file by processing sheets in chunks."""
        sheet_name = self.kwargs.get("sheet_name")

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
                        chunk = df.iloc[start : start + chunk_size].copy()
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
                chunk = df.iloc[start : start + chunk_size].copy()
                if not chunk.empty:
                    yield chunk

    def _stream_json(self, chunk_size: int) -> Generator[pd.DataFrame, None, None]:
        """Stream JSON file by processing in chunks."""
        # For JSON, we need to load and then chunk
        df = pd.read_json(self.file_path, **self.kwargs)

        for start in range(0, len(df), chunk_size):
            chunk = df.iloc[start : start + chunk_size].copy()
            if not chunk.empty:
                yield chunk

    def _stream_generic(self, chunk_size: int) -> Generator[pd.DataFrame, None, None]:
        """Generic streaming for unsupported file types."""
        # Load entire file and chunk it
        # This is not ideal but better than loading everything into memory at once
        logger.warning(
            f"Using generic streaming for {self.file_type} - may not be memory optimal"
        )

        # Estimate file size and adjust chunk size if needed
        file_size = Path(self.file_path).stat().st_size
        if file_size > 100 * 1024 * 1024:  # > 100MB
            chunk_size = max(chunk_size // 2, 10)  # Reduce chunk size for large files

        # Load with appropriate reader (this part should be replaced with proper streaming)
        if self.file_type == "yaml":
            import yaml

            with open(self.file_path, "r") as f:
                data = yaml.safe_load(f)
            df = (
                pd.json_normalize(data)
                if isinstance(data, (list, dict))
                else pd.DataFrame(data)
            )
        else:
            # Default to empty DataFrame
            df = pd.DataFrame()

        if not df.empty:
            for start in range(0, len(df), chunk_size):
                chunk = df.iloc[start : start + chunk_size].copy()
                if not chunk.empty:
                    yield chunk

    def estimate_total_rows(self) -> Optional[int]:
        """Estimate total rows by quick sampling."""
        if self._estimated_rows is not None:
            return self._estimated_rows

        try:
            # Quick estimation based on file type
            if self.file_type == "csv":
                # Count lines in CSV (approximate)
                with open(self.file_path, "r") as f:
                    self._estimated_rows = sum(1 for _ in f) - 1  # Subtract header
            elif self.file_type == "parquet":
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


def create_streaming_source(
    engine: Engine = None,
    query: str = None,
    file_path: str = None,
    file_type: str = None,
    query_analysis: QueryAnalysis = None,
    **kwargs,
) -> StreamingDataSource:
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
        raise ValueError(
            "Must provide either (engine, query) for SQL or (file_path, file_type) for files"
        )
