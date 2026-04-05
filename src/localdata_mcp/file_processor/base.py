"""Base classes and data structures for streaming file processing."""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..streaming import StreamingDataSource


@dataclass
class ProcessingProgress:
    """Progress tracking for file processing operations."""

    file_path: str
    file_type: str
    total_sheets: Optional[int] = None
    sheets_processed: int = 0
    current_sheet: Optional[str] = None
    estimated_rows: Optional[int] = None
    rows_processed: int = 0
    memory_usage_mb: float = 0.0
    processing_time_seconds: float = 0.0
    start_time: float = 0.0


class StreamingFileProcessor(ABC):
    """Abstract base class for streaming file processors."""

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.progress = ProcessingProgress(
            file_path=file_path, file_type=self.get_file_type(), start_time=time.time()
        )

    @abstractmethod
    def get_file_type(self) -> str:
        """Get the file type this processor handles."""
        pass

    @abstractmethod
    def create_streaming_source(
        self, sheet_name: Optional[str] = None
    ) -> StreamingDataSource:
        """Create streaming data source for this file."""
        pass

    @abstractmethod
    def estimate_processing_requirements(self) -> Dict[str, Any]:
        """Estimate memory and processing requirements."""
        pass

    def get_progress(self) -> ProcessingProgress:
        """Get current processing progress."""
        self.progress.processing_time_seconds = time.time() - self.progress.start_time
        return self.progress


class FileProcessorFactory:
    """Factory for creating streaming file processors."""

    _processors: Dict[str, type] = {}

    @classmethod
    def _ensure_processors(cls) -> None:
        """Lazily populate the processors registry to avoid circular imports."""
        if cls._processors:
            return

        from .processors import (
            StreamingCSVProcessor,
            StreamingExcelProcessor,
            StreamingJSONProcessor,
            StreamingNumbersProcessor,
            StreamingODSProcessor,
        )

        cls._processors = {
            "excel": StreamingExcelProcessor,
            "xlsx": StreamingExcelProcessor,
            "xlsm": StreamingExcelProcessor,
            "xls": StreamingExcelProcessor,
            "json": StreamingJSONProcessor,
            "csv": StreamingCSVProcessor,
            "tsv": StreamingCSVProcessor,
            "ods": StreamingODSProcessor,
            "numbers": StreamingNumbersProcessor,
        }

    @classmethod
    def create_processor(
        cls, file_path: str, file_type: str
    ) -> "StreamingFileProcessor":
        """Create appropriate streaming processor for file type.

        Args:
            file_path: Path to the file
            file_type: Type of file to process

        Returns:
            StreamingFileProcessor: Appropriate processor for the file type

        Raises:
            ValueError: If file type is not supported
        """
        cls._ensure_processors()

        if file_type not in cls._processors:
            raise ValueError(f"Unsupported file type for streaming: {file_type}")

        processor_class = cls._processors[file_type]
        return processor_class(file_path)

    @classmethod
    def get_supported_formats(cls) -> List[str]:
        """Get list of supported file formats."""
        cls._ensure_processors()
        return list(cls._processors.keys())

    @classmethod
    def is_supported(cls, file_type: str) -> bool:
        """Check if file type is supported for streaming processing."""
        cls._ensure_processors()
        return file_type in cls._processors
