"""Streaming data sources for CSV, ODS, and Numbers formats."""

import logging
from typing import Generator, Optional

import pandas as pd

from ..streaming import StreamingDataSource
from .base import ProcessingProgress

logger = logging.getLogger(__name__)

# Check for ODS support
try:
    from odf import opendocument, table

    ODFPY_AVAILABLE = True
except ImportError:
    ODFPY_AVAILABLE = False

# Check for Numbers support
try:
    from numbers_parser import Document

    NUMBERS_PARSER_AVAILABLE = True
except ImportError:
    NUMBERS_PARSER_AVAILABLE = False


class StreamingCSVSource(StreamingDataSource):
    """Enhanced CSV streaming source with adaptive chunk sizing."""

    def __init__(
        self,
        file_path: str,
        progress: Optional[ProcessingProgress] = None,
        **kwargs,
    ):
        self.file_path = file_path
        self.progress = progress
        self.kwargs = kwargs
        self._estimated_rows = None

    def get_chunk_iterator(
        self, chunk_size: int
    ) -> Generator[pd.DataFrame, None, None]:
        """Stream CSV with enhanced error handling and adaptive chunking."""
        try:
            chunk_reader = pd.read_csv(
                self.file_path, chunksize=chunk_size, **self.kwargs
            )

            for chunk in chunk_reader:
                if not chunk.empty:
                    chunk.columns = [
                        str(col).strip().replace(" ", "_").replace("-", "_")
                        for col in chunk.columns
                    ]

                    if self.progress:
                        self.progress.rows_processed += len(chunk)

                    yield chunk

        except pd.errors.ParserError:
            logger.warning("CSV parser error, trying without header")
            chunk_reader = pd.read_csv(
                self.file_path, chunksize=chunk_size, header=None, **self.kwargs
            )
            for chunk in chunk_reader:
                if not chunk.empty:
                    if self.progress:
                        self.progress.rows_processed += len(chunk)
                    yield chunk

    def estimate_total_rows(self) -> Optional[int]:
        """Estimate total rows by counting lines."""
        if self._estimated_rows is not None:
            return self._estimated_rows

        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                self._estimated_rows = sum(1 for _ in f) - 1
            return self._estimated_rows
        except Exception:
            return None

    def estimate_memory_per_row(self) -> float:
        """Estimate memory usage per row."""
        return 100.0


class StreamingODSSource(StreamingDataSource):
    """Streaming ODS source using pandas with odf engine."""

    def __init__(
        self,
        file_path: str,
        sheet_name: Optional[str] = None,
        progress: Optional[ProcessingProgress] = None,
    ):
        self.file_path = file_path
        self.sheet_name = sheet_name
        self.progress = progress
        self._sheets_data = None

    def get_chunk_iterator(
        self, chunk_size: int
    ) -> Generator[pd.DataFrame, None, None]:
        """Stream ODS data by loading sheets and chunking."""
        if not ODFPY_AVAILABLE:
            raise ValueError("odfpy library required for ODS files")

        try:
            with pd.ExcelFile(self.file_path, engine="odf") as excel_file:
                sheets_to_process = (
                    [self.sheet_name] if self.sheet_name else excel_file.sheet_names
                )

                for sheet_name in sheets_to_process:
                    if self.progress:
                        self.progress.current_sheet = sheet_name
                        self.progress.sheets_processed += 1

                    df = pd.read_excel(excel_file, sheet_name=sheet_name, engine="odf")

                    if df.empty:
                        continue

                    df.columns = [
                        str(col).strip().replace(" ", "_").replace("-", "_")
                        for col in df.columns
                    ]

                    for start in range(0, len(df), chunk_size):
                        chunk = df.iloc[start : start + chunk_size].copy()
                        if not chunk.empty:
                            if self.progress:
                                self.progress.rows_processed += len(chunk)
                            yield chunk

        except Exception as e:
            logger.error(f"Error streaming ODS file: {e}")
            raise

    def estimate_total_rows(self) -> Optional[int]:
        """Estimate total rows across all sheets."""
        try:
            with pd.ExcelFile(self.file_path, engine="odf") as excel_file:
                total_rows = 0
                sheets_to_process = (
                    [self.sheet_name] if self.sheet_name else excel_file.sheet_names
                )

                for sheet_name in sheets_to_process:
                    df = pd.read_excel(excel_file, sheet_name=sheet_name, engine="odf")
                    total_rows += len(df)

                return total_rows
        except Exception:
            return None

    def estimate_memory_per_row(self) -> float:
        """Estimate memory usage per row."""
        return 200.0


class StreamingNumbersSource(StreamingDataSource):
    """Streaming Numbers source using numbers-parser."""

    def __init__(
        self,
        file_path: str,
        sheet_name: Optional[str] = None,
        progress: Optional[ProcessingProgress] = None,
    ):
        self.file_path = file_path
        self.sheet_name = sheet_name
        self.progress = progress

    def get_chunk_iterator(
        self, chunk_size: int
    ) -> Generator[pd.DataFrame, None, None]:
        """Stream Numbers data by processing tables in chunks."""
        if not NUMBERS_PARSER_AVAILABLE:
            raise ValueError("numbers-parser library required for Numbers files")

        try:
            doc = Document(self.file_path)
            sheets_to_process = (
                [s for s in doc.sheets if s.name == self.sheet_name]
                if self.sheet_name
                else doc.sheets
            )

            for sheet in sheets_to_process:
                if self.progress:
                    self.progress.current_sheet = sheet.name
                    self.progress.sheets_processed += 1

                for table_idx, table in enumerate(sheet.tables):
                    table_data = table.rows(values_only=True)

                    if not table_data or len(table_data) < 2:
                        continue

                    headers = [
                        str(h) if h is not None else f"Column_{i + 1}"
                        for i, h in enumerate(table_data[0])
                    ]
                    headers = [
                        h.strip().replace(" ", "_").replace("-", "_").replace(".", "_")
                        for h in headers
                    ]

                    data_rows = table_data[1:]

                    for start in range(0, len(data_rows), chunk_size):
                        chunk_rows = data_rows[start : start + chunk_size]
                        chunk_df = pd.DataFrame(chunk_rows, columns=headers)

                        if not chunk_df.empty:
                            if self.progress:
                                self.progress.rows_processed += len(chunk_df)
                            yield chunk_df

        except Exception as e:
            logger.error(f"Error streaming Numbers file: {e}")
            raise

    def estimate_total_rows(self) -> Optional[int]:
        """Estimate total rows across all tables."""
        try:
            doc = Document(self.file_path)
            total_rows = 0

            sheets_to_process = (
                [s for s in doc.sheets if s.name == self.sheet_name]
                if self.sheet_name
                else doc.sheets
            )

            for sheet in sheets_to_process:
                for table in sheet.tables:
                    table_data = table.rows(values_only=True)
                    if table_data and len(table_data) > 1:
                        total_rows += len(table_data) - 1

            return total_rows
        except Exception:
            return None

    def estimate_memory_per_row(self) -> float:
        """Estimate memory usage per row."""
        return 250.0
