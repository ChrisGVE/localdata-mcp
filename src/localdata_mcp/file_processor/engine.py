"""Streaming file engine creation for loading files into SQLite."""

import logging
import os
import re
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.pool import StaticPool

from ..streaming import StreamingQueryExecutor
from .base import FileProcessorFactory

logger = logging.getLogger(__name__)


def create_streaming_file_engine(
    file_path: str,
    file_type: str,
    sheet_name: Optional[str] = None,
    temp_files_registry: Optional[List[str]] = None,
) -> Tuple[Engine, Dict[str, Any]]:
    """Create SQLite engine from file using streaming processing.

    This function replaces the batch loading approach with memory-efficient
    streaming that processes files in chunks and uses temporary SQLite storage.

    Args:
        file_path: Path to the file to process
        file_type: Type of file (excel, csv, json, etc.)
        sheet_name: Specific sheet/table to load (if applicable)
        temp_files_registry: Optional list to track temporary files

    Returns:
        Tuple of (SQLite engine, processing_metadata)
    """
    start_time = time.time()

    processor = FileProcessorFactory.create_processor(file_path, file_type)

    requirements = processor.estimate_processing_requirements()
    logger.info(f"Processing {file_type} file '{file_path}': {requirements}")

    file_size_mb = requirements.get("file_size_mb", 0)
    use_temp_file = file_size_mb > 100

    if use_temp_file:
        temp_fd, temp_path = tempfile.mkstemp(
            suffix=".sqlite", prefix="streaming_file_"
        )
        os.close(temp_fd)

        if temp_files_registry is not None:
            temp_files_registry.append(temp_path)

        engine = create_engine(f"sqlite:///{temp_path}")
        logger.info(f"Using temporary SQLite file: {temp_path}")
    else:
        engine = create_engine(
            "sqlite:///:memory:",
            poolclass=StaticPool,
            connect_args={"check_same_thread": False},
        )
        logger.info("Using in-memory SQLite database")

    streaming_source = processor.create_streaming_source(sheet_name)

    streaming_executor = StreamingQueryExecutor()

    tables_created = {}
    sheet_count = 0
    total_rows_processed = 0

    try:
        used_table_names = set()
        chunk_size = 1000
        chunk_number = 0

        for chunk_df in streaming_source.get_chunk_iterator(chunk_size):
            chunk_number += 1

            if chunk_df.empty:
                continue

            current_sheet = getattr(processor.progress, "current_sheet", None)
            if current_sheet:
                table_name = _sanitize_table_name(current_sheet, used_table_names)
            else:
                table_name = "data_table"
                used_table_names.add(table_name)

            if table_name not in tables_created:
                chunk_df.to_sql(table_name, engine, index=False, if_exists="replace")
                tables_created[table_name] = len(chunk_df)
                logger.info(
                    f"Created table '{table_name}' with initial {len(chunk_df)} rows"
                )
            else:
                chunk_df.to_sql(table_name, engine, index=False, if_exists="append")
                tables_created[table_name] += len(chunk_df)

            total_rows_processed += len(chunk_df)

            if chunk_number % 10 == 0:
                logger.info(
                    f"Processed {chunk_number} chunks, "
                    f"{total_rows_processed} total rows"
                )

            if chunk_number > 1 and chunk_number % 5 == 0:
                processing_time = time.time() - start_time
                if processing_time / chunk_number > 2.0:
                    chunk_size = max(chunk_size // 2, 100)
                    logger.info(
                        f"Reduced chunk size to {chunk_size} due to slow processing"
                    )

    except Exception as e:
        logger.error(f"Error during streaming file processing: {e}")
        raise

    processing_time = time.time() - start_time
    progress = processor.get_progress()

    processing_metadata = {
        "file_path": file_path,
        "file_type": file_type,
        "processing_time_seconds": processing_time,
        "total_rows_processed": total_rows_processed,
        "tables_created": tables_created,
        "sheets_processed": progress.sheets_processed,
        "memory_approach": "streaming_chunks",
        "engine_type": "temporary_sqlite" if use_temp_file else "memory_sqlite",
        "requirements_estimate": requirements,
    }

    logger.info(
        f"Streaming file processing completed: {total_rows_processed} rows "
        f"in {processing_time:.3f}s, {len(tables_created)} tables created"
    )

    return engine, processing_metadata


def _sanitize_table_name(name: str, used_names: set) -> str:
    """Sanitize and ensure unique table name."""
    name = str(name).strip()

    name = re.sub(r"[\s\-]+", "_", name)

    name = re.sub(r"[^\w]", "_", name)

    name = re.sub(r"_+", "_", name)

    if name and not re.match(r"^[a-zA-Z_]", name):
        name = "sheet_" + name

    if not name:
        name = "sheet_unnamed"

    original_name = name
    counter = 1
    while name.lower() in {n.lower() for n in used_names}:
        name = f"{original_name}_{counter}"
        counter += 1

    used_names.add(name)
    return name
