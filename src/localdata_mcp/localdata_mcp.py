"""LocalData MCP - Database connection and query management.

This module defines all module-level state (imports, feature flags, MCP server
instance, logging singletons) and re-exports ``DatabaseManager`` and ``main``
from the ``server`` sub-package.  Keeping the state here ensures that
``unittest.mock.patch("localdata_mcp.localdata_mcp.XXX", ...)`` continues to
work for existing tests.
"""

import argparse
import atexit
import configparser
import hashlib
import importlib.metadata
import json
import logging
import os

from .json_utils import safe_dumps
import psutil
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional

if TYPE_CHECKING:
    from .graph_manager import GraphStorageManager
    from .rdf_storage import RDFStorageManager

import pandas as pd
import yaml
from fastmcp import FastMCP
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.sql import quoted_name

# Import SQL query parser for security validation
from .query_parser import parse_and_validate_sql, SQLSecurityError, check_readonly

# Import query analyzer for pre-execution analysis
from .query_analyzer import analyze_query, QueryAnalysis

# Import streaming executor for memory-bounded processing
from .streaming import StreamingQueryExecutor, create_streaming_source

# Import timeout manager for query timeout handling
from .timeout_manager import QueryTimeoutError, get_timeout_manager

# Import streaming file processors for memory-efficient file loading
from .file_processor import create_streaming_file_engine, FileProcessorFactory

# Import enhanced response metadata system
from .response_metadata import (
    get_metadata_generator,
    EnhancedResponseMetadata,
    LLMCommunicationProtocol,
    ResponseMetadataGenerator,
)

# Import structured logging system
from .logging_manager import get_logging_manager, get_logger
from .config_manager import get_config_manager, initialize_config

# Import backward compatibility management
from .compatibility_manager import get_compatibility_manager

# Import staging manager singleton
from .staging_manager import get_staging_manager

# Import structured error classification
from .error_classification import classify_error

# Import tree storage for structured data (TOML, JSON, YAML)
from .tree_storage import TreeStorageManager, create_tree_schema
from .tree_parsers import parse_toml_to_tree, parse_json_to_tree, parse_yaml_to_tree
from .tree_tools import (
    tool_get_node,
    tool_get_children,
    tool_set_node,
    tool_move_node,
    tool_delete_node,
    tool_list_keys,
    tool_get_value,
    tool_set_value,
    tool_delete_key,
)
from .tree_export import tool_export_structured

# Import graph tool functions
from .graph_tools import (
    tool_get_node_graph,
    tool_get_neighbors,
    tool_get_edges,
    tool_set_node_graph,
    tool_delete_node_graph,
    tool_add_edge,
    tool_remove_edge,
    tool_find_path,
    tool_get_graph_stats,
    tool_get_value_graph,
    tool_set_value_graph,
    tool_delete_key_graph,
    tool_list_keys_graph,
    tool_export_graph,
)

# Import data science tool functions
from .datascience_tools import (
    tool_hypothesis_test,
    tool_anova_analysis,
    tool_effect_sizes,
    tool_fit_regression,
    tool_evaluate_model,
    tool_clustering,
    tool_anomaly_detection,
    tool_dimensionality_reduction,
    tool_time_series_analysis,
    tool_time_series_forecast,
    tool_rfm_analysis,
    tool_ab_test,
)

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
    from numbers_parser import Document

    NUMBERS_PARSER_AVAILABLE = True
except ImportError:
    NUMBERS_PARSER_AVAILABLE = False

# DuckDB support
try:
    import duckdb

    DUCKDB_AVAILABLE = True
except ImportError:
    DUCKDB_AVAILABLE = False

# HDF5 support for scientific data
try:
    import h5py

    H5PY_AVAILABLE = True
except ImportError:
    H5PY_AVAILABLE = False

# Modern database support - Redis
try:
    import redis

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

# Modern database support - Elasticsearch
try:
    from elasticsearch import Elasticsearch

    ELASTICSEARCH_AVAILABLE = True
except ImportError:
    ELASTICSEARCH_AVAILABLE = False

# Modern database support - MongoDB
try:
    import pymongo

    PYMONGO_AVAILABLE = True
except ImportError:
    PYMONGO_AVAILABLE = False

# Modern database support - InfluxDB
try:
    from influxdb_client import InfluxDBClient

    INFLUXDB_AVAILABLE = True
except ImportError:
    INFLUXDB_AVAILABLE = False

# Modern database support - Neo4j
try:
    from neo4j import GraphDatabase

    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False

# Modern database support - CouchDB
try:
    import couchdb

    COUCHDB_AVAILABLE = True
except ImportError:
    COUCHDB_AVAILABLE = False

# SPARQL endpoint support
try:
    from SPARQLWrapper import SPARQLWrapper as _SPARQLWrapper

    SPARQLWRAPPER_AVAILABLE = True
except ImportError:
    SPARQLWRAPPER_AVAILABLE = False

# Oracle support (optional enterprise dependency)
try:
    from .oracle_support import ORACLEDB_AVAILABLE
except ImportError:
    ORACLEDB_AVAILABLE = False

# MS SQL Server support (optional enterprise dependency)
try:
    from .mssql_support import PYMSSQL_AVAILABLE, PYODBC_AVAILABLE

    MSSQL_AVAILABLE = PYMSSQL_AVAILABLE or PYODBC_AVAILABLE
except ImportError:
    PYMSSQL_AVAILABLE = False
    PYODBC_AVAILABLE = False
    MSSQL_AVAILABLE = False

# Set up logging
# Initialize structured logging before other components
config_manager = get_config_manager()
logging_config = config_manager.get_logging_config()
logging_manager = get_logging_manager(logging_config)

# Get structured logger (logging configuration handled by LoggingManager)
logger = get_logger(__name__)

# Create the MCP server instance
mcp = FastMCP("localdata-mcp")

# Add metrics endpoint if enabled
if logging_config.enable_metrics:
    from prometheus_client import CONTENT_TYPE_LATEST

    @mcp.tool()
    def get_metrics() -> str:
        """Get Prometheus metrics for monitoring dashboards.

        Returns:
            Prometheus-formatted metrics string
        """
        try:
            with logging_manager.context(
                operation="metrics_export", component="localdata_mcp"
            ):
                logger.debug("Exporting Prometheus metrics")

            metrics_data = logging_manager.get_metrics()
            return metrics_data

        except Exception as e:
            logging_manager.log_error(e, "localdata_mcp", operation="metrics_export")
            return f"Error exporting metrics: {e}"


# ---------------------------------------------------------------------------
# Import the implementation from the server sub-package.
# This MUST come after all module-level state above is initialised, because
# the server modules import this module (by absolute name) to access the
# flags and singletons defined here.
# ---------------------------------------------------------------------------
from .server.database_manager import DatabaseManager, QueryBuffer  # noqa: F401, E402
from .server.cli import main, _get_version, _parse_cli_args  # noqa: F401, E402

__all__ = [
    "mcp",
    "DatabaseManager",
    "QueryBuffer",
    "main",
    "_get_version",
    "_parse_cli_args",
]

if __name__ == "__main__":
    main()
