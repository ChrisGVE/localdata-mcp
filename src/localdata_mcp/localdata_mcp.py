"""LocalData MCP - Database connection and query management."""

import os
import json
import logging
import pandas as pd
import yaml
import toml
import hashlib
import time
import threading
import tempfile
import atexit
from pathlib import Path
from dataclasses import dataclass
from fastmcp import FastMCP
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.sql import quoted_name
from typing import Literal, Optional, Dict, Any, List

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create console handler if none exists
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

# Create the MCP server instance
mcp = FastMCP("localdata-mcp")

@dataclass
class QueryBuffer:
    query_id: str
    db_name: str
    query: str
    results: pd.DataFrame
    timestamp: float
    source_file_path: Optional[str] = None
    source_file_mtime: Optional[float] = None


class DatabaseManager:
    def __init__(self):
        self.connections: Dict[str, Any] = {}
        self.query_history: Dict[str, List[str]] = {}
        
        # Security and connection management
        self.connection_semaphore = threading.Semaphore(10)  # Max 10 concurrent connections
        self.connection_lock = threading.Lock()
        self.connection_count = 0
        
        # Query buffering system
        self.query_buffers: Dict[str, QueryBuffer] = {}
        self.query_buffer_lock = threading.Lock()
        
        # Temporary file management
        self.temp_files: List[str] = []
        self.temp_file_lock = threading.Lock()
        
        # Auto-cleanup for buffers (10 minute expiry)
        self.buffer_cleanup_interval = 600  # 10 minutes
        self.last_cleanup = time.time()
        
        # Register cleanup on exit
        atexit.register(self._cleanup_all)

    def _get_connection(self, name: str):
        if name not in self.connections:
            raise ValueError(
                f"Database '{name}' is not connected. Use 'connect_database' first."
            )
        return self.connections[name]

    def _sanitize_path(self, file_path: str):
        """Enhanced path security - restrict to current working directory and subdirectories only."""
        base_dir = Path(os.getcwd()).resolve()
        try:
            # Resolve the path to handle symlinks and relative paths
            abs_file_path = Path(file_path).resolve()
        except (OSError, ValueError) as e:
            raise ValueError(f"Invalid path '{file_path}': {e}")
        
        # Security check: ensure path is within base directory
        try:
            abs_file_path.relative_to(base_dir)
        except ValueError:
            raise ValueError(f"Path '{file_path}' is outside the allowed directory. Only current directory and subdirectories are allowed.")
        
        # Check if file exists
        if not abs_file_path.is_file():
            raise ValueError(f"File not found at path '{file_path}'.")
        
        return str(abs_file_path)
    
    def _get_file_size(self, file_path: str) -> int:
        """Get file size in bytes."""
        try:
            return os.path.getsize(file_path)
        except OSError as e:
            raise ValueError(f"Cannot get size of file '{file_path}': {e}")
    
    def _is_large_file(self, file_path: str, threshold_mb: int = 100) -> bool:
        """Check if file exceeds the size threshold (default 100MB)."""
        threshold_bytes = threshold_mb * 1024 * 1024
        return self._get_file_size(file_path) > threshold_bytes
    
    def _generate_query_id(self, db_name: str, query: str) -> str:
        """Generate a unique query ID in format: {db}_{timestamp}_{4char_hash}."""
        timestamp = int(time.time())
        query_hash = hashlib.md5(query.encode()).hexdigest()[:4]
        return f"{db_name}_{timestamp}_{query_hash}"
    
    def _cleanup_expired_buffers(self):
        """Remove expired query buffers (older than 10 minutes)."""
        current_time = time.time()
        if current_time - self.last_cleanup < self.buffer_cleanup_interval:
            return  # Skip if not time for cleanup yet
        
        with self.query_buffer_lock:
            expired_ids = [
                query_id for query_id, buffer in self.query_buffers.items()
                if current_time - buffer.timestamp > self.buffer_cleanup_interval
            ]
            for query_id in expired_ids:
                del self.query_buffers[query_id]
        
        self.last_cleanup = current_time
    
    def _cleanup_all(self):
        """Clean up all resources on exit."""
        # Clean up temporary files
        with self.temp_file_lock:
            for temp_file in self.temp_files:
                try:
                    if os.path.exists(temp_file):
                        os.unlink(temp_file)
                except OSError:
                    pass  # Ignore errors during cleanup
            self.temp_files.clear()
        
        # Close all database connections
        with self.connection_lock:
            for name, engine in self.connections.items():
                try:
                    engine.dispose()
                except:
                    pass  # Ignore errors during cleanup
            self.connections.clear()

    def _get_engine(self, db_type: str, conn_string: str):
        if db_type == "sqlite":
            return create_engine(f"sqlite:///{conn_string}")
        elif db_type == "postgresql":
            return create_engine(conn_string)
        elif db_type == "mysql":
            return create_engine(conn_string)
        elif db_type in ["csv", "json", "yaml", "toml"]:
            sanitized_path = self._sanitize_path(conn_string)
            return self._create_engine_from_file(sanitized_path, db_type)
        else:
            raise ValueError(f"Unsupported db_type: {db_type}")
    
    def _create_engine_from_file(self, file_path: str, file_type: str):
        """Create SQLite engine from file, using temporary storage for large files."""
        try:
            # Check if file is large
            is_large = self._is_large_file(file_path)
            
            # Load data based on file type
            if file_type == "csv":
                try:
                    df = pd.read_csv(file_path)
                except pd.errors.ParserError:
                    # Fallback for CSV with no header
                    df = pd.read_csv(file_path, header=None)
            elif file_type == "json":
                df = pd.read_json(file_path)
            elif file_type == "yaml":
                with open(file_path, 'r') as f:
                    data = yaml.safe_load(f)
                df = pd.json_normalize(data) if isinstance(data, (list, dict)) else pd.DataFrame(data)
            elif file_type == "toml":
                with open(file_path, 'r') as f:
                    data = toml.load(f)
                df = pd.json_normalize(data) if isinstance(data, (list, dict)) else pd.DataFrame(data)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
            
            # Create engine - use temporary file for large files, memory for small ones
            if is_large:
                # Create temporary SQLite file
                temp_fd, temp_path = tempfile.mkstemp(suffix='.sqlite', prefix='db_client_')
                os.close(temp_fd)  # Close the file descriptor, SQLAlchemy will handle the file
                
                with self.temp_file_lock:
                    self.temp_files.append(temp_path)
                
                engine = create_engine(f"sqlite:///{temp_path}")
            else:
                engine = create_engine("sqlite:///:memory:")
            
            # Load data into SQLite
            df.to_sql("data_table", engine, index=False, if_exists="replace")
            return engine
            
        except Exception as e:
            raise ValueError(f"Failed to create engine from {file_type} file '{file_path}': {e}")

    def _get_table_metadata(self, inspector, table_name):
        columns = inspector.get_columns(table_name)
        foreign_keys = inspector.get_foreign_keys(table_name)
        primary_keys = inspector.get_pk_constraint(table_name)["constrained_columns"]
        indexes = inspector.get_indexes(table_name)
        table_options = inspector.get_table_options(table_name)

        col_list = []
        for col in columns:
            col_info = {"name": col["name"], "type": str(col["type"])}
            if col["nullable"] is False:
                col_info["not_null"] = True
            if col.get("autoincrement", False) is True:
                col_info["autoincrement"] = True
            if col.get("default"):
                col_info["default"] = str(col["default"])

            if col["name"] in primary_keys:
                col_info["primary_key"] = True

            for fk in foreign_keys:
                if col["name"] in fk["constrained_columns"]:
                    col_info["foreign_key"] = {
                        "referred_table": fk["referred_table"],
                        "referred_column": fk["referred_columns"][0],
                    }
            col_list.append(col_info)

        index_list = []
        for idx in indexes:
            index_list.append(
                {
                    "name": idx["name"],
                    "columns": idx["column_names"],
                    "unique": idx.get("unique", False),
                }
            )

        return {
            "name": table_name,
            "columns": col_list,
            "foreign_keys": [f["name"] for f in foreign_keys],
            "primary_keys": primary_keys,
            "indexes": index_list,
            "options": table_options,
        }

    # =========================================================
    # Requested Tools
    # =========================================================

    @mcp.tool
    def connect_database(self, name: str, db_type: str, conn_string: str):
        """
        Open a connection to a database.

        Args:
            name: A unique name to identify the connection (e.g., "analytics_db", "user_data").
            db_type: The type of the database ("sqlite", "postgresql", "mysql", "csv", "json", "yaml", "toml").
            conn_string: The connection string or file path for the database.
        """
        logger.info(f"Attempting to connect to database '{name}' of type '{db_type}'")
        
        if name in self.connections:
            logger.warning(f"Database '{name}' is already connected")
            return f"Error: A database with the name '{name}' is already connected."
        
        # Check connection limit
        if not self.connection_semaphore.acquire(blocking=False):
            logger.warning(f"Connection limit reached for database '{name}'")
            return f"Error: Maximum number of concurrent connections (10) reached. Please disconnect a database first."
        
        try:
            engine = self._get_engine(db_type, conn_string)
            
            with self.connection_lock:
                self.connections[name] = engine
                self.query_history[name] = []
                self.connection_count += 1
            
            logger.info(f"Successfully connected to database '{name}'. Total connections: {self.connection_count}")
            return f"Successfully connected to database '{name}'."
        except Exception as e:
            # Release semaphore on failure
            self.connection_semaphore.release()
            logger.error(f"Failed to connect to database '{name}': {e}")
            return f"Failed to connect to database '{name}': {e}"

    @mcp.tool
    def disconnect_database(self, name: str):
        """
        Close a connection to a database. All open connections are closed when the script terminates.

        Args:
            name: The name of the database connection to close.
        """
        logger.info(f"Attempting to disconnect from database '{name}'")
        try:
            conn = self._get_connection(name)
            conn.dispose()
            
            with self.connection_lock:
                del self.connections[name]
                del self.query_history[name]
                self.connection_count -= 1
            
            # Release semaphore slot
            self.connection_semaphore.release()
            
            logger.info(f"Successfully disconnected from database '{name}'. Total connections: {self.connection_count}")
            return f"Successfully disconnected from database '{name}'."
        except ValueError as e:
            logger.error(f"Database '{name}' not found for disconnection: {e}")
            return str(e)
        except Exception as e:
            logger.error(f"Error disconnecting from database '{name}': {e}")
            return f"An error occurred while disconnecting: {e}"

    @mcp.tool
    def execute_query(self, name: str, query: str) -> str:
        """
        Execute a SQL query and return results as a markdown table.
        Returns error for results with more than 100 rows.

        Args:
            name: The name of the database connection.
            query: The SQL query to execute.
        """
        try:
            # Clean up expired buffers
            self._cleanup_expired_buffers()
            
            engine = self._get_connection(name)
            df = pd.read_sql_query(query, engine)
            self.query_history[name].append(query)
            
            if df.empty:
                return "Query executed successfully, but no results were returned."
            
            # Check row count limit for markdown queries
            if len(df) > 100:
                return f"Error: Query returned {len(df)} rows, which exceeds the 100-row limit for markdown format. Use execute_query_json() for large result sets."
            
            return df.to_markdown()
        except Exception as e:
            return f"An error occurred while executing the query: {e}"

    @mcp.tool
    def execute_query_json(self, name: str, query: str) -> str:
        """
        Execute a SQL query and return results as JSON.
        For results with >100 rows, returns first 10 rows + metadata + buffering info.

        Args:
            name: The name of the database connection.
            query: The SQL query to execute.
        """
        try:
            # Clean up expired buffers
            self._cleanup_expired_buffers()
            
            engine = self._get_connection(name)
            df = pd.read_sql_query(query, engine)
            self.query_history[name].append(query)
            
            if df.empty:
                return json.dumps([])
            
            # Check row count for large result handling
            if len(df) > 100:
                # Store full result in buffer
                query_id = self._generate_query_id(name, query)
                
                # Check if this is a file-based connection to track modifications
                source_file_path = None
                source_file_mtime = None
                connection = self.connections[name]
                if hasattr(connection, 'url') and connection.url.database and connection.url.database != ':memory:':
                    # This might be a file-based database, but for CSV connections we need to track the original file
                    # For now, we'll leave this as None - more complex file tracking would need connection metadata
                    pass
                
                buffer = QueryBuffer(
                    query_id=query_id,
                    db_name=name,
                    query=query,
                    results=df,
                    timestamp=time.time(),
                    source_file_path=source_file_path,
                    source_file_mtime=source_file_mtime
                )
                
                with self.query_buffer_lock:
                    self.query_buffers[query_id] = buffer
                
                # Return first 10 rows with metadata
                first_10 = df.head(10)
                
                response = {
                    "metadata": {
                        "total_rows": len(df),
                        "showing_rows": f"1-{min(10, len(df))}",
                        "query_id": query_id,
                        "file_modified_since_buffer": False  # Will be updated when we implement file tracking
                    },
                    "data": json.loads(first_10.to_json(orient="records")),
                    "next_options": {
                        "get_next_100": f"get_query_chunk(query_id='{query_id}', start_row=11, chunk_size=100)",
                        "get_all_remaining": f"get_query_chunk(query_id='{query_id}', start_row=11, chunk_size='all')"
                    }
                }
                
                return json.dumps(response, indent=2)
            else:
                # Return all results for small result sets
                return df.to_json(orient="records")
                
        except Exception as e:
            return f"An error occurred while executing the query: {e}"

    @mcp.tool
    def get_query_history(self, name: str) -> str:
        """
        Get the recent query history for a specific database connection.

        Args:
            name: The name of the database connection.
        """
        try:
            history = self.query_history.get(name, [])
            if not history:
                return f"No query history found for database '{name}'."
            return "\n".join(history)
        except Exception as e:
            return f"An error occurred: {e}"

    @mcp.tool
    def list_databases(self) -> str:
        """
        List all available database connections.
        """
        if not self.connections:
            return "No databases are currently connected."
        return json.dumps(list(self.connections.keys()))

    @mcp.tool
    def describe_database(self, name: str) -> str:
        """
        Get detailed information about a database, including its schema in JSON format.

        Args:
            name: The name of the database connection.
        """
        try:
            engine = self._get_connection(name)
            inspector = inspect(engine)

            db_info = {
                "name": name,
                "dialect": engine.dialect.name,
                "version": inspector.get_server_version_info(),
                "default_schema_name": inspector.default_schema_name,
                "schemas": inspector.get_schema_names(),
                "tables": [],
            }

            for table_name in inspector.get_table_names():
                table_info = self._get_table_metadata(inspector, table_name)
                with engine.connect() as conn:
                    safe_table_name = self._safe_table_identifier(table_name)
                    result = conn.execute(text(f"SELECT COUNT(*) FROM {safe_table_name}"))
                    row_count = result.scalar()
                table_info["size"] = row_count
                db_info["tables"].append(table_info)

            return json.dumps(db_info, indent=2)
        except Exception as e:
            return f"An error occurred: {e}"

    @mcp.tool
    def find_table(self, table_name: str) -> str:
        """
        Find which database contains a specific table.

        Args:
            table_name: The name of the table to find.
        """
        found_dbs = []
        for name, engine in self.connections.items():
            inspector = inspect(engine)
            if table_name in inspector.get_table_names():
                found_dbs.append(name)

        if not found_dbs:
            return f"Table '{table_name}' was not found in any connected databases."
        return json.dumps(found_dbs)

    @mcp.tool
    def describe_table(self, name: str, table_name: str) -> str:
        """
        Get a detailed description of a table including its schema in JSON.

        Args:
            name: The name of the database connection.
            table_name: The name of the table.
        """
        try:
            engine = self._get_connection(name)
            inspector = inspect(engine)
            if table_name not in inspector.get_table_names():
                return (
                    f"Error: Table '{table_name}' does not exist in database '{name}'."
                )

            table_info = self._get_table_metadata(inspector, table_name)
            with engine.connect() as conn:
                safe_table_name = self._safe_table_identifier(table_name)
                result = conn.execute(text(f"SELECT COUNT(*) FROM {safe_table_name}"))
                row_count = result.scalar()
            table_info["size"] = row_count

            return json.dumps(table_info, indent=2)
        except Exception as e:
            return f"An error occurred: {e}"

    @mcp.tool
    def get_table_sample(self, name: str, table_name: str, limit: int = 10) -> str:
        """
        Get a sample of data from a table (default size 10 rows).

        Args:
            name: The name of the database connection.
            table_name: The name of the table.
            limit: The number of rows to return.
        """
        try:
            engine = self._get_connection(name)
            safe_table_name = self._safe_table_identifier(table_name)
            # Use parameterized query for LIMIT
            query = text(f"SELECT * FROM {safe_table_name} LIMIT :limit")
            df = pd.read_sql_query(query, engine, params={"limit": limit})
            if df.empty:
                return f"Table '{table_name}' is empty."
            return df.to_markdown()
        except Exception as e:
            return f"An error occurred while getting table sample: {e}"

    @mcp.tool
    def get_table_sample_json(self, name: str, table_name: str, limit: int = 10) -> str:
        """
        Get a sample of data from a table in JSON format.

        Args:
            name: The name of the database connection.
            table_name: The name of the table.
            limit: The number of rows to return.
        """
        try:
            engine = self._get_connection(name)
            safe_table_name = self._safe_table_identifier(table_name)
            # Use parameterized query for LIMIT
            query = text(f"SELECT * FROM {safe_table_name} LIMIT :limit")
            df = pd.read_sql_query(query, engine, params={"limit": limit})
            if df.empty:
                return json.dumps([])
            return df.to_json(orient="records")
        except Exception as e:
            return f"An error occurred while getting table sample: {e}"

    @mcp.tool
    def read_text_file(
        self, file_path: str, format: Literal["json", "yaml", "toml"]
    ) -> str:
        """
        Reads a structured text file (JSON, YAML, or TOML) and returns its content as a JSON string.

        Args:
            file_path: The path to the text file.
            format: The format of the file.
        """
        try:
            abs_file_path = self._sanitize_path(file_path)
            with open(abs_file_path, "r") as f:
                content = f.read()

            if format == "json":
                parsed_data = json.loads(content)
            elif format == "yaml":
                parsed_data = yaml.safe_load(content)
            elif format == "toml":
                parsed_data = toml.loads(content)
            else:
                return f"Error: Unsupported format '{format}'."

            return json.dumps(parsed_data, indent=2)
        except Exception as e:
            return f"An error occurred while reading the file: {e}"

    @mcp.tool
    def get_query_chunk(self, query_id: str, start_row: int, chunk_size: str) -> str:
        """
        Retrieve a chunk of rows from a buffered query result.

        Args:
            query_id: The ID of the buffered query result.
            start_row: The starting row number (1-based indexing).
            chunk_size: Number of rows to retrieve, or 'all' for all remaining rows.
        """
        try:
            # Clean up expired buffers
            self._cleanup_expired_buffers()
            
            with self.query_buffer_lock:
                if query_id not in self.query_buffers:
                    return f"Error: Query buffer '{query_id}' not found. It may have expired or been cleared."
                
                buffer = self.query_buffers[query_id]
            
            df = buffer.results
            total_rows = len(df)
            
            # Validate start_row (1-based indexing)
            if start_row < 1 or start_row > total_rows:
                return f"Error: start_row must be between 1 and {total_rows}."
            
            # Convert to 0-based indexing for pandas
            start_idx = start_row - 1
            
            # Handle chunk_size
            if chunk_size == 'all':
                end_idx = total_rows
                chunk_df = df.iloc[start_idx:]
            else:
                try:
                    chunk_size_int = int(chunk_size)
                    if chunk_size_int <= 0:
                        return "Error: chunk_size must be a positive integer or 'all'."
                    end_idx = min(start_idx + chunk_size_int, total_rows)
                    chunk_df = df.iloc[start_idx:end_idx]
                except ValueError:
                    return "Error: chunk_size must be a positive integer or 'all'."
            
            if chunk_df.empty:
                return json.dumps([])
            
            # Build response
            showing_end = start_idx + len(chunk_df)
            response = {
                "metadata": {
                    "query_id": query_id,
                    "total_rows": total_rows,
                    "showing_rows": f"{start_row}-{showing_end}",
                    "chunk_size": len(chunk_df),
                    "buffer_timestamp": buffer.timestamp,
                    "file_modified_since_buffer": self._check_file_modified(buffer)
                },
                "data": json.loads(chunk_df.to_json(orient="records"))
            }
            
            # Add next options if more rows available
            if showing_end < total_rows:
                next_start = showing_end + 1
                response["next_options"] = {
                    "get_next_100": f"get_query_chunk(query_id='{query_id}', start_row={next_start}, chunk_size=100)",
                    "get_all_remaining": f"get_query_chunk(query_id='{query_id}', start_row={next_start}, chunk_size='all')"
                }
            
            return json.dumps(response, indent=2)
            
        except Exception as e:
            return f"An error occurred while retrieving query chunk: {e}"

    @mcp.tool
    def get_buffered_query_info(self, query_id: str) -> str:
        """
        Get information about a buffered query result.

        Args:
            query_id: The ID of the buffered query result.
        """
        try:
            # Clean up expired buffers
            self._cleanup_expired_buffers()
            
            with self.query_buffer_lock:
                if query_id not in self.query_buffers:
                    return f"Error: Query buffer '{query_id}' not found. It may have expired or been cleared."
                
                buffer = self.query_buffers[query_id]
            
            info = {
                "query_id": query_id,
                "db_name": buffer.db_name,
                "query": buffer.query,
                "total_rows": len(buffer.results),
                "buffer_created": buffer.timestamp,
                "buffer_age_seconds": time.time() - buffer.timestamp,
                "expires_in_seconds": max(0, self.buffer_cleanup_interval - (time.time() - buffer.timestamp)),
                "source_file_path": buffer.source_file_path,
                "file_modified_since_buffer": self._check_file_modified(buffer)
            }
            
            return json.dumps(info, indent=2)
            
        except Exception as e:
            return f"An error occurred while retrieving query buffer info: {e}"

    @mcp.tool  
    def clear_query_buffer(self, query_id: str) -> str:
        """
        Clear a specific query buffer.

        Args:
            query_id: The ID of the buffered query result to clear.
        """
        try:
            with self.query_buffer_lock:
                if query_id not in self.query_buffers:
                    return f"Error: Query buffer '{query_id}' not found."
                
                del self.query_buffers[query_id]
            
            return f"Successfully cleared query buffer '{query_id}'."
            
        except Exception as e:
            return f"An error occurred while clearing query buffer: {e}"
    
    def _check_file_modified(self, buffer: QueryBuffer) -> bool:
        """Check if the source file has been modified since buffer creation."""
        if not buffer.source_file_path or not buffer.source_file_mtime:
            return False
        
        try:
            current_mtime = os.path.getmtime(buffer.source_file_path)
            return current_mtime > buffer.source_file_mtime
        except OSError:
            # File might not exist anymore
            return True
    
    def _safe_table_identifier(self, table_name: str) -> str:
        """Create a safe SQL identifier for table names to prevent injection."""
        # Validate table name contains only safe characters
        import re
        if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', table_name):
            raise ValueError(f"Invalid table name '{table_name}'. Table names must start with a letter or underscore and contain only alphanumeric characters and underscores.")
        
        # Use SQLAlchemy's quoted_name for safe identifier quoting
        return str(quoted_name(table_name, quote=True))

def main(): 
    manager = DatabaseManager()
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
