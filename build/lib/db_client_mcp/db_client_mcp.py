# db-client-mcp.py

# /// script
# dependencies = [
#     "fastmcp",
#     "pandas",
#     "sqlalchemy",
#     "psycopg2-binary",
#     "mysql-connector-python",
#     "pymongo",
#     "pyyaml",
#     "toml"
# ]
# ///

import os
import json
import pandas as pd
import yaml
import toml
from fastmcp import FastMCP
from sqlalchemy import create_engine, inspect
from typing import Literal, Optional, Dict, Any, List

# Create the MCP server instance
mcp = FastMCP("db-client-mcp")


class DatabaseManager:
    def __init__(self):
        self.connections: Dict[str, Any] = {}
        self.query_history: Dict[str, List[str]] = {}

    def _get_connection(self, name: str):
        if name not in self.connections:
            raise ValueError(
                f"Database '{name}' is not connected. Use 'connect_database' first."
            )
        return self.connections[name]

    def _sanitize_path(self, file_path: str):
        base_dir = os.getcwd()
        abs_file_path = os.path.abspath(file_path)
        if not abs_file_path.startswith(base_dir):
            raise ValueError(f"Path '{file_path}' is outside the allowed directory.")
        if not os.path.isfile(abs_file_path):
            raise ValueError(f"File not found at path '{file_path}'.")
        return abs_file_path

    def _get_engine(self, db_type: str, conn_string: str):
        if db_type == "sqlite":
            return create_engine(f"sqlite:///{conn_string}")
        elif db_type == "postgresql":
            return create_engine(conn_string)
        elif db_type == "mysql":
            return create_engine(conn_string)
        elif db_type == "csv":
            self._sanitize_path(conn_string)
            try:
                df = pd.read_csv(conn_string)
            except pd.errors.ParserError:
                # Fallback for CSV with no header
                df = pd.read_csv(conn_string, header=None)
            engine = create_engine("sqlite:///:memory:")
            df.to_sql("data_table", engine, index=False, if_exists="replace")
            return engine
        else:
            raise ValueError(f"Unsupported db_type: {db_type}")

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
            db_type: The type of the database ("sqlite", "postgresql", "mysql", "csv").
            conn_string: The connection string or file path for the database.
        """
        if name in self.connections:
            return f"Error: A database with the name '{name}' is already connected."
        try:
            engine = self._get_engine(db_type, conn_string)
            self.connections[name] = engine
            self.query_history[name] = []
            return f"Successfully connected to database '{name}'."
        except Exception as e:
            return f"Failed to connect to database '{name}': {e}"

    @mcp.tool
    def disconnect_database(self, name: str):
        """
        Close a connection to a database. All open connections are closed when the script terminates.

        Args:
            name: The name of the database connection to close.
        """
        try:
            conn = self._get_connection(name)
            conn.dispose()
            del self.connections[name]
            del self.query_history[name]
            return f"Successfully disconnected from database '{name}'."
        except ValueError as e:
            return str(e)
        except Exception as e:
            return f"An error occurred while disconnecting: {e}"

    @mcp.tool
    def execute_query(self, name: str, query: str) -> str:
        """
        Execute a SQL query and return results as a markdown table.

        Args:
            name: The name of the database connection.
            query: The SQL query to execute.
        """
        try:
            engine = self._get_connection(name)
            df = pd.read_sql_query(query, engine)
            self.query_history[name].append(query)
            if df.empty:
                return "Query executed successfully, but no results were returned."
            return df.to_markdown()
        except Exception as e:
            return f"An error occurred while executing the query: {e}"

    @mcp.tool
    def execute_query_json(self, name: str, query: str) -> str:
        """
        Execute a SQL query and return results as JSON.

        Args:
            name: The name of the database connection.
            query: The SQL query to execute.
        """
        try:
            engine = self._get_connection(name)
            df = pd.read_sql_query(query, engine)
            self.query_history[name].append(query)
            if df.empty:
                return json.dumps([])
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
                    result = conn.execute(f"SELECT COUNT(*) FROM {table_name}")
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
                result = conn.execute(f"SELECT COUNT(*) FROM {table_name}")
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
            query = f"SELECT * FROM {table_name} LIMIT {limit}"
            df = pd.read_sql_query(query, engine)
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
            query = f"SELECT * FROM {table_name} LIMIT {limit}"
            df = pd.read_sql_query(query, engine)
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

def main(): 
    manager = DatabaseManager()
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
