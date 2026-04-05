"""Engine creation logic for the Enhanced Connection Manager.

Provides mixin class with methods for creating and configuring
SQLAlchemy engines for various database types.
"""

import logging
import time
from typing import Any, Dict

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.pool import QueuePool, StaticPool

from ..config_manager import DatabaseConfig, DatabaseType

logger = logging.getLogger(__name__)


class EngineFactoryMixin:
    """Mixin providing engine creation methods for EnhancedConnectionManager."""

    def _create_enhanced_engine(self, name: str, config: DatabaseConfig) -> Engine:
        """Create a SQLAlchemy engine with enhanced configuration.

        Args:
            name: Database name.
            config: Database configuration.

        Returns:
            Configured SQLAlchemy engine.
        """
        engine_args: Dict[str, Any] = {"echo": False}

        if config.type in {DatabaseType.POSTGRESQL, DatabaseType.MYSQL}:
            engine_args.update(
                {
                    "poolclass": QueuePool,
                    "pool_size": min(config.max_connections, 10),
                    "max_overflow": max(0, config.max_connections - 10),
                    "pool_timeout": config.connection_timeout,
                    "pool_recycle": 3600,
                    "pool_pre_ping": True,
                }
            )
        elif config.type in {DatabaseType.SQLITE, DatabaseType.DUCKDB}:
            engine_args.update(
                {
                    "poolclass": StaticPool,
                    "connect_args": {"check_same_thread": False},
                }
            )

        engine = self._create_engine_by_type(config, engine_args)
        self._setup_engine_events(engine, name)
        return engine

    def _create_engine_by_type(
        self, config: DatabaseConfig, engine_args: Dict[str, Any]
    ) -> Engine:
        """Create a SQLAlchemy engine based on database type.

        Args:
            config: Database configuration.
            engine_args: Engine keyword arguments.

        Returns:
            SQLAlchemy engine instance.

        Raises:
            ValueError: If the database type is unsupported or a required
                driver is missing.
        """
        if config.type == DatabaseType.SQLITE:
            return create_engine(f"sqlite:///{config.connection_string}", **engine_args)
        if config.type == DatabaseType.DUCKDB:
            return create_engine(f"duckdb:///{config.connection_string}", **engine_args)
        if config.type in {DatabaseType.POSTGRESQL, DatabaseType.MYSQL}:
            return create_engine(config.connection_string, **engine_args)
        if config.type == DatabaseType.ORACLE:
            return self._create_oracle_engine(config)
        if config.type == DatabaseType.MSSQL:
            return self._create_mssql_engine(config)

        file_types = {
            "csv",
            "json",
            "yaml",
            "excel",
            "ods",
            "numbers",
        }
        if config.type.value in file_types:
            return create_engine("sqlite:///:memory:", **engine_args)

        raise ValueError(f"Unsupported database type: {config.type.value}")

    @staticmethod
    def _create_oracle_engine(config: DatabaseConfig) -> Engine:
        """Create an Oracle database engine.

        Args:
            config: Database configuration.

        Returns:
            Oracle SQLAlchemy engine.

        Raises:
            ValueError: If oracledb is not installed.
        """
        from ..oracle_support import (
            create_oracle_engine,
            ORACLEDB_AVAILABLE,
        )

        if not ORACLEDB_AVAILABLE:
            raise ValueError(
                "Oracle support requires 'oracledb'. "
                "Install with: pip install localdata-mcp[enterprise]"
            )
        auth = config.metadata.get("auth") if config.metadata else None
        return create_oracle_engine(config.connection_string, auth=auth)

    @staticmethod
    def _create_mssql_engine(config: DatabaseConfig) -> Engine:
        """Create a MS SQL database engine.

        Args:
            config: Database configuration.

        Returns:
            MSSQL SQLAlchemy engine.

        Raises:
            ValueError: If neither pymssql nor pyodbc is installed.
        """
        from ..mssql_support import (
            create_mssql_engine,
            PYMSSQL_AVAILABLE,
            PYODBC_AVAILABLE,
        )

        if not PYMSSQL_AVAILABLE and not PYODBC_AVAILABLE:
            raise ValueError(
                "MS SQL support requires 'pymssql' or 'pyodbc'. "
                "Install: pip install localdata-mcp[enterprise]"
            )
        auth = config.metadata.get("auth") if config.metadata else None
        return create_mssql_engine(config.connection_string, auth=auth)

    def _setup_engine_events(self, engine: Engine, database_name: str) -> None:
        """Set up SQLAlchemy event listeners for monitoring.

        Args:
            engine: SQLAlchemy engine.
            database_name: Database name for tracking.
        """

        @event.listens_for(engine, "before_cursor_execute")
        def before_cursor_execute(
            conn, cursor, statement, parameters, context, executemany
        ):
            context._query_start_time = time.time()

        @event.listens_for(engine, "after_cursor_execute")
        def after_cursor_execute(
            conn, cursor, statement, parameters, context, executemany
        ):
            if hasattr(context, "_query_start_time"):
                query_time = time.time() - context._query_start_time
                with self._lock:
                    metrics = self._metrics[database_name]
                    metrics.total_query_time += query_time
                    if metrics.total_queries > 0:
                        metrics.average_query_time = (
                            metrics.total_query_time / metrics.total_queries
                        )
