"""Structlog and stdlib logging configuration for LoggingManager."""

import sys
import logging
import logging.handlers
from pathlib import Path

import structlog

from ..config_manager import LoggingConfig, LogLevel


def configure_structlog(config: LoggingConfig, add_context, add_correlation_id):
    """Configure structlog with processors and renderers.

    Args:
        config: Logging configuration.
        add_context: Processor callback to inject thread-local context.
        add_correlation_id: Processor callback to inject correlation IDs.
    """
    processors = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        add_context,
        add_correlation_id,
        structlog.processors.StackInfoRenderer(),
        structlog.dev.set_exc_info,
    ]

    if config.level != LogLevel.DEBUG:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer(colors=True))

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        context_class=dict,
        cache_logger_on_first_use=True,
    )


def configure_stdlib_logging(config: LoggingConfig):
    """Configure standard library logging integration.

    Args:
        config: Logging configuration.
    """
    root_logger = logging.getLogger()
    root_logger.handlers.clear()

    level_map = {
        LogLevel.DEBUG: logging.DEBUG,
        LogLevel.INFO: logging.INFO,
        LogLevel.WARNING: logging.WARNING,
        LogLevel.ERROR: logging.ERROR,
        LogLevel.CRITICAL: logging.CRITICAL,
    }
    root_logger.setLevel(level_map[config.level])

    if config.console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level_map[config.level])
        if config.level == LogLevel.DEBUG:
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
        else:
            formatter = logging.Formatter("%(message)s")
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    if config.file_path:
        file_path = Path(config.file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.handlers.RotatingFileHandler(
            file_path,
            maxBytes=config.max_file_size,
            backupCount=config.backup_count,
        )
        file_handler.setLevel(level_map[config.level])
        file_formatter = logging.Formatter("%(message)s")
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
