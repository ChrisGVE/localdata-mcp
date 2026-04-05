"""LocalData MCP - CLI argument parsing and main entry point."""

import argparse
import importlib.metadata
import sys
from pathlib import Path

# Access patchable module-level names via the canonical module so that
# ``unittest.mock.patch("localdata_mcp.localdata_mcp.XXX")`` works.
import localdata_mcp.localdata_mcp as _parent  # noqa: E402


def _get_version() -> str:
    """Return package version from metadata, with fallback."""
    try:
        return importlib.metadata.version("localdata-mcp")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _parse_cli_args() -> argparse.Namespace:
    """Parse CLI arguments without consuming stdin (needed for MCP stdio transport)."""
    parser = argparse.ArgumentParser(
        description="LocalData MCP server",
        prog="localdata-mcp",
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default=None,
        help="Path to YAML configuration file (highest precedence)",
    )
    parser.add_argument(
        "--version",
        "-V",
        action="version",
        version=f"localdata-mcp {_get_version()}",
    )
    parser.add_argument(
        "--migrate-config",
        action="store_true",
        default=False,
        help="Migrate legacy configuration to YAML format",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Force overwrite during migration",
    )
    parser.add_argument(
        "--validate-config",
        action="store_true",
        default=False,
        help="Validate configuration and exit",
    )
    parser.add_argument(
        "--show-config",
        action="store_true",
        default=False,
        help="Show resolved configuration and exit",
    )
    parser.add_argument(
        "--init-config",
        action="store_true",
        default=False,
        help="Create default config file and exit",
    )
    args, _ = parser.parse_known_args(sys.argv[1:])
    return args


def main():
    """Main entry point with structured logging initialization."""
    args = _parse_cli_args()

    if args.migrate_config:
        from ..config_paths import get_recommended_path, migrate_config

        try:
            source = Path("~/.localdata.yaml").expanduser()
            dest = get_recommended_path()
            if args.force and dest.exists():
                dest.unlink()
            print(f"Migrating config from {source} to {dest}...")
            migrate_config(source=source, dest=dest)
            print(f"Config migrated successfully to {dest}")
        except FileNotFoundError:
            print("No legacy config found at ~/.localdata.yaml")
            sys.exit(1)
        except FileExistsError:
            print(f"Config already exists at {dest}. Use --force to overwrite.")
            sys.exit(1)
        except Exception as e:
            print(f"Migration failed: {e}")
            sys.exit(1)
        sys.exit(0)

    if args.init_config:
        from ..config_paths import create_default_config

        try:
            path = create_default_config()
            print(f"Default config created at {path}")
        except FileExistsError:
            print(
                f"Config file already exists. Remove it first or use a different path."
            )
            sys.exit(1)
        except Exception as e:
            print(f"Failed to create config: {e}")
            sys.exit(1)
        sys.exit(0)

    if args.validate_config:
        try:
            mgr = _parent.get_config_manager()
            mgr._validate_config()
            print("Configuration is valid.")
        except Exception as e:
            print(f"Configuration validation failed: {e}")
            sys.exit(1)
        sys.exit(0)

    if args.show_config:
        import copy
        import re as _re

        mgr = _parent.get_config_manager()
        data = copy.deepcopy(mgr._config_data)

        def _redact(obj):
            """Recursively redact sensitive values."""
            if isinstance(obj, dict):
                for key in obj:
                    if any(
                        s in key.lower()
                        for s in (
                            "password",
                            "secret",
                            "token",
                            "connection_string",
                            "key_path",
                            "cert_path",
                            "wallet_path",
                        )
                    ):
                        obj[key] = "***REDACTED***"
                    else:
                        _redact(obj[key])
            elif isinstance(obj, list):
                for item in obj:
                    _redact(item)

        _redact(data)
        print(_parent.yaml.dump(data, default_flow_style=False, sort_keys=True))
        sys.exit(0)

    if args.config:
        _parent.initialize_config(config_file=args.config)

    try:
        version = _get_version()
        with _parent.logging_manager.context(
            operation="system_startup", component="localdata_mcp"
        ):
            _parent.logger.info(
                "LocalData MCP starting up",
                version=version,
                structured_logging_enabled=True,
                metrics_enabled=_parent.logging_config.enable_metrics,
                security_logging_enabled=_parent.logging_config.enable_security_logging,
            )

        manager = _parent.DatabaseManager()

        with _parent.logging_manager.context(
            operation="system_ready", component="localdata_mcp"
        ):
            _parent.logger.info(
                "LocalData MCP ready to accept connections",
                transport="stdio",
                logging_level=_parent.logging_config.level.value,
                metrics_endpoint=f"http://localhost:{_parent.logging_config.metrics_port}{_parent.logging_config.metrics_endpoint}"
                if _parent.logging_config.enable_metrics
                else None,
            )

        _parent.mcp.run(transport="stdio")

    except Exception as e:
        _parent.logging_manager.log_error(
            e, "localdata_mcp", operation="system_startup"
        )
        raise


if __name__ == "__main__":
    main()
