"""Enterprise authentication framework for LocalData MCP."""

import logging
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Set

logger = logging.getLogger(__name__)


class AuthMethod(str, Enum):
    """Supported authentication methods."""

    PASSWORD = "password"
    WALLET = "wallet"
    KERBEROS = "kerberos"
    TRUSTED = "trusted"
    AZURE_AD = "azure_ad"
    CERTIFICATE = "certificate"


AUTH_SUPPORT_MATRIX: Dict[str, Set[AuthMethod]] = {
    "oracle": {
        AuthMethod.PASSWORD,
        AuthMethod.WALLET,
        AuthMethod.KERBEROS,
        AuthMethod.CERTIFICATE,
    },
    "mssql": {
        AuthMethod.PASSWORD,
        AuthMethod.TRUSTED,
        AuthMethod.KERBEROS,
        AuthMethod.AZURE_AD,
        AuthMethod.CERTIFICATE,
    },
    "postgresql": {AuthMethod.PASSWORD, AuthMethod.KERBEROS, AuthMethod.CERTIFICATE},
    "mysql": {AuthMethod.PASSWORD, AuthMethod.KERBEROS, AuthMethod.CERTIFICATE},
    "sqlite": set(),
    "duckdb": set(),
}


class AuthConfig:
    """Validated authentication configuration."""

    def __init__(
        self,
        method: str = "password",
        wallet_path: Optional[str] = None,
        cert_path: Optional[str] = None,
        key_path: Optional[str] = None,
        token: Optional[str] = None,
    ):
        self.method = AuthMethod(method)
        self.wallet_path = wallet_path
        self.cert_path = cert_path
        self.key_path = key_path
        self.token = token

    def to_dict(self) -> Dict[str, Any]:
        """Serialize without sensitive data."""
        return {
            "method": self.method.value,
            "has_wallet": self.wallet_path is not None,
            "has_cert": self.cert_path is not None,
            "has_token": self.token is not None,
        }


class AuthSecurityValidator:
    """Validates auth config against security policies."""

    def validate(self, auth: Dict[str, Any], db_type: str) -> AuthConfig:
        """Validate and return typed AuthConfig."""
        allowed_keys = ("method", "wallet_path", "cert_path", "key_path", "token")
        config = AuthConfig(**{k: v for k, v in auth.items() if k in allowed_keys})

        # Check method is supported
        supported = AUTH_SUPPORT_MATRIX.get(db_type, set())
        if config.method != AuthMethod.PASSWORD and config.method not in supported:
            raise ValueError(
                f"Auth method '{config.method.value}' not supported for {db_type}. "
                f"Supported: {[m.value for m in supported]}"
            )

        # Validate paths exist
        for attr in ("wallet_path", "cert_path", "key_path"):
            path_val = getattr(config, attr)
            if path_val and not Path(path_val).exists():
                raise ValueError(f"{attr} does not exist: {path_val}")

        return config


def log_auth_attempt(
    db_type: str,
    auth_method: str,
    success: bool,
    connection_name: Optional[str] = None,
) -> None:
    """Log auth attempt without sensitive data."""
    logger.info(
        "Auth %s: db_type=%s method=%s connection=%s",
        "success" if success else "failure",
        db_type,
        auth_method,
        connection_name or "unknown",
    )
