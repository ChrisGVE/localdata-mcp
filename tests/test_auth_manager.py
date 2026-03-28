"""Tests for enterprise authentication framework."""

import logging
from unittest.mock import patch

import pytest

from localdata_mcp.auth_manager import (
    AUTH_SUPPORT_MATRIX,
    AuthConfig,
    AuthMethod,
    AuthSecurityValidator,
    log_auth_attempt,
)


class TestAuthMethodEnum:
    """Tests for AuthMethod enum."""

    def test_auth_method_enum_values(self):
        """All 6 authentication methods are defined with correct values."""
        assert AuthMethod.PASSWORD == "password"
        assert AuthMethod.WALLET == "wallet"
        assert AuthMethod.KERBEROS == "kerberos"
        assert AuthMethod.TRUSTED == "trusted"
        assert AuthMethod.AZURE_AD == "azure_ad"
        assert AuthMethod.CERTIFICATE == "certificate"
        assert len(AuthMethod) == 6


class TestAuthSupportMatrix:
    """Tests for the AUTH_SUPPORT_MATRIX."""

    def test_auth_support_matrix_oracle(self):
        """Oracle supports password, wallet, kerberos, and certificate."""
        oracle = AUTH_SUPPORT_MATRIX["oracle"]
        assert oracle == {
            AuthMethod.PASSWORD,
            AuthMethod.WALLET,
            AuthMethod.KERBEROS,
            AuthMethod.CERTIFICATE,
        }

    def test_auth_support_matrix_mssql(self):
        """MSSQL supports password, trusted, kerberos, azure_ad, and certificate."""
        mssql = AUTH_SUPPORT_MATRIX["mssql"]
        assert mssql == {
            AuthMethod.PASSWORD,
            AuthMethod.TRUSTED,
            AuthMethod.KERBEROS,
            AuthMethod.AZURE_AD,
            AuthMethod.CERTIFICATE,
        }

    def test_auth_support_matrix_sqlite_empty(self):
        """SQLite has no authentication methods."""
        assert AUTH_SUPPORT_MATRIX["sqlite"] == set()


class TestAuthConfig:
    """Tests for AuthConfig."""

    def test_auth_config_defaults(self):
        """Default config uses password method with no extras."""
        config = AuthConfig()
        assert config.method == AuthMethod.PASSWORD
        assert config.wallet_path is None
        assert config.cert_path is None
        assert config.key_path is None
        assert config.token is None

    def test_auth_config_invalid_method(self):
        """Invalid method string raises ValueError."""
        with pytest.raises(ValueError, match="'bogus' is not a valid AuthMethod"):
            AuthConfig(method="bogus")

    def test_auth_config_to_dict_no_sensitive_data(self):
        """to_dict excludes raw token/password values."""
        config = AuthConfig(
            method="password",
            cert_path="/some/cert",
            token="super-secret-token",
        )
        result = config.to_dict()

        # Contains boolean indicators, not raw values
        assert result["method"] == "password"
        assert result["has_cert"] is True
        assert result["has_token"] is True

        # No sensitive data leaked
        assert "super-secret-token" not in str(result)
        assert "/some/cert" not in str(result)
        assert "token" not in [k for k in result if k != "has_token"]


class TestAuthSecurityValidator:
    """Tests for AuthSecurityValidator."""

    def test_validator_unsupported_method(self):
        """Wallet auth for sqlite raises ValueError."""
        validator = AuthSecurityValidator()
        with pytest.raises(ValueError, match="not supported for sqlite"):
            validator.validate({"method": "wallet"}, "sqlite")

    def test_validator_nonexistent_path(self):
        """Non-existent cert_path raises ValueError."""
        validator = AuthSecurityValidator()
        with pytest.raises(ValueError, match="cert_path does not exist"):
            validator.validate(
                {"method": "password", "cert_path": "/nonexistent/path/cert.pem"},
                "postgresql",
            )

    def test_validator_valid_config(self):
        """Password auth passes validation for any db type."""
        validator = AuthSecurityValidator()
        config = validator.validate({"method": "password"}, "postgresql")
        assert config.method == AuthMethod.PASSWORD


class TestLogAuthAttempt:
    """Tests for log_auth_attempt."""

    def test_log_auth_attempt_no_sensitive_data(self):
        """Logged message contains only db_type, method, connection — no secrets."""
        with patch("localdata_mcp.auth_manager.logger") as mock_logger:
            log_auth_attempt(
                db_type="postgresql",
                auth_method="password",
                success=True,
                connection_name="mydb",
            )

            mock_logger.info.assert_called_once()
            call_args = mock_logger.info.call_args
            # Reconstruct the logged message
            fmt = call_args[0][0]
            args = call_args[0][1:]
            message = fmt % args

            assert "postgresql" in message
            assert "password" in message
            assert "mydb" in message
            assert "success" in message
            # No token, secret, or credential strings
            assert "token" not in message.lower()
            assert "secret" not in message.lower()
