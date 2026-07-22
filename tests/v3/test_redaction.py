"""tests/v3/test_redaction.py — E2.4 structural credential redaction.

NX-4's outbound invariant (ARCHITECTURE.md section 4b/8, NFR-110): no
DSN-shaped string crosses a log edge un-redacted. The credential
shapes have one home — nexus/config/dsn_patterns.py, shared with the
endpoint refusal path — and the structlog chain applies them to every
event: message text, keyword values, nested containers, and rendered
exception text. Fault-injection cases log real credential material and
assert it never reaches stderr. Neighbors: test_config_endpoints.py
covers the refusal half of the shared patterns;
test_observability_bootstrap.py covers the chain's bring-up.
"""

from __future__ import annotations

from typing import Iterator

import pytest

from localdata_mcp.nexus.config.dsn_patterns import redact_credentials_text
from localdata_mcp.nexus.observability import bootstrap, get_logger


@pytest.fixture(autouse=True)
def fresh_logging_state() -> Iterator[None]:
    """Known logging state per test (bootstrap phase, INFO)."""
    bootstrap(force=True)
    yield
    bootstrap(force=True)


class TestRedactCredentialsText:
    def test_url_password_is_redacted_user_and_host_kept(self) -> None:
        redacted = redact_credentials_text(
            "connect failed for postgresql://alice:s3cret@db:5432/sales"
        )
        assert "s3cret" not in redacted
        assert "postgresql://alice:[REDACTED]@db:5432/sales" in redacted

    def test_keyword_password_is_redacted(self) -> None:
        redacted = redact_credentials_text("Server=h;Password=hunter2;Db=x")
        assert "hunter2" not in redacted
        assert "Password=[REDACTED]" in redacted

    def test_pwd_keyword_case_insensitive(self) -> None:
        redacted = redact_credentials_text("driver=odbc PWD = topsecret timeout=3")
        assert "topsecret" not in redacted

    def test_multiple_occurrences_all_redacted(self) -> None:
        text = "a mysql://u:one@h/d then redis://v:two@r/0"
        redacted = redact_credentials_text(text)
        assert "one" not in redacted
        assert "two" not in redacted

    def test_credential_free_text_unchanged(self) -> None:
        text = "loaded sqlite:///data.db and duckdb://analytics in 12ms"
        assert redact_credentials_text(text) == text


class TestOutboundLogEdge:
    def test_dsn_in_message_never_reaches_stderr(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        get_logger("redact-probe").info(
            "connect failed for postgresql://alice:s3cret@db:5432/sales"
        )
        err = capsys.readouterr().err
        assert "s3cret" not in err
        assert "[REDACTED]" in err

    def test_dsn_in_keyword_value_redacted(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        get_logger("redact-probe").info(
            "pool warmed", dsn="mysql://bob:hunter2@h:3306/d"
        )
        err = capsys.readouterr().err
        assert "hunter2" not in err
        assert "mysql://bob:[REDACTED]@h:3306/d" in err

    def test_dsn_nested_in_container_redacted(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        get_logger("redact-probe").info(
            "endpoints declared",
            dsns=["postgresql://a:pw1@x/1", {"backup": "Password=pw2;Server=y"}],
        )
        err = capsys.readouterr().err
        assert "pw1" not in err
        assert "pw2" not in err

    def test_stdlib_native_record_redacted_at_the_edge(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # A third-party library logging via stdlib bypasses the
        # structlog chain; the handler's formatter must still redact.
        import logging

        logging.getLogger("third.party").warning(
            "retrying postgresql://svc:libsecret@db/x"
        )
        err = capsys.readouterr().err
        assert "libsecret" not in err
        assert "postgresql://svc:[REDACTED]@db/x" in err

    def test_exception_text_redacted(self, capsys: pytest.CaptureFixture[str]) -> None:
        log = get_logger("redact-probe")
        try:
            raise ValueError("refused DSN oracle://sys:t0psecret@ora:1521/XE")
        except ValueError:
            log.exception("declaration refused")
        err = capsys.readouterr().err
        assert "t0psecret" not in err
        assert "declaration refused" in err
