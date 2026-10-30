"""Masking a credential where it is one, and leaving every other word alone.

The masker is the one place a password is taken out of text this server hands
back. It used to be a substring replacement, which turned a short password into
a corruption of the message around it (#82): password ``e`` made "connection
refused" read "conn***ction r***fus***d". These pin the rule that replaced it —
a credential is recognised by **where it sits**, not by how it is spelled.
"""

from __future__ import annotations

import pytest
from sqlalchemy.engine import make_url

from localdata_mcp.credentials import mask_credentials, redacted_url


def test_the_password_as_an_ordinary_substring_is_left_alone():
    url = make_url("postgresql://u:e@db.example.com/sales")
    message = "connection to server failed: Connection refused"

    assert mask_credentials(message, connecting_to=url) == message


def test_a_url_in_the_text_loses_its_password_and_nothing_else():
    message = "could not open postgresql://alice:hunter2@db.example.com:5432/sales"

    assert mask_credentials(message) == (
        "could not open postgresql://alice:***@db.example.com:5432/sales"
    )


def test_a_url_flattened_into_a_path_still_loses_its_password():
    """Path normalisation turns ``://`` into ``:/`` before a message is written."""
    message = "No such file: /work/postgresql:/u:hunter2@localhost:5432/db"

    assert mask_credentials(message) == (
        "No such file: /work/postgresql:/u:***@localhost:5432/db"
    )


def test_a_url_without_a_password_is_unchanged():
    message = "could not open postgresql://alice@db.example.com/sales"

    assert mask_credentials(message) == message


@pytest.mark.parametrize(
    ("echoed", "masked"),
    [
        (
            "username='u', password='e', config={}",
            "username='u', password='***', config={}",
        ),
        ('{"user": "u", "password": "x y"}', '{"user": "u", "password": "***"}'),
        ("DRIVER={X};UID=u;PWD=secret;SERVER=h", "DRIVER={X};UID=u;PWD=***;SERVER=h"),
        (
            "host=h user=u password=secret dbname=d",
            "host=h user=u password=*** dbname=d",
        ),
    ],
)
def test_a_password_a_driver_echoes_under_its_key_is_masked(echoed, masked):
    url = make_url("postgresql://u:e@h/d")

    assert mask_credentials(echoed, connecting_to=url) == masked


def test_a_password_key_is_only_a_credential_in_a_connection_complaint():
    """Outside a connection, ``password = 'x'`` is somebody's SQL, not a secret."""
    message = "no such column: password in WHERE password = 'x'"

    assert mask_credentials(message) == message


def test_a_password_holding_url_characters_is_masked_however_it_is_spelled():
    url = make_url("postgresql://u:p%40ss%2Fw%3Ard@h/d")
    decoded = "postgresql://u:p@ss/w:rd@h/d"
    encoded = "postgresql://u:p%40ss%2Fw%3Ard@h/d"

    assert mask_credentials(decoded, connecting_to=url) == "postgresql://u:***@h/d"
    assert mask_credentials(encoded, connecting_to=url) == "postgresql://u:***@h/d"
    assert mask_credentials("password=p@ss/w:rd", connecting_to=url) == "password=***"


def test_a_password_with_a_space_is_masked_whole_under_its_key():
    url = make_url("postgresql://u:a%20b@h/d")

    assert mask_credentials("host=h password=a b user=u", connecting_to=url) == (
        "host=h password=*** user=u"
    )


def test_masking_twice_changes_nothing():
    url = make_url("postgresql://u:e@h/d")
    once = mask_credentials("postgresql://u:e@h/d password='e'", connecting_to=url)

    assert mask_credentials(once, connecting_to=url) == once


def test_a_redacted_url_says_a_password_was_given_only_when_one_was():
    assert redacted_url(make_url("postgresql://u:pw@h/d")) == "postgresql://u:***@h/d"
    assert redacted_url(make_url("postgresql://u@h/d")) == "postgresql://u@h/d"
