"""How the harness addresses an endpoint: from the host, or from inside its network.

Offline. Nothing here needs a container. ``tests/endpoints.py`` has two ways of
reaching a service, and they differ only in the address:

* **host mode**, the default: ``127.0.0.1`` and the port the compose file
  *publishes*. This is how a test process on the developer's machine reaches a
  container.
* **in-network mode**, selected by the environment variable
  :data:`endpoints.IN_NETWORK`: the compose **service name** and the port the
  container listens on *inside* itself. This is how the containerised runner
  started by ``scripts/endpoint-run.sh`` reaches a database on the compose
  network it joins (task 35, #139).

Everything that carries an address has to follow the mode, not only the URL.
That includes the ``.pgpass`` host field, the ODBC DSN's ``Server`` line, the
KDC a Kerberos client asks, the name a TLS certificate vouches for, and the
principal a keytab holds. A mode that moved the URL and left one of those
behind would connect in one mode and fail in the other. So each is asserted
here.
"""

from __future__ import annotations

import socket
import tempfile
from pathlib import Path

import endpoints
import pytest
import yaml
from endpoints import Target, Unavailable
from sqlalchemy.engine import make_url


class _Plain(dict):
    """An environment answering any variable with a harmless value.

    The credential sweep in test_dialects.py uses a hostile one. Addressing is
    the subject here, so the values only have to be valid.
    """

    def __missing__(self, key: str) -> str:
        return "value"


def _services() -> dict:
    return yaml.safe_load(endpoints.COMPOSE.read_text())["services"]


# ---------------------------------------------------------------------------
# Which mode is in force
# ---------------------------------------------------------------------------


def test_host_mode_is_the_default(monkeypatch):
    monkeypatch.delenv(endpoints.IN_NETWORK, raising=False)
    assert endpoints.in_network() is False


def test_in_network_mode_is_selected_by_one_variable(monkeypatch):
    monkeypatch.setenv(endpoints.IN_NETWORK, "1")
    assert endpoints.in_network() is True
    monkeypatch.setenv(endpoints.IN_NETWORK, "0")
    assert endpoints.in_network() is False


def test_an_unrecognised_mode_value_is_refused_rather_than_read_as_host(monkeypatch):
    """``true`` is the likely typo, and read as host mode it would fail open.

    Inside the runner nothing listens on 127.0.0.1, so every endpoint would
    *skip*. The run would be green and would have tested nothing. A value the
    harness does not recognise is refused out loud instead.
    """
    monkeypatch.setenv(endpoints.IN_NETWORK, "true")
    with pytest.raises(RuntimeError, match=endpoints.IN_NETWORK):
        endpoints.in_network()


# ---------------------------------------------------------------------------
# The address each mode resolves to
# ---------------------------------------------------------------------------


def test_host_mode_addresses_the_loopback_and_the_published_port(monkeypatch):
    monkeypatch.delenv(endpoints.IN_NETWORK, raising=False)
    services = _services()
    for target in endpoints.TARGETS:
        host, port = endpoints._address(target, services[target.service])
        assert host == endpoints.HOST, target.name
        assert port == endpoints._published_port(target, services[target.service])


def test_in_network_mode_addresses_the_service_name_and_its_own_port(monkeypatch):
    monkeypatch.setenv(endpoints.IN_NETWORK, "1")
    services = _services()
    for target in endpoints.TARGETS:
        host, port = endpoints._address(target, services[target.service])
        assert host == target.service, target.name
        assert port == target.container_port, target.name


def test_in_network_mode_still_refuses_a_port_the_compose_file_does_not_map(
    monkeypatch,
):
    """In-network mode never uses the published port, but it still checks it.

    The mapping is how this module notices that the endpoint table and the
    compose file disagree about a service's port. Dropping the check in one
    mode would let that drift through in exactly the mode the standard run
    now uses.
    """
    monkeypatch.setenv(endpoints.IN_NETWORK, "1")
    target = Target(endpoints.ENDPOINTS[0])
    with pytest.raises(RuntimeError, match="none of which maps"):
        endpoints._address(target, {"ports": ["1:2"]})


def test_nothing_listening_on_an_unresolvable_name_is_a_skip_not_an_error():
    """In-network, a service that is not up has a name that does not resolve.

    In host mode an absent container is a refused connection on 127.0.0.1. On
    a compose network it is ``socket.gaierror``, because Docker's DNS only
    knows the services that are running. Both mean "not running here", so both
    have to answer False rather than raise out of the probe.
    """
    assert endpoints._listening("localdata-test-no-such-service.invalid", 5432) is False


def test_other_service_follows_the_mode(monkeypatch):
    """The Kerberos mode reads a second service, the KDC, and it moves too."""
    monkeypatch.delenv(endpoints.IN_NETWORK, raising=False)
    _, host, port = endpoints._other_service(endpoints.KDC, 88)
    assert (host, port) == (endpoints.HOST, 1088)

    monkeypatch.setenv(endpoints.IN_NETWORK, "1")
    _, host, port = endpoints._other_service(endpoints.KDC, 88)
    assert (host, port) == (endpoints.KDC, 88)


# ---------------------------------------------------------------------------
# Every builder carries the host it is given
# ---------------------------------------------------------------------------


def test_every_target_builds_a_url_for_the_host_it_is_given():
    """A swept assertion, so a builder added later cannot ignore the host.

    The ODBC DSN mode is the one URL here whose host position names a data
    source rather than an address. For that mode the host has to land in the
    DSN file's ``Server`` line instead, and that is checked below.
    """
    swept = 0
    for target in endpoints.TARGETS:
        if target.auth is not None and target.auth.mode == "kerberos":
            continue  # needs a KDC to answer; its config is checked below
        scratch = Path(tempfile.mkdtemp(prefix="localdata-addressing-"))
        try:
            reached = target.build(
                _Plain(), target.container_port, scratch, host=target.service
            )
        except Unavailable:
            continue  # SQL Server with no ODBC driver on this machine
        swept += 1
        parsed = make_url(reached.url)
        if parsed.port is None:
            odbcini = Path(reached.environ["ODBCINI"]).read_text()
            assert f"Server = {target.service}" in odbcini, target.name
        else:
            assert parsed.host == target.service, target.name
            assert parsed.port == target.container_port, target.name
    assert swept, "no target was swept, so this asserted nothing"


def test_the_builders_default_to_the_loopback_host():
    """Host mode stays what it was: a builder called without a host uses HOST."""
    for endpoint in endpoints.ENDPOINTS:
        try:
            built = endpoint.url(_Plain(), 15432)
        except Unavailable:
            continue
        assert make_url(built).host == endpoints.HOST, endpoint.name


def test_the_pgpass_file_names_the_host_the_url_connects_to():
    """libpq matches the file's host field against the host it is connecting to.

    A file that still said 127.0.0.1 while the URL named a service would match
    nothing. libpq would then report ``no password supplied``, which names the
    credential rather than the address.
    """
    scratch = Path(tempfile.mkdtemp(prefix="localdata-pgpass-host-"))
    reached = endpoints._postgres_pgpassfile(
        _Plain(), 5432, scratch, host="localdata-test-postgres"
    )
    line = Path(reached.environ["PGPASSFILE"]).read_text()
    assert line.startswith("localdata-test-postgres:5432:")


def test_the_kerberos_client_config_names_the_kdc_it_is_given():
    config = endpoints._krb5_conf("LOCALDATA.TEST", "localdata-test-kdc", 88)
    assert "kdc = localdata-test-kdc:88" in config
    assert "default_realm = LOCALDATA.TEST" in config


# ---------------------------------------------------------------------------
# The compose file vouches for the names in-network mode uses
# ---------------------------------------------------------------------------


def _dependants(services: dict, helper: str) -> list[str]:
    return [
        name
        for name, service in services.items()
        if helper in (service.get("depends_on") or {})
    ]


def test_the_tls_certificate_names_every_server_that_uses_it():
    """``verify-full`` compares the name the client asked for with the certificate.

    In host mode that name is 127.0.0.1. In-network it is the service name. A
    certificate that named only the first would make the TLS modes fail under
    the runner with a hostname mismatch, so the name list has to cover every
    service that takes its certificate from the CA, and the loopback address
    as well.
    """
    services = _services()
    names = endpoints._environment(services["localdata-test-tls-ca"])[
        "TLS_SERVER_NAMES"
    ].split()
    assert "127.0.0.1" in names
    for dependant in _dependants(services, "localdata-test-tls-ca"):
        assert dependant in names, dependant


def test_the_keytab_holds_a_principal_for_every_server_that_uses_it():
    """A client asks for ``postgres/<the host it connected to>``.

    Host mode connects to 127.0.0.1 and in-network mode to the service name, so
    the keytab must hold both principals. If either is missing, the KDC answers
    ``Server not found in Kerberos database``.
    """
    services = _services()
    hosts = endpoints._environment(services[endpoints.KDC])[
        "KRB5_SERVICE_HOSTS"
    ].split()
    assert "127.0.0.1" in hosts
    for dependant in _dependants(services, endpoints.KDC):
        assert dependant in hosts, dependant


#: The overlay scripts/endpoint-run.sh applies on top of the compose file.
IN_NETWORK_OVERLAY = (
    endpoints.COMPOSE.parent / "docker" / "test" / "in-network.compose.yml"
)


def test_the_in_network_overlay_names_only_services_the_compose_file_defines():
    overlay = yaml.safe_load(IN_NETWORK_OVERLAY.read_text())
    defined = set(_services())
    for name in overlay.get("services", {}):
        assert name in defined, name


def test_a_node_that_advertises_its_address_advertises_its_service_name_in_network():
    """YDB's client connects to the address the node advertises.

    From the host that address is ``localhost``, with the port published to
    itself. In-network ``localhost`` is the test container, so the overlay has
    to make the node advertise the name the runner resolves, which is its
    service name. Any other value would pass the handshake and then fail every
    statement.
    """
    overlay = yaml.safe_load(IN_NETWORK_OVERLAY.read_text())["services"]
    assert _services()["localdata-test-ydb"]["hostname"] == "localhost"
    for name, service in overlay.items():
        if "hostname" in service:
            assert service["hostname"] == name, name


def test_the_in_network_overlay_gives_each_project_its_own_network():
    """A fixed network name would be shared by every project made from the file.

    The main file's ``localdata-test-net`` is what a host-mode stack uses. A run
    that reused it could join a network another stack on this machine is still
    using.
    """
    overlay = yaml.safe_load(IN_NETWORK_OVERLAY.read_text())
    assert "${COMPOSE_PROJECT_NAME}" in overlay["networks"]["default"]["name"]


# ---------------------------------------------------------------------------
# Which tests a service's run selects
# ---------------------------------------------------------------------------


def test_a_database_service_selects_its_endpoint_by_name():
    """The same selection the host runs used: ``-k <endpoint name>``.

    It matches the endpoint's own target and every mode whose id carries the
    name, and those modes skip when their service is not the one up. That is
    what docs/CONSTRAINTS.md §25.7's per-engine counts were taken with, so the
    in-network counts compare with them directly.
    """
    assert endpoints.selection("localdata-test-postgres") == "postgres"
    assert endpoints.selection("localdata-test-exasol") == "exasol"


def test_an_authentication_service_selects_exactly_the_modes_it_serves():
    assert (
        endpoints.selection("localdata-test-postgres-tls")
        == "postgres[tls-verify-full] or postgres[client-cert]"
    )
    assert endpoints.selection("localdata-test-postgres-krb") == "postgres[kerberos]"
    assert (
        endpoints.selection("localdata-test-clickhouse-noauth")
        == "clickhouse[empty-password]"
    )


def test_a_helper_selects_nothing_and_is_not_runnable_on_its_own():
    """The CA and the KDC have no tests of their own.

    Each comes up as a dependency of the database that needs it, so it is not
    a run of its own.
    """
    assert endpoints.selection("localdata-test-tls-ca") == ""
    assert endpoints.selection("localdata-test-kdc") == ""
    runnable = endpoints.runnable_services()
    assert "localdata-test-tls-ca" not in runnable
    assert "localdata-test-kdc" not in runnable


def test_every_service_a_target_names_is_runnable_and_in_compose_order():
    runnable = endpoints.runnable_services()
    assert set(runnable) == {target.service for target in endpoints.TARGETS}
    order = list(_services())
    assert runnable == [name for name in order if name in runnable]


def test_a_service_the_compose_file_does_not_define_is_refused():
    with pytest.raises(RuntimeError, match="no service"):
        endpoints.selection("localdata-test-no-such-thing")


def test_the_unresolvable_probe_name_really_does_not_resolve():
    """The probe above is only a test if the name it uses really fails to resolve.

    ``.invalid`` is reserved by RFC 2606 so that it never resolves, and this
    test checks that the machine agrees.
    """
    with pytest.raises(socket.gaierror):
        socket.getaddrinfo("localdata-test-no-such-service.invalid", 5432)
