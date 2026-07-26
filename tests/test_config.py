"""What the configuration cascade must do.

The cascade is *first found wins*, not a merge, so most of these tests are
about which candidate is consulted before which other one. The rest are about
refusing a configuration rather than running with a misunderstood one — a
mistyped ``path_limited`` that silently kept its safe default would be a
security setting the user believes they changed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from localdata_mcp import config as config_module
from localdata_mcp.config import (
    DEFAULT_MEMORY_BUDGET_MB,
    MAX_SLOTS,
    Config,
    ConfigError,
)


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch, tmp_path):
    """Cut every path the cascade consults away from the real machine."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    monkeypatch.delenv("LOCALDATA_CONFIG_PATH", raising=False)
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    monkeypatch.delenv("APPDATA", raising=False)
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    config_module.reset()
    yield
    config_module.reset()


def write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


def test_no_configuration_anywhere_yields_defaults():
    loaded = config_module.load()
    assert loaded == Config()
    assert loaded.slots == MAX_SLOTS
    assert loaded.roots == ()
    assert loaded.source is None


def test_the_default_posture_is_closed():
    """Both switches that widen what the server may reach default to off."""
    loaded = config_module.load()
    assert loaded.path_limited is True
    assert loaded.network_enabled is False


# ---------------------------------------------------------------------------
# Cascade order
# ---------------------------------------------------------------------------


def test_xdg_is_consulted_before_the_project_file(tmp_path, monkeypatch):
    xdg = tmp_path / "xdg"
    monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg))
    write(xdg / "localdata" / "config.toml", "[workspace]\nslots = 3\n")
    write(Path("localdata.toml"), "[workspace]\nslots = 7\n")

    loaded = config_module.load()
    assert loaded.slots == 3


def test_the_project_file_is_used_when_xdg_has_none(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "empty-xdg"))
    write(Path("localdata.toml"), "[workspace]\nslots = 7\n")

    loaded = config_module.load()
    assert loaded.slots == 7
    assert loaded.source == Path("localdata.toml").resolve()


def test_the_project_file_is_consulted_before_the_os_native_path(tmp_path, monkeypatch):
    """Order 1, 2, 5, 3, 4 — project-local outranks the platform default."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "empty-xdg"))
    monkeypatch.setattr(config_module.sys, "platform", "darwin")
    native = (
        Path.home() / "Library" / "Application Support" / "localdata" / "config.toml"
    )
    write(native, "[workspace]\nslots = 4\n")
    write(Path("localdata.toml"), "[workspace]\nslots = 7\n")

    assert config_module.load().slots == 7


def test_the_os_native_path_is_the_last_resort(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "empty-xdg"))
    monkeypatch.setattr(config_module.sys, "platform", "darwin")
    native = (
        Path.home() / "Library" / "Application Support" / "localdata" / "config.toml"
    )
    write(native, "[workspace]\nslots = 4\n")

    assert config_module.load().slots == 4


def test_xdg_defaults_to_dot_config_when_unset():
    write(
        Path.home() / ".config" / "localdata" / "config.toml",
        "[workspace]\nslots = 5\n",
    )
    assert config_module.load().slots == 5


def test_windows_looks_in_appdata_and_not_in_application_support(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "empty-xdg"))
    monkeypatch.setattr(config_module.sys, "platform", "win32")
    appdata = tmp_path / "AppData" / "Roaming"
    monkeypatch.setenv("APPDATA", str(appdata))
    write(appdata / "localdata" / "config.toml", "[workspace]\nslots = 6\n")

    candidates = config_module.config_search_path()
    assert appdata / "localdata" / "config.toml" in candidates
    assert not any("Application Support" in str(c) for c in candidates)
    assert config_module.load().slots == 6


def test_the_explicit_pointer_outranks_every_discovered_path(tmp_path, monkeypatch):
    xdg = tmp_path / "xdg"
    monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg))
    write(xdg / "localdata" / "config.toml", "[workspace]\nslots = 3\n")
    explicit = write(tmp_path / "explicit.toml", "[workspace]\nslots = 9\n")
    monkeypatch.setenv("LOCALDATA_CONFIG_PATH", str(explicit))

    loaded = config_module.load()
    assert loaded.slots == 9
    assert loaded.source == explicit


def test_an_explicit_pointer_to_nothing_is_an_error_not_a_fallback(monkeypatch):
    """Falling back would run under a configuration the user did not choose."""
    monkeypatch.setenv("LOCALDATA_CONFIG_PATH", "/nonexistent/localdata.toml")
    with pytest.raises(ConfigError, match="LOCALDATA_CONFIG_PATH"):
        config_module.load()


# ---------------------------------------------------------------------------
# Refusing a configuration we cannot honour
# ---------------------------------------------------------------------------


def test_malformed_toml_is_refused_loudly():
    write(Path("localdata.toml"), "[workspace\nslots = 3\n")
    with pytest.raises(ConfigError, match="localdata.toml"):
        config_module.load()


def test_an_unknown_key_is_refused():
    """A typo in a security switch must not read as the safe default."""
    write(Path("localdata.toml"), "[paths]\npath_limitted = false\n")
    with pytest.raises(ConfigError, match="path_limitted"):
        config_module.load()


def test_an_unknown_section_is_refused():
    write(Path("localdata.toml"), "[pathz]\nroots = []\n")
    with pytest.raises(ConfigError, match="pathz"):
        config_module.load()


def test_the_error_for_an_unknown_key_names_the_alternatives():
    write(Path("localdata.toml"), "[paths]\nrootz = []\n")
    with pytest.raises(ConfigError, match="path_limited"):
        config_module.load()


def test_more_slots_than_the_ceiling_allows_is_refused():
    write(Path("localdata.toml"), f"[workspace]\nslots = {MAX_SLOTS + 1}\n")
    with pytest.raises(ConfigError, match=str(MAX_SLOTS)):
        config_module.load()


def test_zero_slots_is_refused():
    write(Path("localdata.toml"), "[workspace]\nslots = 0\n")
    with pytest.raises(ConfigError):
        config_module.load()


def test_a_wrongly_typed_value_is_refused():
    write(Path("localdata.toml"), '[paths]\npath_limited = "yes"\n')
    with pytest.raises(ConfigError, match="path_limited"):
        config_module.load()


def test_roots_must_be_a_list_of_strings():
    write(Path("localdata.toml"), '[paths]\nroots = "~/data"\n')
    with pytest.raises(ConfigError, match="roots"):
        config_module.load()


def test_a_boolean_is_not_accepted_for_slots():
    """TOML booleans are ints in Python; slots = true must not become 1."""
    write(Path("localdata.toml"), "[workspace]\nslots = true\n")
    with pytest.raises(ConfigError, match="slots"):
        config_module.load()


def test_a_boolean_is_not_accepted_for_the_memory_budget():
    """Same trap as slots: `= true` must not quietly become a 1 MB budget."""
    write(Path("localdata.toml"), "[workspace]\nmemory_budget_mb = true\n")
    with pytest.raises(ConfigError, match="memory_budget_mb"):
        config_module.load()


def test_a_memory_budget_of_zero_is_refused():
    """A budget nothing can fit under would spill on every single operation."""
    write(Path("localdata.toml"), "[workspace]\nmemory_budget_mb = 0\n")
    with pytest.raises(ConfigError, match="memory_budget_mb"):
        config_module.load()


def test_a_typo_in_the_memory_budget_is_refused():
    write(Path("localdata.toml"), "[workspace]\nmemory_budget = 200\n")
    with pytest.raises(ConfigError, match="memory_budget"):
        config_module.load()


# ---------------------------------------------------------------------------
# Values that are honoured
# ---------------------------------------------------------------------------


def test_roots_are_expanded_and_resolved(tmp_path):
    data = Path.home() / "data"
    data.mkdir()
    write(Path("localdata.toml"), '[paths]\nroots = ["~/data"]\n')

    assert config_module.load().roots == (data.resolve(),)


def test_a_root_that_does_not_exist_is_kept_verbatim():
    """Kept so the posture report shows it; containment simply never matches."""
    write(Path("localdata.toml"), '[paths]\nroots = ["~/absent"]\n')
    roots = config_module.load().roots
    assert roots == ((Path.home() / "absent").resolve(),)


def test_the_memory_budget_has_a_default_and_can_be_raised():
    """The budget is a knob, not a constant — a 64 GB host may want more."""
    assert config_module.load().memory_budget_mb == DEFAULT_MEMORY_BUDGET_MB
    write(Path("localdata.toml"), "[workspace]\nmemory_budget_mb = 512\n")
    assert config_module.load().memory_budget_mb == 512


def test_widening_the_boundary_is_honoured_when_asked_for():
    write(
        Path("localdata.toml"),
        "[paths]\npath_limited = false\n\n[network]\nenabled = true\n",
    )
    loaded = config_module.load()
    assert loaded.path_limited is False
    assert loaded.network_enabled is True


def test_an_empty_file_means_defaults_from_a_real_source():
    source = write(Path("localdata.toml"), "")
    loaded = config_module.load()
    assert loaded.slots == MAX_SLOTS
    assert loaded.source == source.resolve()


# ---------------------------------------------------------------------------
# The process-wide active configuration
# ---------------------------------------------------------------------------


def test_the_active_configuration_is_loaded_once_and_reused():
    write(Path("localdata.toml"), "[workspace]\nslots = 5\n")
    first = config_module.active()
    Path("localdata.toml").write_text("[workspace]\nslots = 2\n")
    assert config_module.active() is first


def test_a_configuration_can_be_installed_directly():
    config_module.use(Config(slots=2, path_limited=False))
    assert config_module.active().slots == 2
    config_module.reset()
    assert config_module.active().slots == MAX_SLOTS
