"""localdata_mcp/nexus/config — NX-2, the configuration nexus.

The one dataclass-per-truth ConfigModel (every S8 default, one home),
layered TOML loaders with two-tier trust merge, per-field provenance,
and endpoint declarations (ARCHITECTURE.md sections 5 and 8, NX-2).
Components reach config through this surface; no module reads raw
environment variables or config files directly.
"""

from .default_site_check import check_one_default_site
from .endpoints import EndpointDeclaration, Posture, resolve_credential
from .env_derive import env_var_name
from .errors import (
    ConfigurationError,
    InlineCredentialError,
    IntroductionRefusedError,
    InvalidValueError,
    PinShadowingError,
    TypeMismatchError,
    UnknownFieldError,
)
from .loaders import CONFIG_SEARCH_PATHS, load_config
from .merge import ConfigLoadResult, merge_sources
from .models import ConfigModel
from .provenance import Layer, LayerSource, Provenance

__all__ = [
    "CONFIG_SEARCH_PATHS",
    "ConfigLoadResult",
    "ConfigModel",
    "ConfigurationError",
    "EndpointDeclaration",
    "InlineCredentialError",
    "IntroductionRefusedError",
    "InvalidValueError",
    "Layer",
    "LayerSource",
    "PinShadowingError",
    "Posture",
    "Provenance",
    "TypeMismatchError",
    "UnknownFieldError",
    "check_one_default_site",
    "env_var_name",
    "load_config",
    "merge_sources",
    "resolve_credential",
]
