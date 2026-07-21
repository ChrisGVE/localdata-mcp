"""Three-tier resolution for the ``security`` configuration section.

LocalData reads configuration from several files. Most settings follow the
ordinary rule -- the file with the highest priority wins. The ``security``
section cannot, because one of those files is ``./.localdata.yaml``, which lives
in the directory holding the data. That directory is frequently not authored by
the person running the server: a cloned repository, a shared dataset drop, a
working directory an agent can itself write to. Under a plain last-wins rule, a
config file shipped alongside the data could set ``readonly: false`` and switch
off the protection the operator asked for, which would make the configuration
system the bypass.

The answer is three tiers rather than two:

1. ``global_invariant`` -- the operator's floor. Declarable only in operator
   layers (``LOCALDATA_CONFIG``, ``/etc/localdata/config.yaml``, the user
   config). Never relaxable.
2. ``security`` in an operator layer -- an ordinary default. A project-local
   file may override it completely, because there are legitimate reasons to
   unblock something.
3. ``security`` in ``./.localdata.yaml`` -- wins over operator defaults, then is
   clamped against the floor.

Tier 2 is what distinguishes this from the simpler "a local file may only
tighten" rule. That rule conflates "my hard floor" with "what I would normally
like", freezing the convenience defaults and leaving no route for a legitimate
relaxation. Separating them is strictly more expressive and no less safe.

The floor binds *every* layer, including the operator's own ``security``
section. An operator who declares a floor and then contradicts it in the same
file has written a contradiction; the floor wins and the contradiction is
reported. Only the floor itself can lower the floor.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class InvariantConfigError(ValueError):
    """A ``global_invariant`` section declares something unsupported.

    Raised rather than warned about. A declared-but-unenforced security key is
    the exact defect this module exists to remove; shipping a second one would
    be indefensible.
    """


#: Sections a ``global_invariant`` may constrain. The section mirrors the
#: configuration tree so other sections can be added later, but a key outside
#: this set is an error at load rather than a value that quietly does nothing.
SUPPORTED_INVARIANT_SECTIONS = frozenset({"security"})


@dataclass
class ClampRecord:
    """One security value the floor overrode.

    Carries everything a user needs to understand why their setting did not
    take effect: which key, what they asked for, what is enforced instead, and
    which file asked.
    """

    key: str
    requested: Any
    enforced: Any
    requested_by: Optional[str] = None

    def message(self) -> str:
        source = f" requested by {self.requested_by}" if self.requested_by else ""
        return (
            f"security.{self.key}{source} was overridden by global_invariant: "
            f"requested {self.requested!r}, enforced {self.enforced!r}"
        )


@dataclass
class SecurityResolution:
    """The effective ``security`` settings and how they were arrived at."""

    effective: Dict[str, Any]
    invariant: Dict[str, Any] = field(default_factory=dict)
    clamps: List[ClampRecord] = field(default_factory=list)


def _tighten_or(floor: Any, value: Any) -> Any:
    """A boolean switch the floor may pin on but never off."""
    return bool(floor) or bool(value)


def _tighten_min(floor: Any, value: Any) -> Any:
    """A ceiling the floor sets; a lower request is honoured, a higher one is not."""
    return min(int(floor), int(value))


def _tighten_union(floor: Any, value: Any) -> Any:
    """A denylist: floor entries are permanent, requested entries are added.

    Floor entries come first and order is otherwise preserved, so the effective
    list reads predictably rather than in set order.
    """
    merged = list(floor)
    for entry in value:
        if entry not in merged:
            merged.append(entry)
    return merged


def _normalize(path: str) -> Optional[Path]:
    """Resolve a configured path for comparison, or None if it is unusable."""
    try:
        return Path(path).expanduser().resolve()
    except (OSError, RuntimeError, ValueError):
        return None


def _is_within(candidate: str, boundary: str) -> bool:
    """Is ``candidate`` the same directory as ``boundary``, or inside it?"""
    resolved_candidate = _normalize(candidate)
    resolved_boundary = _normalize(boundary)
    if resolved_candidate is None or resolved_boundary is None:
        return False
    try:
        return resolved_candidate.is_relative_to(resolved_boundary)
    except AttributeError:  # pragma: no cover - Python < 3.9
        return str(resolved_candidate).startswith(str(resolved_boundary))


def _tighten_paths(floor: Any, value: Any) -> Any:
    """An allowlist: a request survives only if it lies within a floor path.

    This is containment, not set intersection. A floor of ``["/data"]`` with a
    request of ``["/data/project1"]`` must keep ``/data/project1`` -- a set
    intersection would find no common element and yield no access at all, which
    reads as a bug rather than as policy.

    If nothing requested lies within the floor, the floor's own paths are used.
    Returning an empty list would leave a server that can read nothing, which is
    a worse answer to a misconfiguration than falling back to the policy that
    the misconfiguration violated. The fallback never grants more than the
    floor.
    """
    within = [
        candidate for candidate in value if any(_is_within(candidate, b) for b in floor)
    ]
    return within if within else list(floor)


#: How each security key is tightened against the floor. A key absent here has
#: no floor semantics and is left to ordinary last-wins resolution.
SECURITY_CLAMPS = {
    "readonly": _tighten_or,
    "restrict_paths": _tighten_or,
    "max_query_length": _tighten_min,
    "blocked_keywords": _tighten_union,
    "allowed_paths": _tighten_paths,
}


def validate_invariant(data: Dict[str, Any], source: str) -> Dict[str, Any]:
    """Check one file's ``global_invariant`` section and return its security keys.

    Args:
        data: The parsed ``global_invariant`` mapping from a single config file.
        source: Path of that file, so an error names the file to edit.

    Raises:
        InvariantConfigError: If the section is malformed, constrains a section
            this release does not enforce, or names an unknown security key.
    """
    if not isinstance(data, dict):
        raise InvariantConfigError(
            f"{source}: 'global_invariant' must be a mapping, got {type(data).__name__}"
        )

    unsupported = sorted(set(data) - SUPPORTED_INVARIANT_SECTIONS)
    if unsupported:
        supported = ", ".join(sorted(SUPPORTED_INVARIANT_SECTIONS))
        raise InvariantConfigError(
            f"{source}: 'global_invariant' cannot constrain {unsupported} in this "
            f"release; only these sections are enforced: {supported}. Remove the "
            f"entry rather than leaving a constraint that does nothing."
        )

    security = data.get("security", {})
    if not isinstance(security, dict):
        raise InvariantConfigError(
            f"{source}: 'global_invariant.security' must be a mapping, "
            f"got {type(security).__name__}"
        )

    unknown = sorted(set(security) - set(SECURITY_CLAMPS))
    if unknown:
        known = ", ".join(sorted(SECURITY_CLAMPS))
        raise InvariantConfigError(
            f"{source}: 'global_invariant.security' names unenforceable keys "
            f"{unknown}; enforceable keys are: {known}."
        )

    return dict(security)


def merge_invariants(declarations: List[Tuple[str, Dict[str, Any]]]) -> Dict[str, Any]:
    """Combine floors declared by several operator layers; the stricter wins.

    ``/etc`` and a user config may both raise the floor, and neither may lower
    the other's. Combining with the same tightening operator used against a
    request gives exactly that.

    Args:
        declarations: ``(source_path, security_invariant)`` pairs, already
            validated, in any order.
    """
    merged: Dict[str, Any] = {}
    for _source, invariant in declarations:
        for key, floor in invariant.items():
            if key not in merged:
                merged[key] = floor
            else:
                merged[key] = SECURITY_CLAMPS[key](merged[key], floor)
    return merged


def resolve_security(
    requested: Dict[str, Any],
    invariant: Dict[str, Any],
    requested_by: Optional[Dict[str, str]] = None,
) -> SecurityResolution:
    """Clamp the merged security settings against the floor.

    Args:
        requested: The ``security`` section after ordinary last-wins merging of
            every layer, project-local included.
        invariant: The merged floor from :func:`merge_invariants`.
        requested_by: Optional map of security key to the path of the file that
            supplied its winning value, used to name the file in the warning.

    Returns:
        The effective settings, the floor applied, and a record of every value
        the floor actually changed.
    """
    effective = dict(requested)
    clamps: List[ClampRecord] = []
    sources = requested_by or {}

    for key, floor in invariant.items():
        tighten = SECURITY_CLAMPS[key]
        if key not in effective:
            # Nothing requested this key, so the floor simply becomes the value.
            effective[key] = floor
            continue

        asked = effective[key]
        enforced = tighten(floor, asked)
        effective[key] = enforced
        if enforced != asked:
            clamps.append(
                ClampRecord(
                    key=key,
                    requested=asked,
                    enforced=enforced,
                    requested_by=sources.get(key),
                )
            )

    return SecurityResolution(
        effective=effective, invariant=dict(invariant), clamps=clamps
    )


def report_clamps(clamps: List[ClampRecord]) -> None:
    """Log every clamp that changed a value.

    Clamping silently would reproduce the defect this module exists to fix: a
    security setting that is accepted, stored, and then quietly ignored.
    """
    for clamp in clamps:
        logger.warning(clamp.message())
