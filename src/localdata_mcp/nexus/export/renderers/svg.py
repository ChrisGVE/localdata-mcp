"""renderers/svg.py — the allow-list SVG sanitizer, sole owner (E7.4).

FR-501 SVG inertness per §7's visualization detail: `defusedxml` does
its REAL job (XML parse hardening — entity expansion, XXE, blowup),
then this NEW code permits only the element/attribute set matplotlib's
Agg SVG backend legitimately emits — an ALLOW-list, because SVG's
active-content surface (`javascript:` URIs anywhere, `<style>`
`url()`, `<animate>`, `data:` URIs) is too large to deny-enumerate.
Value-level constraints wherever a value can carry a reference:
`style` may contain no `url()` or external reference of any form, and
every reference-class attribute (`href`/`xlink:href`, `clip-path`)
must resolve to a LOCAL fragment — the artifact is REFUSED otherwise
(local references are the general rule, not a `use` aside).
Non-allow-listed elements are dropped with their subtree;
non-allow-listed attributes are dropped; reference violations refuse
the whole artifact. Matplotlib's own XML escaping of data-derived text
stays the first defense layer; this list is the second (FR-501's
defense-in-depth). Neighbors: visualize/render/ (E13) hands rendered
bytes here; interface.py routes file output.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ElementTree
from typing import Any

from defusedxml.ElementTree import fromstring as defused_fromstring

from ..interface import ExportError

FORMAT = "svg"

_SVG_NS = "http://www.w3.org/2000/svg"
_XLINK_NS = "http://www.w3.org/1999/xlink"

# The element set matplotlib's Agg SVG backend legitimately emits (§7).
_ALLOWED_ELEMENTS = frozenset(
    {"svg", "g", "path", "rect", "text", "line", "defs", "use"}
)

# Geometric/style attributes plus the reference-class ones the value
# rules below constrain. Anything absent is dropped.
_ALLOWED_ATTRIBUTES = frozenset(
    {
        "version",
        "width",
        "height",
        "viewBox",
        "id",
        "d",
        "style",
        "transform",
        "clip-path",
        "x",
        "y",
        "x1",
        "y1",
        "x2",
        "y2",
        "rx",
        "ry",
        "fill",
        "stroke",
        "stroke-width",
        "stroke-linecap",
        "stroke-linejoin",
        "opacity",
        "fill-opacity",
        "stroke-opacity",
        "font-family",
        "font-size",
        "font-style",
        "font-weight",
        "text-anchor",
        "href",
    }
)

_LOCAL_FRAGMENT = re.compile(r"^#[A-Za-z0-9_.:-]+$")
_LOCAL_URL_REF = re.compile(r"^url\(#[A-Za-z0-9_.:-]+\)$")
_URL_IN_STYLE = re.compile(r"url\s*\(", re.IGNORECASE)
_EXTERNAL_MARKER = re.compile(r"(?:javascript:|data:|https?:|//)", re.IGNORECASE)


class SvgRefusedError(ExportError):
    """A reference-class value violated the local-only rule — the
    whole artifact is refused, never partially kept (FR-501)."""


def render(payload: Any) -> bytes:
    """Sanitized SVG bytes from rendered SVG text/bytes."""
    text = payload.decode("utf-8") if isinstance(payload, bytes) else str(payload)
    try:
        root = defused_fromstring(text)
    except Exception as failure:
        raise ExportError(f"SVG does not parse safely: {failure}") from failure
    if _local_name(root.tag) != "svg":
        raise ExportError("payload is not an SVG document")
    sanitized = _sanitize_element(root)
    if sanitized is None:  # pragma: no cover — root is checked above
        raise ExportError("SVG root element was refused")
    ElementTree.register_namespace("", _SVG_NS)
    ElementTree.register_namespace("xlink", _XLINK_NS)
    return ElementTree.tostring(sanitized, encoding="utf-8")


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _sanitize_element(element: ElementTree.Element) -> ElementTree.Element | None:
    """The element with only allow-listed content — None drops it with
    its whole subtree."""
    if _local_name(element.tag) not in _ALLOWED_ELEMENTS:
        return None
    clean = ElementTree.Element(element.tag)
    clean.text = element.text
    clean.tail = element.tail
    for name, value in element.attrib.items():
        kept = _sanitize_attribute(name, value)
        if kept is not None:
            clean.set(name, value)
    for child in element:
        kept_child = _sanitize_element(child)
        if kept_child is not None:
            clean.append(kept_child)
    return clean


def _sanitize_attribute(name: str, value: str) -> str | None:
    """The attribute name iff allow-listed AND its value passes the
    reference rules; a reference VIOLATION refuses the artifact."""
    local = _local_name(name)
    if local not in _ALLOWED_ATTRIBUTES:
        return None
    if local == "href":
        if not _LOCAL_FRAGMENT.match(value):
            raise SvgRefusedError(
                f"non-local reference {value!r} on href — refused (FR-501: "
                "references must be local fragments)"
            )
        return name
    if local == "clip-path":
        if not _LOCAL_URL_REF.match(value.strip()):
            raise SvgRefusedError(f"non-local clip-path reference {value!r} — refused")
        return name
    if local == "style":
        if _URL_IN_STYLE.search(value) or _EXTERNAL_MARKER.search(value):
            raise SvgRefusedError(
                "style value carries a url()/external reference — refused "
                "(FR-501: no reference of any form in style)"
            )
        return name
    if _EXTERNAL_MARKER.search(value):
        raise SvgRefusedError(
            f"attribute {local} carries an external/active reference — refused"
        )
    return name
