"""XML documents whose root holds repeated elements, one element per row.

The reader is written by hand rather than handed to ``pandas.read_xml``, which
drops nested elements and repeated tags without saying so. It picks which
repeated elements are the tables on the same rules the JSON reader uses. The
module is not called ``xml`` so that it is never mistaken for the standard
library package of that name.
"""

from __future__ import annotations

from pathlib import Path
from xml.etree import ElementTree

from ..errors import LoadError
from ..names import table_names
from .documents import frame_and_nested_note
from .result import NamedFrame, ReadResult, inferred_types


def read_xml(path: Path) -> ReadResult:
    """Read an XML document whose root holds repeated elements, one per row.

    Written rather than delegated to ``pandas.read_xml``, which is fail-open on
    two shapes this corpus contains. Measured against the stdlib parser: a row
    holding a nested element comes back with that column ``NaN`` — the subtree
    silently dropped — and a row with a tag repeated twice keeps only the last
    one. Both are data loss with nothing reported, which is the class this
    server exists to refuse.

    Columns are the row's attributes and the tags of its direct children. A
    child with children of its own is kept as its XML text (as a nested JSON
    value is kept as JSON text); a tag appearing twice in one row is a list
    rather than a column, and is refused by name. Each element that repeats
    under the root is a table, and every one of them lands (see
    :func:`_rows_within`).
    """
    try:
        root = ElementTree.parse(path).getroot()
    except ElementTree.ParseError as exc:
        raise LoadError(f"Could not read {path.name}: {exc}") from exc

    found, notes = _rows_within(root, path.name)

    tables = []
    for tag, rows in found:
        records, nested = _row_records(rows, path.name)
        if nested:
            notes = notes + (_nested_elements_note(nested, tag),)

        frame, _ = frame_and_nested_note(records, None)
        tables.append(NamedFrame(inferred_types(frame), tag))
    return ReadResult(tuple(tables), notes)


def _row_records(
    rows: list[ElementTree.Element], name: str
) -> tuple[list[dict], dict[str, None]]:
    """Every row's record, and the tags held nested in any row, each once."""
    records = []
    nested: dict[str, None] = {}  # insertion-ordered, and deduplicating
    for element in rows:
        record, held = _record_of(element, name)
        records.append(record)
        nested.update(dict.fromkeys(held))
    return records, nested


def _nested_elements_note(nested: dict[str, None], tag: str | None) -> str:
    """The note naming the elements of one table that were kept as XML text."""
    where = ", ".join(nested) if tag is None else f"{', '.join(nested)} of <{tag}>"
    return (
        f"Nested elements in {where} were kept as XML text, "
        f"because SQL has no nested type. The text is the element as it "
        f"stood in the file, so nothing was lost."
    )


def _rows_within(
    root: ElementTree.Element, name: str
) -> tuple[list[tuple[str | None, list[ElementTree.Element]]], tuple[str, ...]]:
    """Pick the repeated elements that are the tables, on JSON's rules.

    One kind of child is the table. Several kinds, one of which repeats, is the
    wrapped shape — a ``<generated>`` beside the rows — and the note names what
    was left out. Several kinds that repeat are several tables, and every one
    lands, named after its tag the way a JSON table is named after its key
    path. A tag is ``None`` for the one table of a document holding one, which
    is named after the file as it always was. A child appearing once is read
    as metadata, for the reason given at :func:`_repeated_tags`.
    """
    groups = _children_by_tag(root)
    if not groups:
        raise LoadError(f"Could not read {name}: <{root.tag}> holds no elements.")
    if len(groups) == 1:
        return [(None, next(iter(groups.values())))], ()

    repeated = _repeated_tags(groups)
    others = ", ".join(f"<{tag}>" for tag in groups if tag not in repeated)
    if len(repeated) == 1:
        tag = repeated[0]
        return [(None, groups[tag])], (_one_repeated_note(name, tag, others),)
    if repeated:
        note = _several_repeated_note(name, root, repeated, others)
        return [(tag, groups[tag]) for tag in repeated], (note,)
    raise LoadError(
        f"Could not read {name}: <{root.tag}> holds one each of "
        f"{', '.join(f'<{tag}>' for tag in groups)}, so nothing in it repeats "
        f"as rows do."
    )


def _children_by_tag(
    root: ElementTree.Element,
) -> dict[str, list[ElementTree.Element]]:
    """The root's children grouped by tag, tags in order of first appearance."""
    groups: dict[str, list[ElementTree.Element]] = {}
    for child in root:
        groups.setdefault(child.tag, []).append(child)
    return groups


def _repeated_tags(groups: dict[str, list[ElementTree.Element]]) -> list[str]:
    """The tags that appear more than once: the ones that can be tables.

    An XML text value carries no record of whether a child appeared once
    because it is metadata or because the table it starts has one row, and
    this rule reads it as metadata — as it did before several tables landed.
    """
    return [tag for tag, members in groups.items() if len(members) > 1]


def _one_repeated_note(name: str, tag: str, others: str) -> str:
    """The note for one repeated element among others: it alone is the table."""
    return (
        f"{name} holds <{tag}> repeated among other elements, and <{tag}> "
        f"was loaded as the table. The rest ({others}) are not part of it."
    )


def _several_repeated_note(
    name: str, root: ElementTree.Element, repeated: list[str], others: str
) -> str:
    """The note for several repeated elements: what each one landed as."""
    _, landing = table_names(repeated, Path(name).stem)
    where = ", ".join(f"<{tag}> as {table}" for tag, table in zip(repeated, landing))
    rest = f" The rest ({others}) are not part of any table." if others else ""
    return (
        f"{name} holds {len(repeated)} tables, each an element repeated "
        f"under <{root.tag}>, and every one of them landed as a table of its "
        f"own, named from its tag made safe for SQL the way a sheet name is: "
        f"{where}. Each loaded entry names its tag as its sheet.{rest}"
    )


def _record_of(element: ElementTree.Element, name: str) -> tuple[dict, list[str]]:
    """One row — attributes then children by tag — and which parts were nested."""
    record: dict[str, object] = dict(element.attrib)
    nested = []

    for child in element:
        if child.tag in record:
            raise LoadError(
                f"Could not read {name}: <{child.tag}> appears more than once "
                f"in a single <{element.tag}>, which makes it a list rather "
                f"than a column. A table cannot hold it without choosing which "
                f"one to keep, and choosing would lose the rest."
            )
        if len(child):
            # Kept whole. Dropping it is what pandas does, and it reports nothing.
            record[child.tag] = ElementTree.tostring(child, encoding="unicode").strip()
            nested.append(child.tag)
        else:
            record[child.tag] = child.text

    if not record:
        raise LoadError(
            f"Could not read {name}: <{element.tag}> has no attributes and no "
            f"child elements, so it names no columns. A row needs named parts."
        )
    return record, nested
