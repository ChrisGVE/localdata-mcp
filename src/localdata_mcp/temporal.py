"""Recognising a date in a flat file, and refusing to guess at one.

A file carries dates as text, and text is compared as text: ``'30.11.2023'``
sorts after ``'01.03.2025'``, so ``ORDER BY`` runs backwards and ``max()``
returns the *earliest* instant. Measured across twenty-four spellings of the
same five instants, seven ordered wrongly and four reported the earliest as the
maximum, with nothing anywhere to say so (CONSTRAINTS §8.1).

The fix is not a better parser. Most of those spellings are **genuinely
ambiguous** — ``01/03/2025`` is the first of March or the third of January
depending on where the file was written, and nothing in the file says which.
A server that guesses is wrong silently, which is the failure it was trying to
prevent. So this module recognises only the two forms that carry their own
meaning:

* **ISO 8601 calendar dates and datetimes, extended format** — ``2024-03-01``,
  ``2024-03-01T14:30:00``, ``2024-03-01T14:30:00Z``, ``2024-03-01 14:30:00``
  (the space separator is RFC 3339's relaxation of ISO 8601-1:2019, and is what
  pandas and every SQL engine emit by default).
* **Unix time**, an integer count of seconds since 1970-01-01T00:00:00Z
  (IEEE Std 1003.1) — which needs nothing done to it. See below.

Everything else is left as text and *reported*, so the caller learns the column
will not order correctly instead of finding out from a wrong answer.

**Why integers are not converted, and are not detected either.** A Unix
timestamp column is already integer ticks, and integer comparison is
chronological comparison — ordering, ``min``/``max``, ranges and joins are all
correct with no intervention. Nor could it be detected if intervention were
wanted: ``1766664000`` is a timestamp, an order number or a population count,
and no examination of the values can tell them apart. Converting it would turn
identifiers into dates. So the numeric variant is supported by leaving it alone.

**The grammar is pandas', not ours.** ``to_datetime(format="ISO8601")``
implements the standard and rejects every ambiguous spelling tried against it
(``01.03.2025``, ``03/01/2025``, ``Mar 01, 2025``, RFC 2822, ``2025-Q4``,
``2025-W52-4``). Two holes it leaves are closed here: it accepts ISO 8601
*basic* format, so a text column of ``20240301``-shaped order numbers would
silently become dates; and it maps the empty string to ``NaT``.

**What a recognised column is stored as, and why it is not integer ticks.**
It stays text, rewritten into one canonical UTC spelling. §1.4 says to store
integer ticks, and its evidence does not reach this case: the timestamp failure
it measured is *offset-preserving* text, where the same instant under two
offsets joins zero rows — which canonicalising to a single offset is precisely
what fixes — and the rest of its evidence is about durations, which no reader
here produces. Canonical ISO 8601 sorts lexically in chronological order, so
ordering, ``min``/``max`` and ranges are all correct.

Ticks were tried first and are worse *here*, for a reason worth recording:
they move the silent wrong answer rather than removing it. Against a tick
column the obvious query — ``WHERE order_date > '2025-01-01'`` — compares an
integer to text, matches nothing, and returns **zero rows with no error**. That
is a likelier query than a cross-file instant join, and SQLite's date functions
(``date()``, ``strftime()``, ``julianday()``) all take ISO 8601 text anyway, so
text is also the form the engine is built for.

**The round-trip contract**, which §1.4 requires be stated explicitly:

* An offset in the source is honoured and the value normalised to UTC, so
  ``2024-03-01T14:30:00+01:00`` is stored as ``2024-03-01T13:30:00Z``.
  **The original offset is not recoverable** — a file that needs it must keep
  it in a column of its own.
* A value with no offset is taken as already UTC and keeps its wall time.
* A column whose values are all plain dates stays ``YYYY-MM-DD``. One that
  carries any time of day is written ``YYYY-MM-DDTHH:MM:SSZ`` throughout, so a
  date-only value in it becomes midnight.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

__all__ = [
    "Spelling",
    "as_canonical",
    "canonical_width",
    "canonical_widths",
    "is_canonical",
    "is_date_shaped",
    "is_standard",
    "parse",
    "spelling_of",
    "standardize",
    "standardize_as",
    "text_values",
    "unparsed_temporal_examples",
    "MAX_TEMPORAL_EXAMPLES",
]

#: The extended-format calendar date every accepted value must begin with.
#:
#: This is what closes the basic-format hole: ``to_datetime(format="ISO8601")``
#: accepts ``20240301``, and a column of eight-digit identifiers read as text
#: would otherwise be converted wholesale into dates. Requiring the hyphens
#: means a value has to *look* like a date before it is treated as one.
_EXTENDED_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}")

#: The spellings :func:`standardize` writes, and the only ones a column can be
#: in for its comparisons to be chronological. Anchored at both ends, so a
#: value that merely *starts* with a date does not qualify.
_CANONICAL = re.compile(r"^\d{4}-\d{2}-\d{2}(?:T\d{2}:\d{2}:\d{2}(?:\.\d{6})?Z)?$")

#: Shapes that read as a date to a person but are not a standard this server
#: accepts. Two families, which is all the live sweep found: numbers separated
#: by ``.``, ``/`` or ``-`` in an order nothing declares, and anything carrying
#: a month name.
#: Anchored at the start, because it is tested with ``str.match``. The month
#: name is allowed a short run-up rather than none, so that a weekday prefix
#: (``Fri, 01 Mar 2024 …``) still reads as the date it is.
_DATE_SHAPED = re.compile(
    r"\s*\d{1,4}[./-]\d{1,2}[./-]\d{1,4}"
    r"|.{0,10}?(?i:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*[\s,.]+\d",
)

#: How many distinct offending values to carry into the warning. Enough to
#: recognise the format from, bounded so a column of unique junk cannot return
#: a copy of itself.
MAX_TEMPORAL_EXAMPLES = 3


#: How many values to test before testing all of them. Every column of every
#: file reaches this module, and almost none of them are dates — a column of
#: product codes should cost a few comparisons, not one per row. The full check
#: still runs on anything that survives the sample, so this bounds the work
#: without changing any answer.
_SAMPLE = 64


def text_values(series: pd.Series) -> pd.Series | None:
    """The non-null values of a text column as strings, or ``None``.

    ``None`` for anything that is not a column of text — an already-typed
    column has nothing to recognise — and for a column with no values at all,
    which would otherwise parse vacuously and be declared temporal on the
    strength of nothing.

    ``infer_dtype`` rather than a comprehension over the values: this runs on
    every column of every file, and an ``isinstance`` loop over a few hundred
    thousand rows costs more than everything else here put together.
    """
    if not (
        pd.api.types.is_object_dtype(series.dtype)
        or isinstance(series.dtype, pd.StringDtype)
    ):
        return None
    present = series.dropna()
    if present.empty:
        return None
    if pd.api.types.infer_dtype(present, skipna=True) != "string":
        return None
    return present.astype("string")


def _matches_throughout(values: pd.Series, pattern: re.Pattern[str]) -> bool:
    """Whether every value matches, deciding it on a sample where it can.

    The sample is not an approximation: a *failure* in it is conclusive, and is
    the case that needs to be cheap, because most columns are not dates. Only a
    column that looks like dates all the way through the sample pays for a full
    scan.
    """
    head = values.iloc[:_SAMPLE]
    if not bool(head.str.match(pattern).all()):
        return False
    if len(values) <= _SAMPLE:
        return True
    return bool(values.str.match(pattern).all())


def is_canonical(present: pd.Series) -> bool:
    """Whether every value is already in a spelling :func:`as_canonical` writes.

    Takes the *present* values — what :func:`text_values` returns — rather than
    the column, so a caller reading a file a chunk at a time can ask the
    question of one chunk without an empty chunk answering ``False`` for the
    whole column.

    **Every value being in *a* canonical spelling does not make them all in the
    *same* one**, because :data:`_CANONICAL` leaves both the time and the
    fractional part optional. Ask :func:`canonical_widths` for that; the two
    questions together are what :func:`is_standard` needs (#75).
    """
    return _matches_throughout(present, _CANONICAL)


def canonical_widths(present: pd.Series) -> frozenset[int]:
    """How many distinct spellings these canonical values are written in.

    The three canonical spellings have three exact widths — 10 for a date, 20 to
    the second, 27 with fractional seconds (:data:`_WIDTHS`) — so a width *is* a
    spelling, and counting the distinct ones asks whether the column is written
    one way. One width means one spelling.

    Composes across chunks by union, which is why this is a set rather than a
    flag: a chunk written one way and a chunk written another are each uniform,
    and the column is not. Only meaningful where :func:`is_canonical` holds; a
    non-canonical value's width says nothing.

    Every value is counted rather than a sample, because it takes exactly one
    odd value at the end of a column to make it two-spelled.
    """
    return frozenset(present.str.len().unique().tolist())


def is_date_shaped(present: pd.Series) -> bool:
    """Whether every value reads as a date, in a standard or not.

    The gate on :func:`unparsed_temporal_examples`, exposed for the same reason
    :func:`is_canonical` is: it composes across chunks, and a chunk holding no
    values must not be able to answer for the column.
    """
    return _matches_throughout(present, _DATE_SHAPED)


def parse(present: pd.Series, whole: pd.Series) -> pd.Series | None:
    """Parse a column if **every** value in it is a standard datetime.

    All or nothing, deliberately. A column that is nine-tenths ISO and
    one-tenth something else is a column with a data problem, and converting
    the part that parses would turn the rest into ``NaT`` — deleting exactly
    the values somebody needs to see. It stays text and gets reported instead.

    **This composes across chunks and the rest of the module's decisions do
    not.** Qualifying is all-or-nothing per *value* — every non-null value must
    match :data:`_EXTENDED_DATE` and parse under an explicit ``format`` — with
    no per-column format inference anywhere in it, so a column qualifies if and
    only if every chunk of it qualifies. There is no day-first ambiguity for two
    chunks to settle differently. Which *spelling* to write back is the
    decision that does not compose; see :class:`Spelling`.
    """
    if not _matches_throughout(present, _EXTENDED_DATE):
        return None
    try:
        parsed = pd.to_datetime(whole, format="ISO8601", utc=True)
    except (ValueError, TypeError):
        return None
    # A value that was present and came back NaT did not parse; `format` should
    # have raised, but a silent NaT would drop a value rather than refuse it.
    if parsed[present.index].isna().any():
        return None
    return parsed


@dataclass(frozen=True)
class Spelling:
    """Which of the canonical forms a temporal column is written in.

    Both flags are properties of the **whole column** rather than of any one
    value: a column is written date-only when *every* value is midnight, and
    grows fractional seconds when *any* value carries them. That is what stops
    the decision being made a chunk at a time — the same column would come out
    ``2024-03-01`` in one chunk and ``2024-03-01T00:00:00Z`` in the next, which
    is two spellings of one column and no longer sorts as one. So a reader that
    works in chunks measures the spelling over the whole column first and
    :func:`as_canonical` is *told* it, rather than each chunk deciding for
    itself.
    """

    date_only: bool
    fractional: bool

    def merged_with(self, other: "Spelling") -> "Spelling":
        """The spelling of two parts of one column, taken together.

        The two flags merge in opposite directions because they are opposite
        quantifiers: date-only holds when it holds *everywhere*, and fractional
        seconds appear when they appear *anywhere*.
        """
        return Spelling(
            date_only=self.date_only and other.date_only,
            fractional=self.fractional or other.fractional,
        )


def spelling_of(parsed: pd.Series) -> Spelling:
    """Measure which canonical spelling these instants call for."""
    present = parsed.dropna()
    return Spelling(
        date_only=bool(
            (
                (present.dt.hour == 0)
                & (present.dt.minute == 0)
                & (present.dt.second == 0)
                & (present.dt.microsecond == 0)
            ).all()
        ),
        fractional=bool((present.dt.microsecond != 0).any()),
    )


#: How wide each canonical spelling is, in characters. Exact rather than
#: measured, because every value in a rewritten column is rendered by one
#: ``strftime`` format and so is exactly this long. A chunked reader needs it:
#: it sizes the column before pass 2 has produced a single rewritten value.
_WIDTHS = {(True, False): 10, (False, False): 20, (False, True): 27}


def canonical_width(spelling: Spelling) -> int:
    """How wide a value written in this spelling is, in characters."""
    return _WIDTHS[(spelling.date_only, spelling.fractional)]


def as_canonical(parsed: pd.Series, spelling: Spelling) -> pd.Series:
    """Write parsed instants back in the spelling this column was measured for.

    Date-only columns keep the date form they arrived in. Anything carrying a
    time of day is written to the second, with fractional seconds only where
    the column actually uses them — a column of whole seconds should not grow
    six zeroes it did not have.
    """
    if spelling.date_only:
        formatted = parsed.dt.strftime("%Y-%m-%d")
    elif spelling.fractional:
        formatted = parsed.dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    else:
        formatted = parsed.dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    # `strftime` renders NaT as the string "NaT"; a missing value stays missing.
    return formatted.where(parsed.notna(), other=None)


def standardize(frame: pd.DataFrame) -> pd.DataFrame:
    """Canonicalise every column that is unambiguously ISO 8601.

    Columns that do not qualify are untouched, including the ambiguous
    spellings — those are reported by :func:`unparsed_temporal_examples`
    rather than guessed at.
    """
    converted: dict[str, pd.Series] = {}
    for name in frame.columns:
        series = frame[name]
        present = text_values(series)
        if present is None:
            continue
        # Already in the form this would rewrite it into, which is the case for
        # any file written the way the documentation asks for. Parsing it only
        # to format it back is the most expensive thing in this module, and it
        # would not change a byte.
        #
        # That last clause is the whole justification, and it only holds while
        # the column is written *one* way. A column mixing canonical spellings
        # would change — into the single spelling that makes it sort — so it
        # takes the parsing branch (#75).
        if is_canonical(present) and len(canonical_widths(present)) == 1:
            continue
        parsed = parse(present, series)
        if parsed is not None:
            converted[name] = as_canonical(parsed, spelling_of(parsed))

    if not converted:
        return frame
    return frame.assign(**converted)


def standardize_as(frame: pd.DataFrame, spellings: dict[str, Spelling]) -> pd.DataFrame:
    """Rewrite the named columns in the spelling each was measured for.

    :func:`standardize` for a reader working in chunks: which columns are
    temporal, and which spelling each is written in, were settled over the whole
    column by an earlier pass, so this applies a decision rather than making one.
    A column named here is one that pass already established parses throughout,
    so a chunk of it that does not parse is a contradiction rather than an
    ordinary miss — it is left alone and the column keeps its text, which is the
    same answer :func:`standardize` gives to anything it cannot parse.
    """
    converted: dict[str, pd.Series] = {}
    for name, spelling in spellings.items():
        series = frame[name]
        present = text_values(series)
        if present is None:
            continue
        parsed = parse(present, series)
        if parsed is not None:
            converted[name] = as_canonical(parsed, spelling)

    if not converted:
        return frame
    return frame.assign(**converted)


def is_standard(series: pd.Series) -> bool:
    """Whether this column is ISO 8601 throughout, and so compares correctly.

    Asked of the column as it stands rather than remembered from
    :func:`standardize`, which means it is equally true of a column that
    arrived canonical and one that was rewritten into canonical form. Either
    way the property being reported is the same one: comparisons on this
    column are chronological.

    Matched against the canonical spellings rather than parsed again. Anything
    :func:`standardize` accepted it has already rewritten into one of them, so
    re-running ``to_datetime`` here would buy nothing and cost a second pass
    over the column.

    One spelling as well as canonical, because the claim being made is about
    *comparisons*: a column holding both ``…:30Z`` and ``…:30.5Z`` is canonical
    at every value and still orders the later one first, since ``.`` sorts below
    ``Z``. Answering ``True`` there is what made that defect silent (#75).
    """
    present = text_values(series)
    if present is None:
        return False
    return is_canonical(present) and len(canonical_widths(present)) == 1


def unparsed_temporal_examples(series: pd.Series) -> tuple[str, ...]:
    """Values that read as dates but are not a standard, for the warning.

    Only reached for a column that :func:`standardize` left as text. Returning
    something here is what turns a silent wrong answer into a stated
    limitation, so the test is deliberately narrow: every non-null value must
    be date-shaped. A column that merely *contains* something date-like is a
    text column, and warning about it would be noise.

    Reports nothing for spellings that are non-standard but happen to order
    correctly anyway — ``2025-Q4`` and ``2025-W52-4`` sort chronologically as
    text, so there is no wrong answer to warn about.
    """
    present = text_values(series)
    if present is None:
        return ()
    if not is_date_shaped(present):
        return ()

    examples: dict[str, None] = {}
    for value in present:
        examples.setdefault(value)
        if len(examples) == MAX_TEMPORAL_EXAMPLES:
            break
    return tuple(examples)
