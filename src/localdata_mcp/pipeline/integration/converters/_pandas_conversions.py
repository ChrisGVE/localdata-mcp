"""
Combined Pandas DataFrame conversion methods mixin.

Combines the from-DataFrame and to-DataFrame conversion mixins into
a single mixin class for use by PandasConverter.
"""

from ._pandas_from_df import PandasFromDataFrameMixin
from ._pandas_to_df import PandasToDataFrameMixin


class PandasConversionsMixin(PandasFromDataFrameMixin, PandasToDataFrameMixin):
    """Combined mixin providing all conversion methods for PandasConverter."""

    pass
