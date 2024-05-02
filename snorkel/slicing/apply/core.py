# Import the LFApplier and PandasLFApplier classes from the snorkel.labeling module.
from snorkel.labeling import LFApplier, PandasLFApplier


# Define a new SFApplier class that inherits from LFApplier.
class SFApplier(LFApplier):
    """SF applier for a list of data points.

    This class is a specialized version of the LFApplier class, designed to
    apply Snorkel's labeling functions to a list of data points. The _use_recarray
    attribute is set to True to use NumPy recarray for better performance.

    For more information about the LFApplier class, refer to
    ``snorkel.labeling.core.LFApplier``.
    """

    _use_recarray = True


# Define a new PandasSFApplier class that inherits from PandasLFApplier.
class PandasSFApplier(PandasLFApplier):
    """SF applier for a Pandas DataFrame.

    This class is a specialized version of the PandasLFApplier class, designed to
    apply Snorkel's labeling functions to a Pandas DataFrame. The _use_recarray
    attribute is set to True to use NumPy recarray for better performance.

    For more information about the PandasLFApplier class, refer to
    ``snorkel.labeling.core.PandasLFApplier``.
    """

    _use_recarray = True
