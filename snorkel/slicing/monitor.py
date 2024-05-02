import numpy as np
import pandas as pd
from snorkel.slicing import PandasSFApplier
from snorkel.slicing.sf import SlicingFunction

def slice_dataframe(df: pd.DataFrame, slicing_function: SlicingFunction) -> pd.DataFrame:
    """Return a dataframe with examples corresponding to specified ``SlicingFunction``.

    Parameters
    ----------
    df
        A pandas DataFrame that will be sliced
    slicing_function
        SlicingFunction which will operate over df to return a subset of examples;
        function returns a subset of data for which ``slicing_function`` output is True

    Returns
    -------
    pd.DataFrame
        A DataFrame including only examples belonging to slice_name
    """
    sf_applier = PandasSFApplier([slicing_function])
    sf_labels = sf_applier.apply(df)

    # Get the index of rows where the slicing_function output is True
    row_indices = sf_labels[slicing_function.name].index[sf_labels[slicing_function.name]].tolist()

    # Return the sliced dataframe
    return df.iloc[row_indices]
