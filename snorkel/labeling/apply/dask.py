from functools import partial
from typing import Union

import numpy as np
import pandas as pd
from dask import dataframe as dd
from dask.distributed import Client

from .core import BaseLFApplier, _FunctionCaller
from .pandas import apply_lfs_to_data_point, rows_to_triplets

# Define the type for the Scheduler parameter
Scheduler = Union[str, Client]


class DaskLFApplier(BaseLFApplier):
    """
    LF applier for a Dask DataFrame.

    Dask DataFrames consist of partitions, each being a Pandas DataFrame.
    This allows for efficient parallel computation over DataFrame rows.
    For more information, see https://docs.dask.org/en/stable/dataframe.html
    """

    def apply(
            self,
            df: dd.DataFrame,  # Dask DataFrame to be labeled
            scheduler: Scheduler = "processes",  # Dask scheduling configuration
            fault_tolerant: bool = False  # Output -1 if LF execution fails
    ) -> np.ndarray:
        """
        Label Dask DataFrame of data points with LFs.

        Parameters
        ----------
        df : dd.DataFrame
            Dask DataFrame containing data points to be labeled by LFs
        scheduler : Scheduler, optional
            A Dask scheduling configuration: either a string option or
            a ``Client``. For more information, see
            https://docs.dask.org/en/stable/scheduling.html#
        fault_tolerant : bool, optional
            Output ``-1`` if LF execution fails?, by default False

        Returns
        -------
        np.ndarray
            Matrix of labels emitted by LFs
        """
        f_caller = _FunctionCaller(fault_tolerant)  # Create a FunctionCaller instance
        apply_fn = partial(apply_lfs_to_data_point, lfs=self._lfs, f_caller=f_caller)  # Partially apply the apply_lfs_to_data_point function
        map_fn = df.map_partitions(lambda p_df: p_df.apply(apply_fn, axis=1))  # Apply the function to each partition
        labels = map_fn.compute(scheduler=scheduler)  # Compute the labels
        labels_with_index = rows_to_triplets(labels)  # Convert the labels to a specific format
        return self._numpy_from_row_data(labels_with_index)  # Convert the labeled data to a numpy array


class PandasParallelLFApplier(DaskLFApplier):
    """
    Parallel LF applier for a Pandas DataFrame.

    Creates a Dask DataFrame from a Pandas DataFrame, then uses
    ``DaskLFApplier`` to label data in parallel. See ``DaskLFApplier``.
    """

    def apply(  # type: ignore
            self,
            df: pd.DataFrame,  # Pandas DataFrame to be labeled
            n_parallel: int = 2,  # Parallelism level for LF application
            scheduler: Scheduler = "processes",  # Dask scheduling configuration
            fault_tolerant: bool = False  # Output -1 if LF execution fails
    ) -> np.ndarray:
        """
        Label Pandas DataFrame of data points with LFs in parallel using Dask.

        Parameters
        ----------
        df : pd.DataFrame
            Pandas DataFrame containing data points to be labeled by LFs
        n_parallel : int, optional
            Parallelism level for LF application. Corresponds to ``npartitions``
            in constructed Dask DataFrame. For ``scheduler="processes"``, number
            of processes launched. Recommended to be no more than the number
            of cores on the running machine., by default 2
        scheduler : Scheduler, optional
            A Dask scheduling configuration: either a string option or
            a ``Client``. For more information, see
            https://docs.dask.org/en/stable/scheduling.html#
            , by default "processes"
        fault_tolerant : bool, optional
            Output ``-1`` if LF execution fails?, by default False

        Returns
        -------
        np.ndarray
            Matrix of labels emitted by LFs
        """
        if n_parallel < 2:
            raise ValueError(
                "n_parallel should be >= 2. "
                "For single process Pandas, use PandasLFApplier."
            )  # Raise an error if n_parallel is less than 2

        df = dd.from_pandas(df, npartitions=n_parallel)  # Convert the Pandas DataFrame to a Dask DataFrame
        return super().apply(df, scheduler=scheduler, fault_tolerant=fault_tolerant)  # Label the Dask DataFrame using the DaskLFApplier
