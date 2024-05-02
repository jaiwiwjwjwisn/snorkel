from typing import Tuple

import numpy as np
from pyspark import RDD

from snorkel.types import DataPoint
from .core import BaseLFApplier, RowData, _FunctionCaller, apply_lfs_to_data_point

class SparkLFApplier(BaseLFApplier):
    r"""
    A class for applying Snorkel's labeling functions (LFs) to an RDD of DataPoints using PySpark.

    This LF applier stores data points as `Row` objects in an RDD and submits a Spark `map` job to execute the LFs.
    An RDD can typically be obtained from a PySpark DataFrame.

    For an example usage with AWS EMR instructions, see the `test/labeling/apply/lf_applier_spark_test_script.py` file.
    """

    def apply(self, data_points: RDD, fault_tolerant: bool = False) -> np.ndarray:
        """
        Label an RDD of DataPoints with LFs and return the resulting label matrix.

        Parameters
        ----------
        data_points : RDD
            A Resilient Distributed Dataset (RDD) containing DataPoints to be labeled by LFs.
        fault_tolerant : bool, optional
            If True, output -1 if LF execution fails, by default False.

        Returns
        -------
        np.ndarray
            A 2D numpy array of labels emitted by LFs.
        """
        f_caller = _FunctionCaller(fault_tolerant)

        def map_fn(args: Tuple[DataPoint, int]) -> RowData:
            """
            A helper function to apply LFs to a single DataPoint.

            Parameters
            ----------
            args : Tuple[DataPoint, int]
                A tuple containing a DataPoint and its index.

            Returns
            -------
            RowData
                A namedtuple containing the labeled data point and its corresponding label.
            """
            return apply_lfs_to_data_point(*args, lfs=self._lfs, f_caller=f_caller)

        labels = data_points.zipWithIndex().map(map_fn).collect()
        return self._numpy_from_row_data(labels)
