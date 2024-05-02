from collections import OrderedDict
from itertools import product
from typing import List, Optional, Union

import numpy as np
import scipy.sparse as sparse
from pandas import DataFrame, Series
from sklearn.metrics import confusion_matrix

from snorkel.utils import to_int_label_array

from .lf import LabelingFunction


class LFAnalysis:
    """Analyze labeling functions using a label matrix.

    Attributes
    ----------
    label_matrix : np.ndarray
        A 2D numpy array where `label_matrix[i, j]` is the label given by the jth
        LF to the ith candidate (using -1 for abstain).
    lfs : Optional[List[LabelingFunction]]
        The labeling functions used to generate `label_matrix`.
    lf_names : Optional[List[str]]
        The names of the labeling functions.
    """

    def __init__(
        self, label_matrix: np.ndarray, lfs: Optional[List[LabelingFunction]] = None
    ) -> None:
        if not isinstance(label_matrix, np.ndarray) or label_matrix.ndim != 2:
            raise ValueError("label_matrix must be a 2D numpy array")

        self.label_matrix = label_matrix
        self._label_matrix_sparse = sparse.csr_matrix(label_matrix + 1)
        self._lf_names = None

        if lfs is not None:
            if len(lfs) != self._label_matrix_sparse.shape[1]:
                raise ValueError(
                    f"Number of LFs ({len(lfs)}) and number of "
                    f"LF matrix columns ({self._label_matrix_sparse.shape[1]}) are different"
                )
            self._lf_names = [lf.name for lf in lfs]

    def _covered_data_points(self) -> np.ndarray:

