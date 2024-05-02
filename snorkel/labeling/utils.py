from typing import Tuple, Union

import numpy as np
import pandas as pd


def filter_unlabeled_dataframe(
    X: pd.DataFrame, y: np.ndarray, L: np.ndarray
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Filter out examples not covered by any labeling function.

    Parameters
    ----------
    X : pd.DataFrame
        Data points in a Pandas DataFrame.
    y : np.ndarray
        Matrix of probabilities output by label model's predict_proba method.
    L : np.ndarray
        Matrix of labels emitted by LFs.

    Returns
    -------
    Tuple[pd.DataFrame, np.ndarray]
        Data points that were labeled by at least one LF in L, along with their
        probabilities matrix.

    Raises
    ------
    ValueError
        If the number of rows in X is not equal to the number of rows in L.
        If the number of columns in L is not equal to the number of labeling
        functions.
        If the labels in L are not either 0 or 1.
    """
    if X.shape[0] != L.shape[0]:
        raise ValueError("The number of rows in X must be equal to the number"
                         " of rows in L.")

    num_labeling_functions = L.shape[1]
    if L.dtype != np.int8:
        raise ValueError("The labels in L must be either 0 or 1.")
    if not np.all(np.logical_or(L == 0, L == 1)):
        raise ValueError("The labels in L must be either 0 or 1.")

    if y.shape[0] != X.shape[0]:
        raise ValueError("The number of rows in y must be equal to the number"
                         " of rows in X.")

    mask = (L != 0).any(axis=1)
    return X.iloc[mask], y[mask]
