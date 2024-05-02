import logging
from collections import defaultdict
from typing import DefaultDict, Dict, List, Tuple

import numpy as np

from snorkel.utils import to_int_label_array


def get_label_buckets(*y: List[np.ndarray]) -> Dict[Tuple[int, ...], np.ndarray]:
    """Return data point indices bucketed by label combinations.

    Parameters
    ----------
    *y
        A list of np.ndarray of (int) labels

    Returns
    -------
    Dict[Tuple[int, ...], np.ndarray]
        A mapping of each label bucket to a NumPy array of its corresponding indices

    Example
    -------
    A common use case is calling ``buckets = label_buckets(Y_gold, Y_pred)`` where
    ``Y_gold`` is a set of gold (i.e. ground truth) labels and ``Y_pred`` is a
    corresponding set of predicted labels.

    >>> Y_gold = np.array([1, 1, 1, 0])
    >>> Y_pred = np.array([1, 1, -1, -1])
    >>> buckets = get_label_buckets(Y_gold, Y_pred)

    The returned ``buckets[(i, j)]`` is a NumPy array of data point indices with
    true label i and predicted label j.

    More generally, the returned indices within each bucket refer to the order of the
    labels that were passed in as function arguments.

    >>> buckets[(1, 1)]  # true positives
    array([0, 1])
    >>> (1, 0) in buckets  # false positives
    False
    >>> (0, 1) in buckets  # false negatives
    False
    >>> (0, 0) in buckets  # true negatives
    False
    >>> buckets[(1, -1)]  # abstained positives
    array([2])
    >>> buckets[(0, -1)]  # abstained negatives
    array([3])
    """
    if len(y) == 0:
        raise ValueError("At least one label list is required")
    if len(set(map(len, y))) != 1:
        raise ValueError("All label lists must have the same number of elements")
    x = y[0]
    if len(x) == 0:
        return {}
    if len(set(x)) < len(x):
        y = [to_int_label_array(z, flatten_vector=True) for z in y]
    buckets: DefaultDict[Tuple[int, int], List[int]] = defaultdict(list)
    for i, labels in enumerate(zip(*y)):
        if len(labels) != len(x):
            raise ValueError(
                "Number of rows in x does not match number of elements in at least one label list"
            )
        buckets[labels].append(i)
    if len(buckets) == 0:
        return {}
    return {k: np.array(v) for k, v in buckets.items()}


def get_label_instances(
    bucket: Tuple[int, ...], x: np.ndarray, *y: List[np.ndarray]
) -> np.ndarray:
    """Return instances in x with the specified combination of labels.

    Parameters
    ----------
    bucket
        A tuple of label values corresponding to which instances from x are returned
    x
        NumPy array of data instances to be returned
    *y
        A list of np.ndarray of (int) labels

    Returns
    -------
    np.ndarray
        NumPy array of instances from x with the specified combination of labels

    Example
    -------
    A common use case is calling ``get_label_instances(bucket, x.to_numpy(), Y_gold, Y_pred)``
    where ``x`` is a NumPy array of data instances that the labels correspond to,
    ``Y_gold`` is a list of gold (i.e. ground truth) labels, and
    ``Y_pred`` is a corresponding list of predicted labels.

    >>> import pandas as pd
    >>> x = pd.DataFrame(data={'col1': ["this is a string", "a second string", "a third string"], 'col2': ["1", "2", "3"]})
    >>> Y_gold = np.array([1, 1, 1])
    >>> Y_pred = np.array([1, 0, 0])
    >>> bucket = (1, 0)

    The returned NumPy array of data instances from ``x`` will correspond to
    the rows where the first list had a 1 and the second list had a 0.
    >>> get_label_instances(bucket, x.to_numpy(), Y_gold, Y_pred)
    array([['a second string', '2'],
           ['a third string', '3']], dtype=object)

    More generally, given bucket ``(i, j, ...)`` and lists ``y1, y2, ...``
    the returned data instances from ``x`` will correspond to the rows where
    y1 had label i, y2 had label j, and so on. Note that ``x`` and ``y``
    must all be the same length.
    """
    if len(y) == 0:
        raise ValueError("At least one label list is required")
    if len(bucket) == 0:
        return np.array([])
    if len(y) != len(bucket):
        raise ValueError("Number of lists must match the amount of labels in bucket")
    if x.shape[0] != len(y[0]):
        # Note: the check for all y having the same number of elements occurs in get_label_buckets
        raise ValueError(
            "Number of rows in x does not match number of elements in at least one label list"
        )
    buckets = get_label_buckets(*y)
    try:
        indices = buckets[bucket]
    except KeyError:
        return np.array([])
    instances = x[indices]
    return instances
