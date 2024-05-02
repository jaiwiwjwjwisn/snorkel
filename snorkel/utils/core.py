import hashlib
from typing import Dict, List, Optional

import numpy as np


def _hash(i: int) -> int:
    """Deterministic hash function.

    This function takes an integer 'i' and returns its deterministic hash value.
    It converts the integer to a byte string, computes the SHA1 hash, and then
    converts the resulting hexadecimal string back to an integer.

    Parameters
    ----------
    i : int
        The integer to be hashed.

    Returns
    -------
    int
        The deterministic hash value of the input integer.
    """
    byte_string = str(i).encode("utf-8")
    return int(hashlib.sha1(byte_string).hexdigest(), 16)


def probs_to_preds(
    probs: np.ndarray, tie_break_policy: str = "random", tol: float = 1e-5
) -> np.ndarray:
    """Convert an array of probabilistic labels into an array of predictions.

    This function takes an array of probabilistic labels and converts it into an
    array of predictions based on the specified tie-break policy. If the probabilities
    for all classes are below a certain tolerance 'tol', the function will break the
    tie using the specified policy.

    Parameters
    ----------
    probs : np.ndarray
        A 2D array of probabilistic labels with shape [num_datapoints, num_classes].
    tie_break_policy : str, optional
        The policy to break ties when converting probabilistic labels to predictions.
        Supported policies are "abstain", "true-random", and "random". Default is "random".
    tol : float, optional
        The minimum difference among probabilities to be considered a tie. Default is 1e-5.

    Returns
    -------
    np.ndarray
        A 1D array of predictions (integers in [0, ..., num_classes - 1]) with shape [num_datapoints].
    """
    num_datapoints, num_classes = probs.shape
    if num_classes <= 1:
        raise ValueError(
            f"probs must have probabilities for at least 2 classes. "
            f"Instead, got {num_classes} classes."
        )

    Y_pred = np.empty(num_datapoints)
    diffs = np.abs(probs - probs.max(axis=1).reshape(-1, 1))

    for i in range(num_datapoints):
        max_idxs = np.where(diffs[i, :] < tol)[0]
        if len(max_idxs) == 1:
            Y_pred[i] = max_idxs[0]
        # Deal with "tie votes" according to the specified policy
        elif tie_break_policy == "random":
            Y_pred[i] = max_idxs[_hash(i) % len(max_idxs)]
        elif tie_break_policy == "true-random":
            Y_pred[i] = np.random.choice(max_idxs)
        elif tie_break_policy == "abstain":
            Y_pred[i] = -1
        else:
            raise ValueError(
                f"tie_break_policy={tie_break_policy} policy not recognized."
            )
    return Y_pred.astype(np.int_)


def preds_to_probs(preds: np.ndarray, num_classes: int) -> np.ndarray:
    """Convert an array of predictions into an array of probabilistic labels.

    This function takes an array of predictions and converts it into an array of
    probabilistic labels with probability 1.0 in the column corresponding to the
    prediction.

    Parameters
    ----------
    preds : np.ndarray
        A 1D array of predictions with shape [num_datapoints].
    num_classes : int
        The number of classes in the dataset.

    Returns
    -------
    np.ndarray
        A 2D array of probabilistic labels with shape [num_datapoints, num_classes].
    """
    if np.any(preds < 0):
        raise ValueError("Could not convert abstained vote to probability")
    return np.eye(num_classes)[preds.squeeze()]


def to_int_label_array(X: np.ndarray, flatten_vector: bool = True) -> np.ndarray:
    """Convert an array to a (possibly flattened) array of ints.

    This function takes an array and casts all values to integers. If the input is a
    2D array with shape [n, 1], it flattens it to a 1D array with shape [n].

    Parameters
    ----------
    X : np.ndarray
        An array to possibly flatten and possibly cast to int.
    flatten_vector : bool, optional
        If True, flatten array into a 1D array. Default is True.

    Returns
    -------
    np.ndarray
        The converted array.

    Raises
    ------
    ValueError
        Provided input could not be converted to an np.ndarray
    """
    if np.any(np.not_equal(np.mod(X, 1), 0)):
        raise ValueError("Input contains at least one non-integer value.")
    X = X.astype(np.dtype(int))
    # Correct shape
    if flatten_vector:
        X = X.squeeze()
        if X.ndim == 0:
            X = np.expand_dims(X, 0)
        if X.ndim != 1:
            raise ValueError("Input could not be converted to 1d np.array")
    return X


def filter_labels(
    label_dict: Dict[str, Optional[np.ndarray]], filter_dict: Dict[str, List[int]]
) -> Dict[str, np.ndarray]:
    """Filter out examples from arrays based on specified labels to filter.

    This function takes a dictionary of label arrays and a dictionary of filters
    and removes examples whose labels match the filter criteria for any label set.

    Parameters
    ----------
    label_dict : Dict[str, Optional[np.ndarray]]
        A mapping from label set name to the array of labels.
    filter_dict : Dict[str, List[int]]
        A mapping from label set name to the labels that should be filtered out for
        that label set.

    Returns
    -------
    Dict[str, np.ndarray]
        A mapping with the same keys as label_dict but with filtered arrays as values.

    Example
    -------
    >>> golds = np.array([-1, 0, 0, 1, 0])
    >>> preds = np.array([0, 0, 0, 1, -1])
    >>> filtered = filter_labels(
    ...     label_dict={"golds": golds, "preds": preds},
    ...     filter_dict={"golds": [-1], "preds": [-1]}
    ... )
    >>> filtered["golds"]
    array([0, 0, 1])
    >>> filtered["preds"]
    array([0, 0, 1])
    """
    masks = []
    for label_name, filter_values in filter_dict.items():
        label_array: Optional[np.ndarray] = label_dict.get(label_name)
        if label_array is not None:
            # _get_mask requires not-null input
            masks.append(_get_mask(label_array, filter_values))
    mask = (np.multiply(*masks) if len(masks) > 1 else masks[0]).squeeze()

    filtered = {}
    for label_name, label_array in label_dict.items():
        filtered[label_name] = label_array[mask] if label_array is not None else None
    return filtered


def _get_mask(label_array: np.ndarray, filter_values: List[int]) -> np.ndarray:
    """Return a boolean mask marking which labels are not in filter_values.

    This function takes an array of labels and a list of values to filter out and
    returns a boolean mask indicating which labels should be kept (1) or filtered (0).

    Parameters
    ----------
    label_array : np.ndarray
        An array of labels.
    filter_values : List[int]
        A list of values that should be filtered out of the label array.

    Returns
    -------
    np.ndarray
        A boolean mask indicating whether to keep (1) or filter (0) each example.
    """
    mask: np.ndarray = np.ones_like(label_array).astype(bool)
    for value in filter_values:
        mask *= np.where(label_array != value, 1, 0).astype(bool)
    return mask
