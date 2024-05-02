from typing import Any, Tuple

import numpy as np

from snorkel.labeling.model.base_labeler import BaseLabeler

class RandomVoter(BaseLabeler):
    """Random vote label model."""

    def predict_proba(self, L: np.ndarray) -> np.ndarray:
        """Assign random votes to the data points.

        Parameters
        ----------
        L : np.ndarray
            An [n, m] matrix of labels

        Returns
        -------
        np.ndarray
            A [n, k] array of probabilistic labels

        Example
        -------
        >>> L = np.array([[0, 0, -1], [-1, 0, 1], [1, -1, 0]])
        >>> random_voter = RandomVoter()
        >>> predictions = random_voter.predict_proba(L)
        """
        n = L.shape[0]
        Y_p = np.random.rand(n, self.cardinality)
        Y_p /= Y_p.sum(axis=1).reshape(-1, 1)
        return Y_p

class MajorityClassVoter(BaseLabeler):
    """Majority class label model."""

    def fit(self, balance: np.ndarray, *args: Any, **kwargs: Any) -> None:
        """Train majority class model.

        Set class balance for majority class label model.

        Parameters
        ----------
        balance : np.ndarray
            A [k] array of class probabilities
        """
        self.balance = balance

    def predict_proba(self, L: np.ndarray) -> np.ndarray:
        """Predict probabilities using majority class.

        Assign majority class vote to each datapoint.
        In case of multiple majority classes, assign equal probabilities among them.

        Parameters
        ----------
        L : np.ndarray
            An [n, m] matrix of labels

        Returns
        -------
        np.ndarray
            A [n, k] array of probabilistic labels

        Example
        -------
        >>> L = np.array([[0, 0, -1], [-1, 0, 1], [1, -1, 0]])
        >>> maj_class_voter = MajorityClassVoter()
        >>> maj_class_voter.fit(balance=np.array([0.8, 0.2]))
        >>> maj_class_voter.predict_proba(L)
        array([[1., 0.],
               [1., 0.],
               [1., 0.]])
        """
        n = L.shape[0]
        max_class = np.argmax(self.balance)
        Y_p = np.zeros((n, self.cardinality))
        Y_p[:, max_class] = 1
        return Y_p

class MajorityLabelVoter(BaseLabeler):
    """Majority vote label model."""

    def predict_proba(self, L: np.ndarray) -> np.ndarray:
        """Predict probabilities using majority vote.

        Assign vote by calculating majority vote across all labeling functions.
        In case of ties, non-integer probabilities are possible.

        Parameters
        ----------
        L : np.ndarray
            An [n, m] matrix of labels

        Returns
        -------
        np.ndarray
            A [n, k] array of probabilistic labels

        Example
        -------
        >>> L = np.array([[0, 0, -1], [-1, 0, 1], [1, -1, 0]])
        >>> maj_voter = MajorityLabelVoter()
        >>> maj_voter.predict_proba(L)
        array([[1. , 0. ],
               [0.5, 0.5],
               [0.5, 0.5]])
        """
        n, m = L.shape
        Y_p = np.zeros((n, self.cardinality))
        for i in range(n):
            unique_labels, counts = np.unique(L[i, np.where(L[i] != -1)], return_counts=True)
            Y_p[i, unique_labels] = counts / counts.sum()
        return Y_p
