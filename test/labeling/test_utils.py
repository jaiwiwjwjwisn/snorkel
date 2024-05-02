import unittest

import numpy as np
import pandas as pd

from snorkel.labeling import filter_unlabeled_dataframe


class TestAnalysis(unittest.TestCase):
    """Tests for the `filter_unlabeled_dataframe` function."""

    def setUp(self) -> None:
        """Create test data."""
        self.X = pd.DataFrame(dict(A=["x", "y", "z"], B=[1, 2, 3]))
        self.y = np.array(
            [[0.25, 0.25, 0.25, 0.25], [1.0, 0.0, 0.0, 0.0], [0.2, 0.3, 0.5, 0.0]]
        )
        self.L = np.array([[0, 1, -1], [-1, -1, -1], [1, 1, 0]])

    def test_filter_unlabeled_dataframe_some_positive_labels(self) -> None:
        """Test filtering unlabeled dataframe with some positive labels."""
        X_filtered, y_filtered = filter_unlabeled_dataframe(self.X, self.y, self.L)
        np.testing.assert_array_equal(X_filtered.values, np.array([["x", 1], ["z", 3]]))
        np.testing.assert_array_almost_equal(
            y_filtered, np.array([[0.25, 0.25, 0.25, 0.25], [0.2, 0.3, 0.5, 0.0]])
        )

    def test_filter_unlabeled_dataframe_no_positive_labels(self) -> None:
        """Test filtering unlabeled dataframe with no positive labels."""
        self.L[:, 0] = -1
        X_filtered, y_filtered = filter_unlabeled_dataframe(self.X, self.y, self.L)
        np.testing.assert_array_equal(X_filtered, pd.DataFrame(columns=self.X.columns))
        np.testing.assert_array_equal(y_filtered, np.zeros((0, self.y.shape[1])))

