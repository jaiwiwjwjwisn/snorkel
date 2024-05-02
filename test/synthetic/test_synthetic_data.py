import unittest
from typing import Optional

import numpy as np

import snorkel.labeling
from snorkel.labeling import LFAnalysis
from snorkel.synthetic.synthetic_data import generate_simple_label_matrix


class TestGenerateSimpleLabelMatrix(unittest.TestCase):
    """Testing the generate_simple_label_matrix function."""

    def setUp(self) -> None:
        """Set constants for the tests."""
        self.n = 1000  # Number of data points
        self.m = 10  # Number of LFs
        self.k = 2  # Cardinality
        self.decimal = 2  # Number of decimals to check element-wise error
        np.random.seed(123)

    def test_generate_L(self) -> None:
        """Test the generated dataset for consistency."""
        P, Y, L = generate_simple_label_matrix(self.n, self.m, self.k)
        P_emp = LFAnalysis(L).lf_empirical_probs(Y, k=self.k)
        np.testing.assert_array_almost_equal(P, P_emp, decimal=self.decimal)

    def test_generate_L_multiclass(self) -> None:
        """Test the generated dataset for consistency with cardinality=3."""
        self.k = 3
        P, Y, L = generate_simple_label_matrix(self.n, self.m, self.k)
        P_emp = LFAnalysis(L).lf_empirical_probs(Y, k=self.k)
        np.testing.assert_array_almost_equal(P, P_emp, decimal=self.decimal)


if __name__ == "__main__":
    unittest.main()
