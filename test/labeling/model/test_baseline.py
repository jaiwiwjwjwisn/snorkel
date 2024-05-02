import unittest

import numpy as np

from snorkel.labeling.model import MajorityClassVoter, MajorityLabelVoter, RandomVoter

class BaselineModelTest(unittest.TestCase):
    """Tests for the baseline label model classes in Snorkel."""

    def setUp(self) -> None:
        """Set up the random seed for the RandomVoter tests."""
        np.random.seed(0)

    def tearDown(self) -> None:
        """Reset the random seed after each test."""
        np.random.seed()

    def test_random_vote(self) -> None:
        """Test the RandomVoter class."""
        L = np.array([[0, 1, 0], [-1, 3, 2], [2, -1, -1], [0, 1, 1]])
        rand_voter = RandomVoter()
        Y_p = rand_voter.predict_proba(L)

        self.assertLessEqual(Y_p.max(), 1.0)
        self.assertGreaterEqual(Y_p.min(), 0.0)
        np.testing.assert_array_almost_equal(
            np.sum(Y_p, axis=1), np.ones(np.shape(L)[0])
        )
        self.assertEqual(np.shape(Y_p), (np.shape(L)[0], 2))

    def test_majority_class_vote(self) -> None:
        """Test the MajorityClassVoter class."""
        L = np.array([[0, 1, 0], [1, 1, 0], [1, 1, 0], [-1, -1, 1]])
        mc_voter = MajorityClassVoter()
        mc_voter.fit(balance=np.array([0.8, 0.2]))
        Y_p = mc_voter.predict_proba(L)

        Y_p_true = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
        np.testing.assert_array_almost_equal(Y_p, Y_p_true)
        self.assertEqual(np.shape(Y_p), (np.shape(L)[0], 2))

    def test_majority_label_vote(self) -> None:
        """Test the MajorityLabelVoter class."""
        L = np.array([[0, 1, 0], [0, 1, 0], [1, 0, 0], [-1, -1, 1]])
        ml_voter = MajorityLabelVoter()
        Y_p = ml_voter.predict_proba(L)

        Y_p_true = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        np.testing.assert_array_almost_equal(Y_p, Y_p_true)
        self.assertEqual(np.shape(Y_p), (np.shape(L)[0], 2))

if __name__ == "__main__":
    unittest.main()
