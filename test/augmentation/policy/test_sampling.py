import unittest

from snorkel.augmentation import MeanFieldPolicy, RandomPolicy

class TestSamplingPolicy(unittest.TestCase):
    """Tests for the Snorkel sampling policies."""

    def setUp(self):
        """Creates the policies to be tested."""
        self.random_policy = RandomPolicy(2, sequence_length=2)
        self.mean_field_policy = MeanFieldPolicy(2, sequence_length=2, p=[1, 0])

    def tearDown(self):
        """Removes the policies after testing."""
        del self.random_policy
        del self.mean_field_policy

    def test_random_policy(self):
        """Tests the RandomPolicy class."""
        n_samples = 1000
        samples = [self.random_policy.generate() for _ in range(n_samples)]
        a_ct = samples.count([0, 0])
        b_ct = samples.count([0, 1])
        c_ct = samples.count([1, 0])
        d_ct = samples.count([1, 1])
        total_ct = a_ct + b_ct + c_ct + d_ct
        self.assertGreater(a_ct, 0)
        self.assertGreater(b_ct, 0)
        self.assertGreater(c_ct, 0)
        self.assertGreater(d_ct, 0)
        self.assertAlmostEqual(total_ct / n_samples, 1, delta=0.01)

    def test_mean_field_policy(self):
        """Tests the MeanFieldPolicy class."""
        n_samples = 1000
        samples = [self.mean_field_policy.generate() for _ in range(n_samples)]
        sequence_ct = samples.count([0, 0])
        total_ct = sequence_ct
        self.assertGreater(total_ct, 0.95 * n_samples)
        self.assertEqual(samples.count([0, 1]), 0)
        self.assertEqual(samples.count([1, 0]), 0)
        self.assertEqual(samples.count([1, 1]), 0)
