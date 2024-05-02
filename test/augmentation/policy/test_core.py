import unittest

from snorkel.augmentation import ApplyAllPolicy, ApplyEachPolicy

class TestPolicy(unittest.TestCase):
    def test_apply_each_policy(self):
        test_cases = [
            (3, True, [[], [0], [1], [2]]),
            (3, False, [[0], [1], [2]]),
        ]
        for num\_choices, keep\_original, expected\_samples in test\_cases:
            with self.subTest(num\_choices=num\_choices, keep\_original=keep\_original):
                policy = ApplyEachPolicy(num\_choices, keep\_original=keep\_original)
                samples = policy.generate\_for\_example()
                self.assertEqual(samples, expected\_samples)

    def test_apply_all_policy(self):
        test\_cases = [
            (3, 2, False, [[0, 1, 2], [0, 1, 2]]),
        ]
        for num\_choices, n\_per\_original, keep\_original, expected\_samples in test\_cases:
            with self.subTest(num\_choices=num\_choices, n\_per\_original=n\_per\_original, keep\_original=keep\_original):
                policy = ApplyAllPolicy(num\_choices, n\_per\_original=n\_per\_original, keep\_original=keep\_original)
                samples = policy.generate\_for\_example()
                self.assertEqual(samples, expected\_samples)
