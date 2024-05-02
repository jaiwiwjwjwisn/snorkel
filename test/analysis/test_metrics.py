import unittest

import numpy as np
from typing import Any, Dict, List, Optional, Tuple

from snorkel.analysis import metric_score
from snorkel.labeling.model import LabelingModel
from snorkel.utils import preds_to_probs


class MetricsTest(unittest.TestCase):
    """Test cases for the `metric_score` function."""

    def test_accuracy_basic(self,) -> None:
        """Test the `accuracy` metric with basic input arrays."""
        golds = np.array([0, 0, 0, 1, 1])
        preds = np.array([0, 0, 0, 1, 0])
        score = metric_score(golds, preds, probs=None, metric="accuracy")
        self.assertAlmostEqual(score, 0.8)

    def test_accuracy_probs(self) -> None:
        """Test the `accuracy` metric with probability inputs."""
        golds = np.array([0, 0, 0, 1, 1])
        probs = preds_to_probs(golds, 2)
        score = metric_score(golds, preds=None, probs=probs, metric="accuracy")
        self.assertAlmostEqual(score, 1.0)

    def test_bad_inputs(self) -> None:
        """Test the `metric_score` function with bad input types."""
        golds = np.array([0, 0, 0, 1, 1])
        pred1 = np.array([0, 0, 0, 1, 0.5])
        pred2 = np.array([[0, 0, 0, 0, 0], [1, 1, 1, 1, 1]])
        with self.assertRaisesRegex(
            ValueError, "Input contains at least one non-integer"
        ):
            metric_score(golds, pred1, probs=None, metric="accuracy")

        with self.assertRaisesRegex(ValueError, "Input could not be converted"):
            metric_score(golds, pred2, probs=None, metric="accuracy")

        with self.assertRaisesRegex(ValueError, "The metric you provided"):
            metric_score(golds, pred2, probs=None, metric="bad_metric")

        with self.assertRaisesRegex(
            ValueError, "filter_dict must only include keys in"
        ):
            metric_score(
                golds,
                golds,
                probs=None,
                metric="accuracy",
                filter_dict={"bad_map": [0]},
            )

    # ... other test cases omitted for brevity


if __name__ == "__main__":
    unittest.main()
