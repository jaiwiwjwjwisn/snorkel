import unittest
from typing import List, Dict, Any, Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch.utils.data as data

from snorkel.analysis import Scorer
from snorkel.classification import DictDataset
from snorkel.slicing import SFApplier, SliceAwareClassifier, slicing_function


@slicing_function()
def num_greater_than_42(x: SimpleNamespace) -> int:
    """Slicing function that returns 1 if x.num > 42, 0 otherwise."""
    return x.num > 42


@slicing_function()
def num_greater_than_10(x: SimpleNamespace) -> int:
    """Slicing function that returns 1 if x.num > 10, 0 otherwise."""
    return x.num > 10


slicing_functions = [num_greater_than_42, num_greater_than_10]
DATA = [3, 43, 12, 9, 3]


def create_dataset(
    X: np.ndarray, Y: np.ndarray, split: str, dataset_name: str, input_name: str, task_name: str
) -> DictDataset:
    """Create a dict dataset with the given inputs and labels."""
    return DictDataset(
        name=dataset_name,
        split=split,
        X_dict={input_name: X},
        Y_dict={task_name: Y},
    )


class SliceCombinerTest(unittest.TestCase):
    """Test cases for the SliceCombiner class."""

    def setUp(self) -> None:
        """Set up the test environment."""
        # Define data points
        data_points = [SimpleNamespace(num=num) for num in DATA]

        # Define slicing functions
        applier = SFApplier(slicing_functions)
        self.S = applier.apply(data_points, progress_bar=False)

        # Define base architecture
        self.hidden_dim = 10
        self.mlp = nn.Sequential(
            nn.Linear(2, self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
        )

        # Define model parameters
        self.data_name = "input_data"
        self.task_name = "test_task"

        # Define datasets
        self.X = torch.FloatTensor([(x, x) for x in DATA])
        # Alternating labels
        self.Y = torch.LongTensor([int(i % 2 == 0) for i in range(len(DATA))])

        dataset_name = "test_dataset"
        splits = ["train", "valid"]
        self.datasets = [
            create_dataset(
                self.X, self.Y, split, dataset_name, self.data_name, self.task_name
            )
            for split in splits
        ]

        self.slice_model = SliceAwareClassifier(
            base_architecture=self.mlp,
            head_dim=self.hidden_dim,
            slice_names=[sf.name for sf in slicing_functions],
            input_data_key=self.data_name,
            task_name=self.task_name,
            scorer=Scorer(metrics=["f1"]),
        )

    def test_slice_tasks(self) -> None:
        """Test if all the desired slice tasks are initialized."""

        expected_tasks = {
            # Base task
            self.task_name,
            # Slice tasks for default base slice
            f"{self.task_name}_slice:base_pred",
            f"{self.task_name}_slice:base_ind",
            # Slice Tasks
            f"{self.task_name}_slice:{num_greater_than_42.name}_pred",
            f"{self.task_name}_slice:{num_greater_than_42.name}_ind",
            f"{self.task_name}_slice:{num_greater_than_10.name}_pred",
            f"{self.task_name}_slice:{num_greater_than_10.name}_ind",
        }
        self.assertEqual(self.slice_model.task_names, expected_tasks)

    def test_make_slice_dataloader(self) -> None:
        """Test if the slice dataloader is constructed correctly."""

        # Test correct construction
        dataloader = self.slice_model.make_slice_dataloader(
            dataset=self.datasets[0], S=self.S
        )
        Y_dict = dataloader.dataset.Y_dict
        self.assertEqual(len(Y_dict), 7)
        self.assertIn(self.task_name, Y_dict)
        self.assertIn(
            f"{self.task_name}_slice:base_pred", Y_dict
        )
        self.assertIn(
            f"{self.task_name}_slice:base_ind", Y_dict
        )
        self.assertIn(
            f"{self.task_name}_slice:{num_greater_than_42.name}_pred", Y_dict
        )
        self.assertIn(
            f"{self.task_name}_slice:{num_greater_than_42.name}_ind", Y_dict
        )
        self.assertIn(
            f"{self.task_name}_slice:{num_greater_than_10.name}_pred", Y_dict
        )
        self.assertIn(
            f"{self.task_name}_slice:{num_greater_than_10.name}_ind", Y_dict
        )

        # Test bad data input
        bad_data_dataset = DictDataset(
            name="test_data",
            split="train",
            X_dict={self.data_name: self.X},
            Y_dict={"bad_labels": self.Y},
        )
        with self.assertRaisesRegex(ValueError, "labels missing"):
            self.slice_model.make_slice_dataloader(
                dataset=bad_data_dataset, S=self.S
            )

    def test_scores_pipeline(self) -> None:
        """Test if the appropriate scores are returned with .score and .score_slices."""
        # Make valid dataloader
        valid_dl = self.slice_model.make_slice_dataloader(
            dataset=self.datasets[1], S=self.S, batch_size=4
        )

        # Eval overall
        scores = self.slice_model.score([valid_dl])
        # All labels should appears in .score() output
        self.assertIn(
            f"{self.task_name}/{self.datasets[1].name}/{self.datasets[1].split}/f1",
            scores,
        )
        self.assertIn(
            f"{self.task_name}_slice:{num_greater_than_42.name}_pred/{self.datasets[1].name}/{self.datasets[1].split}/f1",
            scores,
        )
        self.assertIn(
            f"{self.task_name}_slice:{num_greater_than_42.name}_ind/{self.datasets[1].name}/{self.datasets[1].split}/f1",
            scores,
        )
        self.assertIn(
            f"{self.task_name}_slice:{num_greater_than_10.name}_pred/{self.datasets[1].name}/{self.datasets[1].split}/f1",
            scores,
        )
        self.assertIn(
            f"{self.task_name}_slice:{num_greater_than_10.name}_ind/{self.datasets[1].name}/{self.datasets[1].split}/f1",
            scores,
        )

        # Eval on slices
        slice_scores = self.slice_model.score_slices([valid_dl])
        # Check that we eval on 'pred' labels in .score_slices() output
        self.assertIn(
            f"{self.task_name}/{self.datasets[1].name}/{self.datasets[1].split}/f1",
            slice_scores,
        )
        self.assertIn(
            f"{self.task_name}_slice:{num_greater_than_42.name}_pred/{self.datasets[1].name}/{self.datasets[1].split}/f1",
            slice_scores,
        )
        self.assertIn(
            f"{self.task_name}_slice:{num_greater_than_10.name}_pred/{self.datasets[1].name}/{self.datasets[1].split}/f1",
            slice_scores,
        )

        # No 'ind' labels!
        self.assertNotIn(
            f"{self.task_name}_slice:{num_greater_than_42.name}_ind/{self.datasets[1].name}/{self.datasets[1].split}/f1",
            slice_scores,
        )
        self.assertNotIn(
            f"{self.task_name}_slice:{num_greater_than_10.name}_ind/{self.datasets[1].name}/{self.datasets[1].split}/f1",
            slice_scores,
        )

    def test_score_slices(self) -> None:
        """Test if the appropriate scores are returned with .score_slices."""
        # Make valid dataloader
        valid_dl = self.slice_model.make_slice_dataloader(
            dataset=self.datasets[1], S=self.S, batch_size=4
        )

        # Eval on slices
        slice_scores = self.slice_model.score_slices([valid_dl])
        # Check that we eval on 'pred' labels in .score_slices() output
        self.assertIn(
            f"{self.task_name}/{self.datasets[1].name}/{self.datasets[1].split}/f1",
            slice_scores,
        )
        self.assertIn(
            f"{self.task_name}_slice:{num_greater_than_42.name}_pred/{self.datasets[1].name}/{self.datasets[1].split}/f1",
            slice_scores,
        )
        self.assertIn(
            f"{self.task_name}_slice:{num_greater_than_10.name}_pred/{self.datasets[1].name}/{self.datasets[1].split}/f1",
            slice_scores,
        )

        # No 'ind' labels!
        self.assertNotIn(
            f"{self.task_name}_slice:{num_greater_than_42.name}_ind/{self.datasets[1].name}/{self.datasets[1].split}/f1",
            slice_scores,
        )
        self.assertNotIn(
            f"{self.task_name}_slice:{num_greater_than_10.name}_ind/{self.datasets[1].name}/{self.datasets[1].split}/f1",
            slice_scores,
        )


if __name__ == "__main__":
    unittest.main()
