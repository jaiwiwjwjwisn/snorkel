import random
import unittest
from typing import List

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

from snorkel.analysis import Scorer
from snorkel.classification import (
    DictDataLoader,
    DictDataset,
    MultitaskClassifier,
    Operation,
    Task,
    Trainer,
)

N_TRAIN = 1000  # Number of training examples
N_VALID = 300  # Number of validation examples


class ClassifierConvergenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Ensure deterministic runs
        random.seed(123)
        np.random.seed(123)
        torch.manual_seed(123)

    @pytest.mark.complex  # This test is complex and time-consuming
    def test_convergence(self):
        """Test multitask classifier convergence with two tasks."""

        dataloaders = []  # List to store data loaders for each task and split

        for offset, task_name in zip([0.0, 0.25], ["task1", "task2"]):
            df = create_data(N_TRAIN, offset)  # Create training data for the task
            dataloader = create_dataloader(df, "train", task_name)  # Create train data loader
            dataloaders.append(dataloader)

        for offset, task_name in zip([0.0, 0.25], ["task1", "task2"]):
            df = create_data(N_VALID, offset)  # Create validation data for the task
            dataloader = create_dataloader(df, "valid", task_name)  # Create valid data loader
            dataloaders.append(dataloader)

        task1 = create_task("task1", module_suffixes=["A", "A"])  # Create task 1
        task2 = create_task("task2", module_suffixes=["A", "B"])  # Create task 2
        model = MultitaskClassifier(tasks=[task1, task2])  # Initialize the multitask classifier

        # Train the model
        trainer = Trainer(lr=0.0024, n_epochs=10, progress_bar=False)
        trainer.fit(model, dataloaders)
        scores = model.score(dataloaders)  # Calculate scores for the model

        # Confirm near perfect scores on both tasks
        for idx, task_name in enumerate(["task1", "task2"]):
            self.assertGreater(
                scores[f"{task_name}/TestData/valid/accuracy"], 0.95
            )  # Check if accuracy is greater than 0.95

            # Calculate/check train/val loss
            train_dataset = dataloaders[idx].dataset
            train_loss_output = model.calculate_loss(
                train_dataset.X_dict, train_dataset.Y_dict
            )
            train_loss = train_loss_output[0][task_name].item()
            self.assertLess(train_loss, 0.05)  # Check if train loss is less than 0.05

            val_dataset = dataloaders[2 + idx].dataset
            val_loss_output = model.calculate_loss(
                val_dataset.X_dict, val_dataset.Y_dict
            )
            val_loss = val_loss_output[0][task_name].item()
            self.assertLess(val_loss, 0.05)  # Check if validation loss is less than 0.05


def create_data(n: int, offset=0) -> pd.DataFrame:
    """Create uniform X data from [-1, 1] on both axes.

    Create labels with linear decision boundaries related to the two coordinates of X.
    """
    X = (np.random.random((n, 2)) * 2 - 1).astype(np.float32)
    Y = (X[:, 0] < X[:, 1] + offset).astype(int)

    df = pd.DataFrame({"x1": X[:, 0], "x2": X[:, 1], "y": Y})
    return df


def create_dataloader(df: pd.DataFrame, split: str, task_name: str
