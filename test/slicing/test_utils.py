import unittest

import pandas as pd
import torch
import torch.nn as nn

from snorkel.classification import DictDataLoader, DictDataset, Operation, Task
from snorkel.slicing import (
    PandasSFApplier,
    add_slice_labels,
    convert_to_slice_tasks,
    slicing_function,
)

@slicing_function()  # Decorator to register the function as a slicing function
def f(x):
    """
    Slicing function that checks if the value is less than 0.25.

    :param x: Input tensor
    :return: Boolean tensor with True for values less than 0.25
    """
    return x.val < 0.25


class UtilsTest(unittest.TestCase):
    def test_add_slice_labels(self):
        # Create dummy data
        x = torch.Tensor([0.1, 0.2, 0.3, 0.4, 0.5])
        y = torch.Tensor([0, 1, 1, 0, 1]).long()

        # Initialize a DictDataset with the dummy data
        dataset = DictDataset(
            name="TestData", split="train", X_dict={"data": x}, Y_dict={"TestTask": y}
        )

        # Check the initial number of labelsets in the dataset
        self.assertEqual(len(dataset.Y_dict), 1)

        # Create a DataFrame from the input data
        df = pd.DataFrame({"val": x, "y": y})

        # Initialize a list of slicing functions
        slicing_functions = [f]

        # Initialize a PandasSFApplier with the slicing functions
        applier = PandasSFApplier(slicing_functions)

        # Apply the slicing functions to the DataFrame
        S = applier.apply(df, progress_bar=False)

        # Initialize a DictDataLoader with the dataset
        dataloader = DictDataLoader(dataset)

        # Create a dummy task
        dummy_task = create_dummy_task(task_name="TestTask")

        # Add slice labels to the dataloader using the dummy task and the S
        add_slice_labels(dataloader, dummy_task, S)

        # Check if all the expected labelsets are present
        labelsets = dataloader.dataset.Y_dict
        self.assertIn("TestTask", labelsets)
        self.assertIn("TestTask_slice:base_ind", labelsets)
        self.assertIn("TestTask_slice:base_pred", labelsets)
        self.assertIn("TestTask_slice:f_ind", labelsets)
        self.assertIn("TestTask_slice:f_pred", labelsets)
        self.assertEqual(len(labelsets), 5)

        # Check if the "ind" labelsets contain the correct masks
        self.assertEqual(
            labelsets["TestTask_slice:f_ind"].numpy().tolist(), [1, 1, 0, 0, 0]
        )
        self.assertEqual(
            labelsets["TestTask_slice:base_ind"].numpy().tolist(), [1, 1, 1, 1, 1]
        )

        # Check if the "pred" labelsets contain the correct masked elements
        self.assertEqual(
            labelsets["TestTask_slice:f_pred"].numpy().tolist(), [0, 1, -1, -1, -1]
        )
        self.assertEqual(
            labelsets["TestTask_slice:base_pred"].numpy().tolist(), [0, 1, 1, 0, 1]
        )
        self.assertEqual(labelsets["TestTask"].numpy().tolist(), [0, 1, 1, 0, 1])

    def test_convert_to_slice_tasks(self):
        # Initialize a dummy task
        task_name = "TestTask"
        task = create_dummy_task(task_name)

        # Initialize a list of slice names
        slice_names = ["slice_a", "slice_b", "slice_c"]

        # Convert the task to slice tasks
        slice_tasks = convert_to_slice_tasks(task, slice_names)

        # Check if the original base task is present
        slice_task_names = [t.name for t in slice_tasks]
        self.assertIn(task_name, slice_task_names)

        # Check if there are 2 tasks (pred + ind) per slice, accounting for base slice
        for slice_name in slice_names + ["base"]:
            self.assertIn(f"{task_name}_slice:{slice_name}_pred", slice_task_names)
            self.assertIn(f"{task_name}_slice:{slice_name}_ind", slice_task_names)

        self.assertEqual(len(slice_tasks), 2 * (len(slice_names) + 1) + 1)

        # Check if the modules share the same body flow operations
        body_flow = task.op_sequence[:-1]
        ind_and_pred_tasks = [
            t for t in slice_tasks if "_ind" in t.name or "_pred" in t.name
        ]
        for op in body_flow:
            for slice_task in ind_and_pred_tasks:
                self.assertTrue(
                    slice_task.module_pool[op.module_name]
                    is task.module_pool[op.module_name]
                )

        # Check if the pred tasks share the same predictor head
        pred_tasks = [t for t in slice_tasks if "_pred" in t.name]
        predictor_head_name = pred_tasks[0].op_sequence[-1].module_name
        shared_predictor_head = pred_tasks[0].module_pool[predictor_head_name]
        for pred_task in pred_tasks[1:]:
            self.assertTrue(
                pred_task.module_pool[predictor_head_name] is shared_predictor_head
            )


def create_dummy_task(task_name):
    # Create a dummy task
    module_pool = nn.ModuleDict(
        {"linear1": nn.Linear(2, 10), "linear2": nn.Linear(10, 2)}
    )

    op_sequence = [
        Operation(name="encoder", module_name="linear1", inputs=["_input_"]),
        Operation(name="prediction_head", module_name="linear2", inputs=["encoder"]),
    ]

    task = Task(name=task_name, module_pool=module_pool, op_sequence=op_sequence)
    return task
