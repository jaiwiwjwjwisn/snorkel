import unittest

import torch.nn as nn

from snorkel.classification import Operation, Task

# Define a constant string for the task name
TASK_NAME = "TestTask"


class TaskTest(unittest.TestCase):
    def test_task_creation(self):
        # Initialize a module pool, a dictionary containing various neural network modules
        module_pool = nn.ModuleDict(
            {
                "linear1": nn.Sequential(nn.Linear(2, 10), nn.ReLU()),
                "linear2": nn.Linear(10, 1),
            }
        )

        # Define a sequence of operations for the task
        op_sequence = [
            Operation(
                name="the_first_layer",  # A unique name for the operation
                module_name="linear1",  # The name of the module to be used
                inputs=["_input_"]  # Inputs to the module, "_input_" denotes the initial input
            ),
            Operation(
                name="the_second_layer",  # A unique name for the operation
                module_name="linear2",  # The name of the module to be used
                inputs=["the_first_layer"],  # Inputs to the module, "the_first_layer" denotes the output of the previous operation
            ),
        ]

        # Initialize a Task object with the given name, module pool, and operation sequence
        task = Task(name=TASK_NAME, module_pool=module_pool, op_sequence=op_sequence)

        # Task has no functionality on its own
        # Here we only confirm that the object was initialized
        self.assertEqual(task.name, TASK_NAME)


if __name__ == "__main__":
    unittest.main()

