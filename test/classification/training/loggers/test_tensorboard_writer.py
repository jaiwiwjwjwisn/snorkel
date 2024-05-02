import json
import os
import shutil
import tempfile
import unittest

from snorkel.classification.training.loggers import TensorBoardWriter
from snorkel.types import Config

# Define a temporary configuration class that inherits from the Config class
class TempConfig(Config):
    """
    A temporary configuration class for testing purposes.
    Inherits from the `Config` class and has two attributes: `a` and `b`.
    """
    a: int = 42
    b: str = "foo"


class TestTensorBoardWriter(unittest.TestCase):
    """
    A unittest TestCase class for testing the `TensorBoardWriter` class.
    """

    def setUp(self):
        """
        Create a temporary directory for testing.
        This method is called before each test method is executed.
        """
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        """
        Remove the temporary directory created in the `setUp` method.
        This method is called after each test method is executed.
        """
        shutil.rmtree(self.test_dir)

    def test_tensorboard_writer(self):
        """
        Test the `TensorBoardWriter` class by making API calls and checking the results.
        This method tests the `add_scalar`, `write_config`, and `cleanup` methods.
        """
        # Note: this just tests API calls. We rely on tensorboard's unit tests for correctness.

        # Define a name for the TensorBoard run
        run_name = "my_run"

        # Create a configuration object with an overridden value for the `b` attribute
        config = TempConfig(b="bar")

        # Create a TensorBoardWriter object with the specified run name and log directory
        writer = TensorBoardWriter(run_name=run_name, log_dir=self.test_dir)

        # Add a scalar value to TensorBoard
        writer.add_scalar("my_value", value=0.5, step=2)

        # Write the configuration object to a JSON file in TensorBoard
        writer.write_config(config)

        # Get the path to the JSON file containing the configuration object
        log_path = os.path.join(self.test_dir, run_name, "config.json")

        # Load the JSON file and compare the contents to the original configuration object
        with open(log_path, "r") as f:
            file_config = json.load(f)
        self.assertEqual(config._asdict(), file_config)

        # Clean up the TensorBoard log directory
        writer.cleanup()
