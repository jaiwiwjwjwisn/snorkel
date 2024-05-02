import json
import os
import shutil
import tempfile
import unittest

from snorkel.classification.training.loggers import LogWriter  # Import LogWriter from the snorkel library

# Define a custom configuration class that inherits from snorkel's Config class
class TempConfig(Config):
    a: int = 42  # Integer field 'a' with a default value of 42
    b: str = "foo"  # String field 'b' with a default value of "foo"


class TestLogWriter(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory to store logs for each test case
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        # Remove the temporary directory after each test case
        shutil.rmtree(self.test_dir)

    def test_log_writer(self):
        run_name = "my_run"  # Set a name for the current run
        log_writer = LogWriter(run_name=run_name, log_dir=self.test_dir)  # Initialize LogWriter with the run name and log directory
        log_writer.add_scalar("my_value", value=0.5, step=2)  # Add a scalar value to the log

        log_filename = "my_log.json"  # Set the filename for the log file
        log_writer.write_log(log_filename)  # Write the log to the specified file

        log_path = os.path.join(self.test_dir, run_name, log_filename)  # Construct the full path to the log file
        with open(log_path, "r") as f:
            log = json.load(f)  # Load the log content from the file

        log_expected = dict(my_value=[[2, 0.5]])  # Define the expected log content
        self.assertEqual(log, log_expected)  # Assert that the actual log content matches the expected content

    def test_write_text(self) -> None:
        run_name = "my_run"  # Set a name for the current run
        filename = "my_text.txt"  # Set the filename for the text file
        text = "my log text"  # Set the text content
        log_writer = LogWriter(run_name=run_name, log_dir=self.test_dir)  # Initialize LogWriter with the run name and log directory
        log_writer.write_text(text, filename)  # Write the text content to the specified file
        log_path = os.path.join(self.test_dir, run_name, filename)  # Construct the full path to the text file
        with open(log_path, "r") as f:
            file_text = f.read()  # Load the text content from the file
        self.assertEqual(text, file_text)  # Assert that the actual text content matches the expected content

    def test_write_config(self) -> None:
        run_name = "my_run"  # Set a name for the current run
        config = TempConfig(b="bar")  # Create an instance of TempConfig with a custom value for field 'b'
        log_writer = LogWriter(run_name=run_name, log_dir=self.test_dir)  # Initialize LogWriter with the run name and log directory
        log_writer.write_config(config)  # Write the configuration object to a JSON file
        log_path = os.path.join(self.test_dir, run_name, "config.json")  # Construct the full path to the JSON file
        with open(log_path, "r") as f:
            file_config = json.load(f)  # Load the configuration content from the file
        self.assertEqual(config._asdict(), file_config)  # Assert that the actual configuration content matches the expected content


if __name__ == "__main__":
    unittest.main()  # Run the test suite
