import unittest

from snorkel.labeling.model.logger import Logger  # Import the Logger class from the snorkel library


class LoggerTest(unittest.TestCase):  # Define a test case class for the Logger class

    def test_basic(self):  # Define a basic test case for the Logger class
        metrics_dict = {"train/loss": 0.01}  # Define a dictionary of metrics to log
        logger = Logger(log_freq=1)  # Initialize a new Logger instance with a logging frequency of 1
        logger.log(metrics_dict)  # Log the metrics dictionary

        metrics_dict = {"train/message": "well done!"}  # Define a dictionary of metrics to log
        logger = Logger(log_freq=1)  # Initialize a new Logger instance with a logging frequency of 1
        logger.log(metrics_dict)  # Log the metrics dictionary

    def test_bad_metrics_dict(self):  # Define a test case for invalid metrics dictionaries
        bad_metrics_dict = {"task1/slice1/train/loss": 0.05}  # Define an invalid metrics dictionary
        logger = Logger(log_freq=1)  # Initialize a new Logger instance with a logging frequency of 1
        self.assertRaises(Exception, logger.log, bad_metrics_dict)  # Assert that logging the invalid metrics dictionary raises an exception

    def test_valid_metrics_dict(self):  # Define a test case for valid metrics dictionaries
        mtl_metrics_dict = {"task1/valid/loss": 0.05}  # Define a valid metrics dictionary for multi-task learning
        logger = Logger(log_freq=1)  # Initialize a new Logger instance with a logging frequency of 1
        logger.log(mtl_metrics_dict)  # Log the metrics dictionary


if __name__ == "__main__":  # Run the test suite if this script is executed directly
    unittest.main()  # Run the unittest.main() function to execute all test cases

