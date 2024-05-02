import unittest
from types import SimpleNamespace
from typing import List

from snorkel.preprocess import preprocessor  # Import preprocessor decorator
from snorkel.slicing import SFApplier, slicing_function  # Import SFApplier and slicing_function decorator
from snorkel.types import DataPoint  # Import DataPoint type


@preprocessor()  # Decorate square function with preprocessor
def square(x: DataPoint) -> DataPoint:
    """
    Preprocessor function that squares the 'num' attribute of the input DataPoint.

    Args:
    x (DataPoint): Input DataPoint.

    Returns:
    DataPoint: The same DataPoint with the 'num_squared' attribute set to the square of 'num'.
    """
    x.num_squared = x.num**2
    return x


class SquareHitTracker:
    def __init__(self):
        """
        Initialize the SquareHitTracker class.

        Attributes:
        n_hits (int): The number of times the __call__ method has been called.
        """
        self.n_hits = 0

    def __call__(self, x: float) -> float:
        """
        Callable method that squares the input float and increments the n_hits attribute.

        Args:
        x (float): Input float.

        Returns:
        float: The squared input float.
        """
        self.n_hits += 1
        return x**2


@slicing_function()  # Decorate f function with slicing_function
def f(x: DataPoint) -> int:
    """
    Slicing function that returns 1 if the 'num' attribute of the input DataPoint is greater than 42, 0 otherwise.

    Args:
    x (DataPoint): Input DataPoint.

    Returns:
    int: 1 if x.num > 42, 0 otherwise.
    """
    return 1 if x.num > 42 else 0


@slicing_function(pre=[square])  # Decorate fp function with slicing_function and preprocessor list
def fp(x: DataPoint) -> int:
    """
    Slicing function that returns 1 if the 'num_squared' attribute of the input DataPoint is greater than 42, 0 otherwise.

    Args:
    x (DataPoint): Input DataPoint.

    Returns:
    int: 1 if x.num_squared > 42, 0 otherwise.
    """
    return 1 if x.num_squared > 42 else 0


@slicing_function(resources=dict(db=[3, 6, 9]))  # Decorate g function with slicing_function and resources dictionary
def g(x: DataPoint, db: List[int]) -> int:
    """
    Slicing function that returns 1 if the 'num' attribute of the input DataPoint is in the 'db' list, 0 otherwise.

    Args:
    x (DataPoint): Input DataPoint.
    db (List[int]): List of integers to check for membership.

    Returns:
    int: 1 if x.num in db, 0 otherwise.
    """
    return 1 if x.num in db else 0


DATA = [3, 43, 12, 9, 3]  # Define test data
S_EXPECTED = {"f": [0, 1, 0, 0, 0], "g": [1, 0, 0, 1, 1]}  # Define expected output for S
S_PREPROCESS_EXPECTED = {"f": [0, 1, 0, 0, 0], "fp": [0, 1, 1, 1, 0]}  # Define expected output for S with preprocessing


class TestSFApplier(unittest.TestCase):  # Define test case for SFApplier
    def test_sf_applier(self) -> None:
        """
        Test the SFApplier with the f and g slicing functions.
        """
        data_points = [SimpleNamespace(num=num) for num in DATA]
        applier = SFApplier([f, g])
        S = applier.apply(data_points, progress_bar=False)
        self.assertEqual(S["f"].tolist(), S_EXPECTED["f"])
        self.assertEqual(S["g"].tolist(), S_EXPECTED["g"])
        S = applier.apply(data_points, progress_bar=True)
        self.assertEqual(S["f"].tolist(), S_EXPECTED["f"])
        self.assertEqual(S["g"].tolist(), S_EXPECTED["g"])

    def test_sf_applier_preprocessor(self) -> None:
        """
        Test the SFApplier with the f and fp slicing functions with preprocessing.
        """
        data_points = [SimpleNamespace(num=num) for num in DATA]
        applier = SFApplier([f, fp])
        S = applier.apply(data_points, progress_bar=False)
        self.assertEqual(S["f"].tolist(), S_PREPROCESS_EXPECTED["f"])
        self.assertEqual(S["fp"].tolist(), S_PREPROCESS_EXPECTED["fp"])
