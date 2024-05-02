import unittest
from types import SimpleNamespace
from typing import List, Dict

import pandas as pd

from snorkel.augmentation import (
    ApplyOnePolicy,
    PandasTFApplier,
    RandomPolicy,
    TFApplier,
    transformation_function,
)
from snorkel.types import DataPoint

# Define a square transformation function that takes a DataPoint and squares its num attribute
@transformation_function()
def square(x: DataPoint) -> DataPoint:
    """
    Square the num attribute of the given DataPoint.

    :param x: The DataPoint to square the num attribute of.
    :return: The modified DataPoint with the squared num attribute.
    """
    x.num = x.num**2
    return x


# Define a square transformation function that returns None when num is 2
@transformation_function()
def square_returns_none(x: DataPoint) -> DataPoint:
    """
    Square the num attribute of the given DataPoint, but return None when num is 2.

    :param x: The DataPoint to square the num attribute of.
    :return: The modified DataPoint with the squared num attribute, or None when num is 2.
    """
    if x.num == 2:
        return None
    x.num = x.num**2
    return x


# Define a transformation function that modifies the DataPoint in place
@transformation_function()
def modify_in_place(x: DataPoint) -> DataPoint:
    """
    Modify the DataPoint in place by setting the value of my_key to 0.

    :param x: The DataPoint to modify.
    :return: The modified DataPoint.
    """
    x.d["my_key"] = 
