import unittest

import pandas as pd

from snorkel.slicing import slicing_function  # Import the slicing_function decorator
from snorkel.slicing.monitor import slice_dataframe  # Import the slice_dataframe function

# Define a sample dataset as a list of integers
DATA = [5, 10, 19, 22, 25]

@slicing_function()  # Decorate the function with the slicing_function decorator
def sf(x):
    """
    Define a slicing function that returns True if the 'num' column value is less than 20.
    """
    return x.num < 2
