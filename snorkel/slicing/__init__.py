"""
Module: slicing
----------------

This module contains classes and functions for programmatic data set slicing,
including:

- SlicingFunctions for creating slices
- PandasSFApplier and SFApplier for applying slicing functions to pandas dataframes
- SliceCombinerModule for combining slices
- SliceAwareClassifier for training classifiers on slices
- slice_dataframe for monitoring slice data
- add_slice_labels and convert_to_slice_tasks for data preparation
"""

from .apply.core import PandasSFApplier, SFApplier
from .modules.slice_combiner import SliceCombinerModule
from .monitor import slice_dataframe
from .sf.core import SlicingFunction, slicing_function
from .sliceaware_classifier import SliceAwareClassifier
from .utils import add_slice_labels, convert_to_slice_tasks
