"""Generic model analysis utilities shared across Snorkel."""

# Instead of importing all functions from modules and using `noqa: F401` to suppress unused import warnings,
# we can import only the necessary functions and use aliasing for better readability.
from .error_analysis import get_label_buckets, get_label_instances as get_instances
from .metrics import metric_score as score
from .scorer import Scorer

# Add docstrings to the module and functions for better documentation
"""
This module provides generic model analysis utilities shared across Snorkel.
"""

def get_instances_docstring():
    """
    Get label instances.

    Returns:
        A tuple of (label_buckets, label_instances)
    """
    pass

get_label_buckets.__doc__ = get_instances_docstring()
get_label_instances.__doc__ = get_instances_docstring()

def score_docstring():
    """
    Compute a metric score for the given model.

    Args:
        model (Model): The model to compute the score for.

    Returns:
        float: The metric score.
    """
    pass

score.__doc__ = score_docstring()

Scorer.__doc__ = """
A scorer class for computing metric scores.
"""
