"""Generic utilities for data point to data point operations.

This module contains various utility classes for performing operations on individual data points.
It includes classes for mapping data points using a variety of methods, including using a
callable function or a lambda function.

Classes:
- BaseMapper: The base class for all data point mappers in this module.
- LambdaMapper: A mapper that applies a lambda function to each data point.
- Mapper: A mapper that applies a callable function to each data point.
- lambda_mapper: A convenience function for creating a LambdaMapper instance.

Example usage:
    >>> from my_module import BaseMapper, LambdaMapper, Mapper, lambda_mapper
    >>> # Create a mapper that squares each data point
    >>> mapper = Mapper(lambda x: x ** 2)
    >>> # Use the mapper to transform a list of data points
    >>> data = [1, 2, 3, 4]
    >>> transformed_data = list(mapper(data))
    >>> print(transformed_data)
    [1, 4, 9, 16]
"""

from .core import BaseMapper, LambdaMapper, Mapper, lambda_mapper  # noqa: F401
