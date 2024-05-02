import inspect  # Importing the 'inspect' module for getting function parameters
import pickle  # Importing the 'pickle' module for deepcopy
from collections.abc import Hashable  # Importing Hashable from 'collections.abc'
from types import SimpleNamespace  # Importing SimpleNamespace
from typing import Any, Callable, Dict, List, Mapping, Optional  # Importing necessary types

import numpy as np  # Importing 'numpy'
import pandas as pd  # Importing 'pandas'

from snorkel.types import DataPoint, FieldMap, HashingFunction  # Importing DataPoint, FieldMap, and HashingFunction from 'snorkel.types'

# Defining the MapFunction type as Callable[[DataPoint], Optional[DataPoint]]
MapFunction = Callable[[DataPoint], Optional[DataPoint]]

def get_parameters(f: Callable[..., Any], allow_args: bool = False, allow_kwargs: bool = False) -> List[str]:
    """Get names of function parameters.

    This function checks if the function has *args or **kwargs in its signature
    and raises a ValueError if it does. If not, it returns the names of the
    function parameters.

    Parameters
    ----------
    f : Callable[..., Any]
        The function to get the parameters from
    allow_args : bool
        Allow *args in the function signature (default: False)
    allow_kwargs : bool
        Allow **kwargs in the function signature (default: False)

    Returns
    -------
    List[str]
        The names of the function parameters
    """

def is_hashable(obj: Any) -> bool:
    """Test if object is hashable via duck typing.

    This function checks if the object has a __hash__ method and returns True
    if it does, otherwise it raises an exception.

    Parameters
    ----------
    obj : Any
        The object to test for hashability

    Returns
    -------
    bool
        True if the object is hashable, False otherwise
    """

def get_hashable(obj: Any) -> Hashable:
    """Get a hashable version of a potentially unhashable object.

    This function creates a hashable representation of the object values
    using a frozenset. For dictionaries or pd.Series, it recursively calls
    itself on the values. For NumPy arrays, it hashes the byte representation
    of the data array.

    Parameters
    ----------
    obj : Any
        The object to get the hashable version of

    Returns
    -------
    Hashable
        The hashable representation of the object values

    Raises
    ------
    ValueError
        If the object has no hashing proxy
    """

class BaseMapper:
    """Base class for 'Mapper' and 'LambdaMapper'.

    This class implements nesting, memoization, and deep copy functionality.
    It is used primarily for type checking.

    Parameters
    ----------
    name : str
        The name of the mapper
    pre : List[BaseMapper]
        Mappers to run before this mapper is executed
    memoize : bool
        Memoize mapper outputs (default: False)
    memoize_key : Optional[HashingFunction]
        Hashing function to handle the memoization (default: snorkel.map.core.get_hashable)

    Raises
    ------
    NotImplementedError
        Subclasses need to implement '_generate_mapped_data_point'

    Attributes
    ----------
    memoize : bool
        Memoize mapper outputs
    """

    def __init__(
        self,
        name: str,
        pre: List[BaseMapper],
        memoize: bool,
        memoize_key: Optional[HashingFunction] = None,
    ) -> None:
        """Initialize the BaseMapper class.

        Parameters
        ----------
        name : str
            The name of the mapper
        pre : List[BaseMapper]
            Mappers to run before this mapper is executed
        memoize : bool
            Memoize mapper outputs (default: False)
        memoize_key : Optional[HashingFunction]
            Hashing function to handle the memoization (default: snorkel.map.core.get_hashable)
        """

    def reset_cache(self) -> None:
        """Reset the memoization cache."""

    def _generate_mapped_data_point(self, x: DataPoint) -> Optional[DataPoint]:
        """Generate the mapped data point.

        This method should be implemented by subclasses.

        Parameters
        ----------
        x : DataPoint
            The input data point

        Returns
        -------
        Optional[DataPoint]
            The mapped data point or None if the input data point should be dropped
        """

    def __call__(self, x: DataPoint) -> Optional[DataPoint]:
        """Run mapping function on input data point.

        This method checks if the mapper should memoize the output,
        deep copies the data point, runs the pre mappers, generates
        the mapped data point, and updates the cache.

        Parameters
        ----------
        x : DataPoint
            The input data point

        Returns
        -------
        Optional[DataPoint]
            The mapped data point or None if the input data point should be dropped
        """

    def __repr__(self) -> str:
        """Return a string representation of the mapper."""

class Mapper(BaseMapper):
    """Base class for any data point to data point mapping in the pipeline.

    This class maps data points to new data points by transforming, adding
    additional information, or decomposing into primitives.

    Parameters
    ----------
    name : str
        The name of the mapper
    field_names : Optional[Mapping[str, str]]
        A map from attribute names of the incoming data points
        to the input argument names of the 'run' method (default: None)
    mapped_field_names : Optional[Mapping[str, str]]
        A map from output keys of the 'run' method to attribute
        names of the output data points (default: None)
    pre : Optional[List[BaseMapper]]
        Mappers to run before this mapper is executed (default: None)
    memoize : bool
        Memoize mapper outputs (default: False)
    memoize_key : Optional[HashingFunction]
        Hashing function to handle the memoization (default: snorkel.map.core.get_hashable)

    Raises
    ------
    NotImplementedError
        Subclasses must implement the 'run' method

    Attributes
    ----------
    field_names : Optional[Mapping[str, str]]
        A map from attribute names of the incoming data points
        to the input argument names of the 'run' method
    mapped_field_names : Optional[Mapping[str, str]]
        A map from output keys of the 'run' method to attribute
        names of the output data points
    memoize : bool
        Memoize mapper outputs
    """

    def __init__(
        self,
        name: str,
        field_names: Optional[Mapping[str, str]] = None,
        mapped_field_names: Optional[Mapping[str, str]] = None,
        pre: Optional[List[BaseMapper]] = None,
        memoize: bool = False,
        memoize_key: Optional[HashingFunction] = None,
    ) -> None:
        """Initialize the Mapper class.

        Parameters
        ----------
        name : str
            The name of the mapper
        field_names : Optional[Mapping[str, str]]
            A map from attribute names of the incoming data points
            to the input argument names of the 'run' method (default: None)
        mapped_field_names : Optional[Mapping[str, str]]
            A map from output keys of the 'run' method to attribute
            names of the output data points (default: None)
        pre : Optional[List[BaseMapper]]
            Mappers to run before this mapper is executed (default: None)
        memoize : bool
            Memoize mapper outputs (default: False)
        memoize_key : Optional[HashingFunction]
            Hashing function to handle the memoization (default: snorkel.map.core.get_hashable)
        """

    def run(self, **kwargs: Any) -> Optional[FieldMap]:
        """Run the mapping operation using the input fields.

        This method should be implemented by subclasses.

        Parameters
        ----------
        **kwargs : Any
            The input fields

        Returns
        -------
        Optional[FieldMap]
            The output fields
        """

    def _update_fields(self, x: DataPoint, mapped_fields: FieldMap) -> DataPoint:
        """Update the fields of the data point.

        Parameters
        ----------
        x : DataPoint
            The input data point
        mapped_fields : FieldMap
            The output fields

        Returns
        -------
        DataPoint
            The updated data point
        """

    def _generate_mapped_data_point(self, x: DataPoint) -> Optional[DataPoint]:
        """Generate the mapped data point.

        This method extracts the input fields from the data point,
        calls the 'run' method, updates the fields of the data point,
        and returns the mapped data point.

        Parameters
        ----------
        x : DataPoint
            The input data point

        Returns
        -------
        Optional[DataPoint]
            The mapped data point or None if the input data point should be dropped
        """

class LambdaMapper(BaseMapper):
    """Define a mapper from a function.

    This class is a convenience class for mappers that execute a simple
    function with no set up. The function should map from an input data point
    to a new data point directly, unlike 'Mapper.run'.

    Parameters
    ----------
    name : str
        The name of the mapper
    f : MapFunction
        The function executing the mapping operation
    pre : Optional[List[BaseMapper]]
        Mappers to run before this mapper is executed (default: None)
    memoize : bool
        Memoize mapper outputs (default: False)
    memoize_key : Optional[HashingFunction]
        Hashing function to handle the memoization (default: snorkel.map.core.get_hashable)
    """

    def __init__(
        self,
        name: str,
        f: MapFunction,
        pre: Optional[List[BaseMapper]] = None,
        memoize: bool = False,
        memoize_key: Optional[HashingFunction] = None,
    ) -> None:
        """Initialize the LambdaMapper class.

        Parameters
        ----------
        name : str
            The name of the mapper
        f : MapFunction
            The function executing the mapping operation
        pre : Optional[List[BaseMapper]]
            Mappers to run before this mapper is executed (default: None)
        memoize : bool
            Memoize mapper outputs (default: False)
        memoize_key : Optional[HashingFunction]
            Hashing function to handle the memoization (default: snorkel.map.core.get_hashable)
        """

    def _generate_mapped_data_point(self, x: DataPoint) -> Optional[DataPoint]:
        """Generate the mapped data point.

        This method calls the function with the input data point
        and returns the output data point.

        Parameters
        ----------
        x : DataPoint
            The input data point

        Returns
        -------
        Optional[DataPoint]
            The mapped data point or None if the input data point should be dropped
        """

class lambda_mapper:
    """Decorate a function to define a LambdaMapper object.

    This class is a decorator for creating a LambdaMapper object from a function.

    Parameters
    ----------
    name : Optional[str]
        The name of the mapper (default: None)
    pre : Optional[List[BaseMapper]]
        Mappers to run before this mapper is executed (default: None)
    memoize : bool
        Memoize mapper outputs (default: False)
    memoize_key : Optional[HashingFunction]
        Hashing function to handle the memoization (default: snorkel.map.core.get_hashable)

    Attributes
    ----------
    memoize : bool
        Memoize mapper outputs
    """

    def __init__(
        self,
        name: Optional[str] = None,
        pre: Optional[List[BaseMapper]] = None,
        memoize: bool = False,
        memoize_key: Optional[HashingFunction] = None,
    ) -> None:
        """Initialize the lambda_mapper class.

        Parameters
        ----------
        name : Optional[str]
            The name of the mapper (default: None)
        pre : Optional[List[BaseMapper]]
            Mappers to run before this mapper is executed (default: None)
        memoize : bool
            Memoize mapper outputs (default: False)
        memoize_key : Optional[HashingFunction]
            Hashing function to handle the memoization (default: snorkel.map.core.get_hashable)
        """

    def __call__(self, f: MapFunction) -> LambdaMapper:
        """Create a LambdaMapper object from a function.

        This method creates a LambdaMapper object from the input function.

        Parameters
        ----------
        f : MapFunction
            The function executing the mapping operation

        Returns
        -------
        LambdaMapper
            The LambdaMapper object
        """
