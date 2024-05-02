from typing import Any, Callable, List, Mapping, Optional, TypeVar

from snorkel.labeling.lf import LabelingFunction
from snorkel.preprocess import BasePreprocessor

T = TypeVar('T')


class SlicingFunction(LabelingFunction):
    """Base class for slicing functions.

    See ``snorkel.labeling.lf.LabelingFunction`` for details.
    """

    def __init__(self, name: str, f: Callable[[T], bool], resources: Optional[Mapping[str, Any]] = None, pre: Optional[List[BasePreprocessor]] = None):
        super().__init__(name=name, f=f, resources=resources, pre=pre)

    def __repr__(self):
        return f'SlicingFunction {self.name}, Preprocessors: {self.pre}'

