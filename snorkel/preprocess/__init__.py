"""Preprocessors for LFs, TFs, and SFs."""

from .core import BasePreprocessor, LambdaPreprocessor


def preprocessor(func=None, /, **kwargs):
    """Create a preprocessor that applies the given function or lambda.

    Example:
        To create a preprocessor that uppercases all input strings:

        >>> from preprocessors import preprocessor
        >>> preprocessor(lambda x: x.upper())("hello")
        'HELLO'
    """
    if func is None:
        return lambda x: x
    return LambdaPreprocessor(func, **kwargs)


class Preprocessor(BasePreprocessor):
    """A preprocessor that applies a function or lambda to its input.

    Example:
        To create a preprocessor that applies a custom function to its input:

        >>> from preprocessors import Preprocessor, identity
        >>> preproc = Preprocessor(identity)
        >>> preproc("hello")
        'hello'
    """

    def __init__(self, func, /, **kwargs):
        super().__init__(**kwargs)
        self.func = func

    def __call__(self, obj):
        return self.func(obj)


