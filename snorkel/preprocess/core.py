from snorkel.map import BaseMapper, LambdaMapper, Mapper, lambda_mapper

class BasePreprocessor(BaseMapper):
    """Base class for preprocessors.

    A preprocessor is a data point to data point mapping in a labeling
    pipeline. This allows Snorkel operations (e.g. LFs) to share common
    preprocessing steps that make it easier to express labeling logic.
    """

    def __init_subclass__(cls, **kwargs):
        """Raise an error when BasePreprocessor is subclassed.

        This class is intended to be used as a mixin or as a base class for
        concrete preprocessor classes.
        """
        if cls is not BasePreprocessor:
            raise TypeError("BasePreprocessor cannot be subclassed directly.")

class Preprocessor(BasePreprocessor, Mapper):
    """Base class for preprocessors.

    See ``snorkel.map.core.Mapper`` for details.

    Attributes
    ----------
    process : Callable[[Any], Any]
        The function to apply to each data point.
    """

    def __init__(self, process: Callable[[Any], Any]):
        """Initialize the Preprocessor.

        Parameters
        ----------
        process : Callable[[Any], Any]
            The function to apply to each data point.
        """
        self.process = process

    def process_map(self, data: Iterable[Any]) -> Iterable[Any]:
        """Apply the process function to each data point.

        Parameters
        ----------
        data : Iterable[Any]
            The data points to process.

        Returns
        -------
        Iterable[Any]
            The processed data points.
        """
        return map(self.process, data)

class LambdaPreprocessor(LambdaMapper):
    """Convenience class for defining preprocessors from functions.

    See ``snorkel.map.core.LambdaMapper`` for details.
    """

class preprocessor(lambda_mapper):
    """Decorate functions to create preprocessors.

    See ``snorkel.map.core.lambda_mapper`` for details.

    Example
    -------
    >>> @preprocessor()
    ... def combine_text_preprocessor(x):
    ...     x["article"] = f"{x['title']} {x['body']}"
    ...     return x
    >>> from snorkel.preprocess.nlp import SpacyPreprocessor
    >>> spacy_preprocessor = SpacyPreprocessor("article", "article_parsed")

    We can now add our preprocessors to an LF.

    >>> preprocessors = [combine_text_preprocessor, spacy_preprocessor]
    >>> from snorkel.labeling.lf import labeling_function
    >>> @labeling_function(pre=preprocessors)
    ... def article_mentions_person(x):
    ...     for ent in x["article_parsed"].ents:
    ...         if ent.label_ == "PERSON":
    ...             return ABSTAIN
    ...     return NEGATIVE
    """

    pass
