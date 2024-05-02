from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, ClassVar, Dict, List, Mapping, Optional

from snorkel.preprocess import BasePreprocessor
from snorkel.preprocess.nlp import EN_CORE_WEB_SM, SpacyPreprocessor
from snorkel.types import HashingFunction

from .core import LabelingFunction, labeling_function

class Language(Enum):
    EN = "en_core_web_sm"

class SpacyPreprocessorParameters:
    """Parameters needed to construct a SpacyPreprocessor."""

    text_field: str
    doc_field: str
    language: Language
    disable: Optional[List[str]] = None
    pre: List[BasePreprocessor] = field(default_factory=list)
    memoize: bool = True
    memoize_key: Optional[HashingFunction] = None
    gpu: bool = False

class SpacyPreprocessorConfig:
    """Tuple of SpacyPreprocessor and the parameters used to construct it."""

    nlp: ClassVar[SpacyPreprocessor]
    parameters: SpacyPreprocessorParameters

    @classmethod
    def create(cls, parameters: SpacyPreprocessorParameters) -> "SpacyPreprocessorConfig":
        """Create and store a SpacyPreprocessor instance."""
        nlp = SpacyPreprocessor(**parameters.__dict__)
        cls.nlp = nlp
        return cls(nlp=nlp, parameters=parameters)

    @property
    def preprocessors(self) -> List[BasePreprocessor]:
        """Return the list of preprocessors."""
        return [self.nlp] + self.parameters.pre

class BaseNLPLabelingFunction(LabelingFunction):
    """Base class for spaCy-based LFs."""

    _nlp_config: SpacyPreprocessorConfig

    @classmethod
    def create_preprocessor(
        cls, parameters: SpacyPreprocessorParameters
    ) -> SpacyPreprocessor:
        """Create a SpacyPreprocessor instance."""
        raise NotImplementedError

    @classmethod
    def create_or_check_preprocessor(
        cls,
        text_field: str,
        doc_field: str,
        language: Language,
        disable: Optional[List[str]] = None,
        pre: List[BasePreprocessor] = None,
        memoize: bool = True,
        memoize_key: Optional[HashingFunction] = None,
        gpu: bool = False,
    ) -> None:
        """Create or check a SpacyPreprocessor instance."""
        parameters = SpacyPreprocessorParameters(
            text_field=text_field,
            doc_field=doc_field,
            language=language,
            disable=disable,
            pre=pre or [],
            memoize=memoize,
            memoize_key=memoize_key,
            gpu=gpu,
        )
        if not hasattr(cls, "_nlp_config"):
            cls._nlp_config = SpacyPreprocessorConfig.create(parameters)
        elif parameters != cls._nlp_config.parameters:
            raise ValueError(
                f"{cls.__name__} already configured with different parameters: "
                f"{cls._nlp_config.parameters}"
            )

    def __init__(
        self,
        name: str,
        f: Callable[..., int],
        resources: Optional[Mapping[str, Any]] = None,
        pre: Optional[List[BasePreprocessor]] = None,
        text_field: str = "text",
        doc_field: str = "doc",
        language: Language = Language.EN,
        disable: Optional[List[str]] = None,
        memoize: bool = True,
        memoize_key: Optional[HashingFunction] = None,
        gpu: bool = False,
    ) -> None:
        self.create_or_check_preprocessor(
            text_field,
            doc_field,
            language,
            disable,
            pre,
            memoize,
            memoize_key,
            gpu,
        )
        super().__init__(name, f, resources=resources, pre=self._nlp_config.preprocessors)

class NLPLabelingFunction(BaseNLPLabelingFunction):
    r"""Special labeling function type for spaCy-based LFs.

    This class is a special version of ``LabelingFunction``. It
    has a ``SpacyPreprocessor`` integrated which shares a cache
    with all other ``NLPLabelingFunction`` instances. This makes
    it easy to define LFs that have a text input field and have
    logic written over spaCy ``Doc`` objects. Examples passed
    into an ``NLPLabelingFunction`` will have a new field which
    can be accessed which contains a spaCy ``Doc``. By default,
    this field is called ``doc``. A ``Doc`` object is
    a sequence of ``Token`` objects, which contain information
    on lemmatization, parts-of-speech, etc. ``Doc`` objects also
    contain fields like ``Doc.ents``, a list of named entities,
    and ``Doc.noun_chunks``, a list of noun phrases. For details
    of spaCy ``Doc`` objects and a full attribute listing,
    see https://spacy.io/api/doc.

    Simple ``NLPLabelingFunction``\s can be defined via a
    decorator. See ``nlp_labeling_function``.

    Parameters
    ----------
    name
        Name of the LF
    f
        Function that implements the core LF logic
    resources
        Labeling resources passed in to ``f`` via ``kwargs``
    pre
        Preprocessors to run before SpacyPreprocessor is executed
    text_field
        Name of data point text field to input
    doc_field
        Name of data point field to output parsed document to
    language
        spaCy model to load
        See https://spacy.io/usage/models#usage
    disable
        List of pipeline components to disable
        See https://spacy.io/usage/processing-pipelines#disabling
    memoize
        Memoize preprocessor outputs?
    memoize_key
        Hashing function to handle the memoization (default to snorkel.map.core.get_hashable)
    gpu
        Prefer Spacy GPU processing?

    Raises
    ------
    ValueError
        Calling incorrectly defined preprocessors

    Example
    -------
    >>> def f(x):
    ...     person_ents = [ent for ent in x.doc.ents if ent.label_ == "PERSON"]
    ...     return 0 if len(person_ents) > 0 else -1
    >>> has_person_mention = NLPLabelingFunction(name="has_person_mention", f=f)
    >>> has_person_mention
    NLPLabelingFunction has_person_mention, Preprocessors: [SpacyPreprocessor...]

    >>> from types import SimpleNamespace
    >>> x = SimpleNamespace(text="The movie was good.")
    >>> has_person_mention(x)
    -1

    Attributes
    ----------
    name
        See above
    """

    @classmethod
    def create_preprocessor(
        cls, parameters: SpacyPreprocessorParameters
    ) -> SpacyPreprocessor:
        return SpacyPreprocessor(**parameters.__dict
