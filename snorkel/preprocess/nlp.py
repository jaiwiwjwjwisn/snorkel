from typing import List, Optional

import spacy
from snorkel.types import FieldMap, HashingFunction
from spacy.lang.en.stop_words import STOP_WORDS

class BasePreprocessor(spacy.Language):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.stop_words = set(STOP_WORDS)

    def remove_stop_words(self, doc: spacy.Doc) -> spacy.Doc:
        """Remove stop words from the document."""
        tokens = [token for token in doc if token.text.lower() not in self.stop_words]
        doc.sents = [spacy.tokens.Doc(self.vocab, words=sent) for sent in [tokens[i:j] for i, j in zip([0] + [pos for pos, token in enumerate(doc) if token.dep_ == "ROOT"], [doc.vocab.strings[token.text] for token in doc] + [None])]]
        return doc

class Preprocessor(BasePreprocessor, Preprocessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def preprocess(self, doc: spacy.Doc) -> spacy.Doc:
        """Preprocess the document by removing stop words."""
        doc = self.remove_stop_words(doc)
        return doc
