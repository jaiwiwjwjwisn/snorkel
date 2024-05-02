import unittest
from types import SimpleNamespace

import pytest
from pyspark.sql import Row

from snorkel.labeling.lf.nlp import NLPLabelingFunction
from snorkel.labeling.lf.nlp_spark import (
    SparkNLPLabelingFunction,
    spark_nlp_labeling_function,
)
from snorkel.types import DataPoint

def has_person_mention(x: DataPoint) -> int:
    return 0 if any(ent.label_ == "PERSON" for ent in x.doc.ents) else -1

@pytest.mark.spark
class TestNLPLabelingFunction(unittest.TestCase):
    def _run_lf(self, lf: SparkNLPLabelingFunction, text: str) -> None:
        x = Row(num=8, text=text)
        self.assertEqual(lf(x), has_person_mention(SimpleNamespace(num=8, text=text)))

    def test_nlp_labeling_function(self) -> None:
        lf = SparkNLPLabelingFunction(name="my_lf", f=has_person_mention)
        self._run_lf(lf, "The movie is really great!")
        self._run_lf(lf, "Jane Doe acted well.")

    @pytest.mark.parametrize(
        "text",
        [
            "The movie is really great!",
            "Jane Doe acted well.",
        ],
    )
    def test_nlp_labeling_function_decorator(self, text: str) -> None:
        @spark_nlp_labeling_function()
        def has_person_mention_decorator(x: DataPoint) -> int:
            return has_person_mention(x)

        self.assertIsInstance(has_person_mention_decorator, SparkNLPLabelingFunction)
        self.assertEqual(has_person_mention_decorator.name, "has_person_mention")
        self._run_lf(has_person_mention_decorator, text)

    def test_spark_nlp_labeling_function_with_nlp_labeling_function(self) -> None:
        lf = NLPLabelingFunction(name="my_lf", f=has_person_mention)
        lf_spark = SparkNLPLabelingFunction(name="my_lf_spark", f=has_person_mention)
        self.assertEqual(lf(SimpleNamespace(num=8, text="Jane Doe acted well.")), 0)

        self._run_lf(lf_spark, "The movie is really great!")
        self._run_lf(lf_spark, "Jane Doe acted well.")
