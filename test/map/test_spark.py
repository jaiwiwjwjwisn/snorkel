import unittest
from typing import Optional

import pytest
from pyspark.sql import Row

from snorkel.map import Mapper, lambda_mapper
from snorkel.map.spark import make_spark_mapper
from snorkel.types import DataPoint, FieldMap

# A class that inherits from Mapper and defines a constructor with custom
# arguments. The `run` method is expected to return a FieldMap with the
# transformed fields.
class SplitWordsMapper(Mapper):
    def __init__(
        self, name: str, text_field: str, lower_field: str, words_field: str
    ) -> None:
        super().__init__(
            name, dict(text=text_field), dict(lower=lower_field, words=words_field)
        )

