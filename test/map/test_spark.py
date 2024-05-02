import unittest
from typing import Optional

import pytest
from pyspark.sql import Row
from pyspark.sql.functions import lower

from snorkel.map import Mapper, field_map, lambda_mapper
from snorkel.map.spark import make_spark_mapper
from snorkel.types import DataPoint, FieldMap

