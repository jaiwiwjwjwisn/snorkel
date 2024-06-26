import unittest
from typing import List, Tuple

import numpy as np
import pandas as pd
import pyspark
from pyspark.sql import Row, SparkSession
from pyspark.sql.functions import col, udf
from pyspark.sql.types import IntegerType

from snorkel.labeling import labeling_function
from snorkel.labeling.apply.spark import SparkLFApplier
from snorkel.preprocess import preprocessor
from snorkel.types import DataPoint


@preprocessor()
def square(x: Row) -> Row:
    return Row(num=x.num, num_squared=x.num**2)


@labeling_function()
def f(x: DataPoint) -> int:
    return -1 if x.num is None else 0 if x.num > 42 else -1


@labeling_function(pre=[square])
def fp(x: DataPoint) -> int:
    return -1 if x.num_squared is None else 0 if x.num_squared > 42 else -1


@labeling_function(resources=dict(db=[3, 6, 9]))
def g(x: DataPoint, db: List[int]) -> int:
    return -1 if x.num is None else 0 if x.num in db else -1


@labeling_function()
def f_bad(x: DataPoint) -> int:
    return -1 if x.mum is None else 0 if x.mum > 42 else -1


DATA = [3, 43, 12, 9, 3]
L_EXPECTED = np.array([[-1, 0], [0, -1], [-1, -1], [-1, 0], [-1, 0]])
L_EXPECTED_BAD = np.array([[-1, -1], [0, -1], [-1, -1], [-1, -1], [-1, -1]])
L_PREPROCESS_EXPECTED = np.array([[-1, -1], [0, 0], [-1, 0], [-1, 0], [-1, -1]])

TEXT_DATA = ["Jane", "Jane plays soccer.", "Jane plays soccer."]
L_TEXT_EXPECTED = np.array([[0, -1], [0, 0], [0, 0]])


class TestSparkApplier(unittest.TestCase):
    @pytest.mark.complex
    @pytest.mark.spark
    def setUp(self) -> None:
        self.spark = SparkSession.builder.appName("TestSparkApplier").getOrCreate()

    def tearDown(self) -> None:
        self.spark.stop()

    @pytest.mark.complex
    @pytest.mark.spark
    def test_lf_applier_spark(self) -> None:
        df = self.spark.createDataFrame(pd.DataFrame(dict(num=DATA)))
        applier = SparkLFApplier([f, g])
        L = applier.apply(df.rdd)
        np.testing.assert_equal(L, L_EXPECTED)

    @pytest.mark.complex
    @pytest.mark.spark
    def test_lf_applier_spark_fault(self) -> None:
        df = self.spark.createDataFrame(pd.DataFrame(dict(num=DATA)))
        applier = SparkLFApplier([f, f_bad])
        with self.assertRaises(Exception):
            applier.apply(df.rdd)
        applier.set_fault_tolerant(True)
        L = applier.apply(df.rdd)
        np.testing.assert_equal(L, L_EXPECTED_BAD)

    @pytest.mark.complex
    @pytest.mark.spark
    def test_lf_applier_spark_preprocessor(self) -> None:
        square_udf = udf(square, returnType=pyspark.sql.types.StructType([pyspark.sql.types.StructField("num", IntegerType(), False), pyspark.sql.types.StructField("num_squared", IntegerType(), False)]))
        df = self.spark.createDataFrame(pd.DataFrame(dict(num=DATA)))
        df = df.withColumn("square", square_udf(col("num")))
        applier = SparkLFApplier([f, fp])
        L = applier.apply(df.rdd)
        np.testing.assert_equal(L, L_PREPROCESS_EXPECTED)

    @pytest.mark.complex
    @pytest.mark.spark
    def test_lf_applier_spark_preprocessor_memoized(self) -> None:
        square_memoize_udf = udf(square, returnType=pyspark.sql.types.StructType([pyspark.sql.types.StructField("num", IntegerType(), False), pyspark.sql.types.StructField("num_squared", IntegerType(), False)]))
        df = self.spark.createDataFrame(pd.DataFrame(dict(num=DATA)))
        df = df.withColumn("square_memoize", square_memoize_udf(col("num")))

        @udf(returnType=IntegerType())
        def fp_memoized_udf(square_memoize: Tuple[int, int]) -> int:
            return 0 if square_memoize[1] > 42 else -1

        df = df.withColumn("fp_memoized", fp_memoized_udf(col("square_memoize")))
        applier = SparkLFApplier([f, fp_memoized])
        L = applier.apply(df.rdd)
        np.testing.assert_equal(L, L_PREPROCESS_EXPECTED)
