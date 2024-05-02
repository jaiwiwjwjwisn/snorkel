"""
This script is used to manually test `snorkel.labeling.apply.lf_applier_spark.SparkLFApplier`.
"""

import logging
import sys
import numpy as np
from pyspark import SparkContext
from pyspark.errors import SparkException

from snorkel.labeling.apply.spark import SparkLFApplier
from snorkel.labeling.lf import labeling_function
from snorkel.types import DataPoint

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@labeling_function()
def f(x: DataPoint) -> int:
    return 1 if x > 42 else 0


@labeling_function(resources=dict(db=[3, 6, 9]))
def g(x: DataPoint, db: List[int]) -> int:
    return 1 if x in db else 0


DATA = [3, 43, 12, 9]
L_EXPECTED = np.array([[0, 1], [1, 0], [0, 0], [0, 1]])

def get_spark_context() -> SparkContext:
    try:
        sc = SparkContext()
        sc.addPyFile("snorkel-package.zip")
        return sc
    except SparkException as e:
        logger.error(f"Error creating SparkContext: {e}")
        sys.exit(1)


def build_lf_matrix() -> None:
    logger.info("Getting Spark context")
    sc = get_spark_context()

    logger.info("Creating RDD")
    rdd = sc.parallelize(DATA)

    logger.info("Applying LFs")
    lf_applier = SparkLFApplier([f, g])
    L = lf_applier.apply(rdd)

    np.testing.assert_equal(L.toarray(), L_EXPECTED)


if __name__ == "__main__":
    build_lf_matrix()
