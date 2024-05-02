from snorkel.map.spark import make_spark_mapper  # noqa: F401

def make_spark_preprocessor():
    """
    A function to create a Spark mapper for use in Snorkel.
    
    This function uses the `make_spark_mapper` function from the `snorkel.map.spark` module to create a Spark mapper.
    
    Returns:
        A Spark mapper object.
    """
    return make_spark_mapper()
