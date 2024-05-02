import unittest
import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch
from dask.distributed import Client

from snorkel.labeling import LFApplier, PandasLFApplier, DaskLFApplier, labeling_function
from snorkel.labeling.apply.core import ApplierMetadata
from snorkel.preprocess import preprocessor
from snorkel.preprocess.nlp import SpacyPreprocessor
from snorkel.types import DataPoint
from snorkel.utils import tmp_label_map

