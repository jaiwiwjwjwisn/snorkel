import shutil
import tempfile
import unittest
from typing import List

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.optim as optim

from snorkel.labeling.model import LabelModel
from snorkel.labeling.model.label_model import TrainConfig

