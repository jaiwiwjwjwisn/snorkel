import logging
import random
from collections import Counter, defaultdict
from itertools import chain
from typing import Any, DefaultDict, Dict, List, NamedTuple, Optional, Set, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from munkres import Munkres  # type: ignore
from tqdm import trange

from snorkel.labeling.analysis import LFAnalysis
from snorkel.labeling.model.base_labeler import BaseLabeler
from snorkel.labeling.model.graph_utils import get_clique_tree
from snorkel.labeling.model.logger import Logger
from snorkel.types import Config
from snorkel.utils.config_utils import merge_config
from snorkel.utils.lr_schedulers import LRSchedulerConfig
from snorkel.utils.optimizers import OptimizerConfig

Metrics = Dict[str, float]

