"""PyTorch-based multi-task learning framework for discriminative modeling."""

from .data import (
    DictDataLoader,
    DictDataset,
)
from .loss import cross_entropy_with_probs
from .multitask_classifier import MultitaskClassifier
from .task import Operation, Task

from .training.loggers import (
    Checkpointer,
    CheckpointerConfig,
    LogManager,
    LogManagerConfig,
    LogWriter,
    LogWriterConfig,
    TensorBoardWriter,
)
from .training.trainer import Trainer
