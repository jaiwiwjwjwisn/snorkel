import logging
from typing import Any, Optional

from snorkel.classification.multitask_classifier import MultitaskClassifier
from snorkel.types import Config

from .checkpointer import Checkpointer  # Checkpointer for current model
from .log_writer import LogWriter  # LogWriter for current run logs

