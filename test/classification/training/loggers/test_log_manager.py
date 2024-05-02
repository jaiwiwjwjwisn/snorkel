import pytest
import shutil
from unittest.mock import Mock

import numpy as np
import torch

from snorkel.classification.multitask_classifier import MultitaskClassifier
from snorkel.classification.training.loggers import Checkpointer, LogManager


@pytest.fixture
def tmp_path(request):
    return request.config.rootdir / "tmp"


@pytest.fixture
def checkpointer_factory(tmp_path):
    class MockCheckpointer:
        def __init__(self, **kwargs):
            self.checkpoint_dir = tmp_path
            self.checkpoint_factor = 2
            self.counter_unit = kwargs.get("counter_unit")
            self.evaluation_freq = kwargs.get("evaluation_freq")
            self.load_best_model = Mock()
            self.save_best_model = Mock()

        def save(self, model, epoch, **kwargs):
            pass

        def load_best(self):
            return self.load_best_model()

    return Mock(return_value=MockCheckpointer())


@pytest.fixture
def records(tmp_path):
    return {
        "best_model": {"epoch": 0, "train_loss": 1.0, "val_loss": 0.5},
        "epoch_records": [
            {"epoch": 0, "train_loss": 1.0, "val_loss": 0.5},
            {"epoch": 1, "train_loss": 0.5, "val_loss": 0.25},
            {"epoch": 2, "train_loss": 0.25, "val_loss": 0.125},
        ],
    }


@pytest.mark.unit
@pytest.mark.parametrize(
    "counter_unit,n_batches_per_epoch,evaluation_freq,records,expected_total",
    [
        ("points", 10, 10, {}, 25),
        ("batches", 5, 2, {}, 4),
        ("epochs", 2, 1, {"epoch_records": [{"epoch": 0, "train_loss": 1.0, "val_loss": 0.5}]}, 2),
    ],
)
def test_log_manager(
    counter_unit, n_batches_per_epoch, evaluation_freq, records, expected_total, checkpointer_factory, tmp_path
):
    checkpointer = checkpointer_factory(counter_unit=counter_unit, evaluation_freq=evaluation_freq)
    log_manager = LogManager(
        n_batches_per_epoch=n_batches_per_epoch,
        checkpointer_factory=checkpointer_factory,
        **records,
        counter_unit=counter_unit,
        evaluation_freq=evaluation_freq,
    )

    log_manager.update(5)
    assert not log_manager.trigger_evaluation()
    assert not log_manager.trigger_checkpointing()

    log_manager.update(5)
    assert log_manager.trigger_evaluation()
    assert not log_manager.trigger_checkpointing()

    log_manager.update(10)
    assert log_manager.trigger_evaluation()
    assert log_manager.trigger_checkpointing()

    log_manager.update(5)
    assert not log_manager.trigger_evaluation()
    assert not log_manager.trigger_checkpointing()

    assert log_manager.point_count == 5
    assert log_manager.point_total == expected_total
    assert log_manager.batch_total == log_manager.n_batches_per_epoch * log_manager.epoch_total
    assert log_manager.epoch_total == len(records.get("epoch_records", []))


@pytest.mark.unit
def test_load_on_cleanup(checkpointer_factory, tmp_path):
    classifier = MultitaskClassifier([])
    checkpointer = checkpointer_factory(checkpoint_dir=tmp_path)
    log_manager = LogManager(
        n_batches_per_epoch=2,
        checkpointer_factory=checkpointer_factory,
        log_writer=None,
        checkpointer=checkpointer,
    )

    best_classifier = log_manager.cleanup(classifier)
    assert best_classifier is classifier


@pytest.mark.unit
def test_bad_unit():
    with pytest.raises(ValueError, match="Unrecognized counter_unit"):
        LogManager(n_batches_per_epoch=2, counter_unit="macaroni")


import pytest

from snorkel.classification.training.loggers import Checkpointer


@pytest.fixture
def checkpointer_factory():
    class MockCheckpointer:
        def __init__(self, **kwargs):
            self.checkpoint_dir = None
            self.checkpoint_factor = 2
            self.counter_unit = kwargs.get("counter_unit")
            self.evaluation_freq = kwargs.get("evaluation_freq")
            self.load_best_model = Mock()
            self.save_best_model = Mock()

        def save(self, model, epoch, **kwargs):
            pass

        def load_best(self):
            return self.load_best_model()

    return Mock(return_value=MockCheckpointer())


@pytest.fixture
def checkpointer(checkpointer_factory):
    checkpointer = checkpointer_factory()
    checkpointer.save.return_value = None
    return checkpointer
