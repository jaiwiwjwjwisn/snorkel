from snorkel.types import Config, Float, Int
from typing import Optional

class ExponentialLRSchedulerConfig(Config):
    """Settings for Exponential decay learning rate scheduler."""

    gamma: Optional[Float] = 0.9


class StepLRSchedulerConfig(Config):
    """Settings for Step decay learning rate scheduler."""

    gamma: Optional[Float] = 0.9
    step_size: Int = 5


class LRSchedulerConfig(Config):
    """Settings common to all LRSchedulers.

    Parameters
    ----------
    warmup_steps : float
        The number of warmup_units over which to perform learning rate warmup (a linear
        increase from 0 to the specified lr)
    warmup_unit : str
        The unit to use when counting warmup (one of ["batches", "epochs"])
    warmup_percentage : float
        The percentage of the training procedure to warm up over (ignored if
        warmup_steps is non-zero)
    min_lr : float
        The minimum learning rate to use during training (the learning rate specified
        by a learning rate scheduler will be rounded up to this if it is lower)
    exponential_config : ExponentialLRSchedulerConfig
        Extra settings for the ExponentialLRScheduler
    step_config : StepLRSchedulerConfig
        Extra settings for the StepLRScheduler
    """

    warmup_steps: Float = 0.0  # warm up steps
    warmup_unit: str = "batches"  # [epochs, batches]
    warmup_percentage: Float = 0.0  # warm up percentage
    min_lr: Float = 0.0  # minimum learning rate
    exponential_config: ExponentialLRSchedulerConfig = ExponentialLRSchedulerConfig()  # type:ignore
    step_config: StepLRSchedulerConfig = StepLRSchedulerConfig()  # type:ignore

    def __post_init__(self):
        if self.warmup_unit not in ["batches", "epochs"]:
            raise ValueError(
                f"Invalid value for warmup_unit: {self.warmup_unit}. "
                f"Must be one of ['batches', 'epochs']."
            )
