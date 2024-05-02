from typing import Tuple

from snorkel.types import Config


class SGDOptimizerConfig(Config):
    """Configuration settings for the Stochastic Gradient Descent (SGD) optimizer.

    This class defines the configuration settings for the SGD optimizer,
    including the default value for the momentum parameter.

    Attributes:
        momentum (float): The momentum parameter for the SGD optimizer.
                          The default value is 0.9.
    """

    momentum: float = 0.9


class AdamOptimizerConfig(Config):
    """Configuration settings for the Adam optimizer.

    This class defines the configuration settings for the Adam optimizer,
    including the default values for the amsgrad parameter and the betas tuple.

    Attributes:
        amsgrad (bool): The amsgrad parameter for the Adam optimizer.
                       The default value is False.
        betas (Tuple[float, float]): The betas tuple for the Adam optimizer.
                                      The default values are (0.9, 0.999).
    """

    amsgrad: bool = False
    betas: Tuple[float, float] = (0.9, 0.999)


class AdamaxOptimizerConfig(Config):
    """Configuration settings for the Adamax optimizer.

    This class defines the configuration settings for the Adamax optimizer,
    including the default values for the betas tuple and the eps parameter.

    Attributes:
        betas (Tuple[float, float]): The betas tuple for the Adamax optimizer.
                                      The default values are (0.9, 0.999).
        eps (float): The eps parameter for the Adamax optimizer.
                     The default value is 1e-8.
    """

    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8


class OptimizerConfig(Config):
    """Configuration settings common to all optimizers.

    This class defines the configuration settings that are common to all optimizers,
    including the default configurations for SGD, Adam, and Adamax optimizers.

    Attributes:
        sgd_config (SGDOptimizerConfig): The configuration settings for the SGD optimizer.
                                          The default value is an instance of SGDOptimizerConfig.
        adam_config (AdamOptimizerConfig): The configuration settings for the Adam optimizer.
                                            The default value is an instance of AdamOptimizerConfig.
        adamax_config (AdamaxOptimizerConfig): The configuration settings for the Adamax optimizer.
                                                 The default value is an instance of AdamaxOptimizerConfig.
    """

    sgd_config: SGDOptimizerConfig = SGDOptimizerConfig()  # type:ignore
    adam_config: AdamOptimizerConfig = AdamOptimizerConfig()  # type:ignore
    adamax_config: AdamaxOptimizerConfig = AdamaxOptimizerConfig()  # type:ignore
