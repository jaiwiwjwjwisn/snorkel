import logging
from functools import partial
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import ModuleDict

from snorkel.analysis import Scorer

Outputs = Mapping[str, List[torch.FloatTensor]]


class Operation:
    """A single operation (forward pass of a module) to execute in a Task.

    See ``Task`` for more detail on the usage and semantics of an Operation.

    Parameters
    ----------
    name : str
        The name of this operation (defaults to module_name since for most workflows,
        each module is only used once per forward pass)
    module_name : str
        The name of the module in the module pool that this operation uses
    inputs : Sequence[Union[str, Tuple[str, str]]]
        The inputs that the specified module expects, given as a list of names of
        previous operations (or optionally a tuple of the operation name and a key
        if the output of that module is a dict instead of a Tensor).
        Note that the original input to the model can be referred to as "_input_".

    Example
    -------
    >>> op1 = Operation(module_name="linear1", inputs=[("_input_", "features")])
    >>> op2 = Operation(module_name="linear2", inputs=["linear1"])
    >>> op_sequence = [op1, op2]

    Attributes
    ----------
    name : str
        See above
    module_name : str
        See above
    inputs : Sequence[Union[str, Tuple[str, str]]]
        See above
    """

    def __init__(
        self,
        module_name: str,
        inputs: Sequence[Union[str, Tuple[str, str]]],
        name: Optional[str] = None,
    ) -> None:
        self.name = name or module_name
        self.module_name = module_name
        self.inputs = inputs

    def __repr__(self) -> str:
        return self.repr_str()

    def repr_str(self) -> str:
        """A string representation of the Operation instance."""
        return (
            f"Operation(name={self.name}, "
            f"module_name={self.module_name}, "
            f"inputs={self.inputs})"
        )

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Performs a forward pass of the module specified in this Operation.

        Parameters
        ----------
        inputs : Dict[str, torch.Tensor]
            A dictionary of inputs to the module.

        Returns
        -------
        torch.Tensor
            The output of the module.
        """
        module = self.module_pool[self.module_name]
        if not isinstance(module, nn.Module):
            raise ValueError(f"Module '{self.module_name}' is not an instance of nn.Module.")
        return module(**inputs)


class Task:
    r"""A single task (a collection of modules and specified path through them).

    Parameters
    ----------
    name : str
        The name of the task
    module_pool : ModuleDict
        A ModuleDict mapping module names to the modules themselves
    op_sequence : Sequence[Operation]
        A list of ``Operation``\s to execute in order, defining the flow of information
        through the network for this task
    scorer : Scorer
        A ``Scorer`` with the desired metrics to calculate for this task
    loss_func : Optional[Callable[..., torch.Tensor]]
        A function that converts final logits into loss values.
        Defaults to F.cross_entropy() if none is provided.
        To use probalistic labels for training, use the Snorkel-defined method
        cross_entropy_with_probs() instead.
    output_func : Optional[Callable[..., torch.Tensor]]
        A function that converts final logits into 'outputs' (e.g. probabilities)
        Defaults to F.softmax(..., dim=1).

    Attributes
    ----------
    name : str
        See above
    module_pool : ModuleDict
        See above
    op_sequence : Sequence[Operation]
        See above
    scorer : Scorer
        See above
    loss_func : Optional[Callable[..., torch.Tensor]]
        See above
    output_func : Optional[Callable[..., torch.Tensor]]
        See above
    """

    def __init__(
        self,
        name: str,
        module_pool: ModuleDict,
        op_sequence: Sequence[Operation],
        scorer: Scorer = Scorer(metrics=["accuracy"]),
        loss_func: Optional[Callable[..., torch.Tensor]] = None,
        output_func: Optional[Callable[..., torch.Tensor]] = None,
    ) -> None:
        self.name = name
        self.module_pool = module_pool
        self.op_sequence = op_sequence
        self.scorer = scorer
        self.loss_func = loss_func or F.cross_entropy
        self.output_func = output_func or partial(F.softmax, dim=1)

        logging.info(f"Created task: {self.name}")

    def __call__(self, inputs: torch.Tensor) -> Outputs:
        """Performs a forward pass of the entire task.

        Parameters
        ----------
        inputs : torch.Tensor
            The input to the task.

        Returns
        -------
        Outputs
            A dictionary mapping operation names to lists of outputs.
        """
        inputs = {"_input_": inputs}
        outputs: Outputs = {}
        for op in self.op_sequence:
            op_outputs = op(inputs)
            if not isinstance(op_outputs, torch.Tensor):
                raise ValueError(f"Operation '{op.name}' did not return a Tensor.")
            if op.name not in outputs:
                outputs[op.name] = []
            outputs[op.name].append(op_outputs)
        return outputs

    def __repr__(self) -> str:
        cls_name = type(self).__name__
        return f"{cls_name}(name={self.name})"

