from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch

TensorCollection = Union[
    torch.Tensor, np.ndarray, dict, list, tuple, pd.DataFrame, pd.Series
]


def list_to_tensor(item_list: List[torch.Tensor]) -> torch.Tensor:
    """Convert a list of torch.Tensor into a single torch.Tensor.

    Args:
        item_list (List[torch.Tensor]): List of tensors to convert.

    Returns:
        torch.Tensor: Converted tensor.
    """
    if not item_list:
        raise ValueError("item_list cannot be empty.")

    # Convert single value tensor
    if all(item.dim() == 0 for item in item_list):
        item_tensor = torch.stack(item_list, dim=0)
    # Convert 2 or more-D tensor with the same shape
    elif all(
        (item.size() == item_list[0].size()) and (len(item.size()) != 1) for item in item_list
    ):
        item_tensor = torch.stack(item_list, dim=0)
    # Convert reshape to 1-D tensor and then convert
    else:
        item_tensor, _ = pad_batch([item.view(-1) for item in item_list])

    return item_tensor


def pad_batch(
    batch: List[torch.Tensor],
    max_len: int = 0,
    pad_value: int = 0,
    left_padded: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert the batch into a padded tensor and mask tensor.

    Args:
        batch (List[torch.Tensor]): The data for padding
        max_len (int, optional): Max length of sequence of padding. Defaults to 0.
        pad_value (int, optional): The value to use for padding. Defaults to 0.
        left_padded (bool, optional): If True, pad on the left, otherwise on the right. Defaults to False.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The padded matrix and correspoing mask matrix.
    """
    if not batch:
        raise ValueError("batch cannot be empty.")

    batch_size = len(batch)
    max_seq_len = int(np.max([len(item) for item in batch]))  # type: ignore

    if max_len > 0 and max_len < max_seq_len:
        max_seq_len = max_len

    padded_batch = batch[0].new_full((batch_size, max_seq_len), pad_value)

    for i, item in enumerate(batch):
        length = min(len(item), max_seq_len)  # type: ignore
        if left_padded:
            padded_batch[i, -length:] = item[-length:]
        else:
            padded_batch[i, :length] = item[:length]

    mask_batch = torch.eq(padded_batch.clone().detach(), pad_value).type_as(
        padded_batch
    )

    return padded_batch, mask_batch


def move_to_device(
    obj: TensorCollection, device: Optional[int] = -1
) -> TensorCollection:  # pragma: no cover
    """Recursively move torch.Tensors to a given CUDA device.

    Given a structure (possibly) containing Tensors on the CPU, move all the Tensors
    to the specified GPU (or do nothing, if they should beon the CPU).

    Args:
        obj (TensorCollection): Tensor or collection of Tensors to move
        device (Optional[int], optional): Device to move Tensors to. Defaults to -1.

    Returns:
        TensorCollection: Converted collection of tensors.
    """
    if device is None or device < 0 or not torch.cuda.is_available():
        return obj
    elif isinstance(obj, torch.Tensor):
        return obj.cuda(device)  # type: ignore
    elif isinstance(obj, (dict, list, tuple)):
        return type(obj)(move_to_device(v, device) for v in obj)
    elif isinstance(obj, (np.ndarray, pd.DataFrame, pd.Series)):
        return obj
    else:
        return obj


def collect_flow_outputs_by_suffix(
    output_dict: Dict[str, torch.Tensor], suffix: str
) -> List[torch.Tensor]:
    """Return output_dict outputs specified by suffix, ordered by sorted flow_name.

    Args:
        output_dict (Dict[str, torch.Tensor]): Output dictionary.
        suffix (str): Suffix to filter by.

    Returns:
        List[torch.Tensor]: List of tensors.
    """
    if not output_dict:
        raise ValueError("output_dict cannot be empty.")

    return [
        output_dict[flow_name]
        for flow_name in sorted(output_dict.keys())
        if flow_name.endswith(suffix)
    ]


def metrics_dict_to_dataframe(metrics_dict: Dict[str, float]) -> pd.DataFrame:
    """Format a metrics_dict (with keys 'label/dataset/split/metric') format as a pandas DataFrame.

    Args:
        metrics_dict (Dict[str, float]): Metrics dictionary.

    Returns:
        pd.DataFrame: Dataframe of metrics.
    """
    if not metrics_dict:
        raise ValueError("metrics_dict cannot be empty.")

    metrics = []

    for full_metric, score in metrics_dict.items():
        label_name, dataset_name, split, metric = tuple(full_metric.split("/"))
        metrics.append(
            (
                label_name,
                dataset_name,
                split,
                metric,
                score,
            )
        )

    return pd.DataFrame(metrics, columns=["label", "dataset", "split", "metric", "score"])
