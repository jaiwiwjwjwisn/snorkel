from collections import defaultdict
from typing import Any, Callable, Dict, List, Tuple

from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from .utils import list_to_tensor

# Default string names for initializing a DictDataset
DEFAULT_INPUT_DATA_KEY = "input_data"
DEFAULT_DATASET_NAME = "SnorkelDataset"
DEFAULT_TASK_NAME = "task"


class DictDataset(Dataset):
    """A dataset where both the data fields and labels are stored in as dictionaries.

    This class is initialized with a name, split, X_dict, and Y_dict. The name and split
    are simple strings that describe the dataset and the split of the data, respectively.
    X_dict is a dictionary where the keys are field names and the values are the data
    corresponding to those field names. Y_dict is another dictionary where the keys are
    task names and the values are the labels for those tasks.

    Raises
    ------
    ValueError
        If any value in the Y_dict is not a torch.Tensor.

    Attributes
    ----------
    name: str
        The name of the dataset.
    split: str
        The name of the split of the data.
    X_dict: Dict[str, Any]
        A dictionary where the keys are field names and the values are the data
        corresponding to those field names.
    Y_dict: Dict[str, Tensor]
        A dictionary where the keys are task names and the values are the labels for
        those tasks.
    """

    def __init__(self, name: str, split: str, X_dict: XDict, Y_dict: YDict) -> None:
        self.name = name
        self.split = split
        self.X_dict = X_dict
        self.Y_dict = Y_dict

        for name, label in self.Y_dict.items():
            if not isinstance(label, Tensor):
                raise ValueError(
                    f"Label {name} should be torch.Tensor, not {type(label)}."
                )

    def __getitem__(self, index: int) -> Tuple[XDict, YDict]:
        """Return the input data and labels at the given index.

        Returns
        -------
        Tuple[XDict, YDict]
            A tuple where the first element is a dictionary of input data and the
            second element is a dictionary of labels.
        """
        x_dict = {name: feature[index] for name, feature in self.X_dict.items()}
        y_dict = {name: label[index] for name, label in self.Y_dict.items()}
        return x_dict, y_dict

    def __len__(self) -> int:
        """Return the number of examples in the dataset.

        Returns
        -------
        int
            The number of examples in the dataset.
        """
        try:
            return len(next(iter(self.Y_dict.values())))  # type: ignore
        except StopIteration:
            return 0

    def __repr__(self) -> str:
        """Return a string representation of the dataset.

        Returns
        -------
        str
            A string representation of the dataset.
        """
        return (
            f"{type(self).__name__}"
            f"(name={self.name}, "
            f"X_keys={list(self.X_dict.keys())}, "
            f"Y_keys={list(self.Y_dict.keys())})"
        )

    @classmethod
    def from_tensors(
        cls,
        X_tensor: Tensor,
        Y_tensor: Tensor,
        split: str,
        input_data_key: str = DEFAULT_INPUT_DATA_KEY,
        task_name: str = DEFAULT_TASK_NAME,
        dataset_name: str = DEFAULT_DATASET_NAME,
    ) -> "DictDataset":
        """Create a DictDataset from tensors.

        Parameters
        ----------
        X_tensor : Tensor
            The input data tensor.
        Y_tensor : Tensor
            The label tensor.
        split : str
            The name of the split of the data.
        input_data_key : str, optional
            The name of the input data key in the X_dict dictionary, by default
            "input_data".
        task_name : str, optional
            The name of the task key in the Y_dict dictionary, by default "task".
        dataset_name : str, optional
            The name of the dataset, by default "SnorkelDataset".

        Returns
        -------
        DictDataset
            A DictDataset initialized with the given tensors.
        """
        return cls(
            name=dataset_name,
            split=split,
            X_dict={input_data_key: X_tensor},
            Y_dict={task_name: Y_tensor},
        )


def collate_dicts(batch: List[Batch]) -> Batch:
    """Combine many one-element dicts into a single many-element dict for both X and Y.

    Parameters
    ----------
    batch : List[Batch]
        A list of (x_dict, y_dict) where the values of each are a single element.

    Returns
    -------
    Batch
        A tuple of X_dict, Y_dict where the values of each are a merged list or tensor.
    """
    X_batch: Dict[str, Any] = defaultdict(list)
    Y_batch: Dict[str, Any] = defaultdict(list)

    for x_dict, y_dict in batch:
        for field_name, value in x_dict.items():
            X_batch[field_name].append(value)
        for label_name, value in y_dict.items():
            Y_batch[label_name].append(value)

    for field_name, values in X_batch.items():
        # Only merge list of tensors
        if isinstance(values[0], Tensor):
            X_batch[field_name] = list_to_tensor(values)

    for label_name, values in Y_batch.items():
        Y_batch[label_name] = list_to_tensor(values)

    return dict(X_batch), dict(Y_batch)


class DictDataLoader(DataLoader):
    """A DataLoader that uses the appropriate collate_fn for a ``DictDataset``.

    This class is a subclass of DataLoader that uses the collate_dicts function to
    combine the data from multiple examples into a single batch.

    Parameters
    ----------
    dataset : DictDataset
        The dataset to wrap.
    collate_fn : Callable[..., Any]
        The collate function to use when combining multiple indexed examples for a
        single batch.
    kwargs
        Keyword arguments to pass on to DataLoader.__init__().
    """

    def __init__(
        self,
        dataset: DictDataset,
        collate_fn: Callable[..., Any] = collate_dicts,
        **kwargs: Any,
    ) -> None:
        assert isinstance(dataset, DictDataset)
        super().__init__(dataset, collate_fn=collate_fn, **kwargs)
