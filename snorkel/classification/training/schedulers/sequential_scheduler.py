from typing import Sequence

from snorkel.classification.data import DictDataLoader

from .scheduler import BatchIterator, Scheduler


class SequentialScheduler(Scheduler):
    """
    A class representing a scheduler that returns batches from all dataloaders
    in sequential order.

    Inherits from the Scheduler class and overrides the get_batches method.
    """

    def __init__(self) -> None:
        """
        Initialize the SequentialScheduler object.

        Calls the constructor of the superclass.
        """
        super().__init__()

    def get_batches(self, dataloaders: Sequence[DictDataLoader]) -> BatchIterator:
        """
        Return batches from dataloaders sequentially in the order they were given.

        Parameters
        ----------
        dataloaders : Sequence[DictDataLoader]
            A sequence of dataloaders to get batches from.

        Yields
        ------
        (batch, dataloader)
            batch is a tuple of (X_dict, Y_dict) and dataloader is the dataloader
            that that batch came from. That dataloader will not be accessed by the
            model; it is passed primarily so that the model can pull the necessary
            metadata to know what to do with the batch it has been given.
        """
        for dataloader in dataloaders:
            # Iterate through each dataloader in the given sequence
            for batch in dataloader:
                # Iterate through each batch in the current dataloader
                yield batch, dataloader  # Yield the batch and the current dataloader
