from typing import Any

from torch.utils.tensorboard import SummaryWriter

from snorkel.types import Config

from .log_writer import LogWriter


class TensorBoardWriter(LogWriter):
    """A class for logging to Tensorboard during training process.

    This class uses the SummaryWriter from PyTorch's Tensorboard library to log
    and visualize training metrics. It is a subclass of the LogWriter class and
    passes any additional keyword arguments to the LogWriter initializer.

    Attributes:
        writer (SummaryWriter): SummaryWriter for logging and visualization
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.writer = SummaryWriter(self.log_dir)  # Initialize SummaryWriter

    def add_scalar(self, name: str, value: float, step: float) -> None:
        """Log a scalar variable to TensorBoard.

        This method logs a scalar variable to TensorBoard with the given name,
        value, and step axis value.

        Args:
            name (str): Name of the scalar collection
            value (float): Value of scalar
            step (float): Step axis value
        """
        self.writer.add_scalar(name, value, step)

    def write_config(
        self, config: Config, config_filename: str = "config.json"
    ) -> None:
        """Dump the config to file and add it to TensorBoard.

        This method dumps the config to a file with the given filename and adds
        the config as a text file to TensorBoard.

        Args:
            config (Config): JSON-compatible config to write to TensorBoard
            config_filename (str, optional): File to write config to. Defaults to "config.json".
        """
        super().write_config(config, config_filename)  # Write config to file
        self.writer.add_text(tag="config", text_string=str(config))  # Add config to TensorBoard

    def cleanup(self) -> None:
        """Close the SummaryWriter.

        This method closes the SummaryWriter and saves the logs to disk.
        """
        self.writer.close()
