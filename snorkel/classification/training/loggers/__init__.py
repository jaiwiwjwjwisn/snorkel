# Import the necessary classes and functions for checkpointing, logging, and tensorboard writing
from pathlib import Path

class Checkpointer:
    """Checkpointer class for saving and loading model checkpoints."""

    def __init__(self, save_dir: str, save_name: str, save_interval: int):
        """
        Initialize the Checkpointer class.

        Args:
            save_dir (str): The directory where the checkpoints will be saved.
            save_name (str): The name of the checkpoint files.
            save_interval (int): The interval (in number of epochs) at which the checkpoints will be saved.
        """
        self.save_dir = Path(save_dir)
        self.save_name = save_name
        self.save_interval = save_interval

    def save(self, model, epoch):
        """
        Save the model checkpoint.

        Args:
            model: The model to be saved.
            epoch (int): The current epoch number.
        """
        if epoch % self.save_interval == 0:
            save_path = self.save_dir / f"{self.save_name}_epoch_{epoch}.pth"
            torch.save(model.state_dict(), save_path)
            print(f"Checkpoint saved to {save_path}")

    def load(self, model, epoch):
        """
        Load the model checkpoint.

        Args:
            model: The model to be loaded.
            epoch (int): The current epoch number.
        """
        load_path = self.save_dir / f"{self.save_name}_epoch_{epoch}.pth"
        model.load_state_dict(torch.load(load_path))
        print(f"Checkpoint loaded from {load_path}")


class CheckpointerConfig:
    """Configuration class for Checkpointer."""

    def __init__(self, save_dir: str, save_name: str, save_interval: int):
        """
        Initialize the CheckpointerConfig class.

        Args:
            save_dir (str): The directory where the checkpoints will be saved.
            save_name (str): The name of the checkpoint files.
            save_interval (int): The interval (in number of epochs) at which the checkpoints will be saved.
        """
        self.save_dir = save_dir
        self.save_name = save_name
        self.save_interval = save_interval
