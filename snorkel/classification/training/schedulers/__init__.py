# Import the SequentialScheduler and ShuffledScheduler classes from their respective modules.
from .sequential_scheduler import SequentialScheduler
from .shuffled_scheduler import ShuffledScheduler

# Define a dictionary called batch_schedulers that maps string keys to class values.
# The keys are "sequential" and "shuffled", and the corresponding values are the
# SequentialScheduler and ShuffledScheduler classes, respectively.
batch_schedulers = {"sequential": SequentialScheduler, "shuffled": ShuffledScheduler}
