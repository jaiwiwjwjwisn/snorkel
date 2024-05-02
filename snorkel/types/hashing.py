from collections.abc import Hashable
from typing import Any, Callable

# Define a type alias for a hashing function
HashingFunction = Callable[[Any], Hashable]

# A hashing function is a callable object that takes in any object as an argument
# and returns a hashable object. This is often used in data structures like
# hash tables and hash sets to quickly and efficiently map keys to values.
