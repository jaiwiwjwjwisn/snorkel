from collections import Counter
from typing import List

def check_unique_names(names: List[str]) -> None:
    """
    Check that all operator names in the given list are unique.

    This function takes a list of strings as input, where each string represents
    the name of an operator. It uses a Counter object to count the occurrences
    of each operator name, and then checks if any name has a count greater than 1.
    If so, it raises a ValueError with a message indicating which name(s) are
    not unique.

    Args:
        names: A list of strings representing operator names.

    Raises:
        ValueError: If any operator name is not unique.

    Example:
        >>> check_unique_names(["add", "sub", "add"])
        Traceback (most recent call last):
            ...
        ValueError: Operator names not unique: 2 operators with name add
    """
    k, ct = Counter(names).most_common(1)[0]
    if ct > 1:
        raise ValueError(f"Operator names not unique: {ct} operators with name {k}")
