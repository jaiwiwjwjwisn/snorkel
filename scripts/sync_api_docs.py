#!/usr/bin/env python

"""Check and update API docs under docs/packages.

This script checks and updates the package documentation pages, making sure
that the packages in docs/packages.json are documented and up to date.
Rather than calling this directly, use `tox -e check` or `tox -e fix`.
"""

import json
import os
import sys
from importlib import import_module
from typing import Any, List

# Path to the JSON file containing package information
PACKAGE_INFO_PATH = "docs/packages.json"
# Path to the directory containing package documentation pages
PACKAGE_PAGE_PATH = "docs/packages"

# Template for generating package documentation pages
PACKAGE_DOC_TEMPLATE = """{title}
{underscore}

{docstring}

.. currentmodule:: snorkel.{package_name}

.. autosummary::
   :toctree: _autosummary/{package_name}/
   :nosignatures:

   {members}
"""

def get_title_and_underscore(package_name: str) -> str:
    """Generate the title and underscore string for a package.

    Args:
        package_name (str): The name of the package.

    Returns:
        tuple: A tuple containing the title and underscore strings.
    """
    title = f"Snorkel {package_name.capitalize()} Package"
    underscore = "-" * len(title)
    return title, underscore


def get_package_members(package: Any) -> List[str]:
    """Generate a list of members for a package.

    Args:
        package (Any): The package for which to generate members.

    Returns:
        list: A list of members for the package.
    """
    members = []
    for name in dir(package):
        if name.startswith("_"):
            continue
        obj = getattr(package, name)
        if isinstance(obj, type) or callable(obj):
            members.append(name)
    return members


def main(check: bool) -> None:
    """Check or update package documentation pages.

    Args:
        check (bool): Whether to check or update package documentation pages.

    Returns:
        None
    """
    # Load package information from JSON file
    with open(PACKAGE_INFO_PATH, "r") as f:
        packages_info = json.load(f)
    # Get the names of all packages
    package_names = sorted(packages_info["packages"])
    if check:
        # Check if the expected package files match the actual ones
        f_basenames = sorted(
            [
                os.path.splitext(f_name)[0]
                for f_name in os.listdir(PACKAGE_PAGE_PATH)
                if f_name.endswith(".rst")
            ]
        )
        if f_basenames != package_names:
            raise ValueError(
                "Expected package files do not match actual!\n"
                f"Expected: {package_names}\n"
                f"Actual: {f_basenames}"
            )
    else:
        # Create the package documentation pages directory if it doesn't exist
        os.makedirs(PACKAGE_PAGE_PATH, exist_ok=True)
    for package_name in package_names:
        # Import the package
        package = import_module(f"snorkel.{package_name}")
        # Get the package's docstring
        docstring = package.__doc__
        # Generate the title and underscore strings for the package
        title, underscore = get_title_and_underscore(package_name)
        # Get all members for the package
        all_members = get_package_members(package)
        # Add any extra members for the package
        all_members.extend(packages_info["extra_members"].get(package_name, []))
        # Generate the contents for the package documentation page
        contents = PACKAGE_DOC_TEMPLATE.format(
            title=title,
            underscore=underscore,
            docstring=docstring,
            package_name=package_name,
            members="\n   ".join(sorted(all_members, key=lambda s: s.split(".")[-1])),
        )
        # Generate the file path for the package documentation page
        f_path = os.path.join(PACKAGE_PAGE_PATH, f"{package_name}.rst")
        if check:
            # Check if the contents for the package documentation page match the actual ones
            with open(f_path, "r") as f:
                contents_actual = f.read()
            if contents != contents_actual:
                raise ValueError(f"Contents for {package_name} differ!")
        else:
            # Write the contents to the package documentation page
            with open(f_path, "w") as f:
                f.write(contents)


if __name__ == "__main__":
    # Determine whether to check or update package documentation pages
    check = False if len(sys.argv) == 1 else (sys.argv[1] == "--check")
    # Call the main function with the appropriate argument
    sys.exit(main(check))
