#!/usr/bin/env python

"""Check requirements.txt and setup.py install requirements.

Adapted from https://github.com/allenai/allennlp/tree/master/scripts

This script checks the requirements in requirements.txt against those in setup.py.
In particular, it makes sure that there are no duplicates within either setup.py or
requirements.txt and that each package that is listed in both setup.py
and requirements.txt having matching versions. It also ensures that any requirement
listed as "ESSENTIAL" in requirements.txt appears under the "install_requires"
field in setup.py.
"""

import re
import sys
from typing import Dict, Optional, Set, Tuple

PackagesType = Dict[str, Optional[str]]


def parse_section_name(line: str) -> str:
    return line.replace("####", "").strip()


def parse_package(line: str) -> Tuple[str, Optional[str]]:
    parts = re.split(r"(==|>=|<=|>|<)", line)
    module = parts[0]
    version = line.replace(module, "")
    return (module, version)


def parse_requirements() -> Tuple[PackagesType, PackagesType, Set[str]]:
    """Parse all dependencies out of the requirements.txt file."""
    essential_packages: PackagesType = {}
    other_packages: PackagesType = {}
    duplicates: Set[str] = set()
    with open("requirements.txt", "r") as req_file:
        section: str = ""
        for line in req_file:
            line = line.strip()

            if line.startswith("####"):
                # Line is a section name.
                section = parse_section_name(line)
                continue

            if not line or line.startswith("#"):
                # Line is empty or just regular comment.
                continue

            module, version = parse_package(line)
            if module in essential_packages or module in other_packages:
                duplicates.add(module)

            if section.startswith("ESSENTIAL"):
                essential_packages[module] = version
            else:
                other_packages[module] = version

    return essential_packages, other_packages, duplicates


def parse_setup() -> Tuple[PackagesType, PackagesType, Set[str], Set[str]]:
    """Parse all dependencies out of the setup.py script."""
    essential_packages: PackagesType = {}
    test_packages: PackagesType = {}
    essential_duplicates: Set[str] = set()
    test_duplicates: Set[str] = set()

    with open("setup.py") as setup_file:
        contents = setup_file.read()

    # Parse out essential packages.
    match = re.search(
        r"""install_requires=\[[\s\n]*['"](.*?)['"],?[\s\n]*\]""", contents, re.DOTALL
    )
    if match is not None:
        package_string = match.groups()[0].strip()
        for package in re.split(r"""['"],[\s\n]+['"]""", package_string):
            module, version = parse_package(package)
            if module in essential_packages:
                essential_duplicates.add(module)
            else:
                essential_packages[module] = version

    # Parse packages only needed for testing.
    match = re.search(
        r"""tests_require=\[[\s\n]*['"](.*?)['"],?[\s\n]*\]""", contents, re.DOTALL
    )
    if match is not None:
        package_string = match.groups()[0].strip()
        for package in re.split(r"""['"],[\s\n]+['"]""", package_string):
            module, version = parse_package(package)
            if module in test_packages:
                test_duplicates.add(module)
            else:
                test_packages[module] = version

    return essential_packages, test_packages, essential_duplicates, test_duplicates


def main() -> int:
    exit_code = 0

    (
        requirements_essential,
        requirements_other,
        requirements_duplicate,
    ) = parse_requirements()
    requirements_all = dict(requirements_essential, **requirements_other)
    setup_essential, setup_test, essential_duplicates, test_duplicates = parse_setup()

    print_issues(
        "Requirements.txt",
        requirements_duplicate,
        essential_duplicates,
        requirements_all,
        setup_essential,
    )

    print_issues(
        "Setup.py",
        essential_duplicates,
        test_duplicates,
        setup_essential,
        requirements_all,
    )

    return exit_code


def print_issues(
    header: str,
    duplicates: Set[str],
    section_duplicates: Set[str],
    packages1: PackagesType,
    packages2: PackagesType,
) -> None:
    print(f"\n### {header} issues:")

    if duplicates:
        print_duplicates("Packages", duplicates)

    if section_duplicates:
        print_duplicates("Sections", section_duplicates)

    for module, version in packages1.items():
        if module not in packages2:
            print(f"  ✗ '{module}' is missing in {header}")
        elif packages2[module] != version:
            print(
                f"  ✗ '{module}' has version '{version}' in {header} but "
                f"'{packages2[module]}' in requirements.txt"
            )


def print_duplicates(title: str, duplicates: Set[str]) -> None:
    if duplicates:
        print(f"  ✗ Duplicate {title}:")
        for module in duplicates:
            print(f"      - {module}")


if __name__ == "__main__":
    sys.exit(main())
