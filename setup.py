# Import necessary modules
from typing import Dict
from setuptools import find_packages, setup

# Execute version.py to get the version number
VERSION: Dict[str, str] = {}
with open("snorkel/version.py", "r") as version_file:
    exec(version_file.read(), VERSION)

# Read README.md as the long_description for the package
with open("README.md", "r") as readme_file:
    long_description = readme_file.read()

# Configure the package information
setup(
    # Package name
    name="snorkel",
    # Version number
    version=VERSION["VERSION"],
    # Project URL
    url="https://github.com/snorkel-team/snorkel",
    # Package description
    description="A system for quickly generating training data with weak supervision",
    # Long description in Markdown format
    long_description_content_type="text/markdown",
    long_description=long_description,
    # License information
    license="Apache License 2.0",
    # Classifiers for the package
    classifiers=[
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
    ],
    # Project URLs
    project_urls={
        "Homepage": "https://snorkel.org",
        "Source": "https://github.com/snorkel-team/snorkel/",
        "Bug Reports": "https://github.com/snorkel-team/snorkel/issues",
        "Citation": "https://doi.org/10.14778/3157794.3157797",
    },
    # Packages to include
    packages=find_packages(exclude=("test*",)),
    # Include package data
    include_package_data=True,
    # Required packages
    install_requires=[
        "munkres>=1.0.6",
        "numpy>=1.24.0",
        "scipy>=1.2.0",
        "pandas>=1.0.0",
        "tqdm>=4.33.0",
        "scikit-learn>=0.20.2",
        "torch>=1.2.0",
        "tensorboard>=2.13.0",
        "protobuf>=3.19.6",
        "networkx>=2.2",
    ],
    # Python version requirement
    python_requires=">=3.11",
    # Keywords
    keywords="machine-learning ai weak-supervision",
)
