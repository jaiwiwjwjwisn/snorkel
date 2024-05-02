# Import the version module and assign the version number to a variable
from .version import VERSION

# Assign the variable to the __version__ variable in this module using the
# built-in __version__ variable as a reference
__version__ = VERSION

# Suppress the F401 import warning for the __version__ variable
#
