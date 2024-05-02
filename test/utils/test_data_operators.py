import unittest

from snorkel.utils.data_operators import check_unique_names

class DataOperatorsTest(unittest.TestCase):
    def test_check_unique_names(self):
        """
        Test the check_unique_names function from snorkel.utils.data_operators.
        This function checks that all names in a list are unique.
        """
        # Test passing a list of unique names
        check_unique_names(["alice", "bob", "chuck"])

        # Test passing a list with non-unique names
        with self.assertRaisesRegex(ValueError, "3 operators with name c"):
            check_unique_names(["a", "a", "b", "c", "c", "c"])
            # The function should raise a ValueError with a specific error message
            # when it encounters non-unique names.
