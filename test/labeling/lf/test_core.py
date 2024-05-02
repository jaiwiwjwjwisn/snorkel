import pickle
import unittest
from typing import List, SimpleNamespace

from snorkel.labeling import LabelingFunction, labeling_function
from snorkel.preprocess import preprocessor
from snorkel.types import DataPoint


class TestLabelingFunction(unittest.TestCase):
    """Test cases for the LabelingFunction class."""

    def _run_lf(self, lf: LabelingFunction, x_43: SimpleNamespace, x_19: SimpleNamespace) -> None:
        self.assertEqual(lf(x_43), 0)
        self.assertEqual(lf(x_19), -1)

    def setUp(self) -> None:
        self.x_43 = SimpleNamespace(num=43)
        self.x_19 = SimpleNamespace(num=19)

    def test_labeling_function(self) -> None:
        """Test the LabelingFunction constructor and basic functionality."""
        lf = LabelingFunction(name="my_lf", f=f)
        self._run_lf(lf, self.x_43, self.x_19)

    def test_labeling_function_resources(self) -> None:
        """Test the LabelingFunction constructor with resources."""
        db = [3, 6, 43]
        lf = LabelingFunction(name="my_lf", f=g, resources=dict(db=db))
        self._run_lf(lf, self.x_43, self.x_19)

    def test_labeling_function_preprocessor(self) -> None:
        """Test the LabelingFunction constructor with preprocessors."""
        lf = LabelingFunction(name="my_lf", f=f, pre=[square, square])
        x_6 = SimpleNamespace(num=6)
        x_2 = SimpleNamespace(num=2)
        self._run_lf(lf, self.x_43, x_2)
        self.assertEqual(lf(x_6), 0)

    def test_labeling_function_returns_none(self) -> None:
        """Test the LabelingFunction constructor with a preprocessor that returns None."""
        lf = LabelingFunction(name="my_lf", f=f, pre=[square, returns_none])

        with self.assertRaisesMessage(ValueError, "Labeling function returned None."):
            lf(self.x_43)

    def test_labeling_function_serialize(self) -> None:
        """Test the LabelingFunction serialize/deserialize functionality."""
        db = [3, 6, 43**2]
        lf = LabelingFunction(name="my_lf", f=g, resources=dict(db=db), pre=[square])
        lf_load = pickle.loads(pickle.dumps(lf))
        self._run_lf(lf_load, self.x_43, self.x_19)

    def test_labeling_function_decorator(self) -> None:
        """Test the labeling_function decorator."""

        @labeling_function()
        def lf(x: DataPoint) -> int:
            return 0 if x.num > 42 else -1

        self.assertIsInstance(lf, LabelingFunction)
        self.assertEqual(lf.name, "lf")
        self._run_lf(lf, self.x_43, self.x_19)

    def test_labeling_function_decorator_args(self) -> None:
        """Test the labeling_function decorator with arguments."""
        db = [3, 6, 43**2]

        @labeling_function(name="my_lf", resources=dict(db=db), pre=[square])
        def lf(x: DataPoint, db: List[int]) -> int:
            return 0 if x.num in db else -1

        self.assertIsInstance(lf, LabelingFunction)
        self.assertEqual(lf.name, "my_lf")
        self._run_lf(lf, self.x_43, self.x_19)

    def test_labeling_function_decorator_no_parens(self) -> None:
        """Test that the decorator raises a ValueError when not called with parentheses."""
        with self.assertRaisesRegex(ValueError, "missing parentheses"):

            @labeling_function
            def lf(x: DataPoint) -> int:
                return 0 if x.num > 42 else -1


from snorkel.types import Label

class LabelingFunction:
    # ...

    def __call__(self, *args, **kwargs) -> Label:
        if self.f is None:
            raise ValueError("Labeling function is not defined.")
        result = self.f(*args, **kwargs)
        if result is None:
            raise ValueError("Labeling function returned None.")
        return result
