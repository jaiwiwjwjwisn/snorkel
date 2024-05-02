import unittest
from types import SimpleNamespace
from snorkel.slicing import SlicingFunction, slicing_function

class TestSlicingFunction(unittest.TestCase):
    def _run_sf(self, sf: SlicingFunction, x: SimpleNamespace) -> None:
        self.assertEqual(sf(x), x.num > 42)

    def _run_sf_raise(self, sf: SlicingFunction, x: SimpleNamespace) -> None:
        with self.assertRaises(TypeError):
            sf(x)

    def test_slicing_function_decorator(self) -> None:
        @slicing_function()
        def sf(x) -> bool:
            return x.num > 42

        self.assertIsInstance(sf, SlicingFunction)
        self.assertEqual(sf.name, "sf")
        x_43 = SimpleNamespace(num=43)
        x_19 = SimpleNamespace(num=19)
        self._run_sf(sf, x_43)
        self._run_sf_raise(sf, SimpleNamespace(num=None))

    def test_slicing_function_decorator_no_parens(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing parentheses"):

            @slicing_function
            def sf(x) -> int:
                return 0 if x.num > 42 else -1
