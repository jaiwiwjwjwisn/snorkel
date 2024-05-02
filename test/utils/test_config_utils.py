import unittest
from unittest.mock import patch

from snorkel.types import Config
from snorkel.utils.config_utils import merge_config


class FooConfig(Config):
    a: float = 0.5


class BarConfig(Config):
    a: int = 1
    foo_config: FooConfig = FooConfig()  # type: ignore


class UtilsTest(unittest.TestCase):
    @patch("snorkel.utils.config_utils.merge_config")
    def test_merge_config_updates_a(self, mock_merge_config):
        config_updates = {"a": 2, "foo_config": {"a": 0.75}}
        merge_config.return_value = BarConfig()
        bar_config = merge_config(BarConfig(), config_updates)
        self.assertEqual(bar_config.a, 2)
        self.assertEqual(bar_config.foo_config.a, 0.5)
        mock_merge_config.assert_called_once_with(BarConfig().foo_config, {"a": 0.75})

    @patch("snorkel.utils.config_utils.merge_config")
    def test_merge_config_missing_key(self, mock_merge_config):
        config_updates = {"b": 2, "foo_config": {"a": 0.75}}
        merge_config.return_value = BarConfig()
        bar_config = merge_config(BarConfig(), config_updates)
        self.assertEqual(bar_config.a, 1)
        self.assertEqual(bar_config.foo_config.a, 0.5)
        mock_merge_config.assert_called_once_with(BarConfig().foo_config, {"a": 0.75})

    @patch("snorkel.utils.config_utils.merge_config")
    def test_merge_config_missing_nested_key(self, mock_merge_config):
        config_updates = {"foo_config": {"b": 0.75}}
        merge_config.return_value = BarConfig()
        bar_config = merge_config(BarConfig(), config_updates)
        self.assertEqual(bar_config.a, 1)
        self.assertEqual(bar_config.foo_config.a, 0.5)
        mock_merge_config.assert_called_once_with(BarConfig().foo_config, {"b": 0.75})

