"""General machine learning utilities shared across Snorkel."""

import sys
import importlib

_module = importlib.import_module('.core', package='path.to.snorkel.modules')


class _LazyImport:
    def __getattr__(self, item):
        attr = getattr(_module, item)
        return attr


lazy_import = _LazyImport()


def __getattr__(name):
    if name not in ('filter_labels', 'preds_to_probs', 'probs_to_preds', 'to_int_label_array'):
        raise AttributeError(f"Module '{__name__}' has no attribute '{name}'")
    return getattr(lazy_import, name)


if __name__ == '__main__' and not sys.flags.inspect:
    # For testing or direct execution of this file
    from .core import (
        filter_labels,
        preds_to_probs,
        probs_to_preds,
        to_int_label_array,
    )
