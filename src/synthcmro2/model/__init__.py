"""Synthetic CMRO2 models: training, validation-based selection, prediction, and export of the figure tables.

    from synthcmro2 import model
    cfg = model.load_config("config/model.yaml")

``load_config`` and ``export_tables`` need no PyTorch; ``train_model``, ``select`` and ``predict`` import it on
first use.
"""

from .config import Config, load_config

_LAZY = {
    "train_model": ("train", "train_model"),
    "select": ("validate", "select"),
    "predict": ("predict", "predict"),
    "export_tables": ("export", "export_tables"),
    "cohort_folds": ("folds", "cohort_folds"),
}
__all__ = ["Config", "load_config", *_LAZY]


def __getattr__(name):
    if name in _LAZY:
        from importlib import import_module

        module, attr = _LAZY[name]
        return getattr(import_module(f".{module}", __name__), attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
