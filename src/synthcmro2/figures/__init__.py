"""Figures and statistics of the synthetic CMRO2 manuscript.

    from synthcmro2 import figures
    cfg = figures.load_config("config/figures.yaml")
    out = figures.make_figure2(metrics, voxels, cfg)   # FigureOutput: figure, statistics tables, group maps
"""

from .config import Config, load_config

_LAZY = {
    "make_figure2": ("manuscript", "make_figure2"),
    "make_figure3": ("manuscript", "make_figure3"),
    "make_figure4": ("manuscript", "make_figure4"),
    "make_figure_s2": ("manuscript", "make_figure_s2"),
    "SurfaceContext": ("manuscript", "SurfaceContext"),
    "load_participant_metrics": ("data", "load_participant_metrics"),
    "load_voxel_values": ("data", "load_voxel_values"),
    "noise_benchmark": ("benchmark", "noise_benchmark"),
    "run_pls": ("pls", "run_pls"),
}
__all__ = ["Config", "load_config", *_LAZY]


def __getattr__(name):
    if name in _LAZY:
        from importlib import import_module

        module, attr = _LAZY[name]
        return getattr(import_module(f".{module}", __name__), attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
