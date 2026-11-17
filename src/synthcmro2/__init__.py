"""Synthetic CMRO2 from minimal contrast-free MRI: models and manuscript figures.

    from synthcmro2 import model, figures

model    training, validation-based fold selection, prediction and export of the figure tables (PyTorch)
figures  Figures 2-4 and S2, the PLS of task versus rest, and the test-retest error benchmark (Table S2)

Command line: ``synthcmro2 {model,figures,pls,benchmark,public-tables,crop-atlas} --help``.
"""

__version__ = "2.1.0"
__all__ = ["figures", "model"]
