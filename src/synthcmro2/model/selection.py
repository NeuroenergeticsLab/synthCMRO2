"""Model selection on validation data only (NumPy and pandas; no PyTorch).

The testing notebooks chose, for each test dataset, the fold with the highest mean test r + SSIM, and
the input configurations were compared on test data. Here, both choices use the validation participants
of the cross-validation folds; the test participants are used only once, after selection.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from .config import Config

logger = logging.getLogger("synthcmro2.model")
SELECTION_FILE = "cv_selection.csv"


def summarize_selection(rows: list[dict], fold_rows: list[dict], reduction_labels: set[str] | None = None):
    """Per-participant, per-fold and per-model tables, and the selection.

    - fold per model: lowest validation generator loss (the checkpointing criterion);
    - cross-validated performance: mean and SD over folds of the per-fold median of participants;
    - model per number of inputs: lowest cross-validated MAE among the input-reduction models
      (``reduction_labels``, by input label; comparison models are never selected).
    """
    per_subj = pd.DataFrame(rows)
    per_fold = pd.DataFrame(fold_rows).merge(
        per_subj.groupby(["config", "fold"])[["MAE", "r", "slope"]].median().reset_index(), on=["config", "fold"]
    )
    best_fold = per_fold.loc[per_fold.groupby("config")["val_generator_loss"].idxmin(), ["config", "fold"]]
    cv = per_fold.groupby(["config", "inputs", "n_inputs"])[["val_generator_loss", "MAE", "r", "slope"]].agg(
        ["mean", "std"]
    )
    cv.columns = ["_".join(c) for c in cv.columns]
    cv = cv.reset_index().merge(best_fold.rename(columns={"fold": "selected_fold"}), on="config")
    if reduction_labels is None:
        reduction = ~cv["inputs"].isin(["T1", "fMRI"])
    else:
        reduction = cv["inputs"].isin(reduction_labels)
    best_mae = cv[reduction].groupby("n_inputs")["MAE_mean"].transform("min")
    cv["best_for_n_inputs"] = reduction & (cv["MAE_mean"] == best_mae.reindex(cv.index))
    return per_subj, per_fold, cv.sort_values(["n_inputs", "MAE_mean"]).reset_index(drop=True)


def selected_folds(cfg: Config, selection_file: Path | None = None) -> dict[str, int]:
    """Fold per model key: from cv_selection.csv, overridden by selection.fixed_folds in the config."""
    path = selection_file or cfg.selection_dir / SELECTION_FILE
    folds: dict[str, int] = {}
    if path.exists():
        table = pd.read_csv(path)
        by_name = dict(zip(table["config"], table["selected_fold"], strict=True))
        folds = {k: int(by_name[m.name]) for k, m in cfg.models.items() if m.name in by_name}
    else:
        logger.warning("no selection file %s; run `synthcmro2 model select` first", path)
    for key, fold in cfg.fixed_folds.items():
        if folds.get(key) not in (None, fold):
            logger.warning("%s: config fixes fold %d instead of the validation-selected fold %d", key, fold, folds[key])
        folds[key] = fold
    return folds
