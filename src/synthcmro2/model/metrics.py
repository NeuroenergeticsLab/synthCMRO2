"""Agreement between analytical and synthetic CMRO2 maps (NumPy only)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats



def gm_metrics(real: np.ndarray, synth: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    """Manuscript definitions: gray matter voxels with positive analytical and synthetic CMRO2.

    MAE (µmol/100 g/min), Pearson r, and OLS slope of synthetic on analytical CMRO2. With fewer than three
    such voxels the metrics are NaN (logged by the caller; medians across participants ignore them).
    Also returns the number of gray matter voxels with positive analytical and synthetic values.
    """
    a, s = real[mask].astype(np.float64), synth[mask].astype(np.float64)
    keep = (a > 0) & (s > 0)
    counts = {
        "n_voxels": int(keep.sum()),
        "n_pos_analytical": int((a > 0).sum()),
        "n_pos_synthetic": int((s > 0).sum()),
    }
    if keep.sum() < 3:
        return {"MAE": np.nan, "r": np.nan, "slope": np.nan, **counts}
    lr = stats.linregress(a[keep], s[keep])
    return {"MAE": float(np.mean(np.abs(s[keep] - a[keep]))), "r": float(lr.rvalue), "slope": float(lr.slope), **counts}


def warn_if_empty(metrics: dict, where: str, logger) -> None:
    if metrics["n_voxels"] < 3:
        logger.warning(
            "%s: %d gray matter voxels positive in both maps (analytical %d, synthetic %d); metrics set to NaN",
            where, metrics["n_voxels"], metrics["n_pos_analytical"], metrics["n_pos_synthetic"],
        )  # fmt: skip


def notebook_metrics(real: np.ndarray, synth: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    """Metrics printed by the testing notebooks: all gray matter voxels, no positivity filter."""
    a, s = real[mask].astype(np.float64), synth[mask].astype(np.float64)
    mse = float(np.mean((s - a) ** 2))
    return {
        "MSE_gm": mse,
        "NMSE_gm": mse / max(float(a.var()), 1e-12),
        "r_gm_all": float(stats.pearsonr(a, s)[0]),
    }


def merge_metrics(path, new: pd.DataFrame) -> pd.DataFrame:
    """Replace the rows of the dataset/model pairs in ``new`` in the CSV at ``path``; keep all other rows."""
    table = new
    if path.exists():
        old = pd.read_csv(path)
        keep = ~old.set_index(["dataset", "model"]).index.isin(new.set_index(["dataset", "model"]).index)
        table = pd.concat([old[keep], new], ignore_index=True)
    table.to_csv(path, index=False)
    return table


def completed_pairs(path, ds, folds: dict[str, int]) -> set[str]:
    """Models of ``ds`` already in the metrics file with all participants and the currently selected fold."""
    if path is None or not path.exists():
        return set()
    old = pd.read_csv(path)
    done = set()
    for mkey, rows in old[old.dataset == ds.key].groupby("model"):
        if set(rows.subj_id) == set(ds.ids) and (rows.fold == folds.get(mkey)).all():
            done.add(mkey)
    return done
