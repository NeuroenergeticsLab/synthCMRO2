"""Build the figure tables (synthcmro2_stats_df, cmro2_vox_df) from the maps written by `predict`.

Same columns and definitions as the notebook cell that built them, so the figure code reads the output
unchanged:

- voxel values: gray matter mask voxels in NiftiMasker order (C order), from the masked analytical map
  (``…desc-orig_cmro2_masked_TRAINING…``) and the synthetic map;
- per participant (``pingouin.linear_regression(real, synthetic, alpha=0.1)``): r = sqrt(R²) (unsigned),
  p of the slope, R², adjusted R², slope, 90% CI of the *intercept* (CI_low, CI_up), MAE over all gray
  matter voxels, and the medians of analytical and synthetic CMRO2.

The regression is computed with SciPy (identical to pingouin for one predictor with intercept), so
pingouin is not needed. NumPy, pandas and nibabel only; no PyTorch.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy import stats

from .config import Config, TestDataset
from .data import grid, load_masked, output_paths

logger = logging.getLogger("synthcmro2.model")
STATS_COLUMNS = ["subj_id", "dataset", "sex", "age", "condition", "model", "r", "p", "variance", "adj_r2", "slope",
                 "CI_low", "CI_up", "MAE", "CMRO2", "synthCMRO2"]  # fmt: skip
VOXEL_COLUMNS = ["subj_id", "dataset", "condition", "model", "vox_id", "CMRO2", "synthCMRO2"]


def regression_stats(real: np.ndarray, synth: np.ndarray, alpha: float = 0.1) -> dict[str, float]:
    """OLS of synthetic on analytical CMRO2 with the outputs the notebook took from pingouin."""
    ok = np.isfinite(real) & np.isfinite(synth)  # remove_na=True
    a, s = real[ok].astype(np.float64), synth[ok].astype(np.float64)
    n = a.size
    lr = stats.linregress(a, s)
    r2 = lr.rvalue**2
    half = stats.t.ppf(1 - alpha / 2, n - 2) * lr.intercept_stderr
    return {
        "r": float(np.sqrt(r2)),
        "p": float(lr.pvalue),
        "variance": float(r2),
        "adj_r2": float(1 - (1 - r2) * (n - 1) / (n - 2)),
        "slope": float(lr.slope),
        "CI_low": float(lr.intercept - half),
        "CI_up": float(lr.intercept + half),
        "MAE": float(np.mean(np.abs(a - s))),
    }


def participants(ds: TestDataset) -> pd.DataFrame:
    if ds.participants_info is None:
        return pd.DataFrame(columns=["sex", "age"])
    info = pd.read_csv(ds.participants_info, sep="\t", dtype={"participant_id": str})
    if info["participant_id"].duplicated().any():
        raise ValueError(f"{ds.participants_info}: duplicated participant_id")
    return info.set_index("participant_id")[["sex", "age"]]


def export_tables(
    cfg: Config, mask: np.ndarray, dataset_keys=None, model_keys=None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Tables for the given datasets (default all), restricted to ``model_keys`` if given."""
    stats_rows, vox_tables, ref = [], [], grid(cfg)
    for dkey in dataset_keys or list(cfg.datasets):
        ds = cfg.datasets[dkey]
        info = participants(ds)
        for mkey in ds.export_models:
            if model_keys and mkey not in model_keys:
                continue
            model = cfg.models[mkey]
            for sid in ds.ids:
                paths = output_paths(cfg, ds, model, sid)
                missing = [str(p) for p in paths.values() if not p.exists()]
                if missing:
                    raise FileNotFoundError(f"{dkey}/{mkey}: run `predict` first; missing {missing}")
                # voxels read as stored (never resampled): the notebook's NiftiMasker interpolated maps whose
                # origin was 0.5 mm off the mask, which lowered gray matter values at the mask edge
                real = load_masked(paths["target_masked"], mask, ref)[mask].astype(np.float64)
                synth = load_masked(paths["synthetic"], mask, ref)[mask].astype(np.float64)
                if sid not in info.index:
                    logger.warning("%s: %s not in %s; sex and age left empty", dkey, sid, ds.participants_info)
                key = {"subj_id": sid, "dataset": ds.table_name, "condition": ds.condition, "model": model.table_label}
                stats_rows.append({
                    **key, "sex": info["sex"].get(sid), "age": info["age"].get(sid), **regression_stats(real, synth),
                    "CMRO2": float(np.nanmedian(real)), "synthCMRO2": float(np.nanmedian(synth)),
                })  # fmt: skip
                if ds.export_voxels:
                    vox_tables.append(
                        pd.DataFrame({**key, "vox_id": np.arange(real.size), "CMRO2": real, "synthCMRO2": synth})
                    )
    stats_df = pd.DataFrame(stats_rows, columns=STATS_COLUMNS)
    vox_df = (
        pd.concat(vox_tables, ignore_index=True)[VOXEL_COLUMNS] if vox_tables else pd.DataFrame(columns=VOXEL_COLUMNS)
    )
    if stats_df.duplicated(["subj_id", "dataset", "condition", "model"]).any():
        raise ValueError("duplicated participant/dataset/condition/model rows")
    return stats_df, vox_df
