"""Sensitivity analysis: repeat the figures and statistics without selected Glasser parcels.

``analysis.exclude_parcels`` (or ``synthcmro2 figures --exclude-parcels``) lists Glasser parcel names, for
example ``[TE1m, TE2a, TE1p, TE2p]``. The parcels are removed in both hemispheres (left labels 1-180, right
labels 201-380 in the volumetric MMP atlas), and then:

- their voxels are dropped from the voxel table, so group maps, scatter plots, surface maps (drawn gray),
  PLS inputs, and the test-retest benchmark use the remaining gray matter only;
- the participant metrics (r, p, slope, CIs, MAE, medians) are recomputed from the remaining voxels with
  the definitions of ``synthcmro2 model export``. Datasets exported without voxel values (the out-of-fold
  training predictions used for the age calibration) cannot be recomputed and are kept as exported;
- Figure S2 computes parcel ICCs without the excluded parcels.

Outputs go to ``<output_dir>_excl-<names>`` and ``<pls_dir>_excl-<names>``, so the main results are never
overwritten.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy import stats

from .data import KEY_COLUMNS

logger = logging.getLogger(__name__)

RIGHT_OFFSET = 200  # right-hemisphere label = left label + 200 in the volumetric Glasser (MMP) atlas
METRIC_VALUE_COLUMNS = ["r", "p", "variance", "adj_r2", "slope", "CI_low", "CI_up", "MAE", "CMRO2", "synthCMRO2"]


def suffix(names) -> str:
    """Folder suffix of an exclusion run, e.g. ``_excl-TE1m-TE2a``."""
    return "_excl-" + "-".join(names) if names else ""


def parcel_labels(names, label_gii) -> np.ndarray:
    """Atlas labels (both hemispheres) of the Glasser parcels ``names`` (e.g. ``TE1m``; ``L_``/``_ROI`` optional)."""
    import nibabel as nib

    table = nib.load(str(label_gii)).labeltable.get_labels_as_dict()
    by_name = {str(v).removeprefix("L_").removesuffix("_ROI"): int(k) for k, v in table.items() if 0 < int(k) <= 180}
    clean = [str(n).removeprefix("L_").removeprefix("R_").removesuffix("_ROI") for n in names]
    unknown = [n for n in clean if n not in by_name]
    if unknown:
        raise ValueError(f"unknown Glasser parcels {unknown}; names as in {label_gii} without L_ and _ROI")
    left = np.array([by_name[n] for n in clean], dtype=int)
    return np.concatenate([left, left + RIGHT_OFFSET])


def labels_for(cfg) -> np.ndarray:
    """Atlas labels to exclude for this configuration (empty without ``exclude_parcels``)."""
    if not cfg.exclude_parcels:
        return np.array([], dtype=int)
    if cfg.surface.glasser_label_gii is None or cfg.surface.glasser_atlas is None or cfg.gm_mask is None:
        raise ValueError("exclude_parcels needs paths.gm_mask, surface.glasser_atlas and surface.glasser_label_gii")
    return parcel_labels(cfg.exclude_parcels, cfg.surface.glasser_label_gii)


def excluded_voxels(parcel_ids: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Voxel IDs (gray matter mask order) inside the given atlas labels."""
    return np.flatnonzero(np.isin(parcel_ids, labels))


def drop_voxels(voxels: pd.DataFrame, vox_ids: np.ndarray) -> pd.DataFrame:
    return voxels[~voxels["vox_id"].isin(vox_ids)].reset_index(drop=True)


def regression_stats(real: np.ndarray, synth: np.ndarray, alpha: float = 0.1) -> dict[str, float]:
    """Same definitions as ``synthcmro2.model.export.regression_stats`` (OLS of synthetic on analytical)."""
    ok = np.isfinite(real) & np.isfinite(synth)
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
        "CMRO2": float(np.nanmedian(real)),
        "synthCMRO2": float(np.nanmedian(synth)),
    }


def recompute_metrics(metrics: pd.DataFrame, voxels: pd.DataFrame) -> pd.DataFrame:
    """Participant metrics recomputed from ``voxels`` for every map it contains; other rows unchanged."""
    rows = []
    for key, g in voxels.groupby(KEY_COLUMNS, observed=True, sort=False):
        g = g.sort_values("vox_id")
        rows.append({**dict(zip(KEY_COLUMNS, map(str, key), strict=True)),
                     **regression_stats(g["CMRO2"].to_numpy(), g["synthCMRO2"].to_numpy())})  # fmt: skip
    new = pd.DataFrame(rows).set_index(KEY_COLUMNS)
    out = metrics.copy()
    idx = pd.MultiIndex.from_frame(out[KEY_COLUMNS].astype(str))
    hit = idx.isin(new.index)
    out.loc[hit, METRIC_VALUE_COLUMNS] = new.loc[idx[hit], METRIC_VALUE_COLUMNS].to_numpy()
    kept = sorted(out.loc[~hit, "dataset"].astype(str).unique())
    if kept:
        logger.warning("parcel exclusion: no voxel values for %s; their metrics are kept as exported", kept)
    return out


def apply(metrics: pd.DataFrame, voxels: pd.DataFrame, parcel_ids: np.ndarray, labels: np.ndarray):
    """Drop the voxels of ``labels`` and recompute the participant metrics. Returns (metrics, voxels)."""
    vox = excluded_voxels(parcel_ids, labels)
    logger.info("parcel exclusion: %d atlas labels, %d of %d gray matter voxels removed",
                len(labels), vox.size, parcel_ids.size)  # fmt: skip
    voxels = drop_voxels(voxels, vox)
    return recompute_metrics(metrics, voxels), voxels


def expand(values: np.ndarray, vox_ids: np.ndarray, n_voxels: int, fill: float = 0.0) -> np.ndarray:
    """Vector over ``vox_ids`` back to all ``n_voxels`` mask voxels (``fill`` elsewhere)."""
    out = np.full(n_voxels, fill, dtype=float)
    out[vox_ids] = values
    return out
