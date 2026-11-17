"""Test-retest reliability of analytical and synthetic CMRO2 across Glasser parcels (Figure S2).

Participants scanned twice (session 1: their maps in the main test set; session 2: the ``test_retest``
dataset, two scans within one session) are compared per Glasser parcel:

1. per participant, session, and map (analytical CMRO2 or a model's synthetic CMRO2), the median over the
   parcel's gray matter voxels (non-positive values excluded, as in Figure 2);
2. per parcel, the ICC(2,1) across participants (targets) and the two scans (raters): two-way random
   effects, absolute agreement, single measurement (Pingouin 'ICC2'; Shrout & Fleiss, 1979);
3. each model's parcel ICCs versus those of analytical CMRO2: two-sided Wilcoxon signed-rank test paired by
   parcel, FDR-corrected across models (as for the paired model comparisons of Figures 2b and 3a).

Parcels, rather than voxels, are the unit because the reliability benchmark of analytical CMRO2 (Epp et al.,
2026c) was estimated on parcels, and neighboring voxels are correlated, which inflates voxel-wise statistics.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .stats import fdr_bh, wilcoxon_paired

ANALYTICAL = "analytical"
# Koo & Li (2016) bands for single-measure ICCs
BANDS = {
    "poor": (-np.inf, 0.5),
    "moderate": (0.5, 0.75),
    "good": (0.75, 0.9),
    "excellent": (0.9, np.inf),
}


def icc2_1(y: np.ndarray) -> np.ndarray:
    """ICC(2,1) of every column of ``y`` (targets x raters x columns); NaN marks a missing value.

    Targets with a missing rating are dropped for that column (listwise, like Pingouin with dropped NaN
    rows). Columns with fewer than two complete targets give NaN.
    """
    y = np.asarray(y, dtype=float)
    if y.ndim == 2:
        y = y[:, :, None]
    _, k, _ = y.shape
    ok = np.isfinite(y).all(axis=1)  # targets x columns
    n = ok.sum(axis=0).astype(float)
    x = np.where(ok[:, None, :], y, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        grand = x.sum(axis=(0, 1)) / (n * k)
        row_mean = x.sum(axis=1) / k
        col_mean = x.sum(axis=0) / n
        ss_total = (((x - grand) * ok[:, None, :]) ** 2).sum(axis=(0, 1))
        ss_rows = k * (((row_mean - grand) ** 2) * ok).sum(axis=0)
        ss_cols = n * ((col_mean - grand) ** 2).sum(axis=0)
        ms_r = ss_rows / (n - 1)
        ms_c = ss_cols / (k - 1)
        ms_e = (ss_total - ss_rows - ss_cols) / ((n - 1) * (k - 1))
        icc = (ms_r - ms_e) / (ms_r + (k - 1) * ms_e + k * (ms_c - ms_e) / n)
    icc[n < 2] = np.nan
    return icc


def session_maps(
    session1: pd.DataFrame, session2: pd.DataFrame, positive_only: bool = True
) -> tuple[dict[str, np.ndarray], list[str], list[str]]:
    """Arrays participants x 2 scans x voxels for analytical CMRO2 and each model's synthetic CMRO2.

    ``session1``/``session2``: voxel-table rows (subj_id, model, vox_id, CMRO2, synthCMRO2) of the two scans.
    Only participants and models present in both are used. Analytical CMRO2 must be identical in the rows of
    every model (one target map per participant and scan).
    """
    subjects = sorted(set(session1["subj_id"].astype(str)) & set(session2["subj_id"].astype(str)))
    models = [m for m in pd.unique(session1["model"].astype(str)) if m in set(session2["model"].astype(str))]
    if len(subjects) < 2 or not models:
        raise ValueError(f"need >= 2 participants and >= 1 model in both scans (got {subjects}, {models})")
    n_vox = int(max(session1["vox_id"].max(), session2["vox_id"].max())) + 1

    def wide(d: pd.DataFrame, model: str, column: str) -> np.ndarray:
        t = d[(d["model"].astype(str) == model) & d["subj_id"].astype(str).isin(subjects)]
        t = t.assign(subj_id=t["subj_id"].astype(str))
        t = t.pivot_table(index="subj_id", columns="vox_id", values=column, aggfunc="first", observed=True)
        return t.reindex(index=subjects, columns=range(n_vox)).to_numpy(float)

    maps = {}
    for column, keys in (("CMRO2", [(ANALYTICAL, models[0])]), ("synthCMRO2", [(m, m) for m in models])):
        for key, model in keys:
            y = np.stack([wide(session1, model, column), wide(session2, model, column)], axis=1)
            if positive_only:
                y[y <= 0] = np.nan
            maps[key] = y
    for m in models[1:]:
        for d in (session1, session2):
            if not np.allclose(wide(d, models[0], "CMRO2"), wide(d, m, "CMRO2"), equal_nan=True):
                raise ValueError(f"analytical CMRO2 differs between the rows of {models[0]} and {m}")
    return maps, subjects, models


def parcel_medians(y: np.ndarray, parcel_ids: np.ndarray, parcels: np.ndarray) -> np.ndarray:
    """participants x scans x voxels -> participants x scans x parcels (median over each parcel's voxels)."""
    with warnings.catch_warnings():  # a parcel without valid voxels in a map is NaN (dropped from its ICC)
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.stack([np.nanmedian(y[:, :, parcel_ids == p], axis=2) for p in parcels], axis=2)


def usable_parcels(parcel_ids: np.ndarray, min_voxels: int) -> np.ndarray:
    """Glasser labels (> 0) with at least ``min_voxels`` gray matter mask voxels."""
    labels, counts = np.unique(parcel_ids[parcel_ids > 0], return_counts=True)
    return labels[counts >= min_voxels]


def parcel_icc(maps: dict[str, np.ndarray], parcel_ids: np.ndarray, min_voxels: int) -> pd.DataFrame:
    """Long table: map ('analytical' or model), parcel, ICC, n_targets (participants with both scans)."""
    n_vox = next(iter(maps.values())).shape[2]
    if len(parcel_ids) != n_vox:
        raise ValueError(f"{len(parcel_ids)} parcel labels for {n_vox} voxels: wrong mask or atlas")
    parcels = usable_parcels(parcel_ids, min_voxels)
    rows = []
    for key, y in maps.items():
        yp = parcel_medians(y, parcel_ids, parcels)
        rows.append(pd.DataFrame({"model": key, "parcel": parcels, "ICC": icc2_1(yp),
                                  "n_targets": np.isfinite(yp).all(axis=1).sum(axis=0)}))  # fmt: skip
    return pd.concat(rows, ignore_index=True)


def summarize_icc(icc: pd.DataFrame) -> pd.DataFrame:
    """Per map: number of parcels, median [IQR], mean, and % of parcels in each reliability band."""
    rows = []
    for key, g in icc.groupby("model", sort=False):
        v = g["ICC"].dropna()
        row = {"model": key, "n_parcels": len(v), "median": v.median(), "q1": v.quantile(0.25),
               "q3": v.quantile(0.75), "mean": v.mean(), "sd": v.std(ddof=1)}  # fmt: skip
        for band, (lo, hi) in BANDS.items():
            row[f"pct_{band}"] = 100 * ((v >= lo) & (v < hi)).mean()
        rows.append(row)
    return pd.DataFrame(rows)


def compare_with_analytical(icc: pd.DataFrame) -> pd.DataFrame:
    """Each model's parcel ICCs versus analytical CMRO2 (Wilcoxon signed-rank, FDR across models)."""
    wide = icc.pivot(index="parcel", columns="model", values="ICC")
    rows = []
    for m in [c for c in pd.unique(icc["model"]) if c != ANALYTICAL]:
        pair = wide[[m, ANALYTICAL]].dropna()
        diff = pair[m] - pair[ANALYTICAL]
        rows.append({"model": m, "median_diff": diff.median(), "mean_diff": diff.mean(),
                     "pct_model_higher": 100 * (diff > 0).mean(),
                     **wilcoxon_paired(pair[m].to_numpy(), pair[ANALYTICAL].to_numpy())})  # fmt: skip
    out = pd.DataFrame(rows)
    out["p_fdr"] = fdr_bh(out["p"].to_numpy())
    return out
