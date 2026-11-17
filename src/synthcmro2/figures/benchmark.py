"""Model errors versus the test-retest error of analytical CMRO2 (Table S2; ``synthcmro2 benchmark``).

The test-retest reliability of analytical mqBOLD CMRO2 (ICC(2,1) across Glasser parcels, Young-adults II;
Epp et al., 2026c) is converted into an error in CMRO2 units:

    SEM = SD_between-participants * sqrt(1 - ICC)
    noise floor  = sqrt(2/pi) * SEM  (~0.80 SEM): expected MAE of a perfect model against a noisy analytical map
    retest error = 2/sqrt(pi) * SEM  (~1.13 SEM): expected mean absolute difference between two analytical scans

SD_between-participants is computed for each panel's own participants and analytical maps, per gray matter
voxel (the level of the individual MAEs in Figs. 2-4; the parcel ICC is assumed to hold for single voxels) and
per Glasser parcel median (the level of the ICC, with model MAEs recomputed on parcel medians).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config

# Epp et al. (2026c): CMRO2 ICC, median [IQR] across parcels
ICC = {"within-session": (0.89, 0.85, 0.92), "between-day": (0.61, 0.50, 0.68)}
NOISE_FLOOR = np.sqrt(2 / np.pi)  # E|e| for e ~ N(0, SEM^2)
RETEST = 2 / np.sqrt(np.pi)  # E|e1 - e2| for two independent errors


def panels(cfg: Config) -> list[tuple[str, str, str]]:
    """(figure, dataset, condition) of every panel with individual MAEs."""
    d = cfg.datasets
    return [("Fig. 2", d["test"], "rest"), ("Fig. 2", d["fmri"], "rest"), ("Fig. 3", d["task"], cfg.pls["task"]),
            ("Fig. 4", d["replication"], "rest"), ("Fig. 4", d["reproducibility"], "rest")]  # fmt: skip


def panel_inputs(sub: pd.DataFrame, models: list[str], parcel_ids: np.ndarray, min_voxels: int):
    """Inputs of one panel from the individual voxel table (positive voxels of one dataset and condition).

    Returns the between-participant SD of analytical CMRO2 per voxel and per parcel median (analytical maps are
    identical across models, so the first model's rows are used), and per model the participant-wise MAE between
    parcel medians (parcels with at least ``min_voxels`` voxels).
    """
    sub = sub.assign(parcel=parcel_ids[sub["vox_id"].to_numpy()])
    analytical = sub[sub["model"].astype(str) == models[0]]
    sd_voxel = analytical.pivot_table(index="vox_id", columns="subj_id", values="CMRO2", observed=True).std(
        axis=1, ddof=1
    )
    parcel_med = (
        sub[sub["parcel"] > 0]
        .groupby(["model", "subj_id", "parcel"], observed=True)[["CMRO2", "synthCMRO2"]]
        .agg(["median", "size"])
    )
    parcel_med = parcel_med[parcel_med[("CMRO2", "size")] >= min_voxels]
    sd_parcel = parcel_med.loc[models[0], ("CMRO2", "median")].unstack("subj_id").std(axis=1, ddof=1)  # noqa: PD010
    mae_parcel = {}
    for model in models:
        pm = parcel_med.loc[model]
        mae_parcel[model] = (pm[("synthCMRO2", "median")] - pm[("CMRO2", "median")]).abs().groupby(
            "subj_id", observed=True).mean()  # fmt: skip
    return sd_voxel, sd_parcel, mae_parcel


def _row(figure, dataset, condition, model, level, mae: pd.Series, sd: pd.Series) -> dict:
    row = {"figure": figure, "dataset": dataset, "condition": condition, "model": model, "level": level,
           "n_participants": int(mae.size), "mae_median": mae.median(), "mae_q1": mae.quantile(0.25),
           "mae_q3": mae.quantile(0.75), "sd_between_median": sd.median()}  # fmt: skip
    for name, (icc, icc_lo, icc_hi) in ICC.items():
        key = name.replace("-", "_")
        sem = (sd * np.sqrt(1 - icc)).median()
        row[f"sem_{key}"] = sem
        row[f"noise_floor_{key}"] = NOISE_FLOOR * sem
        row[f"retest_{key}"] = RETEST * sem
        row[f"retest_{key}_lo"] = RETEST * (sd * np.sqrt(1 - icc_hi)).median()  # over the ICC IQR
        row[f"retest_{key}_hi"] = RETEST * (sd * np.sqrt(1 - icc_lo)).median()
    row["mae_over_retest_within"] = row["mae_median"] / row["retest_within_session"]
    row["mae_over_retest_between"] = row["mae_median"] / row["retest_between_day"]
    return row


def noise_benchmark(
    metrics: pd.DataFrame,
    voxels: pd.DataFrame,
    parcel_ids: np.ndarray,
    cfg: Config,
    parcel_sd: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """One row per panel, model and level (voxel, parcel): median MAE next to the noise floor and retest
    errors implied by the within-session and between-day ICCs of analytical CMRO2.

    With the individual voxel table everything is computed here. With the group table (public data), the
    voxel-level SD comes from its ``CMRO2_sd`` column, and the parcel level needs ``parcel_sd`` (dataset,
    condition, parcel, CMRO2_sd) and the participant-wise ``MAE_parcel`` column of the metrics; without them only
    the voxel level is reported.
    """
    from .data import is_group_table

    group = is_group_table(voxels)
    min_voxels = int(cfg.reliability.get("min_parcel_voxels", 5))
    rows = []
    for figure, dataset, condition in panels(cfg):
        sub = voxels[(voxels["dataset"].astype(str) == dataset) & (voxels["condition"].astype(str) == condition)]
        present = set(sub["model"].astype(str))
        models = [m for m in cfg.models if m in present]
        if not models:
            continue
        if group:
            sd_voxel = sub[sub["model"].astype(str) == models[0]].set_index("vox_id")["CMRO2_sd"]
            sd_parcel, mae_parcel = None, {}
            if parcel_sd is not None and "MAE_parcel" in metrics.columns:
                ps = parcel_sd[(parcel_sd["dataset"].astype(str) == dataset)
                               & (parcel_sd["condition"].astype(str) == condition)]  # fmt: skip
                sd_parcel = ps.set_index("parcel")["CMRO2_sd"] if not ps.empty else None
        else:
            sub = sub[(sub["CMRO2"] > 0) & (sub["synthCMRO2"] > 0)]
            sd_voxel, sd_parcel, mae_parcel = panel_inputs(sub, models, parcel_ids, min_voxels)
        for model in models:
            m = metrics[(metrics["dataset"].astype(str) == dataset) & (metrics["model"].astype(str) == model)
                        & (metrics["condition"].astype(str) == condition)]  # fmt: skip
            rows.append(_row(figure, dataset, condition, model, "voxel", m["MAE"], sd_voxel))
            if sd_parcel is None:
                continue
            mae = mae_parcel[model] if model in mae_parcel else m.set_index("subj_id")["MAE_parcel"].dropna()
            rows.append(_row(figure, dataset, condition, model, "parcel", mae, sd_parcel))
    return pd.DataFrame(rows)
