"""Shareable tables for the public repository (``synthcmro2 public-tables``).

The figures and statistics of the manuscript are reproduced from these tables alone. They contain no individual
voxel values and no individual maps:

participants.tsv          one row per analyzed participant: source cohort, role, sex, age, analysis subsets.
                          Age is withheld for the cohorts in ``public.age_withheld``, except for test-set
                          participants (age-related decline, Figure 4b and Table S4).
participant_metrics.csv   participant-wise agreement statistics (r, slope, MAE, median CMRO2, ...) of the
                          evaluated datasets, plus ``MAE_parcel`` (Table S2); the datasets in
                          ``public.drop_datasets`` (training-set predictions, second scans) are not included.
group_voxel_values.csv.gz voxel-wise medians across participants of analytical and synthetic CMRO2 per dataset,
                          condition and model, with the number of participants per voxel and the
                          between-participant SD of analytical CMRO2 (Table S2).
group_parcel_sd.csv       between-participant SD of analytical parcel medians per panel (Table S2, parcel level).
parcel_icc.csv            test-retest ICC(2,1) per Glasser parcel and map (Figure S2).
age_calibration.csv       age calibration fitted in the out-of-fold training predictions (Figure 4b).
pls/*.npz                 PLS results of Figure 3b without the input maps (bootstrap ratios, p value,
                          per-observation scores and labels).

The tables are written after the parcel exclusion of the configuration (``--exclude-parcels``), so the public
figures need no exclusion step.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .benchmark import panel_inputs, panels
from .config import Config
from .data import GROUP_KEY_COLUMNS
from .pls import load_pls_result, match_observations, save_pls_summary, task_subjects
from .stats import age_bias_calibration
from .surfaces import voxel_parcel_ids

logger = logging.getLogger(__name__)
FLOAT_FORMAT = "%.10g"


def group_voxel_table(voxels: pd.DataFrame) -> pd.DataFrame:
    """Voxel-wise median (and analytical SD) across participants of every dataset/condition/model map.

    Non-positive analytical or synthetic values are excluded per participant, as in ``data.group_map``.
    """
    pos = voxels[(voxels["CMRO2"] > 0) & (voxels["synthCMRO2"] > 0)]
    g = pos.groupby([*GROUP_KEY_COLUMNS, "vox_id"], observed=True)
    out = g[["CMRO2", "synthCMRO2"]].median()
    out["n_subjects"] = g["subj_id"].nunique()
    out["CMRO2_sd"] = g["CMRO2"].std(ddof=1)
    return out.reset_index()


def cohort_of(subject: str, cohorts: dict[str, str]) -> str:
    hits = [name for name, pattern in cohorts.items() if re.search(pattern, subject)]
    if len(hits) != 1:
        raise ValueError(f"participant {subject}: matches cohorts {hits} (public.cohorts)")
    return hits[0]


def participants_table(metrics: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """One row per analyzed participant, with age withheld as configured."""
    d, pub = cfg.datasets, cfg.public
    roles = {d["calibration"]: "training", d["test"]: "test", d["replication"]: "same-protocol",
             d["reproducibility"]: "cross-vendor"}  # fmt: skip
    rows = metrics[metrics["dataset"].astype(str).isin(roles)]
    info = rows.groupby("subj_id", observed=True).agg(
        dataset=("dataset", lambda s: roles[str(s.iloc[0])]), sex=("sex", "first"), age=("age", "first")
    )
    if rows.groupby("subj_id", observed=True)["dataset"].nunique().max() > 1:
        raise ValueError("a participant appears in more than one of the training, test and external datasets")
    info.index = info.index.astype(str)
    info.insert(0, "cohort", [cohort_of(s, pub["cohorts"]) for s in info.index])
    info = info.rename(columns={"dataset": "role"})
    for column, dataset in (("calculation_task", d["task"]), ("resting_state_fmri", d["fmri"]),
                            ("second_scan", d.get("retest"))):  # fmt: skip
        members = set(metrics.loc[metrics["dataset"].astype(str) == str(dataset), "subj_id"].astype(str))
        info[column] = info.index.isin(members)
    withheld = info["cohort"].isin(pub.get("age_withheld", [])) & (info["role"] != "test")
    info.loc[withheld, "age"] = np.nan
    info["sex"] = info["sex"].replace({"w": "f"})
    order = list(pub["cohorts"])
    info["_c"] = info["cohort"].map(order.index)
    info["_r"] = info["role"].map(["training", "test", "same-protocol", "cross-vendor"].index)
    info = info.sort_values(["_c", "_r"]).drop(columns=["_c", "_r"])
    info.index.name = "participant_id"
    return info.reset_index()


def write_public_tables(metrics: pd.DataFrame, voxels: pd.DataFrame, cfg: Config, out: Path) -> list[Path]:
    """Write the public tables (module docstring) to ``out``; returns the written paths."""
    if not cfg.public.get("cohorts"):
        raise ValueError("config: the `public` section with `cohorts` is needed for public-tables")
    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    d = cfg.datasets
    parcel_ids = voxel_parcel_ids(cfg.surface)
    min_voxels = int(cfg.reliability.get("min_parcel_voxels", 5))

    # participants
    participants = participants_table(metrics, cfg)
    path = out / "participants.tsv"
    participants.to_csv(path, sep="\t", index=False, na_rep="n/a")
    written.append(path)

    # Table S2 parcel level: participant-wise MAE between parcel medians and the SD of analytical parcel medians
    mae_rows, sd_rows = [], []
    for _, dataset, condition in panels(cfg):
        sub = voxels[(voxels["dataset"].astype(str) == dataset) & (voxels["condition"].astype(str) == condition)]
        sub = sub[(sub["CMRO2"] > 0) & (sub["synthCMRO2"] > 0)]
        models = [m for m in cfg.models if m in set(sub["model"].astype(str))]
        if not models:
            continue
        _, sd_parcel, mae_parcel = panel_inputs(sub, models, parcel_ids, min_voxels)
        sd_rows.append(pd.DataFrame({"dataset": dataset, "condition": condition, "parcel": sd_parcel.index,
                                     "CMRO2_sd": sd_parcel.to_numpy()}))  # fmt: skip
        for model, mae in mae_parcel.items():
            mae_rows.append(pd.DataFrame({"subj_id": mae.index.astype(str), "dataset": dataset,
                                          "condition": condition, "model": model, "MAE_parcel": mae.to_numpy()}))  # fmt: skip
    path = out / "group_parcel_sd.csv"
    pd.concat(sd_rows, ignore_index=True).to_csv(path, index=False, float_format=FLOAT_FORMAT)
    written.append(path)

    # participant metrics
    shared = metrics[~metrics["dataset"].astype(str).isin(cfg.public.get("drop_datasets", []))].copy()
    keys = ["subj_id", "dataset", "condition", "model"]
    shared = shared.merge(pd.concat(mae_rows), on=keys, how="left")
    withheld = set(participants.loc[participants["age"].isna(), "participant_id"])
    shared.loc[shared["subj_id"].astype(str).isin(withheld), "age"] = np.nan
    shared["sex"] = shared["sex"].replace({"w": "f"})
    path = out / "participant_metrics.csv"
    shared.to_csv(path, index=False, float_format=FLOAT_FORMAT)
    written.append(path)

    # group voxel maps of the shared datasets
    group = group_voxel_table(voxels[voxels["dataset"].astype(str).isin(set(shared["dataset"].astype(str)))])
    path = out / "group_voxel_values.csv.gz"
    group.to_csv(path, index=False, float_format=FLOAT_FORMAT)
    written.append(path)

    # Figure S2: parcel ICCs
    if cfg.has_retest:
        from .manuscript import make_figure_s2

        s2 = make_figure_s2(metrics, voxels, cfg)
        plt.close(s2.figure)
        path = out / "parcel_icc.csv"
        s2.tables["figS2_parcel_icc"].to_csv(path, index=False, float_format=FLOAT_FORMAT)
        written.append(path)

    # Figure 4b: age calibration fitted in the out-of-fold training predictions
    train = metrics[(metrics["dataset"].astype(str) == d["calibration"])
                    & (metrics["model"].astype(str) == cfg.minimal_model)]  # fmt: skip
    if not train.empty:
        fit = age_bias_calibration(train)
        fit = {k: v for k, v in fit.items() if k not in ("age_min", "age_max")}  # would reveal individual ages
        path = out / "age_calibration.csv"
        pd.DataFrame([{"dataset": d["calibration"], "model": cfg.minimal_model, **fit}]).to_csv(
            path, index=False, float_format=FLOAT_FORMAT
        )
        written.append(path)

    # Figure 3b: PLS results without the input maps
    p = cfg.pls
    subjects = task_subjects(voxels, cfg)
    datasets = [d["test"], d["task"]]
    first_model = next(iter(p["model_tags"]))
    for tag, model, column in [(p["analytical_tag"], first_model, "CMRO2"),
                               *[(t, m, "synthCMRO2") for m, t in p["model_tags"].items()]]:  # fmt: skip
        result = load_pls_result(cfg.pls_file(tag), p["latent_variable"])
        obs = match_observations(result, voxels, column, model, datasets, subjects)
        path = save_pls_summary(result, obs, out / "pls" / cfg.pls_file(tag).name)
        written.append(path)

    _check_no_individual_voxels(out)
    return written


def _check_no_individual_voxels(out: Path) -> None:
    """Guard: no written table may hold a participant column together with voxel IDs."""
    for path in [*out.glob("*.csv"), *out.glob("*.csv.gz"), *out.glob("*.tsv")]:
        sep = "\t" if path.suffix == ".tsv" else ","
        cols = set(pd.read_csv(path, sep=sep, nrows=0).columns)
        if "vox_id" in cols and ({"subj_id", "participant_id"} & cols):
            raise RuntimeError(f"{path}: individual voxel values must not be shared")

