"""Load, validate, and reshape the two input tables.

participant metrics (one row per participant x dataset x condition x model)
    subj_id, dataset, sex, age, condition, model, r, p, variance, adj_r2, slope,
    CI_low, CI_up, MAE, CMRO2, synthCMRO2 (optional: MAE_parcel, see ``benchmark``)

voxel values, in one of two forms:
    individual (``synthcmro2 model export``; one row per participant x dataset x condition x model x voxel)
        subj_id, dataset, condition, model, vox_id, CMRO2, synthCMRO2
    group (public data, ``synthcmro2 public-tables``; one row per dataset x condition x model x voxel)
        dataset, condition, model, vox_id, CMRO2, synthCMRO2, n_subjects, CMRO2_sd
        CMRO2 and synthCMRO2 are the voxel-wise medians across participants (non-positive values excluded per
        participant), CMRO2_sd the between-participant SD of analytical CMRO2 (Table S2).

Validation is deliberately strict: a figure built on silently malformed input
is worse than no figure.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

KEY_COLUMNS = ["subj_id", "dataset", "condition", "model"]
METRIC_COLUMNS = [
    *KEY_COLUMNS,
    "sex",
    "age",
    "r",
    "p",
    "variance",
    "adj_r2",
    "slope",
    "CI_low",
    "CI_up",
    "MAE",
    "CMRO2",
    "synthCMRO2",
]
VOXEL_COLUMNS = [*KEY_COLUMNS, "vox_id", "CMRO2", "synthCMRO2"]
GROUP_KEY_COLUMNS = ["dataset", "condition", "model"]
GROUP_VOXEL_COLUMNS = [*GROUP_KEY_COLUMNS, "vox_id", "CMRO2", "synthCMRO2", "n_subjects", "CMRO2_sd"]
VOXEL_DTYPES = {
    "subj_id": "category",
    "dataset": "category",
    "condition": "category",
    "model": "category",
    "vox_id": "int32",
    "CMRO2": "float64",
    "synthCMRO2": "float64",
}


class DataValidationError(ValueError):
    """Raised when an input table does not have the expected structure."""


def _require_columns(df: pd.DataFrame, required: list[str], name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise DataValidationError(f"{name}: missing columns {missing}")


def load_participant_metrics(path: str | Path) -> pd.DataFrame:
    """Read the participant-level table and check its structure.

    Checks: required columns, no duplicated (subject, dataset, condition, model)
    keys, metric columns numeric, Pearson r within [-1, 1], MAE non-negative.
    """
    df = pd.read_csv(path)
    _require_columns(df, METRIC_COLUMNS, "participant metrics")

    duplicated = df.duplicated(KEY_COLUMNS, keep=False)
    if duplicated.any():
        raise DataValidationError(
            f"participant metrics: {duplicated.sum()} rows share a subject/dataset/condition/model key"
        )
    numeric = ["age", "r", "slope", "MAE", "CMRO2", "synthCMRO2"]
    for col in numeric:
        df[col] = pd.to_numeric(df[col], errors="raise")
    if not df["r"].between(-1, 1).all():
        raise DataValidationError("participant metrics: r outside [-1, 1]")
    if (df["MAE"] < 0).any():
        raise DataValidationError("participant metrics: negative MAE")

    n_missing_age = df["age"].isna().sum()
    if n_missing_age:  # expected in the public data: age is shared for the test set only
        logger.info(
            "participant metrics: %d row(s) without age (datasets: %s)",
            n_missing_age,
            ", ".join(sorted(df.loc[df["age"].isna(), "dataset"].astype(str).unique())),
        )
    return df


def is_group_table(voxels: pd.DataFrame) -> bool:
    """True for a group voxel table (medians across participants, no participant column)."""
    return "subj_id" not in voxels.columns


def load_group_voxel_values(path: str | Path) -> pd.DataFrame:
    """Read a group voxel table (public data) and check its structure."""
    df = pd.read_csv(path, dtype={"dataset": "category", "condition": "category", "model": "category"})
    _require_columns(df, GROUP_VOXEL_COLUMNS, "group voxel values")
    if df.duplicated([*GROUP_KEY_COLUMNS, "vox_id"]).any():
        raise DataValidationError("group voxel values: duplicated dataset/condition/model/voxel rows")
    n_maps = df.groupby(GROUP_KEY_COLUMNS, observed=True).ngroups
    logger.info("group voxel values: %d group maps (medians across participants), %d voxels",
                n_maps, df["vox_id"].nunique())  # fmt: skip
    return df


def load_voxel_values(path: str | Path) -> pd.DataFrame:
    """Read the voxel-level table, remove exact duplicate rows, and check structure.

    A table without ``subj_id`` is read as a group table (:func:`load_group_voxel_values`).

    Exact duplicates arise when a model is evaluated twice in an interactive
    session and the results are appended. They are removed with a warning;
    any remaining conflicting duplicates (same key and voxel, different values)
    raise an error. Every participant/model map must have the same number of
    voxels, because voxel IDs index a common gray matter mask.
    """
    header = pd.read_csv(path, nrows=0).columns
    if "subj_id" not in header:
        return load_group_voxel_values(path)
    df = pd.read_csv(path, dtype=VOXEL_DTYPES, usecols=VOXEL_COLUMNS)

    n_exact = int(df.duplicated().sum())
    if n_exact:
        dup_keys = df[df.duplicated()].groupby(KEY_COLUMNS, observed=True).size().reset_index()[KEY_COLUMNS]
        logger.warning(
            "voxel values: removed %d exact duplicate rows (%d maps), e.g. %s",
            n_exact,
            len(dup_keys),
            dup_keys.head(3).to_dict("records"),
        )
        df = df.drop_duplicates().reset_index(drop=True)

    if df.duplicated([*KEY_COLUMNS, "vox_id"]).any():
        raise DataValidationError("voxel values: conflicting values for the same map and voxel")

    n_vox = df.groupby(KEY_COLUMNS, observed=True).size()
    if n_vox.nunique() != 1:
        counts = n_vox.value_counts().to_dict()
        raise DataValidationError(f"voxel values: maps differ in voxel count: {counts}")
    logger.info("voxel values: %d maps x %d voxels", len(n_vox), n_vox.iloc[0])
    return df


def check_metrics_match_voxels(
    metrics: pd.DataFrame, voxels: pd.DataFrame, metrics_only: tuple[str, ...] = ()
) -> None:
    """Both tables must describe the same participant/model maps (except datasets in ``metrics_only``, exported
    without voxel values, e.g. the out-of-fold training predictions)."""
    metrics = metrics[~metrics["dataset"].astype(str).isin(metrics_only)]
    if is_group_table(voxels):
        maps_m = set(map(tuple, metrics[GROUP_KEY_COLUMNS].astype(str).drop_duplicates().to_numpy()))
        maps_v = set(map(tuple, voxels[GROUP_KEY_COLUMNS].astype(str).drop_duplicates().to_numpy()))
        if maps_m != maps_v:
            raise DataValidationError(f"tables disagree: group maps only in metrics {sorted(maps_m - maps_v)}, "
                                      f"only in voxel values {sorted(maps_v - maps_m)}")  # fmt: skip
        n_m = metrics.groupby(GROUP_KEY_COLUMNS, observed=True)["subj_id"].nunique()
        n_v = voxels.groupby(GROUP_KEY_COLUMNS, observed=True)["n_subjects"].max()
        if (n_v > n_m.reindex(n_v.index)).any():
            raise DataValidationError("group voxel values: more participants per voxel than in the metrics")
        return
    keys_m = set(map(tuple, metrics[KEY_COLUMNS].astype(str).to_numpy()))
    keys_v = set(map(tuple, voxels[KEY_COLUMNS].drop_duplicates().astype(str).to_numpy()))
    if keys_m != keys_v:
        raise DataValidationError(
            f"tables disagree: {len(keys_m - keys_v)} maps only in metrics, "
            f"{len(keys_v - keys_m)} only in voxel values"
        )


def select(df: pd.DataFrame, **criteria: str | list[str]) -> pd.DataFrame:
    """Row subset by column equality (scalar) or membership (list)."""
    mask = np.ones(len(df), dtype=bool)
    for col, value in criteria.items():
        values = value if isinstance(value, (list, tuple, set)) else [value]
        mask &= df[col].isin(values).to_numpy()
    out = df.loc[mask]
    if out.empty:
        raise DataValidationError(f"no rows match {criteria}")
    return out


def group_map(
    voxels: pd.DataFrame,
    dataset: str,
    model: str,
    positive_only: bool = True,
    summary: str = "median",
) -> pd.DataFrame:
    """Voxel-wise group map: summary of analytical and synthetic CMRO2 across participants.

    Voxels with non-positive analytical or synthetic CMRO2 are excluded per
    participant before summarizing, because such values are not physiological.

    Returns a frame indexed by ``vox_id`` with columns CMRO2, synthCMRO2 and
    n_subjects (participants contributing to each voxel). A group table (public data) already holds these
    maps, computed the same way; they are returned as stored.
    """
    sub = select(voxels, dataset=dataset, model=model)
    if is_group_table(voxels):
        if summary != "median":
            raise DataValidationError(f"group voxel table stores medians; group_summary {summary!r} needs the "
                                      "individual voxel table")  # fmt: skip
        return sub.set_index("vox_id")[["CMRO2", "synthCMRO2", "n_subjects"]].sort_index()
    if positive_only:
        sub = sub[(sub["CMRO2"] > 0) & (sub["synthCMRO2"] > 0)]
    grouped = sub.groupby("vox_id", observed=True)
    out = grouped[["CMRO2", "synthCMRO2"]].agg(summary)
    out["n_subjects"] = grouped["subj_id"].nunique()
    return out.sort_index()


def paired_by_subject(
    metrics: pd.DataFrame, value: str, column: str, levels: list[str], **criteria: str | list[str]
) -> pd.DataFrame:
    """Wide table (one row per subject, one column per level) for paired tests.

    Pairing is done explicitly on subject ID, never on row order.
    Subjects missing any level are dropped.
    """
    sub = select(metrics, **{column: levels}, **criteria)
    wide = sub.pivot_table(index="subj_id", columns=column, values=value, observed=True)
    wide = wide.reindex(columns=levels).dropna()
    if wide.empty:
        raise DataValidationError(f"no subjects with all of {levels} for {value}")
    return wide
