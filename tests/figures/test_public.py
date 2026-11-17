"""Public tables: group voxel maps reproduce the individual-table computations, and nothing individual leaks."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from synthcmro2.figures.data import (
    DataValidationError,
    check_metrics_match_voxels,
    group_map,
    is_group_table,
    load_voxel_values,
)
from synthcmro2.figures.public import cohort_of, group_voxel_table

ROOT = Path(__file__).resolve().parents[2]
PUBLIC = ROOT / "data" / "public"


def _individual(seed=0, n_sub=5, n_vox=40):
    rng = np.random.default_rng(seed)
    rows = []
    for model in ("A", "B"):
        for s in range(n_sub):
            a = rng.uniform(-10, 200, n_vox)  # some non-positive voxels, excluded per participant
            rows.append(pd.DataFrame({"subj_id": f"s{s}", "dataset": "test_set", "condition": "rest", "model": model,
                                      "vox_id": np.arange(n_vox), "CMRO2": a,
                                      "synthCMRO2": a * 0.9 + rng.normal(0, 5, n_vox)}))  # fmt: skip
    return pd.concat(rows, ignore_index=True)


def test_group_table_equals_group_map_of_individual_table():
    vox = _individual()
    group = group_voxel_table(vox)
    assert is_group_table(group) and not is_group_table(vox)
    for model in ("A", "B"):
        expected = group_map(vox, "test_set", model)
        got = group_map(group, "test_set", model)
        pd.testing.assert_frame_equal(got, expected, check_dtype=False)
    with pytest.raises(DataValidationError, match="stores medians"):
        group_map(group, "test_set", "A", summary="mean")


def test_group_table_round_trip_and_consistency_check(tmp_path):
    vox = _individual()
    path = tmp_path / "group.csv.gz"
    group_voxel_table(vox).to_csv(path, index=False)
    group = load_voxel_values(path)
    metrics = vox[["subj_id", "dataset", "condition", "model"]].drop_duplicates()
    check_metrics_match_voxels(metrics, group)
    with pytest.raises(DataValidationError, match="tables disagree"):
        check_metrics_match_voxels(metrics[metrics["model"] == "A"], group)


def test_cohorts_are_assigned_uniquely():
    cohorts = {"Young-adults I": "^sub-p0[0-9]{2}$", "Older-adults I": "^sub-p1[0-9]{2}$"}
    assert cohort_of("sub-p021", cohorts) == "Young-adults I"
    with pytest.raises(ValueError):
        cohort_of("sub-x001", cohorts)


@pytest.mark.skipif(not PUBLIC.exists(), reason="public tables not present")
def test_shared_tables_hold_no_individual_voxels_or_withheld_ages():
    for path in [*PUBLIC.glob("*.csv"), *PUBLIC.glob("*.csv.gz"), *PUBLIC.glob("*.tsv")]:
        cols = set(pd.read_csv(path, sep="\t" if path.suffix == ".tsv" else ",", nrows=0).columns)
        assert not ("vox_id" in cols and {"subj_id", "participant_id"} & cols), path.name
    participants = pd.read_csv(PUBLIC / "participants.tsv", sep="\t")
    withheld = participants["cohort"].isin(["Older-adults I", "Older-adults II", "Cross-vendor"])
    assert participants.loc[withheld & (participants["role"] != "test"), "age"].isna().all()
    metrics = pd.read_csv(PUBLIC / "participant_metrics.csv")
    assert not set(metrics["dataset"]) & {"training_oof", "test_retest"}
    no_age = set(participants.loc[participants["age"].isna(), "participant_id"])
    assert metrics.loc[metrics["subj_id"].isin(no_age), "age"].isna().all()
