"""Tests for input loading and validation, using small synthetic tables."""

import numpy as np
import pandas as pd
import pytest

from synthcmro2.figures.data import (
    METRIC_COLUMNS,
    DataValidationError,
    group_map,
    load_participant_metrics,
    load_voxel_values,
    paired_by_subject,
)


def _voxel_table(n_subjects=3, n_voxels=4):
    rows = []
    for s in range(n_subjects):
        for v in range(n_voxels):
            rows.append(
                {
                    "subj_id": f"sub-{s}",
                    "dataset": "test_set",
                    "condition": "rest",
                    "model": "reference",
                    "vox_id": v,
                    "CMRO2": 100.0 + 10 * v + s,
                    "synthCMRO2": 90.0 + 10 * v + s,
                }
            )
    return pd.DataFrame(rows)


def _metrics_table():
    base = {c: 0.5 for c in METRIC_COLUMNS}
    rows = []
    for s, (age, r_rest, r_calc) in enumerate([(25, 0.9, 0.8), (30, 0.85, 0.86), (70, 0.7, 0.75)]):
        for dataset, cond, r in [("test_set", "rest", r_rest), ("calculation", "calc", r_calc)]:
            rows.append(
                {
                    **base,
                    "subj_id": f"sub-{s}",
                    "dataset": dataset,
                    "sex": "f",
                    "age": age,
                    "condition": cond,
                    "model": "reference",
                    "r": r,
                }
            )
    return pd.DataFrame(rows)


def test_voxel_exact_duplicates_are_removed(tmp_path):
    table = _voxel_table()
    path = tmp_path / "vox.csv"
    pd.concat([table, table.iloc[:4]]).to_csv(path, index=False)
    loaded = load_voxel_values(path)
    assert len(loaded) == len(table)


def test_voxel_conflicting_duplicates_raise(tmp_path):
    table = _voxel_table()
    conflict = table.iloc[[0]].assign(CMRO2=999.0)
    path = tmp_path / "vox.csv"
    pd.concat([table, conflict]).to_csv(path, index=False)
    with pytest.raises(DataValidationError, match="conflicting"):
        load_voxel_values(path)


def test_voxel_unequal_map_sizes_raise(tmp_path):
    path = tmp_path / "vox.csv"
    _voxel_table().iloc[:-1].to_csv(path, index=False)
    with pytest.raises(DataValidationError, match="voxel count"):
        load_voxel_values(path)


def test_metrics_missing_column_raises(tmp_path):
    path = tmp_path / "metrics.csv"
    _metrics_table().drop(columns="MAE").to_csv(path, index=False)
    with pytest.raises(DataValidationError, match="missing columns"):
        load_participant_metrics(path)


def test_metrics_duplicate_key_raises(tmp_path):
    table = _metrics_table()
    path = tmp_path / "metrics.csv"
    pd.concat([table, table.iloc[[0]]]).to_csv(path, index=False)
    with pytest.raises(DataValidationError, match="share"):
        load_participant_metrics(path)


def test_group_map_is_voxelwise_median():
    gmap = group_map(_voxel_table(), "test_set", "reference")
    np.testing.assert_allclose(gmap["CMRO2"], [101, 111, 121, 131])
    assert (gmap["n_subjects"] == 3).all()


def test_group_map_drops_non_positive_values():
    table = _voxel_table()
    table.loc[(table.subj_id == "sub-0") & (table.vox_id == 0), "synthCMRO2"] = -1
    gmap = group_map(table, "test_set", "reference")
    assert gmap.loc[0, "n_subjects"] == 2


def test_pairing_uses_subject_id_not_row_order():
    table = _metrics_table().sample(frac=1, random_state=3)  # shuffle rows
    wide = paired_by_subject(
        table, "r", "condition", ["calc", "rest"], dataset=["test_set", "calculation"], model="reference"
    )
    assert wide.loc["sub-1", "calc"] == pytest.approx(0.86)
    assert wide.loc["sub-1", "rest"] == pytest.approx(0.85)
