"""Figure tables: same definitions as the notebook cell (pingouin regression, NiftiMasker voxel order)."""

from __future__ import annotations

import nibabel as nib
import numpy as np
import pandas as pd
import pytest
from conftest import write_nifti

from synthcmro2.model.config import parse_config
from synthcmro2.model.data import output_paths
from synthcmro2.model.export import STATS_COLUMNS, VOXEL_COLUMNS, export_tables, regression_stats

KEYS = ["subj_id", "dataset", "condition", "model"]


def test_regression_matches_scipy_definitions():
    """OLS of synthetic on analytical CMRO2 (simulated maps): r, slope, intercept CI (90%) and MAE."""
    from scipy import stats

    rng = np.random.default_rng(1)
    a = rng.uniform(50, 250, 2000)
    s = 0.9 * a + 15 + rng.normal(0, 12, a.size)
    got = regression_stats(a, s)
    lr = stats.linregress(a, s)
    half = stats.t.ppf(0.95, a.size - 2) * lr.intercept_stderr
    assert got["r"] == pytest.approx(abs(lr.rvalue))
    assert got["slope"] == pytest.approx(lr.slope)
    assert (got["CI_low"], got["CI_up"]) == pytest.approx((lr.intercept - half, lr.intercept + half))
    assert got["MAE"] == pytest.approx(np.mean(np.abs(a - s)))


def test_r_is_unsigned_and_ci_is_for_the_intercept():
    rng = np.random.default_rng(0)
    a = rng.uniform(50, 200, 500)
    s = 300 - a + rng.normal(0, 5, 500)
    out = regression_stats(a, s)
    assert out["r"] > 0.9 and out["slope"] < 0  # as in the notebook: r = sqrt(R²)
    assert out["CI_low"] < 300 < out["CI_up"]  # intercept, not slope


def test_export_tables(project, tmp_path):
    raw = project["raw"]
    raw["testing"]["datasets"]["calculation"].update(export_models=["cbf_t2s"])
    raw["testing"]["datasets"]["test_set"].update(export_models=["cbf_t2s", "t1"])
    raw["testing"]["datasets"]["fmri_subset"].update(participants_info="participants.tsv")
    cfg, mask, test = parse_config(raw, tmp_path), project["mask"], project["test"]
    pd.DataFrame({"participant_id": test, "sex": ["f", "m"], "age": [30, 70]}).to_csv(
        tmp_path / "participants.tsv", sep="\t", index=False
    )
    rng, targets = np.random.default_rng(3), {}
    with pytest.raises(FileNotFoundError, match="run `predict` first"):
        export_tables(cfg, mask)
    for ds in cfg.datasets.values():
        for m in ds.export_models:
            for sid in ds.ids:  # the analytical map is the same for every model of a dataset
                p = output_paths(cfg, ds, cfg.models[m], sid)
                real = targets.setdefault((ds.key, sid), rng.uniform(50, 200, mask.shape) * mask)
                write_nifti(p["target_masked"], real)
                write_nifti(p["synthetic"], real * 0.8 + 10)
    stats_df, vox_df = export_tables(cfg, mask)
    assert list(stats_df.columns) == STATS_COLUMNS and list(vox_df.columns) == VOXEL_COLUMNS
    assert set(map(tuple, stats_df[["dataset", "model"]].drop_duplicates().to_numpy())) == {
        ("test_set", "CBF & T2*"), ("test_set", "T1"), ("calculation", "CBF & T2*"), ("fmri_subset", "fMRI"),
    }  # fmt: skip
    assert stats_df.slope.to_numpy() == pytest.approx(0.8)
    assert (vox_df.groupby(KEYS).size() == mask.sum()).all()
    fmri = stats_df[stats_df.dataset == "fmri_subset"].set_index("subj_id")
    assert fmri.loc[test[1], "age"] == 70 and fmri.loc[test[0], "sex"] == "f"
    # voxel order = NiftiMasker order (C order of the mask)
    sid = test[0]
    path = output_paths(cfg, cfg.datasets["fmri_subset"], cfg.models["degree_centrality"], sid)["target_masked"]
    real = np.asarray(nib.load(str(path)).get_fdata())[mask]
    one = vox_df[(vox_df.dataset == "fmri_subset") & (vox_df.subj_id == sid)]
    assert np.allclose(one.sort_values("vox_id").CMRO2.to_numpy(), real)
