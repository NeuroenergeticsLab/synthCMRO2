"""Parcel-wise test-retest ICC (Figure S2)."""

import numpy as np
import pandas as pd
import pytest

from synthcmro2.figures import reliability as R

# Shrout & Fleiss (1979), Table 2: 6 targets x 4 judges; ICC(2,1) = 0.29
SHROUT_FLEISS = np.array(
    [[9, 2, 5, 8], [6, 1, 3, 2], [8, 4, 6, 8], [7, 1, 2, 6], [10, 5, 6, 9], [6, 2, 4, 7]], dtype=float
)


def test_icc2_1_reproduces_shrout_fleiss():
    assert R.icc2_1(SHROUT_FLEISS)[0] == pytest.approx(0.2898, abs=1e-4)


def test_icc2_1_columns_with_missing_values_match_listwise_deletion():
    rng = np.random.default_rng(0)
    y = rng.normal(size=(9, 2, 30)) + 2 * rng.normal(size=(9, 1, 30))
    y[3, 1, 5] = np.nan
    y[:8, 0, 9] = np.nan  # one complete target left: undefined
    out = R.icc2_1(y)
    for i in range(30):
        ok = np.isfinite(y[:, :, i]).all(axis=1)
        if ok.sum() < 2:
            assert np.isnan(out[i])
        else:
            assert out[i] == pytest.approx(R.icc2_1(y[ok, :, i])[0])


def test_icc2_1_perfect_agreement_is_one_and_rater_offset_lowers_it():
    t = np.linspace(100, 200, 9)
    assert R.icc2_1(np.column_stack([t, t]))[0] == pytest.approx(1.0)
    assert R.icc2_1(np.column_stack([t, t + 20]))[0] < 1.0  # absolute agreement penalizes a constant offset


def _voxel_rows(dataset, subjects, models, values, condition="rest"):
    rows = []
    for (s, m), (ana, syn) in values.items():
        rows.append(pd.DataFrame({"subj_id": s, "dataset": dataset, "condition": condition, "model": m,
                                  "vox_id": np.arange(ana.size), "CMRO2": ana, "synthCMRO2": syn}))  # fmt: skip
    return pd.concat(rows, ignore_index=True)


def _two_scans(n_sub=8, n_vox=60, noise_model=None, seed=1):
    rng = np.random.default_rng(seed)
    models = ["reference", "CBF"]
    noise_model = noise_model or {"reference": 1.0, "CBF": 20.0}
    subjects = [f"sub-{i:02d}" for i in range(n_sub)]
    truth = 150 + 30 * rng.normal(size=(n_sub, 1)) + 10 * rng.normal(size=(n_sub, n_vox))
    scans = []
    for _ in range(2):
        ana = truth + 5 * rng.normal(size=truth.shape)
        scans.append({(s, m): (ana[i], truth[i] + noise_model[m] * rng.normal(size=n_vox))
                      for i, s in enumerate(subjects) for m in models})  # fmt: skip
    return subjects, models, scans


def test_parcel_icc_and_comparison_detect_less_reliable_model():
    subjects, models, (s1, s2) = _two_scans()
    scan1 = _voxel_rows("test_set", subjects, models, s1)
    scan2 = _voxel_rows("test_retest", subjects, models, s2)
    maps, subs, mods = R.session_maps(scan1, scan2)
    assert subs == subjects and mods == models and maps["analytical"].shape == (8, 2, 60)
    parcel_ids = np.repeat(np.arange(1, 13), 5)  # 12 parcels x 5 voxels
    parcel_ids[:3] = 0  # unlabeled voxels are not used; parcel 1 keeps 2 voxels < 5 and is dropped
    icc = R.parcel_icc(maps, parcel_ids, min_voxels=5)
    assert set(icc["parcel"]) == set(range(2, 13))
    summary = R.summarize_icc(icc).set_index("model")
    assert summary.loc["reference", "median"] > summary.loc["CBF", "median"]
    tests = R.compare_with_analytical(icc).set_index("model")
    assert tests.loc["CBF", "median_diff"] < 0 and tests.loc["CBF", "p_fdr"] < 0.05


def test_session_maps_rejects_differing_analytical_maps():
    subjects, models, (s1, s2) = _two_scans()
    s1[(subjects[0], "CBF")] = (s1[(subjects[0], "CBF")][0] + 1, s1[(subjects[0], "CBF")][1])
    with pytest.raises(ValueError, match="analytical CMRO2 differs"):
        R.session_maps(_voxel_rows("test_set", subjects, models, s1), _voxel_rows("test_retest", subjects, models, s2))


def test_parcel_icc_checks_label_count():
    subjects, models, (s1, s2) = _two_scans()
    maps, _, _ = R.session_maps(_voxel_rows("t", subjects, models, s1), _voxel_rows("r", subjects, models, s2))
    with pytest.raises(ValueError, match="parcel labels"):
        R.parcel_icc(maps, np.ones(10, dtype=int), min_voxels=5)


def test_make_figure_s2_builds_figure_and_tables():
    from pathlib import Path

    import matplotlib.pyplot as plt

    from synthcmro2.figures.config import load_config
    from synthcmro2.figures.manuscript import make_figure_s2

    cfg = load_config(Path(__file__).resolve().parents[2] / "config" / "figures.yaml")
    subjects, models, (s1, s2) = _two_scans()
    out = make_figure_s2(
        None,
        _voxel_rows("test_set", subjects, models, s1),
        cfg,
        retest_voxels=_voxel_rows("test_retest", subjects, models, s2),
        parcel_ids=np.repeat(np.arange(1, 13), 5),
    )
    assert set(out.tables) == {"figS2_parcel_icc", "figS2_icc_summary", "figS2_vs_analytical"}
    assert list(out.tables["figS2_icc_summary"]["model"]) == ["analytical", "reference", "CBF"]
    assert out.figure.axes[0].get_ylim() == tuple(cfg.reliability["ylim"])
    plt.close(out.figure)
