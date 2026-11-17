"""Parcel exclusion (sensitivity analysis): dropped voxels, recomputed metrics, separate output folders."""

from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats

from synthcmro2.figures import exclusion as X
from synthcmro2.figures.config import load_config

ROOT = Path(__file__).resolve().parents[2]


def _write(tmp_path, tag):
    raw = yaml.safe_load((ROOT / "config" / "figures_individual.yaml").read_text(encoding="utf8"))
    raw["paths"]["tag"] = tag
    (tmp_path / "config").mkdir()
    path = tmp_path / "config" / "figures_individual.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf8")
    return path


def _tables(rng):
    rows, mrows = [], []
    for sid in ("s1", "s2"):
        a = rng.uniform(50, 300, 20)
        s = a * 0.9 + rng.normal(0, 10, 20)
        s[:4] += 200  # outlying voxels, as in the excluded parcels
        rows.append(pd.DataFrame({"subj_id": sid, "dataset": "test_set", "condition": "rest", "model": "m",
                                  "vox_id": np.arange(20), "CMRO2": a, "synthCMRO2": s}))  # fmt: skip
        mrows.append({"subj_id": sid, "dataset": "test_set", "condition": "rest", "model": "m", "age": 30,
                      **{c: 0.0 for c in X.METRIC_VALUE_COLUMNS}})  # fmt: skip
    mrows.append({"subj_id": "s9", "dataset": "training_oof", "condition": "rest", "model": "m", "age": 50,
                  **{c: 7.0 for c in X.METRIC_VALUE_COLUMNS}})  # fmt: skip
    return pd.DataFrame(mrows), pd.concat(rows, ignore_index=True)


def test_apply_drops_voxels_and_recomputes_metrics():
    metrics, voxels = _tables(np.random.default_rng(0))
    parcel_ids = np.array([5] * 4 + [1] * 8 + [205] * 2 + [2] * 6)  # voxels 0-3 and 12-13 in parcel 5 (L and R)
    new_m, new_v = X.apply(metrics, voxels, parcel_ids, np.array([5, 205]))
    assert sorted(new_v["vox_id"].unique()) == [*range(4, 12), *range(14, 20)]
    for sid, g in new_v.groupby("subj_id"):
        lr = stats.linregress(g["CMRO2"], g["synthCMRO2"])
        row = new_m[new_m.subj_id == sid].iloc[0]
        assert np.isclose(row["slope"], lr.slope) and np.isclose(row["r"], abs(lr.rvalue))
        assert np.isclose(row["MAE"], np.mean(np.abs(g["CMRO2"] - g["synthCMRO2"])))
        assert np.isclose(row["CMRO2"], np.median(g["CMRO2"]))
    assert (new_m[new_m.dataset == "training_oof"][X.METRIC_VALUE_COLUMNS] == 7.0).all().all()  # no voxels: kept


def test_expand_back_to_mask():
    out = X.expand(np.array([1.0, 2.0]), np.array([1, 3]), 5)
    assert out.tolist() == [0.0, 1.0, 0.0, 2.0, 0.0]


def test_exclusion_outputs_go_to_separate_folders(tmp_path):
    path = _write(tmp_path, "20261107")
    cfg = load_config(path, exclude_parcels=["TE1m", "TE2a"])
    assert cfg.exclude_parcels == ("TE1m", "TE2a")
    assert cfg.output_dir.name == "manuscript_20261107_excl-TE1m-TE2a"
    assert cfg.pls_dir.name == "pls_20261107_excl-TE1m-TE2a"
    assert load_config(path).output_dir.name == "manuscript_20261107"
