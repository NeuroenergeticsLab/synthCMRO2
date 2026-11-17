"""Metrics and validation-based selection."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest
from conftest import DATA, ROOT

from synthcmro2.model.config import load_config
from synthcmro2.model.metrics import gm_metrics, notebook_metrics
from synthcmro2.model.selection import selected_folds, summarize_selection


def test_gm_metrics_known_answer():
    mask = np.ones((4, 4, 4), bool)
    real = np.arange(1, 65, dtype=float).reshape(4, 4, 4)
    m = gm_metrics(real, 2 * real, mask)
    assert m["slope"] == pytest.approx(2) and m["r"] == pytest.approx(1)
    assert m["MAE"] == pytest.approx(real.mean()) and m["n_voxels"] == 64


def test_gm_metrics_exclude_non_positive_voxels():
    mask = np.ones((4, 4, 4), bool)
    real = np.arange(1, 65, dtype=float).reshape(4, 4, 4)
    synth = real.copy()
    synth[0] = -1000  # 16 voxels excluded, otherwise identical
    m = gm_metrics(real, synth, mask)
    assert m["MAE"] == 0 and m["n_voxels"] == 48
    assert notebook_metrics(real, synth, mask)["MSE_gm"] > 0  # notebook metrics keep them


def rows_for(configs):
    rows, fold_rows = [], []
    for cfg, inputs, n, mae in configs:
        for k in range(1, 5):
            fold_rows.append({"config": cfg, "inputs": inputs, "n_inputs": n, "fold": k, "val_generator_loss": 5 - k})
            rows.append({"config": cfg, "inputs": inputs, "n_inputs": n, "fold": k, "subj_id": "s", "MAE": mae + k,
                         "r": 0.5, "slope": 0.5})  # fmt: skip
    return rows, fold_rows


def test_selection_uses_lowest_validation_loss_and_mae():
    rows, fold_rows = rows_for([("_A", "CBF&T2S", 2, 40), ("_B", "CBF&T2", 2, 50), ("_C", "CBF", 1, 55)])
    _, _, cv = summarize_selection(rows, fold_rows, {"CBF&T2S", "CBF&T2", "CBF"})
    assert (cv.selected_fold == 4).all()  # lowest validation generator loss
    assert set(cv.loc[cv.best_for_n_inputs, "config"]) == {"_A", "_C"}
    assert cv.set_index("config").loc["_A", "MAE_mean"] == pytest.approx(42.5)


def test_comparison_models_never_selected():
    rows, fold_rows = rows_for([("_C", "CBF", 1, 50), ("_F", "fMRI", 1, 10), ("_T", "T1", 1, 20)])
    _, _, cv = summarize_selection(rows, fold_rows, {"CBF"})
    assert cv.loc[cv.best_for_n_inputs, "config"].tolist() == ["_C"]


def test_selected_folds_from_file_and_override(caplog):
    cfg = load_config(ROOT / "config" / "model.yaml")
    folds = selected_folds(cfg, DATA / "cv_selection_20261030.csv")
    assert folds == {"reference": 3, "without_cbv": 4, "cbf_t2s": 3, "cbf_t2": 4, "t2_t2s": 3, "cbf": 4,
                     "t2s": 3, "t2": 3, "t1": 2, "degree_centrality": 4}  # fmt: skip
    from dataclasses import replace

    with caplog.at_level(logging.WARNING, logger="synthcmro2.model"):
        assert selected_folds(replace(cfg, fixed_folds={"cbf": 1}), DATA / "cv_selection_20261030.csv")["cbf"] == 1
    assert "instead of the validation-selected fold 4" in caplog.text


def test_user_selection_table_is_consistent():
    cv = pd.read_csv(DATA / "cv_selection_20261030.csv")
    best = cv.loc[cv.best_for_n_inputs, "inputs"].tolist()
    assert sorted(best) == sorted(["CBF", "CBF&T2S", "CBF&T2&T2S", "CBV&CBF&T2&T2S"])


def test_gm_metrics_without_positive_voxels_are_nan():
    mask = np.ones((4, 4, 4), bool)
    m = gm_metrics(np.ones((4, 4, 4)), -np.ones((4, 4, 4)), mask)
    assert np.isnan(m["MAE"]) and m["n_voxels"] == 0 and m["n_pos_analytical"] == 64 and m["n_pos_synthetic"] == 0


def test_metrics_file_is_merged_and_resumable(tmp_path):
    from types import SimpleNamespace

    from synthcmro2.model.metrics import completed_pairs, merge_metrics

    path = tmp_path / "test_metrics.csv"
    ds = SimpleNamespace(key="test_set", ids=("a", "b"))
    merge_metrics(path, pd.DataFrame({"dataset": "test_set", "model": "cbf", "fold": 4, "subj_id": ["a", "b"]}))
    merge_metrics(path, pd.DataFrame({"dataset": "test_set", "model": "t1", "fold": 2, "subj_id": ["a"]}))
    merge_metrics(path, pd.DataFrame({"dataset": "test_set", "model": "cbf", "fold": 4, "subj_id": ["a", "b"]}))
    assert len(pd.read_csv(path)) == 3  # rerun pair replaced, not appended
    assert completed_pairs(path, ds, {"cbf": 4, "t1": 2}) == {"cbf"}  # t1 incomplete
    assert completed_pairs(path, ds, {"cbf": 3}) == set()  # different fold selected now
