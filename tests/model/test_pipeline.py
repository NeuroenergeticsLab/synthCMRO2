"""End to end on a tiny synthetic project: train -> select -> predict (needs PyTorch and MONAI)."""

from __future__ import annotations

import json

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("monai")
pytest.importorskip("pytorch_msssim")

from synthcmro2.model.predict import predict  # noqa: E402
from synthcmro2.model.train import train_model  # noqa: E402
from synthcmro2.model.validate import select  # noqa: E402


def test_train_select_predict(project, tmp_path):
    cfg, mask, test = project["cfg"], project["mask"], project["test"]
    cpu = torch.device("cpu")
    for key in ("cbf_t2s", "degree_centrality"):
        done = train_model(cfg, cfg.models[key], mask, cpu, epochs=2)
        assert sorted(done) == [1, 2] and all(p.exists() for p in done.values())
        hist = pd.read_csv(cfg.checkpoint("history", cfg.models[key], 1))
        assert len(hist) == 2 and hist.saved.any()
        z = json.loads(cfg.checkpoint("zscore", cfg.models[key], 1).read_text())
        assert set(z["zscore"]) == ({"fMRI"} if key == "degree_centrality" else set())
        assert not set(z["train_ids"]) & set(z["val_ids"])

    with pytest.raises(FileExistsError):  # trained models are never overwritten silently
        train_model(cfg, cfg.models["cbf_t2s"], mask, cpu, folds=[1], epochs=1)

    cv = select(cfg, mask, cpu, ["cbf_t2s", "degree_centrality"])
    assert set(cv.config) == {"_OmitCBV_OmitT2_OmitT1", "_OmitCBV_OmitCBF_OmitT2_OmitT2S_OmitT1"}
    assert not cv.loc[cv.inputs == "fMRI", "best_for_n_inputs"].item()  # comparison model

    table = predict(cfg, mask, cpu, model_keys=["cbf_t2s", "degree_centrality"])
    assert set(table.dataset) == {"test_set", "calculation", "fmri_subset"}
    assert set(table.loc[table.dataset == "fmri_subset", "model"]) == {"degree_centrality"}
    assert table[["MAE", "r", "slope", "SSIM"]].notna().all().all()

    sid = test[0]
    synth = (
        cfg.datasets["test_set"].output_dir
        / sid
        / "qmri"
        / f"{sid}_task-rest_space-MNI152_res-4mm_synthetic_cmro2_OmitCBV_OmitT2_OmitT1.nii.gz"
    )
    assert nib.load(str(synth)).shape == mask.shape
    assert np.allclose(nib.load(str(synth)).affine, nib.load(str(cfg.gm_mask)).affine)  # written on the mask grid
    dc = cfg.datasets["fmri_subset"].output_dir / sid / f"resting_preproc_{sid}_ses-1"
    assert (
        dc / f"{sid}_task-rest_space-MNI152_res-4mm_synthetic_cmro2_OmitCBV_OmitCBF_OmitT2_OmitT2S_OmitT1.nii.gz"
    ).exists()
    name = "_OmitCBV_OmitCBF_OmitT2_OmitT2S_OmitT1"
    assert (dc / f"{sid}_task-rest_space-MNI152_res-4mm_desc-orig_cmro2_masked_TRAINING{name}.nii.gz").exists()

    # `apply` (new maps, no analytical CMRO2) gives the same map as `predict` with the selected fold
    from synthcmro2.model.apply import apply_model
    from synthcmro2.model.data import map_path

    ds = cfg.datasets["test_set"]
    inputs = {m: map_path(cfg, ds.data_dir, sid, m, "rest") for m in ("CBF", "T2S")}
    applied = apply_model(cfg, cfg.models["cbf_t2s"], inputs, tmp_path / "applied.nii.gz", device=cpu)
    np.testing.assert_allclose(nib.load(str(applied)).get_fdata(), nib.load(str(synth)).get_fdata(), atol=1e-5)
    with pytest.raises(ValueError, match="z-scored"):
        apply_model(cfg, cfg.models["degree_centrality"], {"fMRI": inputs["CBF"]}, tmp_path / "x.nii.gz")

    from synthcmro2.model.export import export_tables

    # reads exactly what predict wrote (only the two models trained here)
    stats_df, vox_df = export_tables(cfg, mask, ["test_set", "fmri_subset"], ["cbf_t2s", "degree_centrality"])
    assert set(stats_df.model) == {"CBF & T2*", "fMRI"} and len(vox_df) == len(stats_df) * mask.sum()


def test_out_of_fold_prediction_uses_each_participants_validation_fold(tmp_path, raw_config):
    from conftest import make_project

    from synthcmro2.model.config import parse_config
    from synthcmro2.model.folds import cohort_folds

    proj = make_project(tmp_path, raw_config)
    raw = proj["raw"]
    raw["testing"]["datasets"]["training_oof"] = {
        "out_of_fold": True, "output_dir": "oof", "condition": "rest", "models": ["cbf_t2s"], "export_voxels": False,
    }  # fmt: skip
    cfg, mask = parse_config(raw, tmp_path), proj["mask"]
    cpu = torch.device("cpu")
    train_model(cfg, cfg.models["cbf_t2s"], mask, cpu, epochs=1)
    table = predict(cfg, mask, cpu, dataset_keys=["training_oof"], folds={"cbf_t2s": 1})
    folds = cohort_folds(cfg.cohorts["main"], 2, int(cfg.training["seed"]))
    assert sorted(table.subj_id) == sorted(proj["train"])
    for sid, fold in zip(table.subj_id, table.fold, strict=True):
        assert sid in folds[fold - 1][1] and sid not in folds[fold - 1][0]  # its validation fold, never trained on
    assert (cfg.datasets["training_oof"].output_dir / proj["train"][0] / "qmri").exists()
