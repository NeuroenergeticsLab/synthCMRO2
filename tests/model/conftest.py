"""Shared fixtures: the project configuration and a small synthetic project on disk."""

from __future__ import annotations

import copy
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import yaml

from synthcmro2.model.config import parse_config

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "config" / "model.yaml"
DATA = Path(__file__).parent / "data"


@pytest.fixture(scope="session")
def raw_config() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf8"))


@pytest.fixture
def raw(raw_config) -> dict:
    return copy.deepcopy(raw_config)


def write_nifti(path: Path, data: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(data.astype(np.float32), np.diag([4.0, 4.0, 4.0, 1.0])), str(path))


def make_project(tmp_path: Path, raw_config: dict, shape=(24, 24, 24), n_train=4, n_test=2, seed=0) -> dict:
    """Tiny project: synthetic maps for main and fMRI cohorts; CMRO2 depends on CBF and T2*."""
    rng = np.random.default_rng(seed)
    mask = np.zeros(shape, bool)
    mask[4:-4, 4:-4, 4:-4] = True
    write_nifti(tmp_path / "mask.nii.gz", mask.astype(np.float32))
    raw = copy.deepcopy(raw_config)
    train = [f"sub-t{i:02d}" for i in range(n_train)]
    test = [f"sub-x{i:02d}" for i in range(n_test)]
    for d, ids in (("training", train + test), ("testing", test)):
        for sid in ids:
            maps = {
                "CBV": rng.uniform(2, 6, shape), "CBF": rng.uniform(30, 70, shape),
                "T2": rng.uniform(70, 90, shape), "T2S": rng.uniform(40, 60, shape),
                "T1": rng.uniform(200, 900, shape), "fMRI": rng.uniform(0, 50, shape),
            }  # fmt: skip
            maps["CMRO2"] = 2.0 * maps["CBF"] + 0.5 * maps["T2S"]
            for cond in ("rest", "calc"):
                for key, vol in maps.items():
                    rel = raw["files"][key].format(sid=sid, condition=cond)
                    write_nifti(tmp_path / d / sid / rel, vol)
    raw["paths"] = {
        "training_dir": "training", "gm_mask": "mask.nii.gz", "models_dir": "models",
        "selection_dir": "models/model_selection", "results_dir": "results",
    }  # fmt: skip
    raw["cohorts"] = {"main": {"train_ids": train, "fold1_validation": None},
                      "fmri_subset": {"train_ids": train, "fold1_validation": None}}  # fmt: skip
    raw["training"].update(n_folds=2, epochs=2, batch_size=2)
    raw["testing"]["datasets"] = {
        "test_set": {"data_dir": "testing", "condition": "rest", "models": "main", "ids": test},
        "calculation": {"data_dir": "testing", "condition": "calc", "models": "main", "ids": test[:1]},
        "fmri_subset": {"data_dir": "training", "condition": "rest", "models": ["degree_centrality"], "ids": test},
    }
    return {"raw": raw, "cfg": parse_config(raw, tmp_path), "mask": mask, "train": train, "test": test}


@pytest.fixture
def project(tmp_path, raw_config):
    return make_project(tmp_path, raw_config)
