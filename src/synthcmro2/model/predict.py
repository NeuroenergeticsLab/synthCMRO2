"""Apply the validation-selected fold of each model to the test datasets (3Test_512Bottleneck, 5fMRI_Train).

For each dataset and model: one generator (the selected fold), synthetic CMRO2 saved as NIfTI with the
header of the analytical map, under the same file names as the notebooks, and one row of metrics per
participant. Test results are never used to choose anything.
"""

from __future__ import annotations

import json
import logging
from dataclasses import replace

import nibabel as nib
import numpy as np
import pandas as pd
import torch

from .config import Config, ModelSpec, TestDataset
from .data import SampleLoader, compute_zscore_stats, grid, load_masked, map_path, output_paths
from .folds import cohort_folds
from .losses import ssim_value
from .metrics import completed_pairs, gm_metrics, merge_metrics, notebook_metrics, warn_if_empty
from .naming import NAMED
from .selection import selected_folds
from .validate import load_generator

logger = logging.getLogger("synthcmro2.model")


def test_zscore_stats(cfg: Config, model: ModelSpec, fold: int, mask: np.ndarray):
    """Normalization at test time: all training participants of the cohort (notebooks) or the selected fold's."""
    if cfg.testing.get("zscore_stats", "all_training") == "fold":
        path = cfg.checkpoint("zscore", model, fold)
        if path.exists():
            return {k: tuple(v) for k, v in json.loads(path.read_text())["zscore"].items()}
        logger.warning("%s: %s missing; z-scoring with all training participants", model.key, path)
    return compute_zscore_stats(cfg, model, list(cfg.cohorts[model.cohort].used_ids), mask)


def _save(vol: np.ndarray, ref: nib.Nifti1Image, affine: np.ndarray, path, overwrite: bool = True) -> None:
    """Write ``vol`` with the header of ``ref`` and ``affine``. An existing file is kept (overwrite=False) only
    if it is on the same grid; one with another affine (for example 0.5 mm off the mask) is rewritten."""
    if path.exists() and not overwrite and np.allclose(nib.load(str(path)).affine, affine, atol=1e-6):
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(vol, affine=affine, header=ref.header), str(path))


@torch.no_grad()
def predict_dataset(cfg: Config, ds: TestDataset, model: ModelSpec, fold: int, mask: np.ndarray, device) -> list[dict]:
    gen = load_generator(cfg.checkpoint("generator", model, fold), len(model.inputs), device)
    loader = SampleLoader(
        cfg, model, ds.data_dir, mask, test_zscore_stats(cfg, model, fold, mask), ds.condition, cache=False
    )
    rows, mask_grid = [], grid(cfg)
    for sid in ds.ids:
        x, y = loader(sid)
        y_hat = gen(torch.from_numpy(x)[None].to(device))
        synth = y_hat.cpu().numpy()[0, 0]
        ref = nib.load(str(map_path(cfg, ds.data_dir, sid, "CMRO2", ds.condition)))
        affine = mask_grid[0] if cfg.grid["output_affine"] == "mask" else ref.affine
        paths = output_paths(cfg, ds, model, sid)
        _save(synth, ref, affine, paths["synthetic"])
        _save(y[0], ref, affine, paths["target_masked"], overwrite=False)
        if cfg.testing.get("save_masked_inputs", False) and model.output_subdir == "qmri":
            for m in NAMED:  # as the notebook: every available qMRI map, masked
                src = map_path(cfg, ds.data_dir, sid, m, ds.condition)
                if src.exists():
                    name = cfg.outputs["input_masked"].format(sid=sid, condition=ds.condition, modality=m.lower())
                    dst = ds.output_dir / sid / "qmri" / name
                    _save(load_masked(src, mask, mask_grid), ref, affine, dst, overwrite=False)
        metrics = gm_metrics(y[0], synth, mask)
        warn_if_empty(metrics, f"{ds.key} {model.key} {sid}", logger)
        rows.append({
            "dataset": ds.key, "condition": ds.condition, "model": model.key, "config": model.name,
            "inputs": model.label, "fold": fold, "subj_id": sid, **metrics,
            **notebook_metrics(y[0], synth, mask),
            "SSIM": float(ssim_value(y_hat, torch.from_numpy(y)[None].to(device))),
        })  # fmt: skip
    return rows


def out_of_fold_folds(cfg: Config, ds: TestDataset, model: ModelSpec) -> dict[str, int]:
    """Participant -> the fold (1-based) in which it was a validation participant, i.e. the one fold model that
    was not trained on it."""
    folds = cohort_folds(cfg.cohorts[model.cohort], int(cfg.training["n_folds"]), int(cfg.training["seed"]))
    fold_of = {sid: k for k, (_, val) in enumerate(folds, start=1) for sid in val}
    missing = [sid for sid in ds.ids if sid not in fold_of]
    if missing:
        raise RuntimeError(f"{ds.key}: {missing} are in no validation fold of {model.key}")
    return {sid: fold_of[sid] for sid in ds.ids}


def _out_of_fold_done(metrics_file, ds: TestDataset, mkey: str, fold_of: dict[str, int]) -> bool:
    if metrics_file is None or not metrics_file.exists():
        return False
    old = pd.read_csv(metrics_file)
    rows = old[(old.dataset == ds.key) & (old.model == mkey)]
    return set(rows.subj_id) == set(ds.ids) and all(fold_of[s] == f for s, f in zip(rows.subj_id, rows.fold))


def predict(
    cfg: Config, mask: np.ndarray, device, dataset_keys=None, model_keys=None, folds=None,
    metrics_file=None, resume: bool = False,
) -> pd.DataFrame:  # fmt: skip
    """Apply each model's selected fold. With ``metrics_file`` the metrics are saved after every dataset/model
    pair, so an interrupted run keeps what it finished; ``resume`` skips the pairs already complete there.

    Out-of-fold datasets (training participants) are predicted per participant by the fold model for which it
    was a validation participant, instead of the selected fold."""
    chosen = selected_folds(cfg) | (folds or {})
    tables = []
    for dkey in dataset_keys or list(cfg.datasets):
        ds = cfg.datasets[dkey]
        done = completed_pairs(metrics_file, ds, chosen) if resume and not ds.out_of_fold else set()
        for mkey in ds.models:
            if model_keys and mkey not in model_keys:
                continue
            model = cfg.models[mkey]
            if ds.out_of_fold:
                fold_of = out_of_fold_folds(cfg, ds, model)
                if resume and _out_of_fold_done(metrics_file, ds, mkey, fold_of):
                    logger.info("%s: %s already done (out of fold); skipped", dkey, mkey)
                    continue
                rows = []
                for fold in sorted(set(fold_of.values())):
                    ids = tuple(s for s in ds.ids if fold_of[s] == fold)
                    logger.info("%s: %s, fold %d (validation participants), %d participants", dkey, mkey, fold,
                                len(ids))  # fmt: skip
                    rows += predict_dataset(cfg, replace(ds, ids=ids), model, fold, mask, device)
                table = pd.DataFrame(rows)
            else:
                if mkey not in chosen:
                    raise RuntimeError(f"no selected fold for {mkey}; run `select` or set selection.fixed_folds")
                if mkey in done:
                    logger.info("%s: %s already done (fold %d); skipped", dkey, mkey, chosen[mkey])
                    continue
                logger.info("%s: %s, fold %d, %d participants", dkey, mkey, chosen[mkey], len(ds.ids))
                table = pd.DataFrame(predict_dataset(cfg, ds, model, chosen[mkey], mask, device))
            if metrics_file is not None:
                merge_metrics(metrics_file, table)
            tables.append(table)
    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
