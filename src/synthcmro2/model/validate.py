"""Evaluate the saved fold checkpoints on their validation participants and select folds and models.

Replaces select_by_validation.py; the results are identical for the same checkpoints.
"""

from __future__ import annotations

import json
import logging

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .config import Config, ModelSpec
from .data import SampleLoader, compute_zscore_stats
from .folds import cohort_folds
from .metrics import gm_metrics, warn_if_empty
from .networks import Discriminator, UNet
from .selection import SELECTION_FILE, summarize_selection
from .torchdata import CMRO2Dataset, seed_everything
from .train import validation_loss

logger = logging.getLogger("synthcmro2.model")


def load_generator(path, n_inputs: int, device) -> UNet:
    gen = UNet(n_inputs).to(device)
    gen.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    return gen.eval()


def evaluate_model(cfg: Config, model: ModelSpec, mask: np.ndarray, device) -> tuple[list[dict], list[dict]]:
    t = cfg.training
    rows, fold_rows = [], []
    bce = nn.BCEWithLogitsLoss()
    splits = cohort_folds(cfg.cohorts[model.cohort], int(t["n_folds"]), int(t["seed"]))
    for k, (train_ids, val_ids) in enumerate(splits, start=1):
        g_path, d_path = cfg.checkpoint("generator", model, k), cfg.checkpoint("discriminator", model, k)
        if not g_path.exists():
            logger.warning("%s fold %d: missing %s", model.key, k, g_path)
            continue
        gen = load_generator(g_path, len(model.inputs), device)
        disc = None
        if d_path.exists():
            disc = Discriminator(len(model.inputs)).to(device)
            disc.load_state_dict(torch.load(d_path, map_location=device, weights_only=True))
        else:
            logger.warning("%s fold %d: no discriminator; adversarial term set to 0", model.key, k)
        zstats = compute_zscore_stats(cfg, model, train_ids, mask)
        ds = CMRO2Dataset(SampleLoader(cfg, model, cfg.training_dir, mask, zstats), val_ids)
        dl = DataLoader(ds, batch_size=int(t["batch_size"]), shuffle=False)
        loss, preds = validation_loss(gen, disc, dl, t["loss_weights"], bce, device, collect=True)
        common = {
            "config": model.name,
            "model": model.key,
            "inputs": model.label,
            "n_inputs": len(model.inputs),
            "fold": k,
        }
        for sid, (y, y_hat) in zip(val_ids, preds, strict=True):
            m = gm_metrics(y, y_hat, mask)
            warn_if_empty(m, f"{model.key} fold {k} {sid}", logger)
            rows.append({**common, "subj_id": sid, **m})
        fold_rows.append({**common, "n_validation": len(val_ids), "val_generator_loss": loss,
                          "discriminator_used": disc is not None})  # fmt: skip
        logger.info("%s fold %d: validation generator loss %.4f", model.key, k, loss)
    return rows, fold_rows


def select(cfg: Config, mask: np.ndarray, device, model_keys: list[str] | None = None, out_dir=None):
    seed_everything(int(cfg.training["seed"]), bool(cfg.training.get("deterministic", True)))
    rows, fold_rows = [], []
    for key in model_keys or list(cfg.models):
        r, f = evaluate_model(cfg, cfg.models[key], mask, device)
        rows += r
        fold_rows += f
    if not fold_rows:
        raise RuntimeError("no checkpoints found")
    reduction = {m.label for m in cfg.models.values() if m.role == "reduction"}
    per_subj, per_fold, cv = summarize_selection(rows, fold_rows, reduction)
    out = out_dir or cfg.selection_dir
    out.mkdir(parents=True, exist_ok=True)
    per_subj.to_csv(out / "cv_validation_per_participant.csv", index=False)
    per_fold.to_csv(out / "cv_validation_per_fold.csv", index=False)
    cv.to_csv(out / SELECTION_FILE, index=False)
    (out / "selection.json").write_text(json.dumps({
        "criterion_fold": "lowest validation generator loss (checkpointing criterion)",
        "criterion_configuration": "lowest cross-validated gray matter MAE (mean over folds), input-reduction models",
        "selected": cv.loc[cv.best_for_n_inputs, ["config", "inputs", "n_inputs", "selected_fold"]].to_dict("records"),
    }, indent=2))  # fmt: skip
    return cv
