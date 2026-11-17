"""Cross-validated training of one input configuration (2Train_512Bottleneck_gpu0 and 5fMRI_Train).

Per fold: z-scoring statistics from the fold's training participants, MONAI augmentation of the training
participants, Adam (generator 1e-4, discriminator 5e-5), ReduceLROnPlateau on the validation generator
loss, and the checkpoint (generator and discriminator) of the epoch with the lowest validation generator
loss. The update steps are those of ``train_gan`` in the notebooks.
"""

from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from .config import Config, ModelSpec
from .data import SampleLoader, compute_zscore_stats
from .folds import cohort_folds
from .losses import discriminator_loss, generator_loss
from .networks import Discriminator, UNet, n_parameters
from .torchdata import CMRO2Dataset, build_augmentation, seed_everything

logger = logging.getLogger("synthcmro2.model")


@torch.no_grad()
def validation_loss(gen, disc, loader, weights, bce, device, collect: bool = False):
    """Mean over batches of the generator loss (as ``validate`` in the notebooks); optionally the predictions.

    If ``disc`` is None the adversarial term is 0 (selection when a discriminator checkpoint is missing).
    """
    gen.eval()
    if disc is not None:
        disc.eval()
    losses, preds = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        y_hat = gen(x)
        logits = disc(x, y_hat) if disc is not None else None
        if logits is None:
            w = {**weights, "adversarial": 0.0}
            logits = torch.zeros(len(x), 1, device=device)
        else:
            w = weights
        losses.append(float(generator_loss(y, y_hat, logits, w, bce)))
        if collect:
            preds.extend(zip(y.cpu().numpy()[:, 0], y_hat.cpu().numpy()[:, 0], strict=True))
    return float(np.mean(losses)), preds


def train_fold(
    cfg: Config,
    model: ModelSpec,
    fold: int,
    train_ids: list[str],
    val_ids: list[str],
    mask: np.ndarray,
    device: torch.device,
    epochs: int,
    overwrite: bool = False,
) -> pd.DataFrame:
    t = cfg.training
    g_path, d_path = cfg.checkpoint("generator", model, fold), cfg.checkpoint("discriminator", model, fold)
    if g_path.exists() and not overwrite:
        raise FileExistsError(f"{g_path} exists; pass --overwrite to retrain (the trained model would be replaced)")
    seed = int(t["seed"]) + fold  # each fold reproducible on its own
    seed_everything(seed, bool(t.get("deterministic", True)))

    zstats = compute_zscore_stats(cfg, model, train_ids, mask)
    cache = bool(t.get("cache_volumes", True))
    loader_kw = {
        "cfg": cfg,
        "model": model,
        "data_dir": cfg.training_dir,
        "mask": mask,
        "zstats": zstats,
        "cache": cache,
    }
    train_ds = CMRO2Dataset(
        SampleLoader(**loader_kw), train_ids, build_augmentation(cfg.augmentation[model.augmentation], seed)
    )
    val_ds = CMRO2Dataset(SampleLoader(**loader_kw), val_ids)
    bs = int(t["batch_size"])
    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True, generator=torch.Generator().manual_seed(seed))
    val_dl = DataLoader(val_ds, batch_size=bs, shuffle=False)

    n_in = len(model.inputs)
    gen, disc = UNet(n_in).to(device), Discriminator(n_in).to(device)
    logger.info(
        "%s fold %d: %d training, %d validation participants; parameters G %d, D %d; z-score %s",
        model.key, fold, len(train_ids), len(val_ids), n_parameters(gen), n_parameters(disc), zstats,
    )  # fmt: skip
    g_opt = optim.Adam(gen.parameters(), lr=float(t["lr_generator"]))
    d_opt = optim.Adam(disc.parameters(), lr=float(t["lr_discriminator"]))
    sched = optim.lr_scheduler.ReduceLROnPlateau(g_opt, mode="min", **t["scheduler"])
    bce, weights = nn.BCEWithLogitsLoss(), t["loss_weights"]

    for p in (g_path, d_path, cfg.checkpoint("history", model, fold)):
        p.parent.mkdir(parents=True, exist_ok=True)
    cfg.checkpoint("zscore", model, fold).write_text(
        json.dumps({"train_ids": train_ids, "val_ids": val_ids, "zscore": zstats}, indent=2)
    )

    best, history = math.inf, []
    for epoch in range(1, epochs + 1):
        start = time.time()
        gen.train()
        disc.train()
        g_sum = d_sum = 0.0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            # discriminator step: real pair -> 1, generated pair -> 0
            d_opt.zero_grad()
            y_hat = gen(x)
            d_loss = discriminator_loss(disc(x, y), disc(x, y_hat.detach()), bce)
            d_loss.backward()
            d_opt.step()
            # generator step with the discriminator frozen
            for p in disc.parameters():
                p.requires_grad = False
            g_opt.zero_grad()
            g_loss = generator_loss(y, y_hat, disc(x, y_hat), weights, bce)
            g_loss.backward()
            g_opt.step()
            for p in disc.parameters():
                p.requires_grad = True
            g_sum, d_sum = g_sum + float(g_loss), d_sum + float(d_loss)

        val, _ = validation_loss(gen, disc, val_dl, weights, bce, device)
        sched.step(val)
        saved = val < best
        if saved:
            best = val
            torch.save(gen.state_dict(), g_path)
            torch.save(disc.state_dict(), d_path)
        history.append({
            "epoch": epoch, "train_generator_loss": g_sum / len(train_dl),
            "train_discriminator_loss": d_sum / len(train_dl),
            "val_generator_loss": val, "lr_generator": g_opt.param_groups[0]["lr"], "saved": saved,
            "seconds": round(time.time() - start, 2),
        })  # fmt: skip
        if epoch % 25 == 0 or epoch == epochs:
            pd.DataFrame(history).to_csv(cfg.checkpoint("history", model, fold), index=False)
            logger.info("%s fold %d epoch %d: train G %.4f, val G %.4f (best %.4f)",
                        model.key, fold, epoch, history[-1]["train_generator_loss"], val, best)  # fmt: skip
    hist = pd.DataFrame(history)
    logger.info("%s fold %d: checkpoint from epoch %d", model.key, fold, int(hist.loc[hist.saved, "epoch"].max()))
    return hist


def train_model(
    cfg: Config, model: ModelSpec, mask: np.ndarray, device: torch.device, folds: list[int] | None = None,
    epochs: int | None = None, overwrite: bool = False,
) -> dict[int, Path]:  # fmt: skip
    splits = cohort_folds(cfg.cohorts[model.cohort], int(cfg.training["n_folds"]), int(cfg.training["seed"]))
    done = {}
    for k, (tr, va) in enumerate(splits, start=1):
        if folds and k not in folds:
            continue
        train_fold(cfg, model, k, tr, va, mask, device, epochs or cfg.epochs(model), overwrite)
        done[k] = cfg.checkpoint("generator", model, k)
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return done
