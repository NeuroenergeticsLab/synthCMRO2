"""Apply one trained generator to new maps (``synthcmro2 model apply``): no analytical CMRO2 needed.

The inputs must be quantitative maps in physical units (as for training: CBF in ml/100 g/min, T2* and T2 in ms,
CBV in %) on the 4-mm MNI152NLin2009cAsym gray matter mask grid (48 x 56 x 48; see README, Using the models).
Each input is multiplied by the gray matter mask, exactly as in training and testing, and the synthetic CMRO2
(µmol/100 g/min) is written with the mask's affine. Values are meaningful inside the gray matter mask only.
Models with z-scored inputs (T1-weighted, degree centrality) are not supported here, because they need the
normalization statistics of the training participants.
"""

from __future__ import annotations

import logging
from pathlib import Path

import nibabel as nib
import numpy as np

from .config import Config, ModelSpec
from .data import grid, load_mask, load_masked
from .selection import selected_folds

logger = logging.getLogger("synthcmro2.model")


def default_checkpoint(cfg: Config, model: ModelSpec) -> Path:
    """The model's validation-selected generator (fold from ``<selection_dir>/cv_selection.csv``)."""
    folds = selected_folds(cfg)
    if model.key not in folds:
        raise FileNotFoundError(f"no selected fold for {model.key}; pass --checkpoint or add cv_selection.csv")
    return cfg.checkpoint("generator", model, folds[model.key])


def apply_model(cfg: Config, model: ModelSpec, inputs: dict[str, Path], out: Path, checkpoint: Path | None = None,
                device="cpu") -> Path:  # fmt: skip
    """Synthesize CMRO2 from ``inputs`` (map key -> NIfTI path, keys as in ``model.inputs``) and write ``out``."""
    import torch

    from .validate import load_generator

    zscored = [m for m in model.inputs if m in cfg.training["zscore"]]
    if zscored:
        raise ValueError(f"{model.key}: inputs {zscored} are z-scored with training statistics; use `predict`")
    missing = [m for m in model.inputs if m not in inputs]
    extra = [m for m in inputs if m not in model.inputs]
    if missing or extra:
        raise ValueError(f"{model.key} expects inputs {list(model.inputs)}; missing {missing}, unexpected {extra}")
    checkpoint = Path(checkpoint) if checkpoint else default_checkpoint(cfg, model)
    if not checkpoint.exists():
        raise FileNotFoundError(f"{checkpoint} (download the trained models, see README)")
    mask = load_mask(cfg.gm_mask)
    ref = grid(cfg)
    x = np.stack([load_masked(Path(inputs[m]), mask, ref) for m in model.inputs]).astype(np.float32)
    gen = load_generator(checkpoint, len(model.inputs), device)
    with torch.no_grad():
        synth = gen(torch.from_numpy(x)[None].to(device)).cpu().numpy()[0, 0]
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(synth.astype(np.float32), affine=ref[0]), str(out))
    inside = synth[mask]
    logger.info("%s (%s): gray matter median %.1f µmol/100 g/min -> %s", model.key, checkpoint.name,
                float(np.median(inside)), out)  # fmt: skip
    return out
