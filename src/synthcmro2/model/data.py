"""Loading, masking and normalization of the input maps (NumPy and nibabel only; no PyTorch).

Behaviour of the notebooks' ``MRIDataLoader`` that is kept:

- every map is multiplied by the group gray matter mask;
- quantitative maps stay in physical units; only the inputs listed in ``training.zscore`` (T1-weighted
  image, degree centrality) are z-scored, with the mean and unbiased SD over the non-zero gray matter voxels
  of the *training* participants, then multiplied by the mask again;
- channel order CBV, CBF, T2, T2*, T1, fMRI.

What is new: maps are checked against the mask grid, only the inputs a model uses are read, and the
normalization statistics are computed by one function used for training, selection and testing.
"""

from __future__ import annotations

import functools
import logging
from collections.abc import Iterable, Sequence
from pathlib import Path

import nibabel as nib
import numpy as np

from .config import Config, ModelSpec

ZStats = dict[str, tuple[float, float]]
Grid = tuple[np.ndarray, float]  # (mask affine, tolerance in mm)
logger = logging.getLogger("synthcmro2.model")


def load_mask(path: Path) -> np.ndarray:
    return np.asarray(nib.load(str(path)).get_fdata()) > 0


@functools.lru_cache(maxsize=8)
def _affine(path: str) -> np.ndarray:
    return nib.load(path).affine


def grid(cfg: Config) -> Grid:
    """Reference grid: the gray matter mask's affine and the tolerated origin offset."""
    return _affine(str(cfg.gm_mask)), float(cfg.grid["tolerance_mm"])


def grid_offset_mm(affine: np.ndarray, ref: np.ndarray) -> float:
    """Largest origin difference (mm) between two affines with the same voxel axes; inf if the axes differ."""
    if not np.allclose(affine[:3, :3], ref[:3, :3], atol=1e-4):
        return float("inf")
    return float(np.abs(affine[:3, 3] - ref[:3, 3]).max())


def check_grid(img, path: Path, shape: tuple[int, ...], ref: Grid | None) -> float:
    """Stop unless the image is on the mask grid: same shape, same axes, origin within the tolerance."""
    if img.shape[:3] != tuple(shape):
        raise ValueError(f"{path}: shape {img.shape} differs from the gray matter mask {tuple(shape)}")
    if ref is None:
        return 0.0
    offset = grid_offset_mm(img.affine, ref[0])
    if offset > ref[1] + 1e-6:
        raise ValueError(
            f"{path}: not on the gray matter mask grid (origin offset {offset:.3g} mm > {ref[1]} mm); resample it first"
        )
    return offset


def load_masked(path: Path, mask: np.ndarray, ref: Grid | None = None) -> np.ndarray:
    """Map multiplied by the mask, as float32 (the notebooks: get_fdata() * mask, then float32).

    Voxels are read as stored, never interpolated. With ``ref`` the map must be on the mask grid
    (see ``check_grid``); an origin offset within the tolerance is a header convention, not a shift of the data.
    """
    if not path.exists():
        raise FileNotFoundError(path)
    img = nib.load(str(path))
    check_grid(img, path, mask.shape, ref)
    return (np.asarray(img.get_fdata()) * mask).astype(np.float32)


def map_path(cfg: Config, data_dir: Path, sid: str, key: str, condition: str = "rest") -> Path:
    """Path of one map ('CBF', ..., 'fMRI', 'CMRO2') of one participant."""
    return data_dir / sid / cfg.files[key].format(sid=sid, condition=condition)


def zscore_stats_from(volumes: Iterable[np.ndarray]) -> tuple[float, float]:
    """Mean and unbiased SD over the non-zero voxels of all volumes (float64 accumulation)."""
    vals = np.concatenate([v[v != 0].astype(np.float64) for v in volumes])
    if vals.size < 2:
        raise ValueError("fewer than two non-zero voxels for z-scoring")
    return float(vals.mean()), float(vals.std(ddof=1))


def compute_zscore_stats(
    cfg: Config, model: ModelSpec, sids: Sequence[str], mask: np.ndarray, data_dir: Path | None = None
) -> ZStats:
    """Normalization statistics of the z-scored inputs of ``model`` over the participants ``sids``.

    Always computed on resting-state training data (the models were trained at rest), also when
    the test condition is the calculation task.
    """
    data_dir = data_dir or cfg.training_dir
    zscored = [m for m in model.inputs if m in cfg.training["zscore"]]
    ref = grid(cfg)
    return {m: zscore_stats_from(load_masked(map_path(cfg, data_dir, s, m), mask, ref) for s in sids) for m in zscored}


def apply_zscore(x: np.ndarray, mean: float, sd: float, mask: np.ndarray, eps: float) -> np.ndarray:
    return (((x - mean) / (sd + eps)) * mask).astype(np.float32)


class SampleLoader:
    """Returns (inputs[C, X, Y, Z], target[1, X, Y, Z]) for a participant, optionally cached."""

    def __init__(
        self,
        cfg: Config,
        model: ModelSpec,
        data_dir: Path,
        mask: np.ndarray,
        zstats: ZStats | None,
        condition: str = "rest",
        cache: bool = True,
    ):
        missing = [m for m in model.inputs if m in cfg.training["zscore"] and not (zstats and m in zstats)]
        if missing:
            raise ValueError(f"no normalization statistics for {missing}")
        self.cfg, self.model, self.data_dir, self.mask = cfg, model, data_dir, mask
        self.zstats, self.condition, self.cache = zstats or {}, condition, cache
        self._cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.ref = grid(cfg)

    def path(self, sid: str, key: str) -> Path:
        return map_path(self.cfg, self.data_dir, sid, key, self.condition)

    def __call__(self, sid: str) -> tuple[np.ndarray, np.ndarray]:
        if sid in self._cache:
            x, y = self._cache[sid]
            return x.copy(), y.copy()
        channels = []
        for m in self.model.inputs:
            v = load_masked(self.path(sid, m), self.mask, self.ref)
            if m in self.zstats:
                v = apply_zscore(v, *self.zstats[m], self.mask, float(self.cfg.training["zscore_eps"]))
            channels.append(v)
        x = np.stack(channels).astype(np.float32)
        y = load_masked(self.path(sid, "CMRO2"), self.mask, self.ref)[None]
        if self.cache:
            self._cache[sid] = (x, y)
        return x.copy(), y.copy()


def output_paths(cfg: Config, ds, model: ModelSpec, sid: str) -> dict[str, Path]:
    """Where `predict` writes, and `export` reads, a participant's synthetic and masked analytical maps."""
    out_dir = ds.output_dir / sid / model.output_subdir.format(sid=sid)
    fmt = {"sid": sid, "condition": ds.condition, "name": model.name}
    suffix = model.name if model.suffix_masked_target else ""
    return {
        "synthetic": out_dir / cfg.outputs["synthetic"].format(**fmt),
        "target_masked": out_dir / cfg.outputs["target_masked"].format(**fmt, suffix=suffix),
    }
