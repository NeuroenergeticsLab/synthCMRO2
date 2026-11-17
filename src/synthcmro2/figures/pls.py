"""Partial least squares (PLS) task patterns: loading, checks, and overlap (Figure 3b).

The PLS results were saved with ``np.save`` as pickled ``pyls`` result objects.
Unpickling arbitrary files can execute code, and requiring ``pyls`` only to
read stored arrays is unnecessary, so results are loaded with a restricted
unpickler that accepts nothing but NumPy arrays and the ``pyls`` result
containers (mapped to a plain attribute dictionary).

Mean-centered PLS was run on the rest and calculation maps of the Young-adults I
test participants, separately for analytical CMRO2 and for the synthetic maps
of each model. The first latent variable (LV) is analyzed.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .data import DataValidationError, select

_NUMPY_GLOBALS = {
    ("numpy.core.multiarray", "_reconstruct"),
    ("numpy._core.multiarray", "_reconstruct"),
    ("numpy", "ndarray"),
    ("numpy", "dtype"),
}


class _ResultDict(dict):
    """Stand-in for ``pyls.structures.*Results`` (dicts with attribute access)."""

    def __getattr__(self, key: str):
        try:
            return self[key]
        except KeyError as err:
            raise AttributeError(key) from err


class _RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        if (module, name) in _NUMPY_GLOBALS:
            if name == "_reconstruct":
                try:
                    from numpy._core.multiarray import _reconstruct  # NumPy >= 2
                except ImportError:  # NumPy 1.x
                    from numpy.core.multiarray import _reconstruct
                return _reconstruct
            return getattr(np, name)
        if module == "pyls.structures" and name.startswith("PLS"):
            return _ResultDict
        raise pickle.UnpicklingError(f"refusing to load {module}.{name} from a PLS result file")


@dataclass(frozen=True)
class PLSResult:
    """The parts of a ``pyls`` mean-centered PLS result used for the figures.

    ``X`` (the input maps) is present in the full ``pyls`` result; the public summary (``.npz``, written by
    :func:`save_pls_summary`) leaves it out and stores the identity of each observation (``observations``)
    instead.
    """

    name: str
    X: np.ndarray | None  # observations x voxels (input maps); None in the public summary
    bsr: np.ndarray  # bootstrap ratios of the analyzed LV, one per voxel
    p_value: float  # permutation p-value of the analyzed LV
    design_scores: np.ndarray  # per observation, analyzed LV
    brain_scores: np.ndarray  # per observation, analyzed LV
    n_perm: int
    n_boot: int
    observations: pd.DataFrame | None = None  # subj_id, condition per observation (public summary)


def save_pls_summary(result: PLSResult, observations: pd.DataFrame, path: str | Path) -> Path:
    """Write the public PLS summary: everything the figures use except the input maps (``X``)."""
    path = Path(path).with_suffix(".npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        bsr=result.bsr,
        p_value=result.p_value,
        design_scores=result.design_scores,
        brain_scores=result.brain_scores,
        n_perm=result.n_perm,
        n_boot=result.n_boot,
        subj_id=np.asarray(observations["subj_id"].astype(str).tolist(), dtype=str),
        condition=np.asarray(observations["condition"].astype(str).tolist(), dtype=str),
    )
    return path


def load_pls_result(path: str | Path, lv: int = 0) -> PLSResult:
    """Read one PLS result: the pickled ``pyls`` result saved with ``np.save`` (0-d object array), or, if that
    file does not exist, the public summary next to it (same name, ``.npz``; first LV only)."""
    path = Path(path)
    summary = path.with_suffix(".npz")
    if not path.exists() and summary.exists():
        if lv != 0:
            raise DataValidationError(f"{summary.name}: the public PLS summary stores the first LV only")
        with np.load(summary, allow_pickle=False) as z:
            obs = pd.DataFrame({"subj_id": z["subj_id"].astype(str), "condition": z["condition"].astype(str)})
            return PLSResult(name=path.stem, X=None, bsr=z["bsr"].astype(float), p_value=float(z["p_value"]),
                             design_scores=z["design_scores"].astype(float),
                             brain_scores=z["brain_scores"].astype(float), n_perm=int(z["n_perm"]),
                             n_boot=int(z["n_boot"]), observations=obs)  # fmt: skip
    with path.open("rb") as fh:
        version = np.lib.format.read_magic(fh)
        if version != (1, 0):
            raise DataValidationError(f"{path.name}: unsupported .npy format version {version}")
        np.lib.format.read_array_header_1_0(fh)
        obj = _RestrictedUnpickler(fh).load()
    res = obj.item() if isinstance(obj, np.ndarray) else obj
    return PLSResult(
        name=path.stem,
        X=np.asarray(res.inputs["X"], dtype=float),
        bsr=np.asarray(res.bootres["x_weights_normed"])[:, lv],
        p_value=float(res.permres["pvals"][lv]),
        design_scores=np.asarray(res.y_scores)[:, lv],
        brain_scores=np.asarray(res.x_scores)[:, lv],
        n_perm=int(res.inputs["n_perm"]),
        n_boot=int(res.inputs["n_boot"]),
    )


def match_observations(
    result: PLSResult,
    voxels: pd.DataFrame,
    column: str,
    model: str,
    datasets: list[str],
    subjects: list[str],
) -> pd.DataFrame:
    """Identify each PLS observation (row of X) in the voxel table.

    Every row of X must equal exactly one participant/condition map of
    ``column`` (``CMRO2`` for analytical, ``synthCMRO2`` for synthetic) for the
    given model. This checks the provenance of the PLS input and that PLS voxels
    follow the same order as ``vox_id``. It also tells which observations are rest
    and which are task, so nothing relies on an assumed row order. The public summary has no ``X``; its stored
    observation labels are returned instead.
    """
    if result.X is None:
        if result.observations is None:
            raise DataValidationError(f"{result.name}: neither input maps nor observation labels stored")
        return result.observations.copy()
    sub = select(voxels, dataset=datasets, model=model, subj_id=subjects)
    wide = sub.pivot_table(
        index=["subj_id", "dataset", "condition"], columns="vox_id", values=column, observed=True
    )
    if wide.shape[1] != result.X.shape[1]:
        raise DataValidationError(
            f"{result.name}: {result.X.shape[1]} PLS voxels but {wide.shape[1]} voxels in the table"
        )
    maps = wide.to_numpy()
    rows = []
    for i, x in enumerate(result.X):
        hits = np.flatnonzero(np.all(np.isclose(maps, x, rtol=0, atol=1e-6), axis=1))
        if hits.size != 1:
            raise DataValidationError(f"{result.name}: observation {i} matches {hits.size} maps")
        rows.append(dict(zip(wide.index.names, wide.index[hits[0]], strict=True)))
    return pd.DataFrame(rows)


def orientation_sign(result: PLSResult, conditions: pd.Series, task: str, baseline: str) -> float:
    """+1 if task observations have higher design scores than baseline ones, else -1.

    The sign of a PLS latent variable is arbitrary. Multiplying bootstrap ratios
    and brain scores by this sign makes positive values mean task > baseline in
    every map, so maps from different analyses can be compared and colored alike.
    """
    cond = np.asarray(conditions)
    contrast = result.design_scores[cond == task].mean() - result.design_scores[cond == baseline].mean()
    if contrast == 0:
        raise DataValidationError(f"{result.name}: design scores do not separate {task} and {baseline}")
    return float(np.sign(contrast))


def overlap(bsr_a: np.ndarray, bsr_b: np.ndarray, threshold: float = 2.0) -> dict[str, float]:
    """Overlap of reliable voxels (|BSR| > threshold) between two maps.

    ``dice`` and ``jaccard`` ignore the direction of the effect, as in the
    original analysis. ``dice_same_sign`` counts only overlapping voxels whose
    effect has the same direction in both maps.
    """
    a, b = np.abs(bsr_a) > threshold, np.abs(bsr_b) > threshold
    both = a & b
    same_sign = both & (np.sign(bsr_a) == np.sign(bsr_b))
    denom = a.sum() + b.sum()
    return {
        "n_reliable_a": int(a.sum()),
        "n_reliable_b": int(b.sum()),
        "n_overlap": int(both.sum()),
        "dice": float(2 * both.sum() / denom) if denom else np.nan,
        "dice_same_sign": float(2 * same_sign.sum() / denom) if denom else np.nan,
        "jaccard": float(both.sum() / (a | b).sum()) if (a | b).any() else np.nan,
    }


def plot_bsr_slice(
    ax,
    bsr: np.ndarray,
    contour: np.ndarray,
    mask_img: str | Path,
    z: float,
    threshold: float,
    vmax: float,
    background: str | Path | None = None,
):
    """Axial slice of a thresholded BSR map with black contours, drawn into ``ax``.

    ``bsr`` and ``contour`` are voxel vectors in gray matter mask order; they are
    put back into brain space with the same mask that defined ``vox_id``. The
    slice is drawn with matplotlib (nilearn's plotting functions cannot draw into
    subfigures), in neurological orientation (left hemisphere on the left).
    Without ``background``, nilearn's 4-mm MNI152 template is used.

    Returns the overlay image, for a shared colorbar.
    """
    from nilearn import datasets, image
    from nilearn.maskers import NiftiMasker
    from nilearn.plotting import cm

    masker = NiftiMasker(mask_img=str(mask_img)).fit()
    bsr_img = masker.inverse_transform(bsr)
    contour_img = masker.inverse_transform(contour.astype(float))
    bg_img = image.load_img(str(background)) if background else datasets.load_mni152_template(resolution=4)
    # force_resample/copy_header exist from nilearn 0.11 (they silence its deprecation warnings); older
    # versions resample and copy the header the same way without them
    import inspect

    known = inspect.signature(image.resample_to_img).parameters
    extra = {k: True for k in ("force_resample", "copy_header") if k in known}
    bg_img = image.resample_to_img(bg_img, bsr_img, interpolation="continuous", **extra)

    k = round(float((np.linalg.inv(bsr_img.affine) @ np.array([0.0, 0.0, z, 1.0]))[2]))
    bg = bg_img.get_fdata()[:, :, k]
    val = bsr_img.get_fdata()[:, :, k]
    cont = contour_img.get_fdata()[:, :, k]
    if bsr_img.affine[0, 0] < 0:  # make x increase to the right (neurological view)
        bg, val, cont = bg[::-1], val[::-1], cont[::-1]

    brain = np.argwhere(bg > 0.05 * bg.max())
    (i0, j0), (i1, j1) = brain.min(0), brain.max(0) + 1
    bg, val, cont = bg[i0:i1, j0:j1], val[i0:i1, j0:j1], cont[i0:i1, j0:j1]

    outside = bg <= 0.05 * bg.max()  # white, not black, outside the brain
    ax.imshow(np.ma.masked_where(outside, bg).T, cmap="gray", origin="lower", interpolation="nearest")
    overlay = ax.imshow(
        np.ma.masked_where(np.abs(val) <= threshold, val).T,
        cmap=cm.cold_hot,
        vmin=-vmax,
        vmax=vmax,
        origin="lower",
        interpolation="nearest",
    )
    if cont.any():
        ax.contour(cont.T, levels=[0.5], colors="k", linewidths=0.5)
    ax.text(0.02, 0.98, "L", transform=ax.transAxes, fontsize=6, va="top", color="0.3")
    ax.text(0.98, 0.98, "R", transform=ax.transAxes, fontsize=6, va="top", ha="right", color="0.3")
    ax.set_axis_off()
    return overlay


# --------------------------------------------------------------------------
# Computing the PLS results (``synthcmro2 pls``; needs ``pyls``)
# --------------------------------------------------------------------------

ANALYTICAL_COLUMN, SYNTHETIC_COLUMN = "CMRO2", "synthCMRO2"
PLS_SETTINGS = {"mean_centering": 0, "n_perm": 3000, "n_boot": 3000, "rotate": True, "ci": 95}


def task_subjects(voxels: pd.DataFrame, cfg) -> list[str]:
    """Participants with calculation-task data (sorted)."""
    return sorted(select(voxels, dataset=cfg.datasets["task"])["subj_id"].astype(str).unique())


def pls_inputs(voxels: pd.DataFrame, cfg, model: str, column: str, subjects: list[str]) -> np.ndarray:
    """X (2 x n_subjects rows x voxels): baseline (rest) maps first, then task maps, subjects in order."""
    p = cfg.pls
    blocks = []
    for dataset, condition in ((cfg.datasets["test"], p["baseline"]), (cfg.datasets["task"], p["task"])):
        sub = select(voxels, dataset=dataset, model=model, subj_id=subjects)
        sub = sub[sub["condition"].astype(str) == condition]
        wide = sub.pivot_table(index="subj_id", columns="vox_id", values=column, observed=True)
        wide.index = wide.index.astype(str)
        missing = sorted(set(subjects) - set(wide.index))
        if missing:
            raise ValueError(f"{model}, {condition}: no maps for {missing}")
        blocks.append(wide.loc[subjects].to_numpy(dtype=float))
    x = np.vstack(blocks)
    if not np.isfinite(x).all():
        raise ValueError(f"{model}: non-finite values in the PLS input")
    return x


def pls_analyses(cfg) -> list[tuple[str, str, str]]:
    """(file tag, model whose rows are read, column): analytical maps first, then each model's synthetic."""
    tags = cfg.pls["model_tags"]
    return [(cfg.pls["analytical_tag"], next(iter(tags)), ANALYTICAL_COLUMN)] + [
        (tag, model, SYNTHETIC_COLUMN) for model, tag in tags.items()
    ]


def run_pls(cfg, voxels: pd.DataFrame, out_dir: Path, seed: int) -> dict:
    """Mean-centered PLS of calculation versus rest for analytical CMRO2 and each model in ``pls.model_tags``.

    X: 2 x n participants (rest first, then calculation) x voxels, straight from the voxel table.
    ``pyls.meancentered_pls(X, groups=[n], n_cond=2, mean_centering=0, n_perm=3000, n_boot=3000, rotate=True,
    ci=95, seed=seed)``; each result is saved with ``np.save`` under ``pls.file_pattern``, the names the
    figure code reads. Returns the provenance record (also written to ``provenance.json``).
    """
    import hashlib
    import json
    from datetime import UTC, datetime
    from importlib import metadata

    import pyls

    subjects = task_subjects(voxels, cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    record = {}
    for tag, model, column in pls_analyses(cfg):
        x = pls_inputs(voxels, cfg, model, column, subjects)
        res = pyls.meancentered_pls(x, groups=[len(subjects)], n_cond=2, seed=seed, verbose=False, **PLS_SETTINGS)
        path = out_dir / cfg.pls["file_pattern"].format(tag=tag)
        np.save(path, res, allow_pickle=True)
        record[tag] = {"model": model, "column": column, "p_lv1": float(res.permres.pvals[0]),
                       "X_sha256": hashlib.sha256(x.tobytes()).hexdigest(), "file": path.name}  # fmt: skip
    try:
        version = metadata.version("pyls")
    except metadata.PackageNotFoundError:  # e.g. run from a source folder without installing
        version = getattr(pyls, "__version__", None)
    prov = {"created": datetime.now(UTC).isoformat(timespec="seconds"), "voxel_values": str(cfg.voxel_values),
            "subjects": subjects, "seed": seed, "settings": PLS_SETTINGS, "pyls": version, "analyses": record}  # fmt: skip
    (out_dir / "provenance.json").write_text(json.dumps(prov, indent=2))
    return prov
