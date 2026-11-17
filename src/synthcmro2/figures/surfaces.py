"""Cortical surface maps of group CMRO2 (left hemisphere, fs_LR 32k inflated).

Voxel values are summarized per Glasser parcel (median over the gray matter
voxels of each parcel) and painted on the left inflated surface using the
Glasser surface labels (keys 1-180, which match volume labels 1-180 of the
left hemisphere). The surface is drawn with matplotlib directly, so maps can be
placed in any axes of a figure (insets, shared colorbars). This replaces the
earlier combination of ENIGMA Toolbox and matplotlib_surface_plotting.

Voxel IDs in the voxel table follow ``NiftiMasker(mask_img=gm_mask)`` order,
which is how they were created. nilearn and nibabel are imported lazily.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.collections import PolyCollection
from matplotlib.colors import Colormap, Normalize

from .config import SurfacePaths

N_LEFT_PARCELS = 180  # Glasser left hemisphere, labels 1-180


@dataclass(frozen=True)
class Surface:
    vertices: np.ndarray  # (n_vertices, 3), mm
    faces: np.ndarray  # (n_faces, 3), vertex indices
    labels: np.ndarray  # (n_vertices,), Glasser key, 0 = medial wall


def load_surface(paths: SurfacePaths) -> Surface:
    """Inflated left surface and its Glasser vertex labels."""
    import nibabel as nib

    surf = nib.load(str(paths.inflated_surf_gii))
    labels = nib.load(str(paths.glasser_label_gii)).darrays[0].data.astype(int)
    vertices, faces = surf.darrays[0].data, surf.darrays[1].data
    if labels.size != vertices.shape[0]:
        raise ValueError("surface and label files have different numbers of vertices")
    return Surface(vertices.astype(float), faces.astype(int), labels)


def load_atlas_on_mask_grid(paths: SurfacePaths):
    """Mask and atlas images, which must share one voxel grid (no resampling).

    The atlas is expected on the training grid (48 x 56 x 48), i.e. cropped like
    the CMRO2 maps (``synthcmro2 crop-atlas``, equivalent to
    ``fslroi ... 0 -1 0 56 0 -1``). A mismatch raises an error instead of being
    silently resampled.
    """
    from nilearn import image

    mask = image.load_img(str(paths.gm_mask))
    atlas = image.load_img(str(paths.glasser_atlas))
    if atlas.shape[:3] != mask.shape[:3] or not np.allclose(atlas.affine, mask.affine):
        raise ValueError(
            f"atlas grid {atlas.shape[:3]} does not match the mask grid {mask.shape[:3]}; "
            "crop it with `synthcmro2 crop-atlas`"
        )
    return mask, atlas


def voxel_parcel_ids(paths: SurfacePaths) -> np.ndarray:
    """Glasser parcel ID of every gray matter voxel, in voxel-ID order."""
    from nilearn.maskers import NiftiMasker

    mask, atlas = load_atlas_on_mask_grid(paths)
    ids = NiftiMasker(mask_img=mask).fit().transform(atlas).ravel()
    if not np.allclose(ids, np.round(ids)):
        raise ValueError("atlas labels are not integers after masking")
    return ids.astype(int)


def parcel_coverage(paths: SurfacePaths, n_parcels: int = N_LEFT_PARCELS) -> pd.DataFrame:
    """How much of each left-hemisphere Glasser parcel lies inside the gray matter mask.

    Coverage is the fraction of a parcel's atlas voxels (on the mask grid)
    that belong to the mask. The conservative mask excludes, for example,
    orbitofrontal and temporal-pole cortex with low signal; parcels there are
    represented by only a few voxels.
    """
    import nibabel as nib

    mask, atlas = load_atlas_on_mask_grid(paths)
    labels = np.asarray(atlas.get_fdata()).astype(int)
    inside = np.asarray(mask.get_fdata()) > 0
    n_atlas = np.bincount(labels.ravel(), minlength=n_parcels + 1)[1 : n_parcels + 1]
    n_mask = np.bincount(labels[inside].ravel(), minlength=n_parcels + 1)[1 : n_parcels + 1]
    names = nib.load(str(paths.glasser_label_gii)).labeltable.get_labels_as_dict()
    return pd.DataFrame(
        {
            "parcel": np.arange(1, n_parcels + 1),
            "name": [
                names.get(i, str(i)).removeprefix("L_").removesuffix("_ROI") for i in range(1, n_parcels + 1)
            ],
            "n_atlas_voxels": n_atlas,
            "n_mask_voxels": n_mask,
            "coverage": n_mask / np.maximum(n_atlas, 1),
        }
    )


def parcel_medians(values: pd.Series, parcel_ids: np.ndarray, n_parcels: int = N_LEFT_PARCELS) -> np.ndarray:
    """Median value per parcel 1..n_parcels; NaN for parcels without voxels.

    ``values`` is indexed by voxel ID. Voxels outside any parcel (ID 0) are
    excluded explicitly.
    """
    frame = pd.DataFrame({"parcel": parcel_ids[values.index.to_numpy()], "value": values.to_numpy()})
    frame = frame[(frame["parcel"] > 0) & (frame["parcel"] <= n_parcels)]
    return frame.groupby("parcel")["value"].median().reindex(range(1, n_parcels + 1)).to_numpy()


def vertex_values(parcel_values: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Paint parcel values (index 0 = parcel 1) on vertices; NaN on the medial wall."""
    out = np.full(labels.shape, np.nan)
    inside = (labels > 0) & (labels <= parcel_values.size)
    out[inside] = parcel_values[labels[inside] - 1]
    return out


def plot_surface(
    ax: Axes,
    surface: Surface,
    values: np.ndarray,
    cmap: str | Colormap = "magma",
    norm: Normalize | None = None,
    view: str = "lateral",
    nan_color: tuple[float, float, float] = (0.8, 0.8, 0.8),
    ambient: float = 0.6,
) -> Normalize:
    """Draw the left hemisphere (orthographic view, Lambert shading) into ``ax``.

    ``view="lateral"`` looks from the left (anterior to the left of the image);
    ``view="medial"`` looks from the right (anterior to the right).
    Returns the normalization used, for a shared colorbar.
    """
    import matplotlib as mpl

    cmap = mpl.colormaps[cmap] if isinstance(cmap, str) else cmap
    if norm is None:
        norm = Normalize(np.nanmin(values), np.nanmax(values))
    sign = -1.0 if view == "lateral" else 1.0  # camera on -x (lateral) or +x (medial)

    tri = surface.vertices[surface.faces]  # (n_faces, 3, 3)
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12
    facing = sign * normals[:, 0]  # > 0: face points toward the camera
    keep = facing > 0
    order = np.argsort(-sign * tri[keep, :, 0].mean(1))  # far faces first (painter's algorithm)
    faces_kept = np.flatnonzero(keep)[order]

    # screen coordinates: horizontal = -y (lateral) or +y (medial), vertical = z
    xy = np.stack([sign * tri[faces_kept, :, 1], tri[faces_kept, :, 2]], axis=-1)
    # parcel values are constant within parcels; a face takes the value of its first vertex
    face_vals = values[surface.faces[faces_kept, 0]]
    rgb = np.where(
        np.isnan(face_vals)[:, None],
        np.array(nan_color)[None, :],
        cmap(norm(np.nan_to_num(face_vals)))[:, :3],
    )
    shade = ambient + (1 - ambient) * np.sqrt(facing[faces_kept])  # soft Lambert shading
    colors = np.clip(rgb * shade[:, None], 0, 1)

    ax.add_collection(
        PolyCollection(
            xy,
            facecolors=colors,
            edgecolors=colors,
            linewidths=0.3,
            antialiased=False,
            rasterized=True,  # ~30k polygons per map: keep PDF/SVG files small and fast
        )
    )
    ax.set_xlim(xy[..., 0].min(), xy[..., 0].max())
    ax.set_ylim(xy[..., 1].min(), xy[..., 1].max())
    ax.set_aspect("equal")
    ax.set_axis_off()
    return norm


def truncated_cmap(name: str, start: float, stop: float) -> Colormap:
    """Part of a colormap, e.g. without its near-black end."""
    import matplotlib as mpl
    from matplotlib.colors import ListedColormap

    return ListedColormap(mpl.colormaps[name](np.linspace(start, stop, 256)), name=f"{name}_{start}_{stop}")


def shared_norm(reference: np.ndarray, percentiles: tuple[float, float]) -> Normalize:
    """Color limits from percentiles of a reference map (used for all maps it is compared with)."""
    lo, hi = np.nanpercentile(reference, percentiles)
    return Normalize(lo, hi)
