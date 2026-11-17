"""Tests for parcel summaries and surface drawing (no atlas files needed)."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize

from synthcmro2.figures.surfaces import Surface, parcel_medians, plot_surface, shared_norm, vertex_values


def test_parcel_medians_ignore_unlabeled_and_right_hemisphere():
    parcel_ids = np.array([0, 1, 1, 2, 201])
    values = pd.Series([99.0, 1.0, 3.0, 5.0, 77.0], index=range(5))
    out = parcel_medians(values, parcel_ids, n_parcels=3)
    np.testing.assert_allclose(out[:2], [2.0, 5.0])
    assert np.isnan(out[2])  # parcel without voxels


def test_parcel_medians_use_voxel_index():
    parcel_ids = np.array([1, 2, 2])
    values = pd.Series([10.0, 20.0], index=[2, 0])  # only voxels 2 and 0 present
    np.testing.assert_allclose(parcel_medians(values, parcel_ids, n_parcels=2), [20.0, 10.0])


def test_vertex_values_medial_wall_is_nan():
    labels = np.array([0, 1, 2, 2])
    out = vertex_values(np.array([5.0, 7.0]), labels)
    assert np.isnan(out[0])
    np.testing.assert_allclose(out[1:], [5.0, 7.0, 7.0])


def test_shared_norm_uses_percentiles():
    norm = shared_norm(np.arange(101, dtype=float), (5, 95))
    assert (norm.vmin, norm.vmax) == (5.0, 95.0)


def _tetrahedron():
    vertices = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0.3, 0.3]], dtype=float)
    faces = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
    return Surface(vertices, faces, np.array([1, 1, 2, 0]))


def test_plot_surface_draws_only_faces_toward_camera():
    surf = _tetrahedron()
    fig, (ax_l, ax_m) = plt.subplots(1, 2)
    vals = vertex_values(np.array([1.0, 2.0]), surf.labels)
    plot_surface(ax_l, surf, vals, norm=Normalize(0, 3), view="lateral")
    plot_surface(ax_m, surf, vals, norm=Normalize(0, 3), view="medial")
    n_lateral = len(ax_l.collections[0].get_paths())
    n_medial = len(ax_m.collections[0].get_paths())
    assert n_lateral + n_medial == len(surf.faces)  # every face faces exactly one of the two cameras
    plt.close(fig)


def test_poorly_covered_parcels_are_not_painted():
    from synthcmro2.figures.manuscript import SurfaceContext

    shown = np.ones(180, dtype=bool)
    shown[1] = False  # parcel 2 barely covered by the mask
    ctx = SurfaceContext(
        surface=_tetrahedron(),
        parcel_ids=np.array([1, 2, 3, 3]),
        coverage=pd.DataFrame({"shown": shown}),
        cmap=None,
        percentiles=(10, 90),
        ambient=0.75,
        units="",
    )
    out = ctx.parcels(pd.Series([10.0, 20.0, 30.0, 50.0]))
    assert out[0] == 10.0
    assert np.isnan(out[1])  # drawn gray
    assert out[2] == 40.0


def test_atlas_on_a_different_grid_is_rejected(tmp_path):
    import pytest

    nib = pytest.importorskip("nibabel")
    pytest.importorskip("nilearn")
    from synthcmro2.figures.config import SurfacePaths
    from synthcmro2.figures.surfaces import load_atlas_on_mask_grid

    affine = np.diag([4.0, 4.0, 4.0, 1.0])
    nib.save(nib.Nifti1Image(np.ones((4, 5, 4), np.uint8), affine), tmp_path / "mask.nii.gz")
    nib.save(nib.Nifti1Image(np.ones((4, 6, 4), np.int16), affine), tmp_path / "atlas.nii.gz")
    paths = SurfacePaths(gm_mask=tmp_path / "mask.nii.gz", glasser_atlas=tmp_path / "atlas.nii.gz")
    with pytest.raises(ValueError, match="does not match the mask grid"):
        load_atlas_on_mask_grid(paths)
