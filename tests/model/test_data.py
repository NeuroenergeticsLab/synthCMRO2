"""Loading and normalization: same values as the notebooks, statistics from training participants only."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import DATA, write_nifti

from synthcmro2.model.data import SampleLoader, compute_zscore_stats, load_masked, map_path, zscore_stats_from


def notebook_stats(volumes):
    """The notebooks' running sums (float32 tensors in the original): mean and unbiased SD of non-zero voxels."""
    x = np.stack(volumes).astype(np.float64)
    nz = x != 0
    s, ss, n = (x * nz).sum(), (x * x * nz).sum(), nz.sum()
    mean = s / n
    return mean, np.sqrt((ss - n * mean**2) / (n - 1))


def test_zscore_stats_equal_notebook_formula():
    rng = np.random.default_rng(1)
    vols = [rng.normal(500, 100, (10, 10, 10)) * (rng.random((10, 10, 10)) > 0.3) for _ in range(5)]
    assert np.allclose(zscore_stats_from(vols), notebook_stats(vols), rtol=1e-10)


def test_zscore_uses_only_the_given_participants(project):
    cfg, mask, train, test = project["cfg"], project["mask"], project["train"], project["test"]
    model = cfg.models["t1"]
    before = compute_zscore_stats(cfg, model, train, mask)
    # changing a test participant's T1 must not change the statistics
    write_nifti(map_path(cfg, cfg.training_dir, test[0], "T1"), np.full(mask.shape, 1e6))
    assert compute_zscore_stats(cfg, model, train, mask) == before
    assert compute_zscore_stats(cfg, cfg.models["cbf_t2s"], train, mask) == {}  # quantitative maps: none


def test_sample_channels_and_normalization(project):
    cfg, mask, train = project["cfg"], project["mask"], project["train"]
    model = cfg.models["reference"]
    x, y = SampleLoader(cfg, model, cfg.training_dir, mask, {})(train[0])
    assert x.shape == (4, *mask.shape) and y.shape == (1, *mask.shape) and x.dtype == np.float32
    cbf = load_masked(map_path(cfg, cfg.training_dir, train[0], "CBF"), mask)
    assert np.array_equal(x[1], cbf)  # channel order CBV, CBF, T2, T2S; physical units kept
    assert not x[:, ~mask].any() and not y[:, ~mask].any()

    t1 = cfg.models["t1"]
    z = compute_zscore_stats(cfg, t1, train, mask)
    x1, _ = SampleLoader(cfg, t1, cfg.training_dir, mask, z)(train[0])
    raw = load_masked(map_path(cfg, cfg.training_dir, train[0], "T1"), mask)
    m, sd = z["T1"]
    assert np.allclose(x1[0][mask], (raw[mask] - m) / (sd + 1e-8), atol=1e-5)
    assert not x1[0][~mask].any()  # re-zeroed outside gray matter


def test_degree_centrality_path_and_normalization(project):
    cfg, mask, train = project["cfg"], project["mask"], project["train"]
    dc = cfg.models["degree_centrality"]
    path = map_path(cfg, cfg.training_dir, train[0], "fMRI")
    assert f"resting_preproc_{train[0]}_ses-1/afni_centrality_0_degree" in str(path) and path.exists()
    z = compute_zscore_stats(cfg, dc, train, mask)
    assert set(z) == {"fMRI"}
    with pytest.raises(ValueError, match="normalization"):
        SampleLoader(cfg, dc, cfg.training_dir, mask, None)


def test_grid_mismatch_and_missing_file(tmp_path):
    mask = np.ones((4, 4, 4), bool)
    write_nifti(tmp_path / "a.nii.gz", np.ones((4, 4, 5)))
    with pytest.raises(ValueError, match="shape"):
        load_masked(tmp_path / "a.nii.gz", mask)
    with pytest.raises(FileNotFoundError):
        load_masked(tmp_path / "missing.nii.gz", mask)


def test_cache_returns_copies(project):
    cfg, mask, train = project["cfg"], project["mask"], project["train"]
    loader = SampleLoader(cfg, cfg.models["cbf"], cfg.training_dir, mask, {})
    x, _ = loader(train[0])
    x[:] = 0
    assert loader(train[0])[0].any()


def test_check_reports_missing_inputs(project):
    from synthcmro2.model.commands import check

    cfg, train = project["cfg"], project["train"]
    assert check(cfg) == []
    map_path(cfg, cfg.training_dir, train[1], "T2S").unlink()
    problems = check(cfg)
    assert len(problems) == 1 and "T2Smap" in problems[0]


def test_grid_offsets_are_read_without_resampling(tmp_path):
    """Some maps store the grid origin 0.5 mm off the mask's (same voxels). Voxels must be read as stored, never
    resampled (resampling had lowered a participant's gray matter median by about 5%)."""
    import nibabel as nib

    from synthcmro2.model.data import check_grid, grid_offset_mm

    mask_img = nib.load(str(DATA / "gm_mask.nii.gz"))
    mask = np.asarray(mask_img.get_fdata()) > 0
    rng = np.random.default_rng(0)
    data = (rng.uniform(20, 250, mask.shape) * mask).astype(np.float32)
    affine = mask_img.affine.copy()
    affine[:3, 3] += 0.5  # header convention of some maps: origin 0.5 mm off, same voxels
    path = tmp_path / "offset_cmro2.nii.gz"
    nib.save(nib.Nifti1Image(data, affine), str(path))
    img = nib.load(str(path))
    assert grid_offset_mm(img.affine, mask_img.affine) == pytest.approx(0.5)
    v = load_masked(path, mask, (mask_img.affine, 0.5))
    assert np.array_equal(v[mask], data[mask])
    with pytest.raises(ValueError, match="not on the gray matter mask grid"):
        load_masked(path, mask, (mask_img.affine, 0.25))
    shifted = mask_img.affine.copy()
    shifted[0, 3] += 4  # one voxel: a real shift of the data
    with pytest.raises(ValueError, match="not on the gray matter mask grid"):
        check_grid(img, path, mask.shape, (shifted, 0.5))
    flipped = mask_img.affine @ np.diag([-1, 1, 1, 1])
    assert grid_offset_mm(flipped, mask_img.affine) == float("inf")


def test_grid_settings_are_validated(raw, tmp_path):
    from synthcmro2.model.config import ConfigError, parse_config

    assert parse_config(raw, tmp_path).grid == {"tolerance_mm": 0.5, "output_affine": "mask"}
    raw["grid"] = {"tolerance_mm": 0.5, "output_affine": "nearest"}
    with pytest.raises(ConfigError):
        parse_config(raw, tmp_path)
