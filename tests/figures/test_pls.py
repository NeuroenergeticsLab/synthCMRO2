"""Tests for PLS loading, orientation, and overlap."""

import pickle
import sys
import types

import numpy as np
import pandas as pd
import pytest

from synthcmro2.figures.data import DataValidationError
from synthcmro2.figures.pls import (
    PLSResult,
    load_pls_result,
    match_observations,
    orientation_sign,
    overlap,
)


def _fake_pyls_module():
    """A stand-in ``pyls.structures`` module so test pickles reference the real class names."""
    module = types.ModuleType("pyls.structures")
    for name in ["PLSResults", "PLSInputs", "PLSBootResults", "PLSPermResults"]:
        cls = type(name, (dict,), {"__module__": "pyls.structures"})
        setattr(module, name, cls)
    return module


@pytest.fixture
def pyls_file(tmp_path, monkeypatch):
    module = _fake_pyls_module()
    monkeypatch.setitem(sys.modules, "pyls", types.ModuleType("pyls"))
    monkeypatch.setitem(sys.modules, "pyls.structures", module)
    X = np.arange(12, dtype=float).reshape(4, 3)
    res = module.PLSResults(
        inputs=module.PLSInputs(X=X, n_perm=10, n_boot=20),
        bootres=module.PLSBootResults(x_weights_normed=np.array([[3.0, 0], [-2.5, 0], [1.0, 0]])),
        permres=module.PLSPermResults(pvals=np.array([0.01, 0.5])),
        y_scores=np.array([[-1.0, 0], [-1.0, 0], [1.0, 0], [1.0, 0]]),
        x_scores=np.array([[0.1, 0], [0.2, 0], [0.3, 0], [0.4, 0]]),
    )
    path = tmp_path / "result.npy"
    np.save(path, np.array(res, dtype=object), allow_pickle=True)
    return path


def test_load_pls_result_reads_first_latent_variable(pyls_file):
    result = load_pls_result(pyls_file)
    np.testing.assert_allclose(result.bsr, [3.0, -2.5, 1.0])
    assert result.p_value == 0.01
    assert (result.n_perm, result.n_boot) == (10, 20)
    assert result.X.shape == (4, 3)


class _Malicious:
    def __reduce__(self):
        return (print, ("this must never run",))


def test_restricted_unpickler_refuses_other_objects(tmp_path):
    path = tmp_path / "bad.npy"
    np.save(path, np.array(_Malicious(), dtype=object), allow_pickle=True)
    with pytest.raises(pickle.UnpicklingError, match="refusing"):
        load_pls_result(path)


def _result(bsr, design):
    n = len(design)
    return PLSResult(
        "x",
        np.zeros((n, len(bsr))),
        np.asarray(bsr, float),
        0.01,
        np.asarray(design, float),
        np.zeros(n),
        10,
        10,
    )


def test_orientation_sign_follows_design_scores():
    cond = pd.Series(["rest", "rest", "calc", "calc"])
    assert orientation_sign(_result([1], [-1, -1, 1, 1]), cond, "calc", "rest") == 1
    assert orientation_sign(_result([1], [1, 1, -1, -1]), cond, "calc", "rest") == -1


def test_overlap_known_values():
    a = np.array([3, -3, 3, 0, 0])
    b = np.array([3, 3, 0, 3, 0])
    out = overlap(a, b, threshold=2)
    assert out["n_overlap"] == 2
    assert out["dice"] == pytest.approx(2 * 2 / 6)
    assert out["dice_same_sign"] == pytest.approx(2 * 1 / 6)
    assert out["jaccard"] == pytest.approx(2 / 4)


def test_match_observations_identifies_rows():
    voxels = pd.DataFrame(
        [
            {"subj_id": s, "dataset": d, "condition": c, "model": "m", "vox_id": v, "CMRO2": val + v}
            for s, base in [("s1", 0), ("s2", 100)]
            for d, c, val in [("test_set", "rest", base), ("calculation", "calc", base + 50)]
            for v in range(3)
        ]
    )
    X = np.array([[150, 151, 152], [0, 1, 2]], dtype=float)  # s2 calc, s1 rest
    res = PLSResult("x", X, np.zeros(3), 0.01, np.zeros(2), np.zeros(2), 1, 1)
    obs = match_observations(res, voxels, "CMRO2", "m", ["test_set", "calculation"], ["s1", "s2"])
    assert obs[["subj_id", "condition"]].to_numpy().tolist() == [["s2", "calc"], ["s1", "rest"]]

    res_bad = PLSResult("x", X + 0.5, np.zeros(3), 0.01, np.zeros(2), np.zeros(2), 1, 1)
    with pytest.raises(DataValidationError):
        match_observations(res_bad, voxels, "CMRO2", "m", ["test_set", "calculation"], ["s1", "s2"])


def test_public_summary_keeps_results_but_not_the_input_maps(pyls_file, tmp_path):
    import pandas as pd

    from synthcmro2.figures.pls import match_observations, save_pls_summary

    result = load_pls_result(pyls_file)
    obs = pd.DataFrame({"subj_id": ["a", "b", "a", "b"], "condition": ["rest", "rest", "calc", "calc"]})
    path = save_pls_summary(result, obs, tmp_path / "public" / "result.npy")
    assert path.suffix == ".npz" and "X" not in np.load(path).files
    public = load_pls_result(tmp_path / "public" / "result.npy")  # falls back to the .npz summary
    assert public.X is None
    np.testing.assert_allclose(public.bsr, result.bsr)
    np.testing.assert_allclose(public.brain_scores, result.brain_scores)
    assert public.p_value == result.p_value
    pd.testing.assert_frame_equal(match_observations(public, None, "CMRO2", "m", [], []), obs)
