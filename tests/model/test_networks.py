"""Networks and losses (need PyTorch; skipped otherwise).

Set SYNTHCMRO2_FUNCTIONS=/path/to/p_syntheticCMRO2_functions.py to also check that the original classes
have the same state_dict and give the same output for the same weights.
"""

from __future__ import annotations

import importlib.util
import os

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("pytorch_msssim")

from synthcmro2.model.losses import discriminator_loss, generator_loss  # noqa: E402
from synthcmro2.model.networks import Discriminator, UNet, n_parameters  # noqa: E402


@pytest.mark.parametrize("n", [1, 2, 4])
def test_parameter_counts_match_training_logs(n):
    # training logs: 1 input G 22,832,428 / D 1,374,017; 2 inputs G 22,834,222 / D 1,375,745
    assert n_parameters(UNet(n)) == 22_832_428 + 1794 * (n - 1)
    assert n_parameters(Discriminator(n)) == 1_374_017 + 1728 * (n - 1)


def test_output_shape():
    gen, disc = UNet(2).eval(), Discriminator(2).eval()
    x = torch.randn(1, 2, 48, 56, 48)
    with torch.no_grad():
        y = gen(x)
        assert y.shape == (1, 1, 48, 56, 48)
        assert disc(x, y).shape == (1, 1)


def test_losses_match_notebook_formula():
    from pytorch_msssim import ssim

    torch.manual_seed(0)
    y, y_hat, logits = torch.rand(2, 1, 24, 24, 24) * 200, torch.rand(2, 1, 24, 24, 24) * 200, torch.randn(2, 1)
    bce = torch.nn.BCEWithLogitsLoss()
    expected = (
        torch.nn.functional.l1_loss(y_hat, y)
        + 0.01 * bce(logits, torch.ones_like(logits))
        + 0.5 * (1 - ssim(y_hat, y, data_range=float(y.max() - y.min()), size_average=True))
    )
    got = generator_loss(y, y_hat, logits, {"l1": 1.0, "adversarial": 0.01, "ssim": 0.5}, bce)
    assert torch.isclose(got, expected)
    d = discriminator_loss(logits, -logits, bce)
    assert torch.isclose(d, 0.5 * (bce(logits, torch.ones_like(logits)) + bce(-logits, torch.zeros_like(logits))))


@pytest.mark.skipif(not os.environ.get("SYNTHCMRO2_FUNCTIONS"), reason="SYNTHCMRO2_FUNCTIONS not set")
def test_identical_to_original_classes():
    spec = importlib.util.spec_from_file_location("orig", os.environ["SYNTHCMRO2_FUNCTIONS"])
    orig = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(orig)
    for n in (1, 2):
        for new_cls, old_cls, args in (
            (UNet, orig.UNet, (1, n, 48, 56, 48)),
            (Discriminator, orig.Discriminator, None),
        ):
            old, new = old_cls(in_channels=n).eval(), new_cls(n).eval()
            assert list(old.state_dict()) == list(new.state_dict())
            new.load_state_dict(old.state_dict())
            if args:
                x = torch.randn(*args)
                with torch.no_grad():
                    assert torch.allclose(old(x), new(x))
