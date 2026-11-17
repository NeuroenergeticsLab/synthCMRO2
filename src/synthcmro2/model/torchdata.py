"""PyTorch dataset, MONAI augmentation and seeding."""

from __future__ import annotations

import random
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .data import SampleLoader


def seed_everything(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 (MONAI and legacy code draw from the global NumPy state)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def build_augmentation(spec: list[dict[str, Any]], seed: int):
    """MONAI Compose from the config list; seeded (the notebooks left MONAI's random state unseeded)."""
    import monai.transforms as mt

    transforms = []
    for item in spec:
        kwargs = {k: (tuple(v) if isinstance(v, list) and k != "keys" else v) for k, v in item.items() if k != "name"}
        transforms.append(getattr(mt, item["name"])(**kwargs))
    compose = mt.Compose(transforms)
    compose.set_random_state(seed=seed)
    return compose


def _plain(t) -> torch.Tensor:
    t = t.as_tensor() if hasattr(t, "as_tensor") else torch.as_tensor(t)  # MONAI MetaTensor -> Tensor
    return t.float().contiguous()


class CMRO2Dataset(Dataset):
    """(inputs, target) tensors per participant; augmentation (if any) after normalization, as in the notebooks."""

    def __init__(self, loader: SampleLoader, sids: Sequence[str], transform=None):
        self.loader, self.sids, self.transform = loader, list(sids), transform

    def __len__(self) -> int:
        return len(self.sids)

    def __getitem__(self, i: int):
        x, y = self.loader(self.sids[i])
        x, y = torch.from_numpy(x), torch.from_numpy(y)
        if self.transform is not None:
            out = self.transform({"inputs": x, "targets": y})
            x, y = out["inputs"], out["targets"]
        return _plain(x), _plain(y)
