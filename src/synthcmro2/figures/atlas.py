"""Crop an image on the full 4-mm MNI152NLin2009cAsym grid (48 x 57 x 48) to the training grid (48 x 56 x 48).

Drops the last y slice, like ``fslroi in out 0 -1 0 56 0 -1``: voxel values and the affine are unchanged, so
no interpolation is involved. Used once to prepare the Glasser atlas of the surface maps and Figure S2
(``synthcmro2 crop-atlas <image> <reference_mask> <output>``).
"""

from __future__ import annotations

import numpy as np


def crop_to_mask_grid(image: str, reference: str, output: str) -> tuple[tuple[int, ...], tuple[int, ...]]:
    import nibabel as nib

    img, ref = nib.load(image), nib.load(reference)
    nx, ny, nz = ref.shape[:3]
    cropped = img.slicer[:nx, :ny, :nz]  # starts at index 0, so the affine is unchanged
    if cropped.shape[:3] != ref.shape[:3] or not np.allclose(cropped.affine, ref.affine):
        raise ValueError(f"{image}: cannot be cropped onto the grid of {reference}")
    if np.any(np.asarray(img.dataobj)[:, ny:, :]):
        raise ValueError(f"{image}: the dropped slice contains non-zero values")
    nib.save(cropped, output)
    return img.shape, cropped.shape
