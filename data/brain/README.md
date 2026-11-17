# Brain files used by the figures and models

| File | Content | Source |
|---|---|---|
| `task-all_space-MNI152NLin2009cAsym_res-4_SNR_YEO_group_mask.nii.gz` | conservative cortical gray matter mask (8155 voxels) on the 4-mm model grid (48 × 56 × 48); defines `vox_id` | Epp et al., Nat Neurosci 2026 |
| `tpl-MNI152NLin2009cAsym_res-04_desc-bet_T1w.nii` | brain-extracted MNI152NLin2009cAsym T1-weighted template, 4 mm (background of Figure 3b) | TemplateFlow |
| `atlas-MMPGlasser2016_space-MNI152NLin2009cAsym_res-4mm_desc-cropped.nii.gz` | Glasser (HCP-MMP1.0) parcellation in MNI152NLin2009cAsym at 4 mm, cropped to the model grid (`synthcmro2 crop-atlas`) | Glasser et al., Nature 2016 |
| `Glasser_2016.32k.L.label.gii` | Glasser parcellation on the fs_LR 32k left surface | Glasser et al., Nature 2016 |
| `fs_LR.32k.L.inflated.surf.gii` | inflated fs_LR 32k left surface (surface panels) | Human Connectome Project |

The atlas and surface files are redistributed under the terms of their sources; cite them when you use them.
