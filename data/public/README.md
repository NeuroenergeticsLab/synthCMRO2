# Public data

Everything needed to reproduce the figures, tables and statistics of the manuscript, without individual
images. Individual voxel values, individual maps and the ages of participants outside the test set are not
shared (see the Data Sharing statement of the article). The tables were written with
`synthcmro2 public-tables` from the individual tables and already exclude the six Glasser parcels per
hemisphere with outlying analytical CMRO₂ (TE1m, TE2a, TE1p, TE2p, a10p and 11l; 194 of 8155 gray matter
voxels).

Units: CMRO₂ in µmol/100 g/min. Missing values are empty (CSV) or `n/a` (TSV).

## participants.tsv

One row per analyzed participant (n = 131).

| Column | Description |
|---|---|
| `participant_id` | pseudonymized ID, as in all other tables |
| `cohort` | source dataset (Table 1): Young-adults I, Young-adults II, Older-adults I, Older-adults II, Same-protocol, Cross-vendor |
| `role` | `training` (n = 54), `test` (n = 41), `same-protocol` (n = 24) or `cross-vendor` (n = 12) |
| `sex` | `f` or `m` |
| `age` | years; `n/a` for the training participants of Older-adults I and II and for the cross-vendor dataset |
| `calculation_task` | test participant with calculation-task data (Figure 3) |
| `resting_state_fmri` | test participant with resting-state fMRI (fMRI model, Figure 2) |
| `second_scan` | test participant with a second within-session scan (Figure S2) |

## participant_metrics.csv

Participant-wise agreement between synthetic and analytical CMRO₂ across gray matter voxels; one row per
participant, dataset, condition and model (Figures 2–4, Tables S2 and S4).

| Column | Description |
|---|---|
| `subj_id`, `sex`, `age` | as in participants.tsv |
| `dataset` | `test_set`, `calculation` (Young-adults I test participants during the task), `fmri_subset`, `replication` (same-protocol dataset), `reproducibility` (cross-vendor dataset) |
| `condition` | `rest` or `calc` |
| `model` | `reference` (full model: CBV, CBF, T2, T2*), `wo CBV` (CBF, T2, T2*), `CBF & T2*`, `CBF`, `T1` (T1-weighted), `fMRI` (degree centrality) |
| `r`, `p`, `variance`, `adj_r2` | Pearson r, its P value, R² and adjusted R² of the linear regression of synthetic on analytical CMRO₂ |
| `slope`, `CI_low`, `CI_up` | regression slope and 90% CI of the intercept |
| `MAE` | mean absolute error across voxels |
| `MAE_parcel` | mean absolute error between Glasser parcel medians (Table S2, parcel level) |
| `CMRO2`, `synthCMRO2` | gray matter median of analytical and synthetic CMRO₂ (age-related decline, Figure 4b) |

## group_voxel_values.csv.gz

Group maps: one row per dataset, condition, model and gray matter voxel (Figures 2–4).

| Column | Description |
|---|---|
| `dataset`, `condition`, `model` | as above |
| `vox_id` | voxel index in the gray matter mask `data/brain/task-all_space-MNI152NLin2009cAsym_res-4_SNR_YEO_group_mask.nii.gz` (C order of the mask's nonzero voxels, as `nilearn.maskers.NiftiMasker`) |
| `CMRO2`, `synthCMRO2` | voxel-wise median across participants (non-positive values excluded per participant) |
| `n_subjects` | participants contributing to the voxel |
| `CMRO2_sd` | between-participant SD of analytical CMRO₂ (Table S2) |

## Other files

| File | Content |
|---|---|
| `group_parcel_sd.csv` | between-participant SD of analytical Glasser parcel medians per dataset and condition (Table S2, parcel level) |
| `parcel_icc.csv` | test-retest ICC(2,1) per Glasser parcel for analytical CMRO₂ and each model (`n_targets`: participants; Figure S2) |
| `age_calibration.csv` | age calibration fitted in the out-of-fold predictions of the 54 training participants: model error = `a` + `b` × age (Figure 4b) |
| `pls/*.npz` | mean-centered PLS of calculation versus rest (Figure 3b), first latent variable: bootstrap ratios per voxel (7961 voxels after the parcel exclusion), permutation P value, design and brain scores with the participant and condition of each observation; the input maps are not included |
| `model_selection/` | cross-validated validation performance of every input configuration and the selected fold per model (Table S3) |
