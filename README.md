# synthCMRO2: synthetic brain oxygen metabolism from two MRI scans

This is the code for the manuscript *Generative Deep Learning Maps Brain Oxygen Metabolism and Its Age-Related
Decline from Two Standard Noninvasive MRI Scans* (Castrillón, Marcolongo, et al.):
[medRxiv preprint](https://doi.org/10.64898/2026.11.17.26368289) (link to the published article to follow).
It also provides the trained models and the data behind every figure and table.

A conditional generative adversarial network synthesizes the cerebral metabolic rate of oxygen (CMRO₂) from
cerebral blood flow (CBF, arterial spin labeling) and T2* (multi-echo gradient echo), without contrast agent or
blood sampling. The reference standard is analytical CMRO₂ from multiparametric quantitative BOLD (mqBOLD).

This repository lets you:

1. **reproduce every figure, table and statistic** of the article from the shared data (no GPU, minutes);
2. **apply the trained models** to your own CBF and T2* maps;
3. **rerun the full pipeline** (training, model selection, prediction) on mqBOLD data.

## Install

Python ≥ 3.11.

```bash
git clone https://github.com/NeuroenergeticsLab/synthCMRO2.git && cd synthCMRO2
pip install -e ".[figures,dev]"      # 1. figures (and tests)
pip install -e ".[model,figures]"    # 2.–3. models: install PyTorch for your CUDA version first (used: 2.7.0+cu118)
```

`make install-figures` or `make install` do the same with [uv](https://docs.astral.sh/uv/). Run the tests with
`pytest` (the model tests are skipped without PyTorch).

## 1. Reproduce the figures and statistics

```bash
synthcmro2 figures      # Figures 2–4 and S2 -> results/manuscript/figures/, statistics -> results/manuscript/stats/
synthcmro2 benchmark    # Table S2 -> results/manuscript/stats/noise_benchmark.csv
```

The commands read the shared tables in `data/public/` through `config/figures.yaml`. Every number in the
Results is in one CSV file in `results/manuscript/stats/` (for example `fig2a_group_agreement.csv`: group-level
r with 95% CI, slope and MAE; `fig4b_age_slopes.csv`: age-related decline), and your statistics should match
those in `reference/stats/` exactly; `reference/figures/` holds the published figures. Table S3 is
`data/public/model_selection/cv_selection.csv`. `results/manuscript/provenance.json` records the software
versions and input checksums of each run.

## 2. Apply the trained models

Download the trained generators ([model archive URL / DOI]) and unpack them into `models/`:

```
models/generators/_OmitCBV_OmitT2_OmitT1/trained_generator_model_RestCondition_1_OmitCBV_OmitT2_OmitT1.pth   # CBF&T2*
models/model_selection/cv_selection.csv                                                                    # selected fold per model
```

Then, for one participant:

```bash
synthcmro2 model apply --input CBF=sub-01_cbf.nii.gz T2S=sub-01_t2star.nii.gz --out sub-01_synthetic_cmro2.nii.gz
```

The input maps must be

- quantitative and in the units used for training: CBF in ml/100 g/min, T2* in ms;
- in MNI152NLin2009cAsym space at 4-mm resolution, on the grid of the gray matter mask
  `data/brain/task-all_space-MNI152NLin2009cAsym_res-4_SNR_YEO_group_mask.nii.gz` (48 × 56 × 48 voxels).

The output is synthetic CMRO₂ in µmol/100 g/min, valid inside that gray matter mask. Other models take other
inputs: `--model reference` (`CBV= CBF= T2= T2S=`), `--model without_cbv` (`CBF= T2= T2S=`) or `--model cbf`
(`CBF=`). The CBF&T2* model is the one we recommend.

The models were trained on healthy adults scanned at one center with one mqBOLD protocol (Philips 3 T,
pCASL, multi-echo gradient echo). Agreement was lower with another vendor (Figure 4), and other protocols,
such as gradient-echo sequences optimized for susceptibility mapping, may require fine-tuning. Validate the
maps on your own data before interpreting them.

## 3. Full pipeline

With mqBOLD data organized as described in `config/model.yaml` (`paths` and `files`; by default
`data/mqbold/<training|testing>/<participant>/qmri/...`):

```bash
synthcmro2 model check                        # configuration and every input file
synthcmro2 model train --models cbf_t2s --device cuda:0
synthcmro2 model select --device cuda:0       # fold and input configuration chosen on validation data
synthcmro2 model predict --device cuda:0      # test datasets and out-of-fold training predictions
synthcmro2 model export --tag 20261107        # individual tables -> results/export/
synthcmro2 pls --config config/figures_individual.yaml
synthcmro2 figures --config config/figures_individual.yaml --exclude-parcels TE1m TE2a TE1p TE2p a10p 11l
synthcmro2 public-tables --config config/figures_individual.yaml --exclude-parcels TE1m TE2a TE1p TE2p a10p 11l --out data/public
```

`public-tables` writes the shareable tables of `data/public/` from the individual ones. The notebook
`notebooks/synthCMRO2.ipynb` runs the same steps; the `Makefile` lists shortcuts for all of them, and
`synthcmro2 <command> --help` describes every option.

## Data

| Dataset (Table 1) | Participants | Availability |
|---|---|---|
| Young-adults I | 30 | OpenNeuro, [ds004873](https://openneuro.org/datasets/ds004873/) |
| Young-adults II | 29 | OpenNeuro, upon release of the corresponding preprint (early 2027) |
| Older-adults I and II | 36 | on reasonable request to the principal investigator |
| Same-protocol | 24 | on request to the principal investigator |
| Cross-vendor | 12 | OpenNeuro, [ds008786](https://openneuro.org/datasets/ds008786) |

`data/public/` contains no individual images or voxel values: group maps (voxel-wise medians across
participants), participant-wise statistics, parcel-wise test-retest reliability, the fitted age calibration and
the PLS results. Age is withheld for the older-adult training participants and the cross-vendor dataset, and
shared for every test participant, whose ages enter the age analysis.
`data/public/README.md` describes every column. `data/brain/` contains the gray matter mask, the MNI template, the Glasser atlas and the fs_LR
surface used for the figures (sources in `data/brain/README.md`).

## Repository

```
config/        model.yaml (models, participants, training), figures.yaml (public data),
               figures_individual.yaml (individual tables)
data/public/   shared tables (Figures 2–4, S2, Tables S2–S4)
data/brain/    mask, template, atlas and surface
reference/     published figures and the statistics produced from data/public/ (to check a run)
notebooks/     the full workflow as a notebook
src/synthcmro2/model/     training, selection, prediction, application, export (PyTorch, MONAI)
src/synthcmro2/figures/   figures, statistics, PLS, benchmark, public tables
tests/
```

## Citation

If you use this code, the models or the data, please cite the article ([medRxiv preprint](https://doi.org/10.64898/2026.11.17.26368289)) and the datasets
above.

## License

Code: MIT (`LICENSE`). Trained models and data: [license of the model archive and data, e.g. CC BY 4.0].
