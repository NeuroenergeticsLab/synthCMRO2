"""Load and validate config/model.yaml.

Everything the code needs (paths, participant lists, file patterns, hyperparameters, seeds) comes
from this typed view; nothing is hard-coded elsewhere. Validation fails loudly on configurations
that would silently produce wrong models, for example a test participant who is also a training
participant, or an input order that does not match the saved checkpoints.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .naming import MODALITIES, checkpoint_name

ROLES = ("reduction", "comparison")


@dataclass(frozen=True)
class Cohort:
    name: str
    train_ids: tuple[str, ...]
    fold1_validation: tuple[str, ...] | None = None
    # Participants kept in train_ids (so the folds stay as in training) but left out of every fold's training
    # and validation set and of the z-score statistics, e.g. sub-stab035 with an empty reference CMRO2 map.
    exclude_ids: tuple[str, ...] = ()

    @property
    def used_ids(self) -> tuple[str, ...]:
        """Training participants that are actually used (train_ids without exclude_ids)."""
        return tuple(s for s in self.train_ids if s not in self.exclude_ids)


@dataclass(frozen=True)
class ModelSpec:
    """One input configuration (one set of fold checkpoints)."""

    key: str
    inputs: tuple[str, ...]
    role: str = "reduction"
    cohort: str = "main"
    epochs: int | None = None  # None: training.epochs
    augmentation: str = "default"
    output_subdir: str = "qmri"
    suffix_masked_target: bool = False
    table_label: str | None = None  # model name in the figure tables (e.g. 'CBF & T2*')

    @property
    def name(self) -> str:
        """File name used by the notebooks, e.g. '_OmitCBV_OmitT2_OmitT1' (unchanged, so old checkpoints load)."""
        return checkpoint_name(self.inputs)

    @property
    def label(self) -> str:
        return "&".join(self.inputs)


@dataclass(frozen=True)
class TestDataset:
    key: str
    data_dir: Path
    condition: str
    ids: tuple[str, ...]
    models: tuple[str, ...]
    output_dir: Path
    table_name: str  # dataset name in the figure tables (default: the key, e.g. 'calculation')
    participants_info: Path | None = None
    export_models: tuple[str, ...] = ()
    # out_of_fold: training participants, each predicted by the fold model for which it was a validation
    # participant (never by a model trained on it); used to calibrate the age dependence of the model error
    out_of_fold: bool = False
    export_voxels: bool = True  # False: participant metrics only (no rows in the voxel table)


@dataclass(frozen=True)
class Config:
    root: Path
    training_dir: Path
    gm_mask: Path
    models_dir: Path
    selection_dir: Path
    results_dir: Path
    cohorts: dict[str, Cohort]
    files: dict[str, str]
    outputs: dict[str, str]
    checkpoints: dict[str, str]
    training: dict[str, Any]
    augmentation: dict[str, list[dict[str, Any]]]
    models: dict[str, ModelSpec]
    fixed_folds: dict[str, int]
    testing: dict[str, Any]
    datasets: dict[str, TestDataset] = field(default_factory=dict)
    grid: dict[str, Any] = field(default_factory=lambda: {"tolerance_mm": 0.5, "output_affine": "mask"})

    def checkpoint(self, kind: str, model: ModelSpec, fold: int) -> Path:
        return self.models_dir / self.checkpoints[kind].format(name=model.name, fold=fold)

    def epochs(self, model: ModelSpec) -> int:
        return int(model.epochs or self.training["epochs"])

    def model_by_name(self, name: str) -> ModelSpec:
        for spec in self.models.values():
            if spec.name == name:
                return spec
        raise KeyError(f"no model with file name {name!r} in the config")


class ConfigError(ValueError):
    """Raised when the configuration is inconsistent."""


def _path(root: Path, value: str | None) -> Path:
    if not value:
        raise ConfigError("missing path")
    p = Path(value).expanduser()
    return p if p.is_absolute() else (root / p).resolve()


def _unique(ids: list[str], what: str) -> tuple[str, ...]:
    dup = sorted({i for i in ids if ids.count(i) > 1})
    if dup:
        raise ConfigError(f"duplicated participants in {what}: {dup}")
    return tuple(ids)


def load_config(config_file: str | Path) -> Config:
    """Read the YAML file; relative paths are resolved against the parent of its folder."""
    config_file = Path(config_file).resolve()
    raw = yaml.safe_load(config_file.read_text(encoding="utf8"))
    return parse_config(raw, root=config_file.parent.parent)


def parse_config(raw: dict[str, Any], root: Path) -> Config:
    p = raw["paths"]
    cohorts = {
        k: Cohort(
            k,
            _unique(list(v["train_ids"]), f"cohorts.{k}.train_ids"),
            tuple(v["fold1_validation"]) if v.get("fold1_validation") else None,
            tuple(v.get("exclude_ids") or ()),
        )
        for k, v in raw["cohorts"].items()
    }
    for c in cohorts.values():
        unknown = sorted(set(c.exclude_ids) - set(c.train_ids))
        if unknown:
            raise ConfigError(f"cohorts.{c.name}.exclude_ids not in train_ids: {unknown}")
    files = dict(raw["files"])
    missing = [m for m in (*MODALITIES, "CMRO2") if m not in files]
    if missing:
        raise ConfigError(f"files: no pattern for {missing}")

    models = {}
    for key, v in raw["models"].items():
        spec = ModelSpec(key=key, inputs=tuple(v["inputs"]), **{k: v[k] for k in v if k != "inputs"})
        if spec.role not in ROLES:
            raise ConfigError(f"models.{key}.role must be one of {ROLES}")
        if spec.cohort not in cohorts:
            raise ConfigError(f"models.{key}.cohort {spec.cohort!r} is not defined")
        if spec.augmentation not in raw["augmentation"]:
            raise ConfigError(f"models.{key}.augmentation {spec.augmentation!r} is not defined")
        spec.name  # validates the inputs (known modalities, channel order)  # noqa: B018
        models[key] = spec
    names = [m.name for m in models.values()]
    if len(set(names)) != len(names):
        raise ConfigError("two models have the same inputs")

    fixed = {k: int(v) for k, v in (raw.get("selection", {}).get("fixed_folds") or {}).items()}
    n_folds = int(raw["training"]["n_folds"])
    for k, v in fixed.items():
        if k not in models or not 1 <= v <= n_folds:
            raise ConfigError(f"selection.fixed_folds: invalid entry {k}: {v}")

    testing = dict(raw.get("testing", {}))
    if testing.get("zscore_stats", "all_training") not in ("all_training", "fold"):
        raise ConfigError("testing.zscore_stats must be 'all_training' or 'fold'")
    datasets = {}
    for key, v in (testing.pop("datasets", None) or {}).items():
        sel = v.get("models", "main")
        keys = [m for m, s in models.items() if s.cohort == "main"] if sel == "main" else list(sel)
        unknown = [m for m in keys if m not in models]
        if unknown:
            raise ConfigError(f"testing.datasets.{key}: unknown models {unknown}")
        export = v.get("export_models", keys)
        if not set(export) <= set(keys):
            raise ConfigError(f"testing.datasets.{key}.export_models must be a subset of its models")
        missing_label = [m for m in export if not models[m].table_label]
        if missing_label:
            raise ConfigError(f"models {missing_label} are exported but have no table_label")
        out_of_fold = bool(v.get("out_of_fold", False))
        if out_of_fold:
            # training participants of the models' cohort; each is predicted by its validation fold's model
            model_cohorts = {models[m].cohort for m in keys}
            if len(model_cohorts) != 1:
                raise ConfigError(f"testing.datasets.{key}: out_of_fold models must share one cohort")
            cohort = cohorts[model_cohorts.pop()]
            ids = list(v.get("ids") or cohort.used_ids)
            outside = sorted(set(ids) - set(cohort.used_ids))
            if outside:
                raise ConfigError(f"testing.datasets.{key}: out_of_fold ids not used in training of {cohort.name}: "
                                  f"{outside}")  # fmt: skip
            if not v.get("output_dir"):
                raise ConfigError(f"testing.datasets.{key}: out_of_fold needs its own output_dir")
            data_dir = v.get("data_dir", p["training_dir"])
        else:
            ids, data_dir = list(v["ids"]), v["data_dir"]
        ds = TestDataset(
            key=key,
            data_dir=_path(root, data_dir),
            condition=v.get("condition", "rest"),
            ids=_unique(ids, f"testing.datasets.{key}.ids"),
            models=tuple(keys),
            output_dir=_path(root, v.get("output_dir", data_dir)),
            table_name=v.get("table_name", key),
            participants_info=_path(root, v["participants_info"]) if v.get("participants_info") else None,
            export_models=tuple(export),
            out_of_fold=out_of_fold,
            export_voxels=bool(v.get("export_voxels", True)),
        )
        # no participant may be tested with a model that was trained on them (out-of-fold predictions use, for
        # each participant, the one fold model that did not train on it; see predict.out_of_fold_folds)
        for m in keys if not out_of_fold else []:
            leak = sorted(set(ds.ids) & set(cohorts[models[m].cohort].train_ids))
            if leak:
                raise ConfigError(f"testing.datasets.{key}: training participants of {m} in the test set: {leak}")
        datasets[key] = ds

    return Config(
        root=root,
        training_dir=_path(root, p["training_dir"]),
        gm_mask=_path(root, p["gm_mask"]),
        models_dir=_path(root, p["models_dir"]),
        selection_dir=_path(root, p["selection_dir"]),
        results_dir=_path(root, p.get("results_dir", "results")),
        cohorts=cohorts,
        files=files,
        outputs=dict(raw["outputs"]),
        checkpoints=dict(raw["checkpoints"]),
        training=dict(raw["training"]),
        augmentation={k: list(v) for k, v in raw["augmentation"].items()},
        models=models,
        fixed_folds=fixed,
        testing=testing,
        datasets=datasets,
        grid=_grid(raw.get("grid") or {}),
    )


def _grid(raw: dict[str, Any]) -> dict[str, Any]:
    grid = {"tolerance_mm": 0.5, "output_affine": "mask", **raw}
    if grid["output_affine"] not in ("mask", "source"):
        raise ConfigError("grid.output_affine must be 'mask' or 'source'")
    if not 0 <= float(grid["tolerance_mm"]) < 4:
        raise ConfigError("grid.tolerance_mm must be in [0, 4) (voxel size 4 mm)")
    return grid
